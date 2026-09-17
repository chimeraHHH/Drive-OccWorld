"""Fixed-budget source displacement pretraining; no O model or occupancy GT load.

Only the authenticated native cache's prev_bev_input[:, -1] enters the head.
Sparse future rigid-box displacements enter the loss/evaluator separately.
No resume, best checkpoint, sample replacement, or automatic retry is supported.
Preflight performs four updates on the first 16 examples of the seed11 order;
its weights are discarded rather than saved as a training checkpoint.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np

SCHEMA = 'source-motion-training-v1'
SHAPE = (200, 200, 16)
EXTENT = (-51.2, -51.2, -5., 51.2, 51.2, 3.)
GROUPS = ('stationary', 'ambiguous', 'moving')
TRAINING = dict(seed=11, train_samples=512, development_samples=200,
                passes=4, accumulate=4, updates=512, lr=1e-3,
                weight_decay=.01, grad_clip=10., warmup_updates=25,
                final_lr_ratio=.1, smooth_l1_beta_m=.5,
                optimizer='AdamW', betas=[.9, .999], eps=1e-8,
                lr_schedule='zero_based_u: u<25=>lr*(0.1+0.9*u/25); otherwise lr*(0.1+0.45*(1+cos(pi*(u-25)/486)))',
                sample_order='continuous_numpy_RandomState11_permutation_each_pass',
                loss='point_xyz_mean_then_object_mean_within_group_then_present_group_mean_then_valid_horizon_mean')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def planned_orders():
    rng = np.random.RandomState(11)
    return [rng.permutation(512).tolist() for _ in range(4)]


def schedule_lr(update):
    """Zero-based update: 0=.1lr, 25=lr, 511=.1lr; no off-by-one step()."""
    require(isinstance(update, int) and 0 <= update < 512, 'Invalid update')
    if update < 25:
        return 1e-3 * (.1 + .9 * update / 25)
    return 1e-3 * (.1 + .9 * .5 * (1 + math.cos(math.pi * (update - 25) / 486)))


def protocol_template():
    """Draft keys for the coordinator to bind and freeze; never auto-freezes."""
    return dict(schema=SCHEMA, status='REVIEW_REQUIRED', training=dict(TRAINING),
        source_shape_xyz=list(SHAPE), extent_xyz=list(EXTENT),
        numerical_policy=dict(dtype='float32', matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False),
        sources_sha256={name: None for name in
            ('train_source_motion_v1.py', 'motion_prediction_head.py', 'native_state_cache.py')},
        cache_index_sha256=dict(train=None, development=None),
        labels=dict(manifest_sha256=None, complete_sha256=None),
        resources=dict(preflight=dict(max_seconds=600, max_allocated_gib=16),
                       train=dict(max_seconds=7200, max_allocated_gib=16)))


def source_path(name):
    require(Path(name).name == name, 'Source bindings must use basenames')
    candidates = [Path(__file__).parent / name,
                  Path(__file__).parent.parent / 'm0_improvement_20260915' / name]
    found = [p for p in candidates if p.is_file()]
    require(bool(found), 'Missing bound source: ' + name)
    return found[0]


def import_bound(name, expected):
    path = source_path(name + '.py')
    require(sha(path) == expected, 'Import source SHA mismatch: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_protocol(a):
    protocol = read(a.protocol)
    require(protocol['schema'] == SCHEMA, 'Wrong training protocol schema')
    require(protocol['status'] == 'FROZEN', 'A frozen protocol is required')
    require(protocol['training'] == TRAINING, 'Fixed training recipe changed')
    require(protocol['source_shape_xyz'] == list(SHAPE) and
            protocol['extent_xyz'] == list(EXTENT), 'Source geometry changed')
    require(protocol['numerical_policy'] == dict(dtype='float32', matmul_tf32=False,
            cudnn_tf32=True, cudnn_benchmark=False), 'Numerical policy changed')
    required = {Path(__file__).name, 'motion_prediction_head.py', 'native_state_cache.py'}
    require(required <= set(protocol['sources_sha256']), 'Incomplete source bindings')
    for name, digest in protocol['sources_sha256'].items():
        require(sha(source_path(name)) == digest, 'Source mismatch: ' + name)
    mode = 'preflight' if a.preflight else 'train'
    require(protocol['resources'][mode] == dict(max_seconds=a.max_seconds,
            max_allocated_gib=a.max_allocated_gib), 'CLI resource cap must equal protocol')
    require(a.max_seconds > 0 and a.max_allocated_gib > 0, 'Positive resource ceilings required')
    return protocol


def cache_index(path, split, protocol):
    root = Path(path).resolve()
    index_path = root / 'index.json'
    require(sha(index_path) == protocol['cache_index_sha256'][split], 'Wrong native cache index')
    index, complete = read(index_path), read(root / 'complete.json')
    require(index['schema'] == 'm0-native-state-cache-v1' and index['status'] == 'COMPLETE', 'Incomplete native cache')
    require(index['split'] == split and complete['status'] == 'COMPLETE_NATIVE_STATE_CACHE', 'Cache role/status mismatch')
    require(complete['index_sha256'] == sha(index_path), 'Cache completion does not bind index')
    count = 512 if split == 'train' else 200
    rows = index['records']
    require(len(rows) == complete['samples'] == count, 'Fixed sample count mismatch')
    require(all(r['split'] == split for r in rows), 'Mixed cache split')
    require(len({r['sample_token'] for r in rows}) == count, 'Duplicate cache identity')
    return root, rows, dict(index_sha256=sha(index_path), complete_sha256=sha(root / 'complete.json'))


def labels_manifest(root, protocol):
    root = Path(root).resolve()
    binding = protocol['labels']
    require(sha(root / 'manifest.json') == binding['manifest_sha256'] and
            sha(root / 'complete.json') == binding['complete_sha256'], 'Sparse label source changed')
    manifest, complete = read(root / 'manifest.json'), read(root / 'complete.json')
    require(manifest['schema'] == 'sparse-rigid-box-motion-manifest-v1' and
            complete['schema'] == 'sparse-rigid-box-motion-complete-v1' and
            manifest['status'] == complete['status'] == 'COMPLETE', 'Wrong/incomplete sparse labels')
    require(complete['manifest_sha256'] == sha(root / 'manifest.json'), 'Label complete/manifest mismatch')
    require(len(manifest['records']) == manifest['samples'] == complete['samples'] == 712 and
            manifest['train_samples'] == 512 and manifest['development_samples'] == 200 and
            manifest['train_dev_scenes_disjoint'], 'Require complete 512+200 sparse label manifest')
    require([r['ordinal'] for r in manifest['records']] == list(range(712)) and
            [r['identity']['split'] for r in manifest['records']] == ['train'] * 512 + ['development'] * 200,
            'Sparse label manifest order changed')
    bindings = {r['identity']['sample_token']: r for r in manifest['records']}
    require(len(bindings) == 712, 'Duplicate sparse label token')
    return root, bindings


def check_identity(record, descriptor):
    for key in ('sample_token', 'scene_token', 'split', 'official_index'):
        require(record[key] == descriptor['identity'][key], 'Label/cache identity differs: ' + key)


def load_sparse(root, descriptor, record):
    check_identity(record, descriptor)
    path = (root / descriptor['file']).resolve()
    require(root in path.parents and path.suffix == '.npz', 'Invalid sparse label path')
    require(path.stat().st_size == descriptor['bytes'] and sha(path) == descriptor['sha256'], 'Sparse label size/SHA mismatch')
    with np.load(path, allow_pickle=False) as z:
        label = {k: z[k].copy() for k in z.files}
    for key in ('sample_token', 'scene_token', 'split'):
        require(str(label[key].item()) == record[key], 'NPZ identity mismatch: ' + key)
    require(str(label['schema'].item()) == 'sparse-rigid-box-motion-supervision-v1' and
            int(label['ordinal']) == descriptor['ordinal'] and
            str(label['label_source_sha256'].item()) == descriptor['label_source_sha256'], 'NPZ provenance changed')
    require(np.array_equal(label['grid_shape_xyz'], SHAPE) and np.array_equal(label['extent_xyz_m'], EXTENT), 'NPZ geometry changed')
    require(set(label) == set(descriptor['shapes']) == set(descriptor['dtypes']), 'NPZ key set differs from manifest')
    for key, value in label.items():
        require(list(value.shape) == descriptor['shapes'][key] and str(value.dtype) == descriptor['dtypes'][key],
                'NPZ descriptor shape/dtype mismatch: ' + key)
    idx, obj = label['source_flat_indices'], label['object_index']
    target, valid = label['target_displacement_m'], label['valid']
    groups, future = label['object_speed_group'], label['object_future_valid']
    n, k = len(idx), len(label['instance_tokens'])
    require(idx.dtype == np.int64 and idx.shape == (n,) and obj.dtype == np.int64 and obj.shape == (n,), 'Sparse index dtype/shape')
    require(n == 0 or (idx[0] >= 0 and idx[-1] < np.prod(SHAPE) and np.all(np.diff(idx) > 0)), 'Source indices not unique XYZ C-order')
    require(n == 0 or (obj.min() >= 0 and obj.max() < k), 'Object index out of range')
    require(np.array_equal(np.unique(obj), np.arange(k)), 'Object map not compact/retained')
    require(target.dtype == np.float32 and target.shape == (4, n, 3) and np.isfinite(target).all(), 'Target dtype/shape/nonfinite')
    require(valid.dtype == np.bool_ and valid.shape == (4, n), 'Point valid mask dtype/shape')
    require(groups.dtype == np.int8 and groups.shape == (4, k) and np.isin(groups, [-1, 0, 1, 2]).all(), 'Invalid speed group')
    require(future.dtype == np.bool_ and future.shape == (4, k), 'Object future mask dtype/shape')
    require(np.array_equal(valid, future[:, obj]) and np.array_equal(groups >= 0, future), 'Object/point valid support differs')
    require(np.all(target[~valid] == 0), 'Missing future labels not zero-filled')
    require(len(set(label['instance_tokens'].tolist())) == k, 'Duplicate instance token')
    require(label['dt_future_seconds'].dtype == np.float64 and label['dt_future_seconds'].shape == (4,), 'Horizon dt contract')
    require(np.isfinite(label['dt_future_seconds']).all() and np.all(np.diff(label['dt_future_seconds']) > 0) and
            label['dt_future_seconds'][0] > 0, 'Future time not positive/increasing')
    return label


def load_tokens(root, record, native, device):
    """Read only authenticated inputs.pt; never call load_sample or open targets.npy."""
    import torch
    directory = native._sample_directory(root, record)
    descriptor = record['files']['inputs']
    path = directory / descriptor['file']
    native._verify_file(path, descriptor)
    # Native metadata contains importable mmdet3d classes; file identity is checked first.
    inputs = torch.load(str(path), map_location='cpu', weights_only=False)
    native.validate_inputs(inputs)
    state = inputs['prev_bev_input']
    require(torch.is_tensor(state) and state.dtype == torch.float32 and
            tuple(state.shape) == (1, 1, 40000, 256), 'Native state dtype/shape changed')
    require(native._tensor_digest(state) == record['reference_bev_tensor_sha256'], 'Native measured BEV tensor SHA mismatch')
    require(bool(torch.isfinite(state).all()), 'Nonfinite observed BEV')
    tokens = state[:, -1].detach().to(device=device)
    del inputs, state
    require(not tokens.requires_grad and tuple(tokens.shape) == (1, 40000, 256), 'Unexpected trainable input')
    return tokens


def gather_sparse(prediction, label):
    import torch
    require(tuple(prediction.shape) == (1, 4, 3, *SHAPE), 'Head output XYZ layout changed')
    indices = torch.as_tensor(label['source_flat_indices'], device=prediction.device)
    return prediction[0].reshape(4, 3, -1).index_select(2, indices).permute(0, 2, 1)


def object_group_loss(sparse_prediction, label):
    """Each supported object gets equal weight within its speed group at each h."""
    import torch
    import torch.nn.functional as F
    target = torch.as_tensor(label['target_displacement_m'], device=sparse_prediction.device)
    object_index = torch.as_tensor(label['object_index'], device=sparse_prediction.device)
    point_losses = F.smooth_l1_loss(sparse_prediction, target, beta=.5, reduction='none').mean(-1)
    zero = sparse_prediction.sum() * 0.
    horizons, audit = [], []
    for h in range(4):
        valid = torch.as_tensor(label['valid'][h], device=sparse_prediction.device)
        groups = torch.as_tensor(label['object_speed_group'][h], device=sparse_prediction.device)
        nobj = len(groups)
        counts = torch.zeros(nobj, device=sparse_prediction.device, dtype=point_losses.dtype)
        sums = torch.zeros_like(counts)
        counts.index_add_(0, object_index[valid], torch.ones_like(point_losses[h, valid]))
        sums.index_add_(0, object_index[valid], point_losses[h, valid])
        means = sums / counts.clamp_min(1)
        present = []
        row = dict(horizon_seconds=(h + 1) * .5, groups={})
        for g, name in enumerate(GROUPS):
            selected = (counts > 0) & (groups == g)
            num = int(selected.sum().item())
            value = means[selected].mean() if num else zero
            row['groups'][name] = dict(objects=num, points=int(counts[selected].sum().item()),
                                       loss=None if not num else float(value.detach()))
            if num:
                present.append(value)
        row['supported'] = bool(present)
        audit.append(row)
        if present:
            horizons.append(torch.stack(present).mean())
    loss = torch.stack(horizons).mean() if horizons else zero
    return loss, dict(valid_horizons=len(horizons), empty_target=not horizons, horizons=audit)


def epe_records(sparse_prediction, label, record):
    pred = sparse_prediction.detach().cpu().numpy().astype(np.float64)
    target = label['target_displacement_m'].astype(np.float64)
    result = []
    for h in range(4):
        for obj, token in enumerate(label['instance_tokens']):
            chosen = label['valid'][h] & (label['object_index'] == obj)
            n = int(chosen.sum())
            if not n:
                continue
            delta, baseline = pred[h, chosen] - target[h, chosen], target[h, chosen]
            result.append(dict(sample_token=record['sample_token'], scene_token=record['scene_token'],
                instance_token=str(token), horizon_seconds=(h + 1) * .5,
                dt_seconds=float(label['dt_future_seconds'][h]), group=GROUPS[int(label['object_speed_group'][h, obj])],
                source_points=n, epe_3d_m=float(np.linalg.norm(delta, axis=1).mean()),
                epe_xy_m=float(np.linalg.norm(delta[:, :2], axis=1).mean()),
                zero_epe_3d_m=float(np.linalg.norm(baseline, axis=1).mean()),
                zero_epe_xy_m=float(np.linalg.norm(baseline[:, :2], axis=1).mean())))
    return result


def epe_summary(rows):
    def stats(values):
        v = np.asarray(values, np.float64)
        return dict(count=len(v), mean=None if not len(v) else float(v.mean()),
                    median=None if not len(v) else float(np.median(v)),
                    p90=None if not len(v) else float(np.quantile(v, .9)))
    out = []
    for h in (.5, 1., 1.5, 2.):
        for group in ('all', *GROUPS):
            subset = [r for r in rows if r['horizon_seconds'] == h and (group == 'all' or r['group'] == group)]
            out.append(dict(horizon_seconds=h, group=group, objects=len(subset),
                source_points=sum(r['source_points'] for r in subset),
                metrics={key: stats([r[key] for r in subset]) for key in
                         ('epe_3d_m', 'epe_xy_m', 'zero_epe_3d_m', 'zero_epe_xy_m')}))
    return dict(unit='metres', aggregation='equal anchor-instance weight after mean point EPE; no voxel weighting',
                groups=out, same_support_for_prediction_and_zero=True,
                zero_is_baseline_not_O_flow=True, no_O_EPE_claim=True,
                labels='rigid-box source-point displacement proxy; not measured dense optical/scene flow',
                unlabelled_and_overlapping_sources_excluded=True,
                confidence_interval=None, single_training_seed=11)


def state_digest(head):
    h = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode())
        h.update(json.dumps(list(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


class Budget:
    def __init__(self, a, start):
        self.a, self.start = a, start
        self.reason = None

    def signal(self, signum, frame):
        self.reason = 'signal_' + str(signum)

    def check(self):
        import torch
        if self.reason:
            raise RuntimeError(self.reason)
        require(time.monotonic() - self.start < self.a.max_seconds, 'Wall-clock ceiling reached')
        if str(self.a.device).startswith('cuda'):
            require(torch.cuda.max_memory_allocated(self.a.device) <= self.a.max_allocated_gib * 2**30,
                    'CUDA allocated-memory ceiling exceeded')


def resources(a, start):
    import torch
    cuda = str(a.device).startswith('cuda')
    return dict(elapsed_seconds=time.monotonic() - start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device) if cuda else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device) if cuda else 0,
                max_seconds=a.max_seconds, max_allocated_gib=a.max_allocated_gib)


def run(a, out, start):
    protocol = validate_protocol(a)
    train_root, train, train_receipt = cache_index(a.train_cache, 'train', protocol)
    label_root, bindings = labels_manifest(a.labels, protocol)
    for ordinal, r in enumerate(train):
        check_identity(r, bindings[r['sample_token']])
        require(bindings[r['sample_token']]['ordinal'] == ordinal, 'Training cache order differs from frozen label order')
    # Preflight does not open the dev cache/index/labels or compute a dev metric.
    dev_root, dev, dev_receipt = None, [], None
    if not a.preflight:
        require(a.dev_cache is not None, 'Formal training requires --dev-cache')
        dev_root, dev, dev_receipt = cache_index(a.dev_cache, 'development', protocol)
        for ordinal, r in enumerate(dev):
            check_identity(r, bindings[r['sample_token']])
            require(bindings[r['sample_token']]['ordinal'] == 512 + ordinal, 'Development order differs from frozen labels')
        require(not ({r['scene_token'] for r in train} & {r['scene_token'] for r in dev}), 'Train/development scenes overlap')
    native = import_bound('native_state_cache', protocol['sources_sha256']['native_state_cache.py'])
    # Needed solely for trusted input pickle metadata. No model/config/checkpoint is constructed.
    repo = Path(a.repo).resolve()
    require((repo / 'projects/mmdet3d_plugin').is_dir(), 'Invalid native repository')
    sys.path.insert(0, str(repo))
    import torch
    from torch import nn
    module = import_bound('motion_prediction_head', protocol['sources_sha256']['motion_prediction_head.py'])
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    if str(a.device).startswith('cuda'):
        torch.cuda.manual_seed_all(11)
        torch.cuda.set_device(a.device)
        torch.cuda.reset_peak_memory_stats(a.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    head = module.MotionPredictionHead(SHAPE).to(a.device).train()
    require(all(p.requires_grad for p in head.parameters()), 'Head parameter unexpectedly frozen')
    require(torch.count_nonzero(head.readout.weight).item() == 0 and torch.count_nonzero(head.readout.bias).item() == 0,
            'Final conv must initialize at exactly zero')
    initial_sha = state_digest(head)
    initial_parameters = {name: p.detach().cpu().clone() for name, p in head.named_parameters()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8,
                                 weight_decay=.01, amsgrad=False)
    require(len(optimizer.state) == 0, 'Fresh optimizer required')
    orders = planned_orders()
    limit = 4 if a.preflight else 512
    source_receipt = dict(protocol_sha256=sha(a.protocol), sources_sha256=protocol['sources_sha256'],
        train_cache=train_receipt, development_cache=dev_receipt, labels=protocol['labels'])
    manifest = dict(schema=SCHEMA, mode='preflight' if a.preflight else 'train', seed=11,
        source_receipt=source_receipt, training=TRAINING, numerical_policy=protocol['numerical_policy'],
        initial_head_sha256=initial_sha, head_parameters=sum(p.numel() for p in head.parameters()),
        trainable_names=[name for name, _ in head.named_parameters()], sample_orders_sha256=json_hash(orders),
        planned_updates=limit, input='authenticated native inputs.prev_bev_input[:, -1] only',
        full_model_or_checkpoint_loaded=False, occupancy_GT_or_predictions_loaded=False,
        future_labels_in_model_input=False, best_checkpoint_selection=False,
        preflight_weights_eligible_for_formal_initialization=False,
        source_shape_xyz=list(SHAPE), native_token_flat_order='y*X+x; output physical XYZ C-order')
    write(out / 'manifest.json', manifest)
    optimizer.zero_grad(set_to_none=True)
    budget = Budget(a, start)
    signal.signal(signal.SIGTERM, budget.signal); signal.signal(signal.SIGINT, budget.signal)
    updates, examples, micro = 0, 0, 0
    last_batch = []
    eval_rows, evaluated = [], 0
    def checkpoint(name, status):
        payload = dict(schema=SCHEMA, status=status, mode=manifest['mode'], head=head.state_dict(),
            optimizer=optimizer.state_dict(), update=updates, examples=examples,
            incomplete_accumulation_examples=micro, sample_orders=orders,
            sample_orders_sha256=json_hash(orders), head_state_sha256=state_digest(head),
            initial_head_sha256=initial_sha, sources=source_receipt,
            manifest_sha256=sha(out / 'manifest.json'), resume_supported=False,
            eligible_for_formal_initialization=updates == 512 and micro == 0,
            torch_rng_state=torch.get_rng_state(), numpy_rng_state=np.random.get_state(),
            cuda_rng_state_all=torch.cuda.get_rng_state_all() if str(a.device).startswith('cuda') else None)
        tmp = out / (name + '.tmp')
        torch.save(payload, str(tmp)); tmp.replace(out / name)
        return dict(file=name, sha256=sha(out / name), head_state_sha256=payload['head_state_sha256'])
    def sync():
        if str(a.device).startswith('cuda'):
            torch.cuda.synchronize(a.device)
    try:
        for pass_index, order in enumerate(orders):
            for ordinal in order:
                budget.check()
                if micro == 0:
                    sync(); step_start = time.monotonic(); last_batch = []
                    for group in optimizer.param_groups:
                        group['lr'] = schedule_lr(updates)
                record = train[ordinal]
                tokens = load_tokens(train_root, record, native, a.device)
                label = load_sparse(label_root, bindings[record['sample_token']], record)
                prediction = head(tokens)
                require(bool(torch.isfinite(prediction).all()), 'Nonfinite displacement prediction')
                if updates == 0 and micro == 0:
                    require(torch.count_nonzero(prediction).item() == 0, 'Initial prediction not exact zero')
                sparse = gather_sparse(prediction, label)
                loss, audit = object_group_loss(sparse, label)
                require(bool(torch.isfinite(loss)), 'Nonfinite motion loss')
                (loss / 4).backward()
                examples += 1; micro += 1
                last_batch.append(dict(ordinal=ordinal, sample_token=record['sample_token'],
                    scene_token=record['scene_token'], raw_loss=float(loss.detach()), audit=audit,
                    inputs_sha256=record['files']['inputs']['sha256'],
                    labels_sha256=bindings[record['sample_token']]['sha256']))
                del prediction, sparse, loss, tokens, label
                if micro < 4:
                    continue
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()),
                        'Missing/nonfinite parameter gradient')
                norm = float(nn.utils.clip_grad_norm_(head.parameters(), 10., error_if_nonfinite=True))
                budget.check()
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                updates += 1; micro = 0
                require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), 'Nonfinite updated weights')
                require(all(bool(torch.isfinite(state[key]).all()) for state in optimizer.state.values()
                            for key in ('exp_avg', 'exp_avg_sq')), 'Nonfinite AdamW moments')
                sync()
                row = dict(update=updates, pass_index=pass_index, examples=examples,
                    lr=optimizer.param_groups[0]['lr'], loss=float(np.mean([r['raw_loss'] for r in last_batch])),
                    preclip_grad_norm=norm, clip_factor=min(1., 10. / (norm + 1e-6)),
                    seconds=time.monotonic()-step_start, samples=last_batch,
                    sample_group_sha256=json_hash([r['sample_token'] for r in last_batch]),
                    resources=resources(a, start), all_gradients_finite=True)
                append(out / 'training.jsonl', row)
                print(json.dumps(dict(event='UPDATE', update=updates, loss=row['loss'], seconds=row['seconds']), allow_nan=False), flush=True)
                budget.check()
                if updates == limit:
                    break
            if updates == limit:
                break
        require(updates == limit and examples == limit * 4 and micro == 0, 'Fixed training budget not completed')
        parameter_max_abs_change = max(float((p.detach().cpu() - initial_parameters[name]).abs().max())
                                       for name, p in head.named_parameters())
        require(parameter_max_abs_change > 0, 'No parameter learned on the fixed examples')
        final = None if a.preflight else checkpoint('final.pth', 'FIXED_FINAL_512')
        before_eval_sha = state_digest(head)
        if not a.preflight:
            head.eval()
            with torch.no_grad():
                for record in dev:
                    budget.check(); sync(); begin = time.monotonic()
                    tokens = load_tokens(dev_root, record, native, a.device)
                    label = load_sparse(label_root, bindings[record['sample_token']], record)
                    prediction = head(tokens)
                    require(bool(torch.isfinite(prediction).all()), 'Nonfinite evaluation prediction')
                    sparse = gather_sparse(prediction, label)
                    rows = epe_records(sparse, label, record)
                    for row in rows:
                        append(out / 'development_objects.jsonl', row)
                    eval_rows.extend(rows)
                    loss, audit = object_group_loss(sparse, label)
                    require(bool(torch.isfinite(loss)), 'Nonfinite evaluation loss')
                    sync(); evaluated += 1
                    append(out / 'development_records.jsonl', dict(
                        sample_token=record['sample_token'], scene_token=record['scene_token'],
                        official_index=record['official_index'], split=record['split'],
                        inputs_sha256=record['files']['inputs']['sha256'],
                        labels_sha256=bindings[record['sample_token']]['sha256'], objects=len(rows),
                        loss=float(loss), audit=audit, seconds=time.monotonic()-begin))
                    del tokens, label, prediction, sparse, loss
                    budget.check()
            require(evaluated == 200 and state_digest(head) == before_eval_sha, 'Evaluation incomplete or weights changed')
        summary = dict(schema=SCHEMA, mode=manifest['mode'], updates=updates, examples=examples,
            evaluated_samples=evaluated, development=None if a.preflight else epe_summary(eval_rows),
            final_head_sha256=before_eval_sha, weights_changed_from_initial=before_eval_sha != initial_sha,
            parameter_max_abs_change=parameter_max_abs_change, all_gradients_finite=True,
            initial_head_sha256=initial_sha, weights_persisted=not a.preflight,
            resources=resources(a, start), preflight_is_not_scientific_result=a.preflight,
            no_O_EPE_claim=True, no_candidate_occupancy_metric=True)
        write(out / 'summary.json', summary)
        filenames = ['manifest.json', 'training.jsonl', 'summary.json']
        if not a.preflight:
            # Valid even if every sample has no label support: preserve an empty stream.
            (out / 'development_objects.jsonl').touch(exist_ok=True)
            filenames += ['final.pth', 'development_objects.jsonl', 'development_records.jsonl']
        budget.check()
        write(out / 'complete.json', dict(schema=SCHEMA,
            status='PASS_SOURCE_MOTION_PREFLIGHT' if a.preflight else 'COMPLETE_SOURCE_MOTION_TRAINING',
            mode=manifest['mode'], updates=updates, examples=examples, evaluated_samples=evaluated,
            files_sha256={name: sha(out / name) for name in filenames}, final_checkpoint=final,
            resources=resources(a, start), resume_supported=False))
    except BaseException as exc:
        failed_checkpoint = None
        try:
            if not a.preflight:
                failed_checkpoint = checkpoint('interrupted.pth', 'INCOMPLETE_NOT_RESUMABLE')
        except BaseException as checkpoint_exc:
            failed_checkpoint = dict(error=repr(checkpoint_exc))
        write(out / 'failed.json', dict(schema=SCHEMA, status='FAILED_NO_RETRY',
            error=repr(exc), updates=updates, examples=examples, evaluated_samples=evaluated,
            partial_accumulation_examples=micro, checkpoint=failed_checkpoint,
            seconds=time.monotonic()-start))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train-cache', 'labels', 'protocol', 'repo', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--dev-cache')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--max-allocated-gib', type=float, required=True)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args(argv)
    require(not (a.preflight and a.dev_cache), '--preflight forbids --dev-cache')
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        run(a, out, start)
    except BaseException as exc:
        if not (out / 'failed.json').exists():
            write(out / 'failed.json', dict(schema=SCHEMA, status='FAILED_NO_RETRY',
                error=repr(exc), phase='initialization', seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
