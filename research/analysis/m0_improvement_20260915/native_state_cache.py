"""Native M0 observed-state cache; no optimizer, common-source override, or retry.

Public API:
  build_native_model(config, checkpoint, device='cuda', repo=None) -> model
  load_sample(cache_dir, record, device='cuda') -> {inputs, targets, record}
  replay(model, sample, training=False) -> original future_pred tuple

``inputs.pt`` contains the INITIAL future_pred call boundary (CPU tensors,
metadata, prescribed future ego/action); ``targets.npy`` contains the original
segmentation separately. load_sample never reads native_preds.npy. It verifies
input/target file hashes once per unchanged local file identity in this process.
Pickle inputs are trusted only after the frozen file hash has been checked.

Extraction uses ds._prepare_data_info at the exact usable index and temporarily
observes future_pred/evaluate_occ_records; neither radar loader nor fusion is
overridden. The original methods are restored after capture. Every sample is
replayed from its persisted input before entering index.json. No missing sample
is replaced. CUDA preflight runs a last-horizon probe loss without full-GT loss.
"""

import argparse
import contextlib
import copy
import datetime
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import types

SCHEMA = 'm0-native-state-cache-v1'
M0_SHA256 = '0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
SOURCE_MODEL_SHA256 = '67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5'
_VERIFIED_FILES = {}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def event(name, **values):
    print(json.dumps(dict(event=name, utc=datetime.datetime.now(
        datetime.timezone.utc).isoformat(), **values), allow_nan=False), flush=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _prepare_repo(repo=None):
    if repo is not None:
        directory = Path(repo).resolve()
        require((directory / 'projects/mmdet3d_plugin').is_dir(), 'Invalid --repo')
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    elif importlib.util.find_spec('projects') is None:
        candidates = [Path.cwd(), Path(__file__).resolve().parents[2] /
                      'code/Drive-OccWorld-sota-p2']
        found = next((p for p in candidates if (p / 'projects/mmdet3d_plugin').is_dir()), None)
        require(found is not None, 'Pass repo= / --repo or add the native M0 repo to sys.path')
        sys.path.insert(0, str(found))
    importlib.import_module('projects.mmdet3d_plugin')


def _tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def _parameter_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(_tensor_digest(value).encode())
    return h.hexdigest()


def build_native_model(config, checkpoint, device='cuda', repo=None):
    """Strict M0 model-only load; preserve normal trainability until caller freezes.

    Does not restore optimizer/counters, install overrides, or read datasets.
    The returned model is in eval mode. Caller owns its subsequent mode/device.
    """
    import torch
    from mmcv import Config
    from mmdet3d.models import build_model
    _prepare_repo(repo)
    config_path = Path(config).resolve() if isinstance(config, (str, Path)) else None
    cfg = Config.fromfile(str(config_path)) if config_path else copy.deepcopy(config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.scientific_eval = True
    for name in ['motion_residual', 'doppler_posterior', 'doppler_advection',
                 'doppler_flow_loss', 'turn_on_plan', 'turn_on_flow']:
        require(not cfg.model.get(name), 'Native M0 requires disabled ' + name)
    head = cfg.model.future_pred_head
    require(head.type == 'WorldHeadV1' and not head.soft_weight,
            'Require native WorldHeadV1 with independent intermediate heads')
    require(head.num_classes == 2 and head.num_pred_height == 16,
            'Native binary 16-height output required')
    require(cfg.model.memory_queue_len == 1 and head.memory_queue_len == 1,
            'Cache source must use original one-slot M0')
    require(cfg.model.future_pred_frame_num == 4 and cfg.model.test_future_frame_num == 4,
            'Four future horizons required')
    require(cfg.model.bev_h == 200 and cfg.model.bev_w == 200,
            'Native 200x200 BEV required')
    require(cfg.model.get('radar_encoder') and cfg.get('fp16') is None,
            'Require native radar encoder and FP32 configuration')
    checkpoint = Path(checkpoint).resolve()
    checkpoint_sha = sha256(checkpoint)
    require(checkpoint_sha == M0_SHA256, 'M0 checkpoint SHA mismatch')
    before = checkpoint.stat()
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    payload = torch.load(str(checkpoint), map_location='cpu')
    require(payload.get('meta', {}).get('epoch') == 24, 'Require M0 epoch24')
    model.load_state_dict(payload['state_dict'], strict=True)
    del payload
    after = checkpoint.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Checkpoint changed while loading')
    source = Path(sys.modules[type(model).__module__].__file__).resolve()
    require(sha256(source) == SOURCE_MODEL_SHA256, 'Frozen native detector source SHA mismatch')
    model.to(device).eval()
    model._native_state_provenance = dict(
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        checkpoint_epoch=24, config=str(config_path) if config_path else None,
        config_sha256=sha256(config_path) if config_path else None,
        source_model=str(source), source_model_sha256=sha256(source),
        model_only=True, optimizer_loaded=False,
        radar_contract='native_loader_B_without_common_source_override',
        head_parameter_count=sum(p.numel() for p in model.future_pred_head.parameters()))
    return model


def _tree_map(value, tensor_fn):
    import torch
    if torch.is_tensor(value):
        return tensor_fn(value)
    if isinstance(value, dict):
        return {k: _tree_map(v, tensor_fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_tree_map(v, tensor_fn) for v in value]
    if isinstance(value, tuple):
        return tuple(_tree_map(v, tensor_fn) for v in value)
    return copy.deepcopy(value)


def validate_inputs(inputs):
    """Forbid occupancy labels in the model-input tree; given ego/action is allowed."""
    expected = {'prev_bev_input', 'action_condition_dict', 'cond_norm_dict',
                'plan_dict', 'valid_frames', 'img_metas', 'prev_img_metas',
                'num_frames', 'occ_flow', 'radar_bev', 'motion_state'}
    require(set(inputs) == expected, 'future_pred boundary keys mismatch: ' + str(set(inputs) ^ expected))
    require(inputs['occ_flow'] == 'occ' and inputs['motion_state'] is None,
            'Native occupancy replay must not contain auxiliary motion state')
    require(inputs['cond_norm_dict'].get('occ_gts') is None,
            'Occupancy targets cannot enter conditional normalization')
    require(inputs['plan_dict'].get('sem_occupancy') is None,
            'Occupancy targets cannot enter planning inputs')
    forbidden = {'segmentation', 'gt_occ', 'targets', 'target', 'gt_future_boxes',
                 'segmentation_bev', 'instance', 'flow', 'occupancy_labels',
                 'gt_bboxes_3d', 'gt_labels_3d', 'gt_boxes'}
    def visit(node, prefix):
        if isinstance(node, dict):
            for key, value in node.items():
                require(str(key) not in forbidden, 'Forbidden target key in inputs: ' + prefix + str(key))
                if str(key) in {'occ_gts', 'sem_occupancy'}:
                    require(value is None, 'Nonempty occupancy conditional input')
                visit(value, prefix + str(key) + '.')
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                visit(value, prefix + str(index) + '.')
    visit(inputs, '')


def _verify_file(path, descriptor):
    path = Path(path).resolve()
    st = path.stat()
    key = (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
           descriptor['sha256'])
    require(st.st_size == descriptor['bytes'], 'Cache file size mismatch: ' + str(path))
    if key not in _VERIFIED_FILES:
        require(sha256(path) == descriptor['sha256'], 'Cache hash mismatch: ' + str(path))
        _VERIFIED_FILES[key] = True


def _sample_directory(cache_dir, record):
    root = Path(cache_dir).resolve()
    if root.is_file():
        root = root.parent
    directory = (root / record['directory']).resolve()
    require(root in directory.parents, 'Sample path escapes cache directory')
    return directory


def load_sample(cache_dir, record, device='cuda'):
    """Load input and targets only; native predictions remain on disk/CPU.

    ``targets`` has the original full segmentation shape [1,7,512,512,40].
    ``inputs`` retains the exact original future_pred kwargs, with tensors on
    device but NumPy metadata left as NumPy. No labels are merged into inputs.
    """
    import numpy as np
    import torch
    directory = _sample_directory(cache_dir, record)
    for key in ('inputs', 'targets'):
        _verify_file(directory / record['files'][key]['file'], record['files'][key])
    inputs = torch.load(str(directory / record['files']['inputs']['file']), map_location='cpu')
    validate_inputs(inputs)
    inputs = _tree_map(inputs, lambda x: x.to(device=device))
    target = np.load(directory / record['files']['targets']['file'], allow_pickle=False)
    require(list(target.shape) == record['files']['targets']['shape'] and target.dtype == np.uint8,
            'Target shape/dtype changed')
    targets = torch.from_numpy(target).to(device=device, dtype=torch.long)
    return dict(inputs=inputs, targets=targets, record=copy.deepcopy(record))


def replay(model, sample, training=False):
    """Replay original future_pred; no cache/label mutation and no detached rollout."""
    import torch
    validate_inputs(sample['inputs'])
    # Original future_pred mutates action/conditional/plan dictionaries. Clone
    # every call while retaining autograd links to explicitly differentiable input.
    inputs = _tree_map(sample['inputs'], lambda x: x.clone())
    inputs['valid_frames'] = [1, 2, 3, 4] if training else []
    model.train(bool(training))
    with torch.enable_grad() if training else torch.no_grad():
        return model.future_pred(**inputs)


def _cpu_numpy(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@contextlib.contextmanager
def _capture_native(model, context):
    """Observe original boundaries. No changes to fusion, loader, or numerics."""
    import numpy as np
    import torch
    native_future = model.future_pred
    native_eval = model.evaluate_occ_records
    signature = inspect.signature(native_future)
    def future(self, *args, **kwargs):
        require('inputs' not in context, 'More than one future_pred call in native M0')
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = dict(bound.arguments)
        validate_inputs(inputs)
        # BEFORE original call: its dictionaries are mutated during rollout.
        context['inputs'] = _tree_map(inputs, lambda x: x.detach().cpu().clone())
        result = native_future(*args, **kwargs)
        require(result[0].dtype == torch.float32, 'Native source must emit float32 predictions')
        context['preds'] = result[0].detach().cpu().clone().numpy()
        return result
    def evaluate(self, preds, targets, metas):
        result = native_eval(preds, targets, metas)
        require(len(result) == 1, 'Extraction is one exact sample at a time')
        values = targets.detach().cpu()
        require(bool(((values == 0) | (values == 1) | (values == 255)).all()),
                'Noncanonical target label before uint8 storage')
        context['targets'] = values.to(torch.uint8).numpy()
        context['native_record'] = dict(result[0], hist_by_horizon=np.asarray(
            result[0]['hist_by_horizon'], dtype=np.int64).tolist())
        return result
    model.future_pred = types.MethodType(future, model)
    model.evaluate_occ_records = types.MethodType(evaluate, model)
    try:
        yield
    finally:
        model.future_pred = native_future
        model.evaluate_occ_records = native_eval


def _describe_file(path, value=None):
    result = dict(file=path.name, sha256=sha256(path), bytes=path.stat().st_size)
    if value is not None:
        result.update(shape=list(value.shape), dtype=str(value.dtype))
    return result


def _save_sample(out, row, context, max_bytes_remaining):
    import numpy as np
    import torch
    directory = out / 'samples' / row['sample_token']
    directory.mkdir(parents=True, exist_ok=False)
    require(tuple(context['inputs']['prev_bev_input'].shape) == (1, 1, 40000, 256),
            'Native initial memory shape mismatch')
    require(context['targets'].shape == (1, 7, 512, 512, 40), 'Original segmentation shape mismatch')
    require(context['preds'].shape == (5, 3, 1, 1, 40000, 16, 2), 'Original prediction layout mismatch')
    require(np.isfinite(context['preds']).all(), 'Nonfinite native prediction')
    require(bool(torch.isfinite(context['inputs']['prev_bev_input']).all()), 'Nonfinite reference state')
    files = {}
    path = directory / 'inputs.pt'
    torch.save(context['inputs'], str(path))
    files['inputs'] = _describe_file(path)
    for key, name in [('targets', 'targets.npy'), ('preds', 'native_preds.npy')]:
        path = directory / name
        np.save(path, context[key], allow_pickle=False)
        files[key] = _describe_file(path, context[key])
    size = sum(v['bytes'] for v in files.values())
    require(size <= max_bytes_remaining, 'Cache byte ceiling reached; sample retained incomplete, no replacement')
    gt_counts = [[int((gt == cls).sum()) for cls in (0, 1, 255)]
                 for gt in context['targets'][0]]
    record = dict(row, directory=str(directory.relative_to(out)), files=files,
                  native_hist=context['native_record']['hist_by_horizon'],
                  target_counts_0_1_255_by_frame=gt_counts,
                  target_frame_seconds=[-1., -.5, 0., .5, 1., 1.5, 2.],
                  output_frame_seconds=[0., .5, 1., 1.5, 2.],
                  reference_bev_tensor_sha256=_tensor_digest(context['inputs']['prev_bev_input']),
                  native_radar_tensor_sha256=_tensor_digest(context['inputs']['radar_bev']),
                  bytes=size)
    write_json(directory / 'identity.json', record)
    record['identity_file_sha256'] = sha256(directory / 'identity.json')
    return record


def parity_check(model, cache_dir, record, device):
    """Full original 5-time/3-layer CPU logits plus exact native GT confusion."""
    import numpy as np
    import torch
    directory = _sample_directory(cache_dir, record)
    descriptor = record['files']['preds']
    _verify_file(directory / descriptor['file'], descriptor)
    expected = np.load(directory / descriptor['file'], mmap_mode='r', allow_pickle=False)
    sample = load_sample(cache_dir, record, device)
    predicted = replay(model, sample, training=False)[0]
    observed = predicted.detach().cpu().numpy()
    require(observed.shape == expected.shape and observed.dtype == expected.dtype,
            'Replay output shape/dtype mismatch')
    maximum = 0.0
    equal = True
    for h in range(5):
        maximum = max(maximum, float(np.max(np.abs(observed[h] - expected[h]))))
        equal = equal and np.array_equal(observed[h], expected[h])
    require(equal, 'Native cached replay is not bitwise equal; max_abs=' + str(maximum))
    with torch.no_grad():
        rows = model.evaluate_occ_records(predicted, sample['targets'], sample['inputs']['img_metas'])
    require(len(rows) == 1, 'Unexpected replay record count')
    require(rows[0]['sample_token'] == record['sample_token'] and
            rows[0]['scene_token'] == record['scene_token'], 'Replay identity mismatch')
    hist = np.asarray(rows[0]['hist_by_horizon'], dtype=np.int64)
    require(np.array_equal(hist, np.asarray(record['native_hist'], dtype=np.int64)),
            'Native full-GT confusion mismatch')
    for h in range(5):
        require(hist[h].sum(1).tolist() == record['target_counts_0_1_255_by_frame'][h+2][:2],
                'Native GT class counts disagree with confusion')
    return dict(status='PASS', sample_token=record['sample_token'],
                bitwise_all_five_horizons_three_layers=True, maximum_absolute_error=maximum,
                exact_native_confusion=True, inputs_targets_separate=True)


def gradient_check(model, cache_dir, record, device):
    """No optimizer: one final-horizon sampled logit loss reaches all 4 states."""
    import torch
    sample = load_sample(cache_dir, record, device)
    # Large targets are not used by this test and never enter replay inputs.
    del sample['targets']
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.future_pred_head.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    reference = sample['inputs']['prev_bev_input'].detach().requires_grad_(True)
    sample['inputs']['prev_bev_input'] = reference
    states = []
    def capture(module, args, output):
        state = output[0]
        require(state.requires_grad, 'A future state was computed under no_grad')
        state.retain_grad()
        states.append(state)
    handle = model.future_pred_head.register_forward_hook(capture)
    parameters_before = _parameter_digest(model.future_pred_head)
    try:
        torch.manual_seed(11)
        if str(device).startswith('cuda'):
            torch.cuda.manual_seed_all(11)
        predictions = replay(model, sample, training=True)[0]
        require(len(states) == 4, 'Require all four future state calls')
        # Probe the final future only, all three classifier layers. Sampled
        # x/y token and height pairs avoid any full-GT interpolation/loss.
        token = torch.linspace(0, 39999, 64, device=predictions.device).long()
        depth = torch.arange(64, device=predictions.device) % 16
        final = predictions[-1, :, 0, 0, token, depth, :]
        margin = final[..., 1] - final[..., 0]
        loss = (margin.square() + .013 * margin).mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite gradient probe loss')
        loss.backward()
        gradients = []
        for horizon, state in enumerate(states, 1):
            require(state.grad is not None and bool(torch.isfinite(state.grad).all()),
                    'Missing/nonfinite future-state gradient at ' + str(horizon))
            norm = float(state.grad.float().norm())
            require(norm > 0, 'Zero future-state gradient at ' + str(horizon))
            gradients.append(dict(horizon=horizon, norm=norm))
        require(reference.grad is not None and bool(torch.isfinite(reference.grad).all()) and
                float(reference.grad.norm()) > 0, 'Final horizon does not reach t0 reference')
        nonzero, none = [], []
        for name, p in model.future_pred_head.named_parameters():
            if p.grad is None:
                none.append(name)
            else:
                require(bool(torch.isfinite(p.grad).all()), 'Nonfinite parameter gradient: ' + name)
                if bool((p.grad != 0).any()):
                    nonzero.append(name)
        require(any('transformer' in name for name in nonzero), 'No future transformer gradient')
        require(any('bev_pred_head' in name for name in nonzero), 'No occupancy decoder gradient')
        require(_parameter_digest(model.future_pred_head) == parameters_before,
                'Gradient-only preflight changed model parameters')
        return dict(status='PASS', sample_token=record['sample_token'],
                    training_valid_frames=[1,2,3,4], probe='final_horizon_64_token_height_margins_all_3_layers',
                    probe_loss=float(loss.detach()), state_gradients=gradients,
                    reference_gradient_norm=float(reference.grad.norm()),
                    nonzero_parameter_gradient_names=nonzero, missing_gradient_names=none,
                    source_parameters_unchanged=True, optimizer_steps=0,
                    full_GT_backward=False, targets_used=False)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        for name, p in model.named_parameters():
            p.requires_grad_(flags[name])
        model.eval()


def _identity_from_metas(metas):
    current = metas[max(metas)]
    result = dict(sample_token=str(current.get('sample_idx', current.get(
        'sample_token', current.get('lidar_token')))), scene_token=str(current['scene_token']))
    if current.get('lidar_token') is not None:
        result['lidar_token'] = str(current['lidar_token'])
    return result


def _selected_rows(selection, split, dataset):
    require(isinstance(selection, dict) and isinstance(selection.get('records'), list),
            'Selection requires top-level records list')
    rows = [copy.deepcopy(r) for r in selection['records'] if r.get('split', split) == split]
    require(rows, 'No selected rows for requested split')
    require(len({r['sample_token'] for r in rows}) == len(rows), 'Duplicate selected sample token')
    require(len({r['official_index'] for r in rows}) == len(rows), 'Duplicate selected official_index')
    for row in rows:
        index = row['official_index']
        require(isinstance(index, int) and 0 <= index < len(dataset.usable_index), 'Invalid usable index')
        actual_index = int(dataset.usable_index[index])
        info = dataset.data_infos[actual_index]
        require(str(info['token']) == row['sample_token'] and str(info['scene_token']) == row['scene_token'],
                'Selection identity differs from exact dataset usable index')
        if 'data_info_index' in row:
            require(row['data_info_index'] == actual_index, 'Raw data_info_index mismatch')
        row['data_info_index'] = actual_index
        row['split'] = split
    return rows


def extract(a, out, started):
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    from mmdet3d.datasets import build_dataset
    _prepare_repo(a.repo)
    cfg = Config.fromfile(a.config)
    dc = copy.deepcopy(cfg.data.test)
    dc.test_mode = True
    dc.pop('samples_per_gpu', None)
    if a.split == 'train':
        dc.ann_file = cfg.data.train.ann_file
    require(dc.radar_cfg.cache_readonly is True and dc.radar_cfg.nsweeps == 5,
            'Require native readonly five-sweep radar cache')
    require(dc.get('radar_observation_cfg') is None and not dc.get('allow_dual_radar_inputs'),
            'Native cache must not add per-return observation inputs')
    require(dc.future_metadata_only is True, 'Require metadata-only future loading')
    event('DATASET_BUILD', split=a.split, ann_file=dc.ann_file)
    dataset = build_dataset(dc)
    selection = json.loads(Path(a.selection).read_text())
    rows = _selected_rows(selection, a.split, dataset)
    model = build_native_model(a.config, a.checkpoint, a.device, repo=a.repo)
    for p in model.parameters():
        p.requires_grad_(False)
    require(str(a.device).startswith('cuda'), 'Native image extraction currently requires CUDA')
    device_index = torch.device(a.device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    wrapped = MMDataParallel(model, device_ids=[device_index])
    manifest = dict(schema=SCHEMA, status='RUNNING', split=a.split,
        seed=11, records=[], selected_records=rows,
        selection_sha256=sha256(a.selection), script_sha256=sha256(__file__),
        extractor_sha256=sha256(__file__), m0_sha256=M0_SHA256,
        source_model_sha256=SOURCE_MODEL_SHA256,
        config_sha256=sha256(a.config), ann_file=str(dc.ann_file), ann_sha256=sha256(dc.ann_file),
        native_model=model._native_state_provenance, native_radar_cfg=dict(dc.radar_cfg),
        future_ego_action_conditioned=True, future_occupancy_is_input=False,
        inputs_targets_separate=True, extraction_model_mode='eval_no_grad',
        input_precision='original_float32', target_storage_dtype='uint8_original_labels',
        target_loading_dtype='torch.long', target_shape=[1,7,512,512,40],
        native_output_shape=[5,3,1,1,40000,16,2],
        numerical_policy=dict(tf32_matmul=True, tf32_cudnn=True, cudnn_benchmark=False),
        max_cache_bytes=a.max_bytes, cache_bytes=0)
    write_json(out / 'index.json', manifest)
    for ordinal, row in enumerate(rows):
        begin = time.monotonic()
        event('EXACT_SAMPLE_PREPARE', ordinal=ordinal, sample_token=row['sample_token'])
        example = dataset._prepare_data_info(row['data_info_index'], rand_interval=None)
        require(example is not None, 'Preselected native sample failed preparation; no replacement')
        observed = _identity_from_metas(example['img_metas'].data)
        require(all(observed[k] == row[k] for k in ('sample_token', 'scene_token')),
                'Prepared sample identity mismatch')
        if 'lidar_token' in row:
            require(observed.get('lidar_token') == row['lidar_token'], 'Prepared lidar identity mismatch')
        row.update({k: v for k, v in observed.items() if k not in row})
        data = collate([example], samples_per_gpu=1)
        data.pop('instance', None)
        context = {}
        with _capture_native(model, context):
            with torch.no_grad():
                wrapped(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize(device_index)
        require(all(context['native_record'][k] == row[k] for k in ('sample_token', 'scene_token')),
                'Native evaluation identity mismatch')
        record = _save_sample(out, row, context, a.max_bytes - manifest['cache_bytes'])
        del data, example, context
        record['parity'] = parity_check(model, out, record, a.device)
        record['seconds'] = time.monotonic() - begin
        manifest['records'].append(record)
        manifest['cache_bytes'] += record['bytes']
        manifest['elapsed_seconds'] = time.monotonic() - started
        write_json(out / 'index.json', manifest)
        write_json(out / 'progress.json', dict(status='RUNNING', completed=len(manifest['records']),
                   planned=len(rows), cache_bytes=manifest['cache_bytes'], seconds=time.monotonic()-started))
        event('NATIVE_SAMPLE_COMPLETE', ordinal=ordinal, sample_token=row['sample_token'],
              bitwise_replay=True, seconds=record['seconds'])
    manifest['status'] = 'COMPLETE'
    write_json(out / 'index.json', manifest)
    write_json(out / 'complete.json', dict(status='COMPLETE_NATIVE_STATE_CACHE', samples=len(rows),
               index_sha256=sha256(out/'index.json'), cache_bytes=manifest['cache_bytes'],
               seconds=time.monotonic()-started, optimizer_steps=0,
               native_radar_unchanged=True, all_samples_bitwise_replayed=True,
               maximum_cuda_memory_bytes=torch.cuda.max_memory_allocated(device_index)))


def preflight(a, out, started):
    cache = Path(a.cache)
    if cache.is_file():
        cache = cache.parent
    complete = json.loads((cache/'complete.json').read_text())
    require(complete['status'] == 'COMPLETE_NATIVE_STATE_CACHE', 'Cache is incomplete')
    require(sha256(cache/'index.json') == complete['index_sha256'], 'Cache index SHA mismatch')
    index = json.loads((cache/'index.json').read_text())
    require(index['schema'] == SCHEMA and index['status'] == 'COMPLETE', 'Native cache schema/status mismatch')
    require(index['config_sha256'] == sha256(a.config), 'Replay config differs from extraction config')
    require(index['native_model']['checkpoint_sha256'] == sha256(a.checkpoint),
            'Replay source checkpoint differs')
    require(complete['samples'] == len(index['records']), 'Incomplete index coverage')
    model = build_native_model(a.config, a.checkpoint, a.device, repo=a.repo)
    selected = index['records'][:a.samples]
    require(selected and len(selected) == a.samples, 'Requested preflight samples exceed complete cache')
    results = []
    for record in selected:
        event('PARITY_PREFLIGHT', sample_token=record['sample_token'])
        parity = parity_check(model, cache, record, a.device)
        event('GRADIENT_PREFLIGHT', sample_token=record['sample_token'])
        gradient = gradient_check(model, cache, record, a.device)
        results.append(dict(parity=parity, gradient=gradient))
        write_json(out/'progress.json', dict(status='RUNNING', results=results))
    write_json(out/'complete.json', dict(status='NATIVE_REPLAY_GRADIENT_PREFLIGHT_PASS',
               index_sha256=sha256(cache/'index.json'), script_sha256=sha256(__file__),
               config_sha256=sha256(a.config), checkpoint_sha256=sha256(a.checkpoint),
               samples=len(results), results=results, optimizer_steps=0,
               seconds=time.monotonic()-started))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=['extract', 'preflight'])
    p.add_argument('--repo')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--selection')
    p.add_argument('--split', choices=['train', 'development'])
    p.add_argument('--cache')
    p.add_argument('--samples', type=int, default=1, help='first N complete records in preflight')
    p.add_argument('--out', required=True, help='must be a NEW directory')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seconds', type=int, default=1200)
    p.add_argument('--max-bytes', type=int, default=200_000_000_000)
    a = p.parse_args(argv)
    require(a.max_seconds > 0 and a.max_bytes > 0 and a.samples > 0, 'Positive bounds required')
    require(a.mode != 'extract' or (a.selection and a.split), 'Extraction requires --selection and --split')
    require(a.mode != 'preflight' or a.cache, 'Preflight requires --cache')
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def stop(signum, frame):
        raise InterruptedError('Native cache bound/interruption signal ' + str(signum) + '; no automatic retry')
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(number, stop)
    signal.alarm(a.max_seconds)
    try:
        event('DEPENDENCY_IMPORT_BEGIN', mode=a.mode)
        import torch
        import numpy as np
        import random
        torch.set_num_threads(2)
        try:
            import cv2
            cv2.setNumThreads(2)
        except ImportError:
            pass
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        if str(a.device).startswith('cuda'):
            require(torch.cuda.is_available(), 'CUDA is unavailable')
            torch.cuda.set_device(torch.device(a.device))
            torch.cuda.manual_seed_all(11)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        event('DEPENDENCY_IMPORT_COMPLETE', torch=torch.__version__)
        if a.mode == 'extract':
            extract(a, out, started)
        else:
            preflight(a, out, started)
    except BaseException as exc:
        write_json(out/'failed.json', dict(status='FAILED_OR_INTERRUPTED', error=repr(exc),
                   seconds=time.monotonic()-started, optimizer_steps=0, automatic_retry=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
