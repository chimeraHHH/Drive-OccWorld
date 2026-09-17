"""Bounded native1 loss-resolution and gradient probe; NEVER an optimizer run.

coarse: first two distinct train anchors + explicitly posthoc rare anchor;
full: only the original first two anchors. Every model is frozen initial M0
except that future-head parameters require gradients. Model.eval() disables
dropout; valid_frames=[1,2,3,4] enables the native four-step gradient route.
The only candidate changes are removing loss_occ's target downsampling and
replacing loss_voxel's fixed H/W/D by the supplied complete GT shape.
No gradient vectors, predictions, checkpoints, or optimizer state are saved.
"""
import argparse
import ast
import copy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import textwrap
import time
import types

import numpy as np

PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
HEAD_SOURCE_SHA = '4738c03a28e353efd4dec68adf6c9407aa73b9273bd13c2f67021795d6240a7a'
INITIAL_HEAD_SHA = '6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'
POSTHOC_SOURCE_SHA = '15a7f27f03d15a6e980358f6f0be1415c80e34abc91dcf83cc2caec5baaf7048'
TRAIN_INDEX_SHA = '1dc14643df86f85c5ae87b17b83c60cada99f189a025675d150062acb046e7b1'
PAIR_TOKENS = ('464fe0be05a74ec9852573cb9e3afd89', '1dacdffab5c240f1b20e05566b940ef6')
RARE_TOKEN = '706d7015b80d4329b80d4a8c52139ab4'
FAMILIES = ('ce', 'sem_scal', 'geo_scal', 'lovasz')
LOSS_AST_SHA = dict(loss_voxel='33f62c6606d947b1f670842b844845f4cf059495d9810f8426950bb0cbb5a84d',
                   loss_occ='039bca62c86c0cace39e959a3ed5e7972b3fb48423531cddd71b2fb7c5cdc8d8')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def event(name, **values):
    print(json.dumps(dict(event=name, **values), allow_nan=False), flush=True)


def canonical(node):
    node = copy.deepcopy(node)
    for item in ast.walk(node):
        if 'type_params' in item._fields:
            require(not getattr(item, 'type_params', []), 'Unreviewed generic syntax')
            item._fields = tuple(k for k in item._fields if k != 'type_params')
    options = dict(include_attributes=False)
    if 'show_empty' in inspect.signature(ast.dump).parameters:
        options['show_empty'] = True
    return ast.dump(node, **options)


def ast_sha(node):
    return hashlib.sha256(canonical(node).encode()).hexdigest()


def derive_method(method, name):
    path = Path(inspect.getsourcefile(method)).resolve()
    require(sha(path) == HEAD_SOURCE_SHA, 'Original loss method bytes differ')
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and
            tree.body[0].name == name and not tree.body[0].decorator_list, 'Unexpected original loss method')
    original = copy.deepcopy(tree); node = tree.body[0]
    require(ast_sha(node) == LOSS_AST_SHA[name], 'Original inspect/dedent loss AST changed')
    if name == 'loss_occ':
        matches = [(i, x) for i, x in enumerate(node.body) if isinstance(x, ast.Assign) and
            len(x.targets) == 1 and isinstance(x.targets[0], ast.Name) and x.targets[0].id == 'target_voxels' and
            isinstance(x.value, ast.Call) and isinstance(x.value.func, ast.Name) and x.value.func.id == '_downsample_occ_target']
        require(len(matches) == 1 and len(matches[0][1].value.args) == 1 and
                not matches[0][1].value.keywords and isinstance(matches[0][1].value.args[0], ast.Name) and
                matches[0][1].value.args[0].id == 'target_voxels', 'Unexpected target-downsampling statement')
        index, old = matches[0]; del node.body[index]
        reconstructed = copy.deepcopy(tree); reconstructed.body[0].body.insert(index, copy.deepcopy(old))
        change = 'remove sole target_voxels=_downsample_occ_target(target_voxels)'
    else:
        require(name == 'loss_voxel', 'Only the two original loss methods may change')
        matches = [(i, x) for i, x in enumerate(node.body) if isinstance(x, ast.Assign) and
            len(x.targets) == 1 and isinstance(x.targets[0], ast.Tuple) and
            [v.id if isinstance(v, ast.Name) else None for v in x.targets[0].elts] == ['H', 'W', 'D']]
        require(len(matches) == 1 and isinstance(matches[0][1].value, ast.Tuple) and
                [v.value if isinstance(v, ast.Constant) else None for v in matches[0][1].value.elts] == [256, 256, 20],
                'Original fixed loss resolution changed')
        index, old = matches[0]; old = copy.deepcopy(old)
        node.body[index].value = ast.parse('target_voxels.shape[-3:]', mode='eval').body
        reconstructed = copy.deepcopy(tree); reconstructed.body[0].body[index] = old
        change = 'replace sole H,W,D=(256,256,20) with target_voxels.shape[-3:]'
    require(canonical(reconstructed) == canonical(original), 'Unexpected additional loss AST change')
    namespace = dict(method.__func__.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<full-resolution:'+name+'>', 'exec'), namespace)
    return namespace[name], dict(original_file=str(path), original_file_sha256=sha(path),
        original_ast_sha256=ast_sha(original.body[0]), derived_ast_sha256=ast_sha(tree.body[0]),
        modification=change, exact_inverse_restores_original_AST=True,
        extraction='inspect.getsource(bound method)+dedent; canonical Python 3.10 empty-fields contract')


def install_full_resolution(head):
    require(not hasattr(head, '_full_resolution_supervision_receipt'), 'Already adapted loss head')
    originals = {name: getattr(head, name) for name in ('loss_occ', 'loss_voxel')}
    methods, receipt = {}, {}
    for name, method in originals.items():
        methods[name], receipt[name] = derive_method(method, name)
    for name, method in methods.items():
        setattr(head, name, types.MethodType(method, head))
    head._full_resolution_supervision_receipt = receipt
    return receipt


def tensor_digest(tensor):
    tensor = tensor.detach().cpu().contiguous(); h = hashlib.sha256()
    h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
    h.update(tensor.numpy().tobytes()); return h.hexdigest()


def vector_dot(left, right):
    # Float64 chunk accumulation; avoid retaining a second full float64 copy.
    result = 0.
    for offset in range(0, len(left), 262144):
        a = left[offset:offset+262144].astype(np.float64)
        b = right[offset:offset+262144].astype(np.float64)
        result += float(np.dot(a, b))
    return result


def comparison(left, right):
    aa, bb = vector_dot(left, left), vector_dot(right, right)
    dot = vector_dot(left, right)
    return dict(dot=dot, left_L2=math.sqrt(aa), right_L2=math.sqrt(bb),
        cosine=None if aa == 0 or bb == 0 else max(-1., min(1., dot/math.sqrt(aa*bb))),
        zero_norm_cosine_is_undefined=(aa == 0 or bb == 0))


def get_gradient(loss, named_parameters, retain_graph):
    import torch
    torch.cuda.synchronize(); begin = time.monotonic()
    gradients = torch.autograd.grad(loss, tuple(p for _, p in named_parameters),
                                    retain_graph=retain_graph, create_graph=False, allow_unused=True)
    torch.cuda.synchronize(); vjp_seconds = time.monotonic()-begin
    vector = np.zeros(sum(p.numel() for _, p in named_parameters), dtype=np.float32)
    offset = 0; missing = []; zero = []; nonzero = 0
    for (name, parameter), gradient in zip(named_parameters, gradients):
        size = parameter.numel()
        if gradient is None:
            missing.append(name)
        else:
            require(torch.isfinite(gradient).all().item(), 'Nonfinite gradient: '+name)
            values = gradient.detach().float().cpu().contiguous().numpy().reshape(-1)
            vector[offset:offset+size] = values
            if np.any(values != 0):
                nonzero += 1
            else:
                zero.append(name)
        offset += size
    del gradients
    return vector, dict(L2=math.sqrt(vector_dot(vector, vector)), parameter_tensors=len(named_parameters),
        None_parameter_names=missing, zero_gradient_parameter_names=zero,
        nonzero_parameter_tensors=nonzero, vjp_device_seconds=vjp_seconds,
        retain_graph=retain_graph, gradient_vector_saved=False,
        gradient_vector_dtype='float32; scalar dot/norm accumulated in float64 chunks')


def forward_with_grad(model, sample):
    """Native no-dropout values with every future transition in the grad path."""
    import torch
    from native_state_cache import _tree_map, validate_inputs
    validate_inputs(sample['inputs'])
    inputs = _tree_map(sample['inputs'], lambda value: value.clone())
    inputs['valid_frames'] = [1, 2, 3, 4]
    model.eval()
    require(model.future_pred_frame_num == model.test_future_frame_num == 4 and
            inputs['cond_norm_dict']['occ_gts'] is None, 'Native horizon or GT-input contract differs')
    with torch.enable_grad():
        result = model.future_pred(**inputs)
    require(result[0].requires_grad and not model.training, 'No-dropout gradient route missing')
    return result


def checked_loss(model, prediction, targets, mode):
    """Observe actual loss arguments and GT, not a separately guessed permute."""
    import torch
    head = model.future_pred_head; native_occ, native_voxel = head.loss_occ, head.loss_voxel
    captured = dict(layers=[])
    def observe_occ(self, output_voxels=None, target_voxels=None, **kwargs):
        require('full' not in captured, 'Repeated loss_occ entry')
        expected = targets[:, self.history_queue_length:].transpose(0, 1).reshape(5, 512, 512, 40)
        require(torch.equal(target_voxels, expected), 'Actual loss GT differs from native current+four future slice')
        captured['full'] = target_voxels
        captured['prediction_shape_from_compute_occ_loss'] = list(output_voxels.shape)
        require(output_voxels.shape == (3, 5, 2, 200, 200, 16), 'Unexpected actual native loss input axes')
        return native_occ(output_voxels, target_voxels, **kwargs)
    def observe_voxel(self, output_voxels, target_voxels, tag, target_voxels_prepared=False):
        require(tag == 'inter_'+str(len(captured['layers'])) and target_voxels_prepared is True, 'Wrong layer/prepared GT contract')
        shape = (5, 512, 512, 40) if mode == 'full' else (5, 256, 256, 20)
        require(tuple(target_voxels.shape) == shape, 'Wrong actual supervision resolution')
        if mode == 'full':
            require(torch.equal(target_voxels, captured['full']), 'Full loss altered GT/ignore voxels')
        captured['layers'].append(dict(tag=tag, GT_shape=list(target_voxels.shape),
            GT_counts_0_1_255=[[int((target_voxels[h] == c).sum().item()) for c in (0, 1, 255)] for h in range(5)]))
        return native_voxel(output_voxels, target_voxels, tag, target_voxels_prepared=True)
    head.loss_occ, head.loss_voxel = types.MethodType(observe_occ, head), types.MethodType(observe_voxel, head)
    try:
        losses = model.compute_occ_loss(prediction, targets)
    finally:
        head.loss_occ, head.loss_voxel = native_occ, native_voxel
    captured.pop('full')
    require(len(captured['layers']) == 3 and set(losses) ==
            {'loss_voxel_'+family+'_inter_'+str(i) for family in FAMILIES for i in range(3)}, 'Original 12-loss schema changed')
    require(all(value.numel() == 1 and value.requires_grad and torch.isfinite(value).item() for value in losses.values()),
            'Missing/nonfinite/nondifferentiable original loss')
    return losses, captured


def verify_contract(a):
    require(sha(a.protocol) == PARENT_SHA, 'Only frozen parent protocol_v2 is accepted')
    parent = read(a.protocol); pkg = Path(__file__).resolve().parent
    require(sha(a.config) == parent['config_sha256'] and sha(a.checkpoint) == parent['m0_sha256'], 'Wrong native config/M0 checkpoint')
    sources = dict(parent['source_sha256'], **{'posthoc_train_diagnostic.py': POSTHOC_SOURCE_SHA})
    for name, digest in sources.items():
        require(Path(name).name == name and sha(pkg/name) == digest, 'Frozen package source changed: '+name)
    runtime = {}
    for name, digest in parent['runtime_source_sha256'].items():
        pieces = name.split('/projects/', 1); require(len(pieces) == 2, 'Unexpected runtime source')
        path = (Path(a.repo)/'projects'/pieces[1]).resolve()
        require(sha(path) == digest, 'Native runtime source changed: '+str(path)); runtime[str(path)] = digest
    require(sha(Path(a.cache)/'index.json') == TRAIN_INDEX_SHA, 'Wrong fixed training cache')
    return parent, sources, runtime


def metric_record(model, predictions, sample):
    import torch
    with torch.no_grad():
        records = model.evaluate_occ_records(predictions.detach(), sample['targets'], sample['inputs']['img_metas'])
    require(len(records) == 1, 'Expected one original evaluator record')
    row = records[0]; row['hist_by_horizon'] = np.asarray(row['hist_by_horizon'], dtype=np.int64).tolist()
    require(row['horizon_seconds'] == [0., .5, 1., 1.5, 2.], 'Wrong native horizon identity')
    return row


def measure(model, sample, mode, role, named_parameters, expected=None):
    import torch
    from memory_experiment import seed_all
    seed_all(11); model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    begin = time.monotonic(); output = forward_with_grad(model, sample)
    torch.cuda.synchronize(); forward_seconds = time.monotonic()-begin
    prediction = output[0]; digest = tensor_digest(prediction)
    native_hist = metric_record(model, prediction, sample)
    require(native_hist['sample_token'] == sample['record']['sample_token'] and
            native_hist['scene_token'] == sample['record']['scene_token'], 'Native evaluator identity changed')
    if expected is not None:
        require(digest == expected['prediction_sha256'] and native_hist == expected['native_evaluator_record'],
                'Changing loss methods changed paired predictions/full-GT histograms')
    torch.cuda.synchronize(); begin = time.monotonic()
    losses, loss_contract = checked_loss(model, prediction, sample['targets'], mode)
    torch.cuda.synchronize(); loss_seconds = time.monotonic()-begin
    scalar = {name: float(value.detach().item()) for name, value in losses.items()}
    family_losses = {family: sum(losses['loss_voxel_'+family+'_inter_'+str(i)] for i in range(3)) for family in FAMILIES}
    total = sum(losses.values()); family_gradients = {}; family_vectors = {}; pairwise = []
    before_vjp = dict(allocated_bytes=int(torch.cuda.memory_allocated()), reserved_bytes=int(torch.cuda.memory_reserved()))
    if mode == 'coarse':
        for family in FAMILIES:
            event('FAMILY_VJP', role=role, family=family, token=sample['record']['sample_token'])
            family_vectors[family], family_gradients[family] = get_gradient(family_losses[family], named_parameters, retain_graph=True)
        for i, first in enumerate(FAMILIES):
            for second in FAMILIES[i+1:]:
                pairwise.append(dict(left=first, right=second, **comparison(family_vectors[first], family_vectors[second])))
    peak_with_family_vjps = dict(allocated_bytes=int(torch.cuda.max_memory_allocated()), reserved_bytes=int(torch.cuda.max_memory_reserved()))
    event('TOTAL_VJP', mode=mode, role=role, token=sample['record']['sample_token'])
    total_vector, total_gradient = get_gradient(total, named_parameters, retain_graph=False)
    require(math.isfinite(total_gradient['L2']) and total_gradient['L2'] > 0,
            'Complete future-head gradient is absent/nonfinite')
    if mode == 'coarse':
        for family in FAMILIES:
            family_gradients[family]['versus_total'] = comparison(family_vectors[family], total_vector)
        summed = sum(family_vectors.values(), start=np.zeros_like(total_vector))
        reconstruction = dict(max_abs_error=float(np.max(np.abs(summed-total_vector))),
                              comparison=comparison(summed, total_vector),
                              note='Floating summation and repeated VJP paths can differ; this is an audit statistic, not exact equality')
        del summed
    else:
        reconstruction = None
    require(all(parameter.grad is None for _, parameter in named_parameters), 'autograd.grad unexpectedly wrote parameter .grad buffers')
    row = dict(mode=mode, role=role, sample_token=sample['record']['sample_token'], scene_token=sample['record']['scene_token'],
        prediction_sha256=digest, native_evaluator_record=native_hist, original_loss_components=scalar,
        original_loss_family_sums={k: float(v.detach().item()) for k, v in family_losses.items()},
        original_loss_total=float(total.detach().item()), actual_loss_contract=loss_contract,
        family_gradients=family_gradients, family_pairwise_cosines=pairwise, total_gradient=total_gradient,
        family_sum_vs_total=reconstruction, before_VJP_memory=before_vjp,
        peak_through_family_VJP=peak_with_family_vjps,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()), peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        forward_device_seconds=forward_seconds, loss_device_seconds_including_GT_checks=loss_seconds,
        total_VJP_device_seconds=total_gradient['vjp_device_seconds'],
        extra_family_VJP_device_seconds=sum(x['vjp_device_seconds'] for x in family_gradients.values()),
        resource_scope='independent eval/grad-enabled forward per mode; coarse family VJPs retain one graph, total VJP frees it; no AdamW/.grad accumulation',
        pairing_verified=expected is not None, performance_confirmation_sample=False)
    del output, prediction, losses, family_losses, total, family_vectors
    gc.collect(); torch.cuda.empty_cache()
    return row, total_vector


def report(out, result):
    lines = ['# 完整分辨率监督的无 optimizer 预检', '',
        '本运行没有训练或保存 checkpoint。所有模型均为初始 M0/native1，eval 模式关闭 dropout，以 valid_frames=[1,2,3,4] 和 enable_grad 保留原生未来链梯度。coarse/full 在两条固定训练 anchor 上独立前向，预测摘要与完整 GT 混淆矩阵精确一致。', '',
        '| 模式/角色 | 原12项loss总和 | 总梯度L2 | Forward秒 | Loss秒含标签核验 | Total VJP秒 | Peak allocated GiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in result['records']:
        lines.append(f"| {r['mode']}/{r['role']} | {r['original_loss_total']:.6g} | {r['total_gradient']['L2']:.6g} | {r['forward_device_seconds']:.3f} | {r['loss_device_seconds_including_GT_checks']:.3f} | {r['total_VJP_device_seconds']:.3f} | {r['peak_allocated_bytes']/2**30:.3f} |")
    lines += ['', 'coarse 额外分项 VJP 的重复开销单独列于 JSON；其 retain_graph 峰值不是普通训练一次 backward 的峰值。full 只计算 total VJP，但仍没有 optimizer moment、.grad 累积和训练模式 dropout buffer，因此不能把表中峰值/秒数直接称正式训练预算。', '',
        'rare anchor 是依据已完成 Train16 诊断明确加入的后验极端损失样本，只作 coarse 梯度来源检查；不用于 full 资源外推或性能确认，不从任何已有统计中删除。三条 coarse 样本之间的总梯度 cosine 是三个固定样本的局部诊断，不能等同原训练的四样本累积、clip 或 AdamW 更新方向。', '',
        'coarse/full 均保留原 CE 权重、三层/五时域、sem/geo/Lovász 公式及 FP32 概率路径。没有采样 Lovász、稳定化、坐标改造或 teacher。分项零梯度对应 cosine=null，完整总梯度必须有限且非零。', '',
        '是否开展正式 512 更新由另行冻结的协议及资源门决定；本程序不会自动启动训练，PASS 仅表示本次工程检查完成。', '']
    (out/'report.md').write_text('\n'.join(lines))


def run(a, out, started):
    parent, sources, runtime = verify_contract(a)
    sys.path.insert(0, str(Path(a.repo).resolve()))
    event('DEPENDENCY_IMPORT_BEGIN')
    import torch
    from memory_experiment import cached_records, prepare_model, seed_all, state_digest
    from native_state_cache import load_sample
    from posthoc_train_diagnostic import select_training_diagnostic
    require(a.device.startswith('cuda') and torch.cuda.is_available(), 'GPU is required for the native resource preflight')
    torch.cuda.set_device(torch.device(a.device)); torch.set_num_threads(2)
    total_memory = torch.cuda.get_device_properties(torch.device(a.device)).total_memory
    cap = int(a.max_allocated_gib*2**30)
    require(0 < cap <= total_memory, 'Allocator cap exceeds device capacity')
    torch.cuda.set_per_process_memory_fraction(cap/total_memory, device=torch.device(a.device))
    seed_all(11); event('DEPENDENCY_IMPORT_COMPLETE', allocator_cap_bytes=cap)
    cache = Path(a.cache); index = read(cache/'index.json'); records = cached_records(cache)
    require(index['split'] == 'train' and len(records) == 512 and index['config_sha256'] == parent['config_sha256'] and
            index['selection_sha256'] == parent['selection_sha256'] and index['extractor_sha256'] == parent['source_sha256']['native_state_cache.py'], 'Wrong native training source')
    require(index['inputs_targets_separate'] is True and index['future_occupancy_is_input'] is False and
            index['native_model']['radar_contract'] == 'native_loader_B_without_common_source_override', 'Native input/GT contract changed')
    selection_path = Path(a.protocol).with_name('selection_v1.json')
    require(sha(selection_path) == parent['selection_sha256'], 'Frozen selection changed')
    keys = ('sample_token', 'scene_token', 'official_index', 'split')
    planned = [r for r in read(selection_path)['records'] if r['split'] == 'train']
    require([[r[k] for k in keys] for r in records] == [[r[k] for k in keys] for r in planned], 'Original training sequence differs')
    selected16 = select_training_diagnostic(records); normal = selected16[:2]; rare = selected16[15]
    require(tuple(r['sample_token'] for r in normal) == PAIR_TOKENS and rare['sample_token'] == RARE_TOKEN,
            'Fixed first-two/explicit posthoc rare identity mismatch')
    manifest = dict(schema='m0-full-supervision-preflight-manifest-v1', script_sha256=sha(__file__),
        parent_protocol_sha256=PARENT_SHA, m0_checkpoint_sha256=sha(a.checkpoint), config_sha256=sha(a.config),
        train_cache_index_sha256=TRAIN_INDEX_SHA, source_sha256=sources, runtime_source_sha256=runtime,
        source_selection_sha256=parent['selection_sha256'],
        selected=[dict(**{k: r[k] for k in keys}, role='normal_'+str(i+1) if i < 2 else 'posthoc_rare',
            input_sha256=r['files']['inputs']['sha256'], target_sha256=r['files']['targets']['sha256']) for i, r in enumerate(normal+[rare])],
        rare_selection='explicit posthoc extreme-loss anchor, original first16 last member; not performance confirmation or exclusion',
        precision=parent['numerical_policy'], seed=11, dropout=False, valid_frames=[1, 2, 3, 4],
        maximum_seconds=a.max_seconds, allocator_cap_bytes=cap, total_device_bytes=total_memory,
        optimizer_created=False, gradient_vectors_saved=False, automatic_training=False)
    write(out/'manifest.json', manifest)
    rows = []; coarse_by_token = {}; coarse_vectors = {}; adapter_receipt = None; local_cross_sample = []
    for mode, current in [('coarse', normal+[rare]), ('full', normal)]:
        seed_all(11); model, migration, params = prepare_model(a.config, a.checkpoint, 'native1')
        require(state_digest(model.future_pred_head) == INITIAL_HEAD_SHA and
                model.memory_queue_len == model.future_pred_head.memory_queue_len == 1 and
                model.future_pred_head.history_queue_length == 2, 'Native single-slot initialization/history changed')
        named_parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        require([name for name, _ in named_parameters] == list(params) and
                sum(p.numel() for _, p in named_parameters) == 13274016, 'Wrong full future-head gradient scope')
        if mode == 'full':
            adapter_receipt = install_full_resolution(model.future_pred_head)
            require(state_digest(model.future_pred_head) == INITIAL_HEAD_SHA, 'Loss adapter changed head weights')
        for i, record in enumerate(current):
            role = 'normal_'+str(i+1) if i < 2 else 'posthoc_rare'
            sample = load_sample(cache, record, device=a.device)
            event('SAMPLE_BEGIN', mode=mode, role=role, token=record['sample_token'])
            row, vector = measure(model, sample, mode, role, named_parameters,
                                  coarse_by_token.get(record['sample_token']) if mode == 'full' else None)
            if mode == 'coarse':
                coarse_by_token[record['sample_token']] = row
                if i < 2:
                    coarse_vectors[record['sample_token']] = vector
                else:
                    for other in normal:
                        local_cross_sample.append(dict(left='posthoc_rare', right=other['sample_token'],
                            **comparison(vector, coarse_vectors[other['sample_token']])))
                    local_cross_sample.append(dict(left=normal[0]['sample_token'], right=normal[1]['sample_token'],
                        **comparison(coarse_vectors[normal[0]['sample_token']], coarse_vectors[normal[1]['sample_token']])))
            else:
                row['full_vs_same_anchor_coarse_total_gradient'] = comparison(vector, coarse_vectors.pop(record['sample_token']))
            rows.append(row); write(out/'progress.json', dict(status='RUNNING', records=rows, coarse_cross_sample_gradients=local_cross_sample,
                seconds=time.monotonic()-started))
            del sample, vector
        require(state_digest(model.future_pred_head) == INITIAL_HEAD_SHA and all(p.grad is None for _, p in named_parameters),
                'Preflight modified M0 tensors or accumulated .grad')
        del model, params, named_parameters; gc.collect(); torch.cuda.empty_cache()
    require(not coarse_vectors, 'Temporary gradient vectors not released')
    require({k: sha(Path(__file__).with_name(k)) for k in sources} == sources and sha(a.checkpoint) == manifest['m0_checkpoint_sha256'],
            'Frozen source/initial checkpoint changed')
    result = dict(schema='m0-full-supervision-preflight-v1', status='PASS_NO_OPTIMIZER_PREFLIGHT',
        manifest_sha256=sha(out/'manifest.json'), adapter_receipt=adapter_receipt, records=rows,
        coarse_posthoc_cross_sample_total_gradients=local_cross_sample,
        paired_prediction_digest_and_original_fullGT_histograms_exact=True, full_GT_not_downsampled=True,
        unchanged_initial_head_sha256=INITIAL_HEAD_SHA, optimizer_updates=0, checkpoint_saved=False,
        automatic_training=False, formal_training_resource_gate='NOT_DECIDED_BY_THIS_PREFLIGHT',
        resource_limitations=['Eval/dropout-free VJP, not original training-mode backward or optimizer step.',
            'Coarse peak includes four retained-graph family VJPs; full peak includes total VJP only.',
            'No .grad accumulation buffers or AdamW moment states; CPU gradient vectors were transient only.',
            'Two normal anchors are not a worst-case full-training memory/time guarantee; rare is excluded from resource extrapolation.'],
        seconds=time.monotonic()-started)
    write(out/'summary.json', result); report(out, result)
    write(out/'complete.json', dict(status=result['status'], summary_sha256=sha(out/'summary.json'), manifest_sha256=sha(out/'manifest.json'),
        report_sha256=sha(out/'report.md'), optimizer_updates=0, checkpoint_saved=False, automatic_training=False,
        seconds=time.monotonic()-started))
    event('COMPLETE', seconds=time.monotonic()-started)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'config', 'checkpoint', 'cache', 'protocol', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--max-seconds', type=int, default=600)
    p.add_argument('--max-allocated-gib', type=float, default=64.)
    a = p.parse_args(argv)
    require(0 < a.max_seconds <= 600 and math.isfinite(a.max_allocated_gib) and 0 < a.max_allocated_gib <= 80,
            'Only a bounded <=600s, <=80GiB allocator preflight is supported')
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    def stop(number, _frame):
        raise InterruptedError('No-optimizer preflight deadline/signal '+str(number)+'; no retry')
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(number, stop)
    signal.alarm(a.max_seconds)
    try:
        run(a, out, started)
    except BaseException as exc:
        write(out/'failed.json', dict(status='FAILED_OR_INTERRUPTED_NO_RETRY', error=repr(exc),
            optimizer_updates=0, checkpoint_saved=False, automatic_training=False, seconds=time.monotonic()-started))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
