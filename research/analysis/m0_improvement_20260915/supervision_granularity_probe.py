"""Read-only supervision audit on the exact completed posthoc train16 sample set.

Four fixed models, eval/no_grad, no optimizer or tensors saved. Full-resolution
confusions must exactly reproduce the existing posthoc record for every model
and sample. Actual compute_occ_loss/loss_voxel arguments determine axis/layout;
an exact source prefix supplies the native loss interpolation and coarse GT.
Per-horizon soft statistics are descriptive decompositions, NOT an additive
decomposition of the original five-horizon pooled nonlinear losses.
"""
import argparse
import ast
import copy
import datetime
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

MODELS = ('M0_fp32', 'native1', 'persistent2', 'rolling2')
HORIZONS = (0., .5, 1., 1.5, 2.)
PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
POSTHOC_SOURCE_SHA = '15a7f27f03d15a6e980358f6f0be1415c80e34abc91dcf83cc2caec5baaf7048'
POSTHOC_SUMMARY_SHA = 'e589ff427824d90b40a286ec3b9ba77301c42b9be31d0b7cb4edea49791642d2'
INITIAL_HEAD_SHA = '6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def event(name, **values):
    print(json.dumps(dict(event=name, **values), allow_nan=False), flush=True)


def tensor_digest(value):
    value = value.detach().to(device='cpu').contiguous()
    h = hashlib.sha256(); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes()); return h.hexdigest()


def native_loss_prefix(bound_method, expected_source_sha):
    """Compile unchanged native loss_voxel prefix ending BEFORE loss_dict={}."""
    path = Path(inspect.getsourcefile(bound_method)).resolve()
    require(sha(path) == expected_source_sha, 'Native loss implementation source changed')
    node = ast.parse(textwrap.dedent(inspect.getsource(bound_method))).body[0]
    require(isinstance(node, ast.FunctionDef) and node.name == 'loss_voxel' and not node.decorator_list,
            'Expected the ordinary original loss_voxel method')
    stop = [i for i, item in enumerate(node.body) if isinstance(item, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == 'loss_dict' for target in item.targets)]
    require(len(stop) == 1, 'Ambiguous native loss prefix boundary')
    node.body = copy.deepcopy(node.body[:stop[0]]) + [ast.Return(value=ast.Tuple(
        elts=[ast.Name(id='output_voxels', ctx=ast.Load()), ast.Name(id='target_voxels', ctx=ast.Load())], ctx=ast.Load()))]
    namespace = dict(bound_method.__func__.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['loss_voxel'], dict(source_file=str(path), source_sha256=sha(path),
        transformation='unchanged source prefix before loss_dict assignment; appended tensor return only',
        original_method='WorldHeadV1.loss_voxel', interpolation='actual native prefix, not reconstructed axes')


def safe_ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator/denominator)


def probability_stats(prob, truth, epsilon=0.):
    """Float32 sums, matching native scaling arithmetic; undefineds explicit."""
    p = prob.float(); y = truth.float()
    intersection = float((p*y).sum().item()); mass = float(p.sum().item())
    gt_mass = float(y.sum().item()); count = int(y.numel())
    correct_negative = float(((1-p)*(1-y)).sum().item()); negative_mass = count-gt_mass
    return dict(intersection=intersection, prediction_mass=mass, GT_mass=int(gt_mass), valid_voxels=count,
        correct_negative_mass=correct_negative, GT_negative_mass=int(negative_mass),
        precision=safe_ratio(intersection, mass+epsilon), recall=safe_ratio(intersection, gt_mass+epsilon),
        specificity=safe_ratio(correct_negative, negative_mass+epsilon), denominator_epsilon=epsilon,
        class_present=gt_mass > 0,
        note='Counters/reductions float32 as in native loss; ratios shown from serialized sufficient statistics')


def soft_statistics(logits, targets):
    import torch
    p = torch.softmax(logits, dim=1)
    valid = targets != 255; gt = targets[valid]
    require(gt.numel() > 0, 'No valid coarse targets; do not omit an empty group')
    p0, p1 = p[:, 0][valid], p[:, 1][valid]
    geo = 1-p0; positive = gt == 1
    mismatch = (p1 > 0) & (geo == 0)
    classes = [probability_stats(p0, gt == 0), probability_stats(p1, positive)]
    geometry = probability_stats(geo, positive, epsilon=1e-5)
    # geo_scal's specificity uses p0 itself, not a recomputed 1-(1-p0).
    geometry['correct_negative_mass'] = float((p0*(~positive).float()).sum().item())
    geometry['specificity'] = safe_ratio(geometry['correct_negative_mass'], geometry['GT_negative_mass']+1e-5)
    return dict(semantic_softmax_classes=classes, geometry_GMO_one_minus_p0=geometry,
        numerical_difference=dict(softmax1_positive_but_one_minus_p0_zero_voxels=int(mismatch.sum().item()),
            on_GT_GMO=int((mismatch & positive).sum().item()),
            max_absolute_probability_difference=float((p1-geo).abs().max().item()),
            semantic_minus_geometric_prediction_mass=classes[1]['prediction_mass']-geometry['prediction_mass']),
        reduction_dtype='float32', differentiable=False)


def label_structure(full, coarse):
    """Exact native 2x2x2 block footprint, with no prediction interpolation."""
    import torch
    require(full.ndim == coarse.ndim == 4 and len(full) == len(coarse) == 5, 'Expected actual five-frame loss GT')
    require(tuple(full.shape[1:]) == (512, 512, 40) and tuple(coarse.shape[1:]) == (256, 256, 20),
            'Unexpected actual native training resolution')
    rows = []
    for h in range(5):
        H, W, D = coarse.shape[1:]
        # Same block grouping expression as _downsample_occ_target, one horizon
        # at a time. These are explicit GT axes supplied by compute_occ_loss.
        blocks = full[h:h+1].reshape(1, H, 2, W, 2, D, 2).permute(0, 1, 3, 5, 2, 4, 6).reshape(-1, 8)
        fg, ign = (blocks == 1).sum(-1), (blocks == 255).sum(-1)
        bg = (blocks == 0).sum(-1)
        require(torch.all(fg+ign+bg == 8).item(), 'Unexpected full-GT class beyond 0/1/255')
        labels = coarse[h].reshape(-1)
        require(torch.all((labels == 0) | (labels == 1) | (labels == 255)).item(), 'Unexpected coarse-GT class')
        full_counts = [int(bg.sum().item()), int(fg.sum().item()), int(ign.sum().item())]
        coarse_counts = [int((labels == c).sum().item()) for c in (0, 1, 255)]
        joint = []
        for c in (0, 1, 255):
            mask = labels == c
            joint.append([int(bg[mask].sum().item()), int(fg[mask].sum().item()), int(ign[mask].sum().item())])
        require(np.array_equal(np.asarray(joint).sum(0), full_counts) and
                np.array_equal(np.asarray(joint).sum(1), np.asarray(coarse_counts)*8), 'Invalid block footprint accounting')
        rows.append(dict(horizon_seconds=HORIZONS[h], full_GT_counts_0_1_255=full_counts,
            coarse_GT_counts_0_1_255=coarse_counts,
            block_GMO_count_histogram_0_to_8=torch.bincount(fg, minlength=9).cpu().tolist(),
            block_joint_GMO_count_rows_ignore255_count_columns=torch.bincount(fg*9+ign, minlength=81).reshape(9, 9).cpu().tolist(),
            blocks_containing_ignore255=int((ign > 0).sum().item()),
            coarse_label_footprint_on_full_GT=joint,
            footprint_axes='rows coarse 0/1/255, columns original 0/1/255; counts original voxels',
            footprint_operation='each native coarse-label block covers its exact eight source GT positions; this is NOT trilinear prediction upsampling'))
        del blocks, fg, ign, bg, labels
    return dict(full_shape=list(full.shape), coarse_shape=list(coarse.shape), block_size=[2, 2, 2],
        coarse_GT_uint8_sha256=tensor_digest(coarse.to(dtype=torch.uint8)), horizons=rows)


def coarse_histograms(logits, targets):
    import torch
    predictions = logits.argmax(1); histograms = []
    for h in range(len(targets)):
        gt = targets[h].long(); valid = (gt >= 0) & (gt < logits.shape[1])
        histograms.append(torch.bincount(2*gt[valid]+predictions[h][valid], minlength=4).reshape(2, 2).cpu().tolist())
    return histograms


def observed_native_loss(model, prediction, targets, prefix, expected_GT_structure=None):
    """Per-instance observers preserve both original methods' return values."""
    import torch
    head = model.future_pred_head; native_occ, native_voxel = head.loss_occ, head.loss_voxel
    capture = dict(layers=[])
    def observe_occ(self, output_voxels=None, target_voxels=None, **kwargs):
        require('full' not in capture, 'Multiple native loss_occ calls')
        capture['full'] = target_voxels
        capture['actual_compute_loss_shape'] = list(output_voxels.shape)
        return native_occ(output_voxels, target_voxels, **kwargs)
    def observe_voxel(self, output_voxels, target_voxels, tag, target_voxels_prepared=False):
        require(target_voxels_prepared is True and tag == 'inter_'+str(len(capture['layers'])),
                'Native layer/prepared-target call order changed')
        coarse_logits, coarse_GT = prefix(self, output_voxels, target_voxels, tag,
                                           target_voxels_prepared=target_voxels_prepared)
        require(coarse_logits.shape == (5, 2, 256, 256, 20) and coarse_GT.shape == (5, 256, 256, 20),
                'Actual native loss-resolution output differs')
        require(torch.isfinite(coarse_logits).all().item(), 'Nonfinite coarse logits')
        if not capture['layers']:
            if expected_GT_structure is None:
                capture['GT_structure'] = label_structure(capture['full'], coarse_GT)
            else:
                require(tensor_digest(coarse_GT.to(dtype=torch.uint8)) == expected_GT_structure['coarse_GT_uint8_sha256'],
                        'Model-dependent coarse targets detected')
                capture['GT_structure'] = expected_GT_structure
        else:
            require(tensor_digest(coarse_GT.to(dtype=torch.uint8)) == capture['GT_structure']['coarse_GT_uint8_sha256'],
                    'Decoder layers used different coarse targets')
        capture['layers'].append(dict(layer_index=len(capture['layers']), tag=tag,
            raw_logit_shape=list(output_voxels.shape), native_loss_logit_shape=list(coarse_logits.shape),
            by_horizon=[dict(horizon_seconds=HORIZONS[h], **soft_statistics(coarse_logits[h:h+1], coarse_GT[h:h+1])) for h in range(5)],
            pooled_current_plus_four_future=soft_statistics(coarse_logits, coarse_GT)))
        if len(capture['layers']) == 3:
            capture['last_layer_coarse_hist_by_horizon'] = coarse_histograms(coarse_logits, coarse_GT)
        del coarse_logits
        return native_voxel(output_voxels, target_voxels, tag, target_voxels_prepared=target_voxels_prepared)
    head.loss_occ = types.MethodType(observe_occ, head); head.loss_voxel = types.MethodType(observe_voxel, head)
    try:
        losses = model.compute_occ_loss(prediction, targets)
    finally:
        head.loss_occ, head.loss_voxel = native_occ, native_voxel
    require(len(capture['layers']) == 3, 'Missing native intermediate losses')
    capture.pop('full')
    capture['original_loss_components'] = {k: float(v.detach().item()) for k, v in losses.items()}
    require(len(losses) == 12 and all(math.isfinite(v) for v in capture['original_loss_components'].values()),
            'Native 12-loss computation failed')
    return capture


def validate_sources(a):
    pkg = Path(__file__).resolve().parent
    require(sha(a.protocol) == PARENT_SHA, 'Only the frozen protocol_v2 is accepted')
    protocol = read(a.protocol)
    require(sha(a.config) == protocol['config_sha256'] and sha(a.checkpoint) == protocol['m0_sha256'],
            'Frozen M0/config identity differs')
    sources = dict(protocol['source_sha256'], **{'posthoc_train_diagnostic.py': POSTHOC_SOURCE_SHA})
    for name, digest in sources.items():
        require(Path(name).name == name and sha(pkg/name) == digest, 'Frozen producer source differs: '+name)
    runtime = {}
    for path, digest in protocol['runtime_source_sha256'].items():
        relative = path.split('/projects/', 1)
        require(len(relative) == 2, 'Unexpected frozen runtime path')
        local = Path(a.repo)/'projects'/relative[1]
        require(sha(local) == digest, 'Native runtime source differs: '+str(local))
        runtime[str(local.resolve())] = digest
    previous = Path(a.posthoc_root)
    require(not (previous/'failed.json').exists(), 'Previous posthoc diagnostic failed')
    complete, summary = read(previous/'complete.json'), read(previous/'summary.json')
    require(complete['status'] == 'COMPLETE' and complete['optimizer_updates'] == 0 and complete['posthoc_only'] is True,
            'Require a completed no-update posthoc diagnostic')
    require(sha(previous/'summary.json') == complete['summary_sha256'] == POSTHOC_SUMMARY_SHA,
            'Only the completed original train16 summary is accepted')
    require(summary['protocol_sha256'] == PARENT_SHA and summary['script_sha256'] == POSTHOC_SOURCE_SHA,
            'Posthoc protocol/source differs')
    require(summary['train_samples'] == summary['train_scenes'] == 16 and
            summary['semantics']['parameter_updates'] == 0 and summary['semantics']['inference_mode'] == 'eval_no_grad',
            'Posthoc sample/mode contract differs')
    prior_rows = {}; prior_files = {'summary.json': POSTHOC_SUMMARY_SHA, 'complete.json': sha(previous/'complete.json')}
    for model in MODELS:
        name = model+'_records.jsonl'; digest = complete['records_sha256'][model]
        require(sha(previous/name) == digest == summary['training'][model]['record_sha256'], 'Previous per-sample record hash differs')
        prior_rows[model] = [json.loads(line) for line in (previous/name).read_text().splitlines() if line.strip()]
        require(len(prior_rows[model]) == 16, 'Previous model record coverage differs')
        prior_files[name] = digest
    return protocol, sources, runtime, summary, prior_rows, prior_files


def model_summaries(rows):
    """Observed coarse and full metrics remain separately named and scored."""
    from posthoc_train_diagnostic import _summarize_hist
    full = np.sum(np.asarray([r['native_full_hist_by_horizon'] for r in rows], dtype=np.int64), axis=0)
    coarse = np.sum(np.asarray([r['last_layer_coarse_hist_by_horizon'] for r in rows], dtype=np.int64), axis=0)
    return dict(full_native_GT=_summarize_hist(full), coarse_native_training_GT=_summarize_hist(coarse),
        full_histogram_exact_previous_posthoc=True,
        metric_warning='Coarse and full IoUs use different voxel populations and ignore rules; neither replaces the other')


def render_report(out, result):
    lines = ['# 固定训练样本的监督粒度诊断', '',
        '同一原始 16 个训练场景各首个 anchor，四个固定最终模型；所有操作为 eval/no_grad。全部完整 GT 混淆矩阵与上一轮逐样本记录精确相同。本结果为事后训练内诊断，不是新验证集成绩。', '',
        '| 模型 | 完整 GT future macro GMO % | 原生粗 GT future macro GMO % |', '|---|---:|---:|']
    for model in MODELS:
        values = result['models'][model]
        lines.append(f"| {model} | {100*values['full_native_GT']['future_macro_gmo']:.4f} | {100*values['coarse_native_training_GT']['future_macro_gmo']:.4f} |")
    rows = result['GT_structure_by_sample']; footprints = np.zeros((5, 3, 3), dtype=np.int64)
    for row in rows:
        footprints += np.asarray([h['coarse_label_footprint_on_full_GT'] for h in row['structure']['horizons']], dtype=np.int64)
    lines += ['', '| 时域秒 | 粗 ignore 覆盖的原有效 GMO | 粗 GMO 覆盖的原 non-GMO | 粗 non-GMO 覆盖的原 GMO |', '|---|---:|---:|---:|']
    for h, second in enumerate(HORIZONS):
        lines.append(f'| {second:g} | {footprints[h,2,1]} | {footprints[h,1,0]} | {footprints[h,0,1]} |')
    lines += ['', '这里的覆盖仅指每个 2×2×2 GT block 的原始位置归属，不是把模型预测作 nearest 或 trilinear 上采样；完整评测没有改 mask。原/粗标签计数、每块 nGMO=0…8 与 ignore255 的联合直方图见 summary.json。', '',
        '每个模型的 records.jsonl 保存三层、五时域的 soft intersection、prediction mass、GT mass、precision/recall/specificity，并另外保存原来跨五时域池化的同类统计。逐时域非线性指标不能相加还原原 loss。分子分母使用原生 float32 归约，展示比值在转为 Python float 后计算，并非原生 FP32 除法/BCE 的逐位重建；原 12 项 loss 由原函数独立计算与核验。', '',
        'geo_scal 使用 1−softmax(class0)，sem_scal 使用 softmax(class1)；两者的 FP32 相消差异独立记录，包括前景标签处 softmax1>0 但 1−p0=0 的数量。概率统计不是硬 argmax precision/recall。原 12 项 loss 仍调用原函数计算，没有损失替换。', '',
        '未做反向传播、optimizer、阈值或 checkpoint 选择，没有删除极端样本，也没有保存大张量。粗网格改善或差异集中本身仍不能证明训练梯度的因果来源或总体性能提升。', '']
    (out/'report.md').write_text('\n'.join(lines))


def run(a, out, started):
    protocol, sources, runtime, previous, prior_rows, prior_files = validate_sources(a)
    sys.path.insert(0, str(Path(a.repo).resolve()))
    event('DEPENDENCY_IMPORT_BEGIN')
    import torch
    from aggregate_memory import load_cache, load_arm, load_precision_reference
    from memory_experiment import cached_records, prepare_model, seed_all, state_digest
    from native_state_cache import load_sample, replay
    from posthoc_train_diagnostic import select_training_diagnostic, verify_final_payload
    require(a.device.startswith('cuda') and torch.cuda.is_available(), 'Native fixed-model diagnostic requires CUDA')
    torch.cuda.set_device(torch.device(a.device)); torch.set_num_threads(2); seed_all(11)
    event('DEPENDENCY_IMPORT_COMPLETE', torch=torch.__version__)
    train = Path(a.train_cache); train_rows = cached_records(train); index = read(train/'index.json')
    require(sha(train/'index.json') == previous['train_cache_index_sha256'], 'Training cache differs from original posthoc')
    require(index['split'] == 'train' and len(train_rows) == 512 and
            index['config_sha256'] == protocol['config_sha256'] and index['selection_sha256'] == protocol['selection_sha256'] and
            index['extractor_sha256'] == protocol['source_sha256']['native_state_cache.py'], 'Native train-cache provenance differs')
    require(index['inputs_targets_separate'] is True and index['future_occupancy_is_input'] is False and
            index['native_model']['radar_contract'] == 'native_loader_B_without_common_source_override' and
            index['numerical_policy']['tf32_matmul'] is True and index['numerical_policy']['tf32_cudnn'] is True,
            'Original native observer/GT contract differs')
    selection_path = Path(a.protocol).with_name('selection_v1.json')
    require(sha(selection_path) == protocol['selection_sha256'], 'Original selection SHA differs')
    planned = [r for r in read(selection_path)['records'] if r['split'] == 'train']
    keys = ('sample_token', 'scene_token', 'official_index', 'split')
    require([[r[k] for k in keys] for r in train_rows] == [[r[k] for k in keys] for r in planned], 'Training identities/order differ')
    selected = select_training_diagnostic(train_rows)
    for model in MODELS:
        require([(r['sample_token'], r['scene_token']) for r in selected] ==
                [(r['sample_token'], r['scene_token']) for r in prior_rows[model]], 'Exact posthoc selection/order differs')
    dev, _, dev_rows, native_dev, _ = load_cache(a.dev_cache, protocol, a.protocol)
    dev_sha = sha(dev/'index.json'); require(dev_sha == previous['development_cache_index_sha256'], 'Development provenance differs')
    receipts = {}
    for model in MODELS[1:]:
        _, receipts[model] = load_arm(a.runs_root, model, dev_rows, native_dev, protocol, a.protocol, dev_sha)
        require(receipts[model]['train_index_sha256'] == sha(train/'index.json'), 'Completed model trained on different cache')
        require(receipts[model]['checkpoint_sha256'] == previous['loaded_checkpoints'][model]['checkpoint_sha256'],
                'Model is not the exact previous posthoc fixed-final checkpoint')
    _, receipts['M0_fp32'] = load_precision_reference(a.runs_root, dev_rows, native_dev, protocol, a.protocol, dev_sha)
    manifest = dict(schema='m0-supervision-granularity-manifest-v1', protocol_sha256=PARENT_SHA,
        script_sha256=sha(__file__), config_sha256=sha(a.config), m0_sha256=sha(a.checkpoint),
        frozen_sources_sha256=sources, runtime_sources_sha256=runtime, previous_files_sha256=prior_files,
        train_cache_index_sha256=sha(train/'index.json'), development_cache_index_sha256=dev_sha,
        selection_source_sha256=protocol['selection_sha256'], sample_selection_rule='Exact original first 16 distinct train scenes, first anchor; old per-model identity sequence required',
        selected=[dict(**{k: r[k] for k in keys}, input_sha256=r['files']['inputs']['sha256'],
                       target_sha256=r['files']['targets']['sha256']) for r in selected],
        numerical_policy=protocol['numerical_policy'], source_receipts=receipts,
        maximum_seconds=a.max_seconds, optimizer_updates=0, gradients=False, large_tensors_saved=False)
    write(out/'manifest.json', manifest)
    structure = {}; loaded = {}; all_rows = {}; extraction_receipt = None
    for model_name in MODELS:
        seed_all(11); mode = 'native1' if model_name == 'M0_fp32' else model_name
        model, migration, _ = prepare_model(a.config, a.checkpoint, mode)
        if model_name == 'M0_fp32':
            require(state_digest(model.future_pred_head) == INITIAL_HEAD_SHA, 'Frozen M0 head differs')
            loaded[model_name] = dict(head_tensor_sha256=INITIAL_HEAD_SHA, optimizer_updates=0, strict=True)
        else:
            directory = Path(a.runs_root)/model_name; path = directory/'latest.pth'
            digest = receipts[model_name]['checkpoint_sha256']; require(sha(path) == digest, 'Final checkpoint hash differs')
            before = path.stat(); payload = torch.load(str(path), map_location='cpu'); after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Checkpoint changed while reading')
            payload.pop('optimizer', None)
            loaded[model_name] = verify_final_payload(payload, read(directory/'manifest.json'), protocol,
                                                      model_name, digest, model, state_digest)
            require(loaded[model_name]['loaded_head_tensor_sha256'] == previous['loaded_checkpoints'][model_name]['loaded_head_tensor_sha256'],
                    'Actual fixed-final head tensor hash differs from previous posthoc')
            del payload
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval(); initial_state = state_digest(model.future_pred_head); seed_all(11)
        head_file = Path(inspect.getsourcefile(model.future_pred_head.loss_voxel)).resolve()
        require(str(head_file) in runtime, 'Actual loss head source is outside frozen runtime contract')
        prefix, receipt = native_loss_prefix(model.future_pred_head.loss_voxel, runtime[str(head_file)])
        if extraction_receipt is None:
            extraction_receipt = receipt
        else:
            require(receipt == extraction_receipt, 'Different loss geometry among models')
        rows = []; torch.cuda.reset_peak_memory_stats(); event('MODEL_BEGIN', model=model_name)
        with torch.no_grad(), (out/(model_name+'_records.jsonl')).open('x') as f:
            for ordinal, record in enumerate(selected):
                begin = time.monotonic(); token = record['sample_token']
                sample = load_sample(train, record, device=a.device)
                output = replay(model, sample, training=False)
                require(output[0].dtype == torch.float32, 'Native prediction dtype changed')
                native_records = model.evaluate_occ_records(output[0], sample['targets'], sample['inputs']['img_metas'])
                require(len(native_records) == 1, 'Expected single native sample')
                raw = native_records[0]; expected = prior_rows[model_name][ordinal]
                require(raw['sample_token'] == token and raw['scene_token'] == record['scene_token'] and
                        raw['horizon_seconds'] == list(HORIZONS), 'Native evaluator identity/horizon differs')
                full_hist = np.asarray(raw['hist_by_horizon'], dtype=np.int64)
                require(np.array_equal(full_hist, expected['hist_by_horizon']), 'Full-GT prediction differs from completed posthoc; reject changed replay')
                capture = observed_native_loss(model, output[0], sample['targets'], prefix, structure.get(token))
                if token not in structure:
                    structure[token] = capture['GT_structure']
                gt_rows = structure[token]['horizons']
                require(np.array_equal(full_hist.sum(2), [h['full_GT_counts_0_1_255'][:2] for h in gt_rows]), 'Full-GT class count mismatch')
                coarse_hist = np.asarray(capture['last_layer_coarse_hist_by_horizon'], dtype=np.int64)
                require(np.array_equal(coarse_hist.sum(2), [h['coarse_GT_counts_0_1_255'][:2] for h in gt_rows]), 'Coarse-GT class count mismatch')
                old_loss = expected['original_loss_components']; new_loss = capture['original_loss_components']
                require(set(old_loss) == set(new_loss) and all(math.isclose(new_loss[k], old_loss[k], rel_tol=1e-5, abs_tol=1e-6) for k in old_loss),
                        'Original loss no longer matches previous fixed-model diagnostic')
                capture.pop('GT_structure')
                row = dict(sample_token=token, scene_token=record['scene_token'], official_index=record['official_index'],
                    native_full_hist_by_horizon=full_hist.tolist(), full_hist_exact_previous_posthoc=True,
                    coarse_GT_uint8_sha256=structure[token]['coarse_GT_uint8_sha256'],
                    original_loss_max_abs_delta_vs_previous=max(abs(new_loss[k]-old_loss[k]) for k in old_loss),
                    original_loss_parity_tolerance=dict(rtol=1e-5, atol=1e-6),
                    seconds=time.monotonic()-begin, **capture)
                f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush(); rows.append(row)
                event('SAMPLE', model=model_name, completed=ordinal+1, planned=16, seconds=row['seconds'])
                del output, sample, capture, native_records
        require(state_digest(model.future_pred_head) == initial_state, 'Diagnostic changed model weights')
        loaded[model_name]['parameters_unchanged'] = True
        loaded[model_name]['maximum_cuda_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        all_rows[model_name] = rows
        write(out/'progress.json', dict(status='RUNNING', completed_models=list(all_rows), seconds=time.monotonic()-started))
        del model; gc.collect(); torch.cuda.empty_cache()
    require({name: sha(Path(__file__).with_name(name)) for name in sources} == sources, 'Frozen source changed during diagnostic')
    require(sha(a.checkpoint) == manifest['m0_sha256'], 'M0 checkpoint changed')
    result = dict(schema='m0-supervision-granularity-v1', status='COMPLETE_READ_ONLY_DIAGNOSTIC',
        manifest_sha256=sha(out/'manifest.json'), loaded_models=loaded, loss_geometry_extraction=extraction_receipt,
        samples=16, scenes=16, models={m: model_summaries(rows) for m, rows in all_rows.items()},
        GT_structure_by_sample=[dict(sample_token=r['sample_token'], scene_token=r['scene_token'], structure=structure[r['sample_token']]) for r in selected],
        records_sha256={m: sha(out/(m+'_records.jsonl')) for m in MODELS},
        semantics=dict(inference='eval/no_grad', optimizer_updates=0, gradients=False, checkpoint_selection=False,
            threshold_selection=False, posthoc=True, training_set_is_in_sample=True, input_GT_mask_loss_unchanged=True,
            per_horizon_soft_statistics_are_not_additive_native_loss_decomposition=True,
            original_loss_scope='three layers, current plus four future pooled per layer',
            label_footprints_are_block_membership_not_interpolated_prediction=True,
            coarse_scores_do_not_replace_full_GT_score=True, large_tensors_saved=False),
        seconds=time.monotonic()-started)
    write(out/'summary.json', result); render_report(out, result)
    write(out/'complete.json', dict(status='COMPLETE', summary_sha256=sha(out/'summary.json'),
        manifest_sha256=sha(out/'manifest.json'), report_sha256=sha(out/'report.md'), records_sha256=result['records_sha256'],
        optimizer_updates=0, gradients=False, full_confusion_exact_previous_posthoc=True, seconds=time.monotonic()-started))
    event('COMPLETE', out=str(out), seconds=time.monotonic()-started)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'config', 'checkpoint', 'train-cache', 'dev-cache', 'runs-root', 'protocol', 'posthoc-root', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--max-seconds', type=int, default=600)
    a = p.parse_args(argv); require(0 < a.max_seconds <= 600, 'Single diagnostic wall cap is 600 seconds')
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    def stop(number, _frame):
        raise InterruptedError('Read-only supervision probe deadline/signal '+str(number)+'; no retry')
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(number, stop)
    signal.alarm(a.max_seconds)
    try:
        run(a, out, started)
    except BaseException as exc:
        write(out/'failed.json', dict(status='FAILED_OR_INTERRUPTED', error=repr(exc),
            seconds=time.monotonic()-started, automatic_retry=False, optimizer_updates=0))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
