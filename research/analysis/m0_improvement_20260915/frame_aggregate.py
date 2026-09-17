"""CPU-only, four-model development aggregation for the reference-frame trial.

No predictions, labels, thresholds, weights or selection files are changed.
All identities and producer receipts are checked before computing effects.
The frozen parent supplies metric arithmetic and its existing cache readers;
this module implements scene resampling for exactly four real models.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np

MODELS = ('M0', 'M0_fp32', 'native1', 'reference')
CONTRASTS = (('reference', 'M0_fp32'), ('reference', 'native1'),
             ('reference', 'M0'), ('M0_fp32', 'M0'))
PRIMARY = set(CONTRASTS[:2])
PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
METRIC_KEYS = ('future_macro_gmo', 'future_macro_binary_miou',
               'future_pooled_gmo', 'future_pooled_binary_miou')
HORIZONS = (0., .5, 1., 1.5, 2.)


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


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def frozen_parent(parent):
    path = Path(__file__).with_name('aggregate_memory.py')
    require(sha(path) == parent['source_sha256']['aggregate_memory.py'],
            'Frozen parent aggregate source differs')
    spec = importlib.util.spec_from_file_location('_frozen_memory_metrics', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_contracts(a):
    require(sha(a.parent_protocol) == PARENT_SHA, 'Wrong frozen parent protocol')
    parent, frame = read(a.parent_protocol), read(a.protocol)
    require(frame['schema'] == 'frame-consistent-training-protocol-v1' and
            frame['status'] == 'FROZEN_BEFORE_REFERENCE_TRAINING' and frame['arm'] == 'reference',
            'Only the frozen reference-frame protocol is accepted')
    require(frame['parent_protocol_sha256'] == PARENT_SHA, 'Frame/parent binding differs')
    for key in ('training', 'numerical_policy', 'config_sha256', 'm0_sha256', 'selection_sha256'):
        require(frame[key] == parent[key], 'Frame/C contract mismatch: '+key)
    hp = frame['training']
    require(hp['seed'] == 11 and hp['passes'] == 4 and hp['train_samples'] == 512 and
            hp['development_samples'] == 200 and hp['accumulate'] == 4,
            'This summary requires the fixed 512-update development screen')
    require('frame_aggregate.py' in frame['source_sha256'] and 'frame_train.py' in frame['source_sha256'],
            'Summary/trainer source must be frozen')
    pkg = Path(__file__).resolve().parent
    sources = dict(parent['source_sha256'])
    for name, digest in frame['source_sha256'].items():
        require(name not in sources or sources[name] == digest, 'Conflicting parent/frame source SHA: '+name)
        sources[name] = digest
    for name, digest in sources.items():
        require(Path(name).name == name and sha(pkg/name) == digest, 'Frozen package source differs: '+name)
    require(parent['runtime_source_sha256'].items() <= frame['runtime_source_sha256'].items(),
            'Frame runtime contract omits parent sources')
    for path, digest in frame['runtime_source_sha256'].items():
        require(Path(path).is_absolute() and sha(path) == digest, 'Frozen runtime source differs: '+path)
    module = frozen_parent(parent)
    spec = importlib.util.spec_from_file_location('_frozen_frame_trainer', pkg/'frame_train.py')
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    evidence = trainer.validate_engineering(frame, a.protocol, parent)
    return parent, frame, module, trainer, evidence


def check_log(directory, trainer, hp):
    rows = jsonl(directory/'training.jsonl')
    require(len(rows) == 512, 'Training log must contain exactly 512 updates')
    for index, row in enumerate(rows):
        require(row['update'] == index+1 and row['examples'] == 4*(index+1) and
                row['pass_index'] == index//128+1, 'Training update/exposure sequence differs')
        require(math.isclose(row['lr'], trainer.schedule_lr(index, hp), rel_tol=1e-14, abs_tol=0),
                'Training LR differs from original schedule')
        require(all(math.isfinite(row[k]) for k in ('loss_mean', 'loss_min', 'loss_max', 'grad_norm')),
                'Nonfinite training log')
    return dict(updates=512, examples=2048, training_log_sha256=sha(directory/'training.jsonl'),
                all_update_LRs_match_original=True)


def check_payload(directory, manifest, expected_sha, arm, planned, expected_state_sha=None):
    """Check actual fixed final CPU tensors; never instantiate a model/optimizer."""
    import torch
    torch.set_num_threads(2)
    path = directory/'latest.pth'
    require(path.is_file() and sha(path) == expected_sha, 'Required actual final checkpoint differs: '+arm)
    payload = torch.load(str(path), map_location='cpu')
    require(payload['arm'] == arm and payload['metadata'] == manifest and
            payload['pass_index'] == 4 and payload['update'] == 512 and payload['sample_orders'] == planned,
            'Final checkpoint identity/endpoint/exact sample orders differ: '+arm)
    tensors = payload['future_pred_head']; digest = hashlib.sha256(); shapes = {}
    require(tensors, 'Empty future head')
    for name, value in sorted(tensors.items()):
        require(torch.is_tensor(value) and torch.isfinite(value).all().item(), 'Nonfinite/missing final tensor')
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        shapes[name] = [list(value.shape), str(value.dtype)]
    final_state = digest.hexdigest()
    if expected_state_sha is not None:
        require(final_state == expected_state_sha, 'Final head tensor digest differs from producer receipt')
    groups = payload['optimizer']['param_groups']
    steps = [int(value['step'].item() if torch.is_tensor(value['step']) else value['step'])
             for value in payload['optimizer']['state'].values()]
    require(len(groups) == 1 and steps and min(steps) > 0 and max(steps) == 512,
            'Invalid/missing final optimizer step counters')
    order_sha = hashlib.sha256(json.dumps(planned, separators=(',', ':')).encode()).hexdigest()
    del payload
    return dict(checkpoint_sha256=expected_sha, checkpoint_rehashed=True, final_head_state_sha256=final_state,
                actual_payload_checked_on_CPU=True, sample_orders_sha256=order_sha,
                updates=512, passes=4, tensor_shapes=shapes, optimizer_groups=groups,
                optimizer_tensor_step_min=min(steps), optimizer_tensor_step_max=max(steps))


def load_reference(a, frame, parent, module, trainer, records, m0_hist, control_manifest):
    directory = Path(a.reference_run)
    require(not (directory/'failed.json').exists(), 'Reference has a failure receipt')
    done, manifest, trained = [read(directory/name) for name in ('complete.json', 'manifest.json', 'training_complete.json')]
    require(done['status'] == 'TRAINED_AND_DEVELOPMENT_EVALUATED' and trained['status'] == 'TRAINED_FIXED_FINAL',
            'Reference did not complete its fixed final training/evaluation')
    for receipt in (done, trained):
        require(receipt['updates'] == 512 and receipt['examples'] == 2048 and
                receipt['checkpoint_sha256'] == done['checkpoint_sha256'], 'Reference endpoint differs')
    require(done['frame_protocol_sha256'] == manifest['protocol_sha256'] == sha(a.protocol) and
            done['parent_protocol_sha256'] == manifest['parent_protocol_sha256'] == PARENT_SHA,
            'Reference protocol receipt differs')
    for field, filename in [('manifest_sha256', 'manifest.json'), ('control_reuse_sha256', 'control_reuse.json'),
                            ('initial_development_sha256', 'initial_development_records.jsonl'),
                            ('training_log_sha256', 'training.jsonl')]:
        require(done[field] == sha(directory/filename), 'Reference receipt hash differs: '+field)
    require(manifest['arm'] == 'reference' and manifest['seed'] == 11 and
            manifest['m0_sha256'] == parent['m0_sha256'] and
            manifest['initial_head_sha256'] == control_manifest['initial_head_sha256'] == module.NATIVE_INITIAL_HEAD_SHA,
            'Reference initial head/source differs from C')
    for key in ('numerical_policy', 'trainable_parameters', 'trainable_parameter_names', 'passes', 'examples_per_pass'):
        require(manifest[key] == control_manifest[key], 'Reference/C training scope differs: '+key)
    require(manifest['trainable_parameters'] == 13274016 and manifest['passes'] == 4 and manifest['examples_per_pass'] == 512,
            'Reference/C matched capacity or exposure differs')
    require(manifest['numerical_policy'] == parent['numerical_policy'] and
            manifest['source_sha256'] == frame['source_sha256'] and
            manifest['parent_source_sha256'] == parent['source_sha256'] and
            manifest['runtime_source_sha256'] == frame['runtime_source_sha256'], 'Reference source/precision differs')
    for role, key in [('train', 'train_index_sha256'), ('development', 'dev_index_sha256')]:
        require(manifest[key] == control_manifest[key] == frame['cache_index_sha256'][role], 'Reference/C cache differs')
    require(manifest['migration']['mode'] == 'reference' and
            manifest['migration']['adapter_sha256'] == frame['source_sha256']['frame_consistent_adapter.py'],
            'Reference did not use the fixed coordinate adapter')
    require(manifest['training_block']['modified_native_statements'] == 0 and
            manifest['training_block']['full_task_losses_unchanged'] is True and
            manifest['training_block']['additional_loss_terms'] == 0, 'Native training/loss block changed')
    reuse = read(directory/'control_reuse.json')
    require(manifest['control_reuse'] == reuse and reuse['files_sha256'] == frame['control_files_sha256'] and
            reuse['reused'] is True and reuse['new_control_training'] is False, 'Reference control-reuse receipt differs')
    initial = manifest['initial_development']
    require(initial['optimizer_updates'] == 0 and initial['head_state_sha256'] == module.NATIVE_INITIAL_HEAD_SHA and
            initial['samples'] == 200 and initial['sha256'] == done['initial_development_sha256'],
            'Initialization-impact diagnostic is not frozen before training')
    require(done['evaluation']['samples'] == 200 and
            done['evaluation']['sha256'] == sha(directory/'development_records.jsonl'), 'Reference evaluation hash differs')
    final_rows = jsonl(directory/'development_records.jsonl')
    initial_rows = jsonl(directory/'initial_development_records.jsonl')
    matrices = []
    for label, rows in [('final', final_rows), ('initial_diagnostic', initial_rows)]:
        require(len(rows) == len(records), 'Reference evaluation coverage differs: '+label)
        for i, (row, expected) in enumerate(zip(rows, records)):
            require(row['sample_token'] == expected['sample_token'] and row['scene_token'] == expected['scene_token'] and
                    row['horizon_seconds'] == list(HORIZONS), 'Reference evaluation identity/order/horizon differs')
            hist = module.hist_array(row['hist_by_horizon'])
            require(np.array_equal(hist.sum(2), m0_hist[i].sum(2)), 'Reference GT row counts differ')
            if label == 'final':
                matrices.append(hist)
    log = check_log(directory, trainer, parent['training'])
    payload = check_payload(directory, manifest, done['checkpoint_sha256'], 'reference',
                            trainer.planned_orders(parent['training']), done['final_head_state_sha256'])
    receipt = dict(files_sha256={name: sha(directory/name) for name in
        ('manifest.json', 'training_complete.json', 'complete.json', 'control_reuse.json',
         'training.jsonl', 'development_records.jsonl', 'initial_development_records.jsonl', 'latest.pth')},
        training=log, checkpoint=payload,
        initial_development='Identity/hash/GT counts verified; not added as a fifth model or used for selection')
    return np.stack(matrices), receipt, manifest


def interval(values):
    require(np.isfinite(values).all(), 'Nonfinite bootstrap score')
    return np.quantile(values, [.025, .975], axis=0).tolist()


def four_model_bootstrap(histograms, scene_tokens, metric_arrays):
    """Paired scene multinomial resampling; no artificial fifth model."""
    require(set(histograms) == set(MODELS), 'Exactly four registered models required')
    scenes = sorted(set(scene_tokens)); n = len(scenes)
    require(n >= 2, 'Scene intervals require at least two scenes')
    lookup = {scene: i for i, scene in enumerate(scenes)}
    grouped = np.zeros((4, n, 5, 2, 2), dtype=np.int64)
    for m, name in enumerate(MODELS):
        values = histograms[name]
        require(values.shape == (len(scene_tokens), 5, 2, 2) and
                np.issubdtype(values.dtype, np.integer) and np.all(values >= 0),
                'Invalid per-sample histogram array: '+name)
        for row, scene in zip(values, scene_tokens):
            grouped[m, lookup[scene]] += row
    require(int(grouped.sum())*n < 2**53, 'Bootstrap count precision bound exceeded')
    weights = np.random.RandomState(11).multinomial(n, np.full(n, 1./n), size=10000)
    flat = grouped.transpose(1, 0, 2, 3, 4).reshape(n, -1)
    counts = (weights.astype(np.float64) @ flat.astype(np.float64)).reshape(10000, 4, 5, 2, 2)
    return metric_arrays(counts), dict(replicates=10000, seed=11, unit='scene',
        scene_count=n, scene_order=scenes, paired=True,
        draws_sha256=hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest(),
        interval='two-sided percentile 95%',
        all_anchors_within_scene_kept_together=True,
        uncertainty='fixed-model scene sampling, not neural training-seed or model-selection uncertainty',
        multiple_comparison_adjustment=False)


def aggregate(histograms, scene_tokens, metric_arrays, minimum_effect_pp):
    boot, receipt = four_model_bootstrap(histograms, scene_tokens, metric_arrays)
    totals = np.stack([histograms[name].sum(0) for name in MODELS])
    point = metric_arrays(totals); models = {}; comparisons = []
    for m, name in enumerate(MODELS):
        horizons = []
        for h, seconds in enumerate(HORIZONS):
            cm = totals[m, h]
            horizons.append(dict(horizon_seconds=seconds,
                confusion_GT_rows_prediction_columns=cm.tolist(),
                TN=int(cm[0, 0]), FP=int(cm[0, 1]), FN=int(cm[1, 0]), TP=int(cm[1, 1]),
                GT_non_GMO=int(cm[0].sum()), GT_GMO=int(cm[1].sum()),
                gmo_iou=float(point['gmo_iou_by_horizon'][m, h]),
                non_gmo_iou=float(point['non_gmo_iou_by_horizon'][m, h]),
                binary_miou=float(point['binary_miou_by_horizon'][m, h]),
                gmo_iou_ci95=interval(boot['gmo_iou_by_horizon'][:, m, h]),
                binary_miou_ci95=interval(boot['binary_miou_by_horizon'][:, m, h])))
        models[name] = dict(horizons=horizons,
            metrics={k: dict(estimate=float(point[k][m]), ci95=interval(boot[k][:, m]), unit='ratio') for k in METRIC_KEYS},
            future_FP=int(totals[m, 1:, 0, 1].sum()), future_FN=int(totals[m, 1:, 1, 0].sum()))
    for left, right in CONTRASTS:
        i, j = MODELS.index(left), MODELS.index(right); metrics = {}; horizons = []
        for key in METRIC_KEYS:
            delta = float(point[key][i]-point[key][j]); ci = interval(boot[key][:, i]-boot[key][:, j])
            metrics[key] = dict(delta_ratio=delta, delta_pp=100*delta, ci95_ratio=ci, ci95_pp=[100*x for x in ci])
        for h, seconds in enumerate(HORIZONS):
            ci = interval(boot['gmo_iou_by_horizon'][:, i, h]-boot['gmo_iou_by_horizon'][:, j, h])
            delta = float(point['gmo_iou_by_horizon'][i, h]-point['gmo_iou_by_horizon'][j, h])
            horizons.append(dict(horizon_seconds=seconds, gmo_delta_ratio=delta, gmo_delta_pp=100*delta,
                gmo_ci95_pp=[100*x for x in ci], delta_FP=int(totals[i, h, 0, 1]-totals[j, h, 0, 1]),
                delta_FN=int(totals[i, h, 1, 0]-totals[j, h, 1, 0])))
        main = metrics['future_macro_gmo']
        comparisons.append(dict(contrast=left+' - '+right, left=left, right=right,
            primary_contrast=(left, right) in PRIMARY, precision_only_contrast=(left, right)==CONTRASTS[-1],
            metrics=metrics, horizons=horizons, registered_minimum_meaningful_effect_pp=minimum_effect_pp,
            point_meets_registered_minimum=None if minimum_effect_pp is None else main['delta_pp'] >= minimum_effect_pp,
            ci_lower_meets_registered_minimum=None if minimum_effect_pp is None else main['ci95_pp'][0] >= minimum_effect_pp,
            ci_lower_positive=main['ci95_pp'][0] > 0))
    return models, comparisons, receipt


def render(out, result):
    with (out/'metrics.csv').open('x', newline='') as f:
        fields = ['model', 'horizon_seconds', 'GMO_ratio', 'GMO_percent', 'binary_mIoU_ratio',
                  'binary_mIoU_percent', 'TN', 'FP', 'FN', 'TP', 'GT_non_GMO', 'GT_GMO']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for name, model in result['models'].items():
            for h in model['horizons']:
                writer.writerow(dict(model=name, horizon_seconds=h['horizon_seconds'], GMO_ratio=h['gmo_iou'],
                    GMO_percent=100*h['gmo_iou'], binary_mIoU_ratio=h['binary_miou'], binary_mIoU_percent=100*h['binary_miou'],
                    **{k: h[k] for k in fields[6:]}))
    with (out/'contrasts.csv').open('x', newline='') as f:
        fields = ['contrast', 'primary_contrast', 'metric', 'delta_ratio', 'delta_pp', 'ci95_lower_pp', 'ci95_upper_pp']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in result['comparisons']:
            for metric, value in row['metrics'].items():
                writer.writerow(dict(contrast=row['contrast'], primary_contrast=row['primary_contrast'], metric=metric,
                    delta_ratio=value['delta_ratio'], delta_pp=value['delta_pp'],
                    ci95_lower_pp=value['ci95_pp'][0], ci95_upper_pp=value['ci95_pp'][1]))
    lines = ['# 固定参考系状态实验：开发集比较', '',
        f"本结果覆盖 {result['samples']} 个 anchors、{result['scenes']} 个 scenes。它不是完整 5119 样本验证集成绩，也不证明多训练种子稳健性。", '',
        '主指标先在每个未来时域汇总全部样本的混淆矩阵，计算 GMO IoU，再对 0.5/1/1.5/2 秒取算术平均。GMO 是可移动语义类别，并非真实运动标注；binary mIoU 是 GMO/non-GMO 两类均值。', '',
        'reference 与 native1 使用相同观测缓存、训练样本、seed11、参数结构及固定最后 512 次更新。reference 改变显式空间读写坐标约定，真实未来 ego/action 条件保留。训练初始权重相同不意味着非零 ego 时初始预测相同。', '',
        'M0 保留原生 TF32 缓存预测；M0_fp32 是零更新的同权重未来 head 精度参照。后者和训练臂仅关闭未来 head 的 matmul TF32，未关闭原生观测缓存或 cuDNN 的 TF32。', '',
        '| 模型 | Future macro GMO % [95% CI] | Future macro binary mIoU % | Future FP | Future FN |',
        '|---|---:|---:|---:|---:|']
    for name in MODELS:
        m = result['models'][name]; v = m['metrics']['future_macro_gmo']
        lines.append(f"| {name} | {100*v['estimate']:.4f} [{100*v['ci95'][0]:.4f}, {100*v['ci95'][1]:.4f}] | {100*m['metrics']['future_macro_binary_miou']['estimate']:.4f} | {m['future_FP']} | {m['future_FN']} |")
    lines += ['', '| 配对比较 | 主比较 | Future macro GMO Δ pp | 95% CI pp |', '|---|---|---:|---:|']
    for row in result['comparisons']:
        v = row['metrics']['future_macro_gmo']
        lines.append(f"| {row['contrast']} | {'是' if row['primary_contrast'] else '否'} | {v['delta_pp']:+.4f} | [{v['ci95_pp'][0]:+.4f}, {v['ci95_pp'][1]:+.4f}] |")
    lines += ['', '区间使用固定 seed11 的 10,000 次场景配对 bootstrap，四模型共享抽样，同场景全部 anchors 一起重采样。区间只反映固定模型的场景抽样不确定性，不含训练随机性、此前开发选择和多重比较调整。', '',
        '两项主比较是 reference−M0_fp32 和 reference−native1；reference−原生 M0 与 M0_fp32−M0 为保留对照。空间契约修正是否有用须由这些真实结果判断，不能从几何审计通过直接推出精度收益。本程序不选择 checkpoint、阈值或额外样本。', '',
        '全部来源核验、整数混淆矩阵、逐时域差异和预登记实际效应门槛见 summary.json。未自动宣告研究目标完成或投稿就绪。', '']
    (out/'report.md').write_text('\n'.join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'reference-run', 'native-runs-root', 'protocol', 'parent-protocol', 'out'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args(argv); started = time.monotonic()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        parent, frame, module, trainer, evidence = load_contracts(a)
        cache, index, records, m0_hist, selection = module.load_cache(a.cache, parent, a.parent_protocol)
        require(sha(cache/'index.json') == frame['cache_index_sha256']['development'],
                'Frame development index differs from frozen cache')
        require(index['future_occupancy_is_input'] is False and index['inputs_targets_separate'] is True,
                'Native input/GT separation changed')
        control_dir = Path(a.native_runs_root)/'native1'
        require(not (control_dir/'failed.json').exists(), 'Control has a failure receipt')
        require(set(frame['control_files_sha256']) == set(trainer.CONTROL_FILES), 'Incomplete fixed native1 artifact contract')
        for name, digest in frame['control_files_sha256'].items():
            require(sha(control_dir/name) == digest, 'Registered C artifact differs: '+name)
        native_hist, native_receipt = module.load_arm(a.native_runs_root, 'native1', records, m0_hist,
            parent, a.parent_protocol, sha(cache/'index.json'))
        fp32_hist, fp32_receipt = module.load_precision_reference(a.native_runs_root, records, m0_hist,
            parent, a.parent_protocol, sha(cache/'index.json'))
        control_manifest = read(control_dir/'manifest.json')
        require(control_manifest['train_index_sha256'] == frame['cache_index_sha256']['train'],
                'Control training cache differs')
        native_receipt['training'] = check_log(control_dir, trainer, parent['training'])
        native_receipt['checkpoint'] = check_payload(control_dir, control_manifest,
            frame['control_files_sha256']['latest.pth'], 'native1', trainer.planned_orders(parent['training']))
        ref_hist, ref_receipt, ref_manifest = load_reference(a, frame, parent, module, trainer,
            records, m0_hist, control_manifest)
        require(native_receipt['checkpoint']['tensor_shapes'] == ref_receipt['checkpoint']['tensor_shapes'],
                'Actual reference/C checkpoint key/shape/dtype structures differ')
        require(native_receipt['checkpoint']['optimizer_groups'] == ref_receipt['checkpoint']['optimizer_groups'],
                'Reference/C final AdamW groups/parameter order/hyperparameters differ')
        require(native_receipt['checkpoint']['sample_orders_sha256'] == ref_receipt['checkpoint']['sample_orders_sha256'],
                'Actual reference/C sample orders differ')
        histograms = dict(M0=m0_hist, M0_fp32=fp32_hist, native1=native_hist, reference=ref_hist)
        minimum = module._minimum_effect(frame)
        models, comparisons, bootstrap = aggregate(histograms, [r['scene_token'] for r in records], module.metric_arrays, minimum)
        targets = [{k: row[k] for k in ('sample_token', 'scene_token', 'official_index', 'split')} |
                   dict(targets_sha256=row['files']['targets']['sha256'],
                        target_counts_0_1_255_by_frame=row['target_counts_0_1_255_by_frame']) for row in records]
        result = dict(schema='m0-frame-aggregate-v1', status='COMPLETED_DEVELOPMENT_COMPARISON',
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), script_sha256=sha(__file__),
            frame_protocol_sha256=sha(a.protocol), parent_protocol_sha256=PARENT_SHA,
            selection_sha256=index['selection_sha256'], cache_index_sha256=sha(cache/'index.json'),
            cache_complete_sha256=sha(cache/'complete.json'), m0_checkpoint_sha256=parent['m0_sha256'],
            source_model_sha256=index['native_model']['source_model_sha256'],
            source_receipts=dict(native1=native_receipt, M0_fp32=fp32_receipt, reference=ref_receipt),
            engineering_evidence=evidence, ordered_identities_and_GT=targets,
            audit=dict(status='PASS', contract='source/config/selection/cache identities; actual final checkpoint payloads; exact training schedule/orders; producer hashes; same full-GT class counts',
                all_four_models_same_GT_row_counts=True, all_exactly_200_same_order=True,
                target_identity_basis='SHA-bound common native cache and original target file SHA in cache index, plus all five-horizon class row sums; old evaluation rows themselves do not contain target-file SHA',
                target_voxel_files_reread=False, source_runtime_files_rehashed=True,
                reference_C_same_weight_structure_and_initialization=True,
                final_checkpoint_selection='fixed final 512 updates; initial reference scores not used for selection'),
            samples=len(records), scenes=bootstrap['scene_count'], models=models, comparisons=comparisons, bootstrap=bootstrap,
            primary_metric='four-future-horizon arithmetic mean of separately pooled GMO IoU',
            secondary_metric='four-future pooled GMO IoU',
            registered_minimum_meaningful_effect_pp=minimum, goal_complete_decision='NOT_MADE_BY_AGGREGATOR',
            semantics=dict(matrix='GT rows, prediction columns', class0='non-GMO', class1='GMO: movable semantic classes, not measured true motion',
                binary_miou='mean of GMO and non-GMO IoU', score_units='ratio; pp=100*ratio',
                historical_val_exposure=selection.get('historical_val_exposure', True),
                full_validation_result=False, training_seeds=1, training_seed_robustness_claim=False,
                threshold_fitting=False, R4_calibration_reused=False, frozen_references=['M0', 'M0_fp32'],
                trained_arms=['native1', 'reference'], spatial_treatment='reference frame for latent spatial memory/query; physical ego/action conditioning unchanged',
                precision=frame['numerical_policy']), seconds=time.monotonic()-started)
        write(out/'summary.json', result); render(out, result)
        files = {name: sha(out/name) for name in ('summary.json', 'report.md', 'metrics.csv', 'contrasts.csv')}
        write(out/'complete.json', dict(status='COMPLETE', schema='m0-frame-aggregate-complete-v1',
            summary_sha256=files['summary.json'], files_sha256=files,
            frame_protocol_sha256=sha(a.protocol), samples=len(records), models=list(MODELS),
            seconds=time.monotonic()-started))
        print(json.dumps(dict(status='COMPLETE', out=str(out), samples=len(records), models=list(MODELS))), flush=True)
    except BaseException as exc:
        write(out/'failed.json', dict(status='FAILED', error=repr(exc), seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
