"""Strict CPU aggregation of M0, M0_fp32, native1, F and O on fixed dev200.

Only fixed final checkpoints are accepted. No model forward, optimizer update,
threshold fitting, label editing or additional data selection is performed.
Metric arithmetic and receipt helpers are loaded from hash-bound frozen sources;
the paired bootstrap below accepts explicit model names, never mutating another
module's model constants. F/O are separate interventions, not a factorial design.
"""
import argparse
from collections import Counter
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

MODELS = ('M0', 'M0_fp32', 'native1', 'F', 'O')
PRIMARY = (('F', 'M0_fp32'), ('F', 'native1'), ('O', 'M0_fp32'), ('O', 'native1'))
CONTRASTS = PRIMARY + (('native1', 'M0_fp32'), ('M0_fp32', 'M0'))
HORIZONS = (0., .5, 1., 1.5, 2.)
METRIC_KEYS = ('future_macro_gmo', 'future_macro_binary_miou',
               'future_pooled_gmo', 'future_pooled_binary_miou')
PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def frozen_module(filename, digest):
    path = Path(__file__).resolve().with_name(filename)
    require(Path(filename).name == filename and sha(path) == digest, 'Frozen source differs: '+filename)
    spec = importlib.util.spec_from_file_location('_objective_aggregate_'+path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_contracts(a):
    require(sha(a.protocol) == PARENT_SHA, 'Wrong frozen native training protocol')
    parent, protocol = read(a.protocol), read(a.objective_protocol)
    require(sha(__file__) == protocol['source_sha256']['objective_supervision_aggregate.py'],
            'Aggregator source is not the frozen objective source')
    trainer = frozen_module('objective_supervision_train.py', protocol['source_sha256']['objective_supervision_train.py'])
    checked_parent, checked_protocol, helper = trainer.load_contracts(a.objective_protocol, a.protocol)
    require(checked_parent == parent and checked_protocol == protocol, 'Protocol changed while loading')
    metric = frozen_module('aggregate_memory.py', parent['source_sha256']['aggregate_memory.py'])
    audit = frozen_module('frame_aggregate.py', protocol['source_sha256']['frame_aggregate.py'])
    evidence = trainer.validate_engineering(protocol, a.objective_protocol, parent)
    evaluation = protocol['evaluation']
    require(evaluation['primary_contrasts'] == [list(x) for x in PRIMARY], 'Registered primary contrasts differ')
    require(evaluation['historical_validation_exposure'] is True and
            evaluation['multiple_comparison_adjustment'] is False,
            'The historical-development exposure and unadjusted intervals must be explicit')
    return parent, protocol, trainer, helper, metric, audit, evidence


def check_selection(records, selection):
    train = [row for row in selection['records'] if row['split'] == 'train']
    require(len(records) == 200 and len(train) == 512, 'Only the registered dev200/train512 screen is accepted')
    for label, rows, scenes in [('development', records, 100), ('train', train, 256)]:
        counts = Counter(row['scene_token'] for row in rows)
        require(len(counts) == scenes and set(counts.values()) == {2}, 'Expected exactly two anchors per '+label+' scene')
        require(len({row['sample_token'] for row in rows}) == len(rows), 'Duplicate '+label+' sample token')
    require(not {r['scene_token'] for r in records} & {r['scene_token'] for r in train}, 'Train/development scene overlap')
    require(not {r['sample_token'] for r in records} & {r['sample_token'] for r in train}, 'Train/development token overlap')


def adapter_contract(receipt, arm, protocol):
    families = ('ce', 'sem_scal', 'geo_scal', 'lovasz') if arm == 'F' else ('ce', 'lovasz')
    keys = sorted('loss_voxel_'+family+'_inter_'+str(i) for family in families for i in range(3))
    require(receipt['schema'] == 'm0-objective-supervision-adapter-v1' and receipt['arm'] == arm and
            receipt['adapter_sha256'] == protocol['source_sha256']['objective_supervision_adapters.py'] and
            receipt['full_resolution_source_sha256'] == protocol['source_sha256']['full_resolution_supervision_preflight.py'],
            'Wrong objective adapter/source: '+arm)
    require(receipt['native_memory_mode'] == 'native1' and receipt['history_queue_length'] == 2 and
            receipt['GT_resolution'] == ([512,512,40] if arm == 'F' else [256,256,20]) and
            receipt['optimized_keys'] == keys and receipt['all_original_twelve_evaluated'] is True and
            receipt['omitted_family_forward_cost_retained'] is (arm == 'O') and
            receipt['forward_space_action_evaluator_unchanged'] is True and
            receipt['simultaneous_frame_or_memory_intervention'] is False,
            'Objective/GT/forward contract differs: '+arm)
    require((isinstance(receipt['full_resolution_adapter'], dict) and bool(receipt['full_resolution_adapter']))
            if arm == 'F' else receipt['full_resolution_adapter'] is None, 'Wrong resolution adapter: '+arm)
    runtime_shas = set(protocol['runtime_source_sha256'].values())
    require(receipt['detector_source_sha256'] in runtime_shas and receipt['head_source_sha256'] in runtime_shas,
            'Objective detector/head loss source is not bound')


def load_candidate(a, arm, parent, protocol, helper, metric, audit, records, m0_hist, control, expected_block):
    directory = Path(a.runs_root)/arm
    require(not (directory/'failed.json').exists() and not (directory/'interrupted.pth').exists(),
            'Candidate has a failure/interruption artifact: '+arm)
    done, trained, manifest = [read(directory/name) for name in ('complete.json', 'training_complete.json', 'manifest.json')]
    require(done['status'] == 'TRAINED_AND_DEVELOPMENT_EVALUATED' and trained['status'] == 'TRAINED_FIXED_FINAL',
            'Candidate fixed final training/evaluation is incomplete: '+arm)
    for receipt in (done, trained):
        require(receipt['updates'] == 512 and receipt['examples'] == 2048 and
                receipt['checkpoint_sha256'] == done['checkpoint_sha256'], 'Candidate endpoint differs: '+arm)
    require(done['objective_protocol_sha256'] == manifest['protocol_sha256'] == sha(a.objective_protocol) and
            done['parent_protocol_sha256'] == manifest['parent_protocol_sha256'] == PARENT_SHA,
            'Candidate protocol binding differs: '+arm)
    for field, filename in [('manifest_sha256', 'manifest.json'), ('control_reuse_sha256', 'control_reuse.json'),
                            ('training_log_sha256', 'training.jsonl')]:
        require(done[field] == sha(directory/filename), 'Candidate producer hash differs: '+arm+'/'+field)
    require(manifest['arm'] == arm and manifest['seed'] == 11 and
            manifest['m0_sha256'] == parent['m0_sha256'] and
            manifest['initial_head_sha256'] == control['initial_head_sha256'] == metric.NATIVE_INITIAL_HEAD_SHA,
            'Candidate initialization differs from original M0: '+arm)
    for key in ('numerical_policy', 'trainable_parameters', 'trainable_parameter_names', 'passes', 'examples_per_pass'):
        require(manifest[key] == control[key], 'Candidate/C training scope differs: '+arm+'/'+key)
    require(manifest['trainable_parameters'] == 13274016 and manifest['passes'] == 4 and manifest['examples_per_pass'] == 512,
            'Candidate scope/exposure differs: '+arm)
    require(manifest['migration'] == control['migration'], 'Candidate must preserve the original native1 geometry/memory: '+arm)
    require(manifest['numerical_policy'] == parent['numerical_policy'] and
            manifest['source_sha256'] == protocol['source_sha256'] and
            manifest['parent_source_sha256'] == parent['source_sha256'] and
            manifest['runtime_source_sha256'] == protocol['runtime_source_sha256'], 'Candidate source/precision differs: '+arm)
    for role, key in [('train', 'train_index_sha256'), ('development', 'dev_index_sha256')]:
        require(manifest[key] == control[key] == protocol['cache_index_sha256'][role], 'Candidate/C cache differs: '+arm)
    adapter_contract(manifest['objective_adapter'], arm, protocol)
    wanted = dict(expected_block, task_loss_intervention=arm)
    require(manifest['training_block'] == wanted, 'Candidate native optimizer/training block receipt differs: '+arm)
    reuse = read(directory/'control_reuse.json')
    require(manifest['control_reuse'] == reuse and reuse['files_sha256'] == protocol['control_files_sha256'] and
            reuse['reused'] is True and reuse['new_control_training'] is False and
            reuse['exact_updates'] == 512 and reuse['exact_examples'] == 2048 and
            reuse['training_log_all_LRs_verified'] is True, 'Candidate reused a different or incomplete C: '+arm)
    require(manifest['reused_frozen_precision_reference'] == control['frozen_precision_reference'] and
            manifest['initial_development_repeated'] is False and done['initial_development_repeated'] is False and
            manifest['loss_log_comparable_across_objectives'] is False and
            manifest['full_validation_automatically_requested'] is False and done['automatic_followup'] is False and
            done['new_training_seeds'] == 1 and done['training_seed'] == 11, 'Candidate scope/reference disclosure differs: '+arm)
    path = directory/'development_records.jsonl'
    require(done['evaluation']['samples'] == 200 and done['evaluation']['sha256'] == sha(path), 'Candidate evaluation hash differs: '+arm)
    rows = jsonl(path); require(len(rows) == len(records), 'Candidate development coverage differs: '+arm)
    matrices = []
    for i, (row, expected) in enumerate(zip(rows, records)):
        require(row['sample_token'] == expected['sample_token'] and row['scene_token'] == expected['scene_token'] and
                row['horizon_seconds'] == list(HORIZONS), 'Candidate identity/order/horizon differs: '+arm)
        matrix = metric.hist_array(row['hist_by_horizon'])
        require(np.array_equal(matrix.sum(2), m0_hist[i].sum(2)), 'Candidate original full-GT class counts differ: '+arm)
        matrices.append(matrix)
    log = audit.check_log(directory, helper, parent['training'])
    payload = audit.check_payload(directory, manifest, done['checkpoint_sha256'], arm,
                                  helper.planned_orders(parent['training']), done['final_head_state_sha256'])
    require(reuse['sample_orders_sha256'] == payload['sample_orders_sha256'], 'Candidate/C exact sample orders differ: '+arm)
    receipt = dict(files_sha256={name: sha(directory/name) for name in
        ('manifest.json', 'training_complete.json', 'complete.json', 'control_reuse.json',
         'training.jsonl', 'development_records.jsonl', 'latest.pth')}, training=log, checkpoint=payload,
        objective_adapter=manifest['objective_adapter'], no_initial_development_selection=True)
    return np.stack(matrices), receipt


def interval(values):
    require(np.isfinite(values).all(), 'Nonfinite bootstrap score')
    return np.quantile(values, [.025, .975], axis=0).tolist()


def paired_bootstrap(histograms, scene_tokens, names, metric_arrays):
    """Explicit names, common scene draws, all anchors in each scene together."""
    require(len(names) == len(set(names)) and set(histograms) == set(names), 'Model identity set differs')
    scenes = sorted(set(scene_tokens)); n = len(scenes)
    require(n >= 2, 'Scene intervals require at least two scenes')
    lookup = {scene: i for i, scene in enumerate(scenes)}
    grouped = np.zeros((len(names), n, 5, 2, 2), dtype=np.int64)
    for m, name in enumerate(names):
        values = histograms[name]
        require(values.shape == (len(scene_tokens), 5, 2, 2) and
                np.issubdtype(values.dtype, np.integer) and np.all(values >= 0), 'Invalid histogram array: '+name)
        require(int(values.max())*len(scene_tokens)*len(names)*20*n < 2**53, 'Integer bootstrap precision bound exceeded')
        for row, scene in zip(values, scene_tokens):
            grouped[m, lookup[scene]] += row
    weights = np.random.RandomState(11).multinomial(n, np.full(n, 1./n), size=10000)
    flat = grouped.transpose(1, 0, 2, 3, 4).reshape(n, -1)
    counts = (weights.astype(np.float64) @ flat.astype(np.float64)).reshape(10000, len(names), 5, 2, 2)
    return metric_arrays(counts), dict(replicates=10000, seed=11, unit='scene', scene_count=n,
        scene_order=scenes, model_order=list(names), paired=True,
        draws_sha256=hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest(),
        interval='two-sided percentile 95%', all_anchors_within_scene_kept_together=True,
        uncertainty='fixed-model scene sampling, not neural training-seed or model-selection uncertainty',
        primary_contrasts=len(PRIMARY), multiple_comparison_adjustment=False)


def aggregate(histograms, scene_tokens, metric_arrays, minimum_effect_pp,
              names=MODELS, contrasts=CONTRASTS, primary=PRIMARY):
    """Parameters permit checks against prior real results; CLI fixes F/O names."""
    require(all(left in names and right in names and left != right for left, right in contrasts),
            'Invalid contrast identities')
    require(set(primary) <= set(contrasts), 'Primary contrasts must be included')
    boot, receipt = paired_bootstrap(histograms, scene_tokens, names, metric_arrays)
    receipt['primary_contrasts'] = len(primary)
    totals = np.stack([histograms[name].sum(0) for name in names])
    point = metric_arrays(totals); models = {}; comparisons = []
    for m, name in enumerate(names):
        horizons = []
        for h, seconds in enumerate(HORIZONS):
            cm = totals[m, h]
            horizons.append(dict(horizon_seconds=seconds, confusion_GT_rows_prediction_columns=cm.tolist(),
                TN=int(cm[0,0]), FP=int(cm[0,1]), FN=int(cm[1,0]), TP=int(cm[1,1]),
                GT_non_GMO=int(cm[0].sum()), GT_GMO=int(cm[1].sum()),
                gmo_iou=float(point['gmo_iou_by_horizon'][m,h]), non_gmo_iou=float(point['non_gmo_iou_by_horizon'][m,h]),
                binary_miou=float(point['binary_miou_by_horizon'][m,h]),
                gmo_iou_ci95=interval(boot['gmo_iou_by_horizon'][:,m,h]),
                binary_miou_ci95=interval(boot['binary_miou_by_horizon'][:,m,h])))
        models[name] = dict(horizons=horizons,
            metrics={key: dict(estimate=float(point[key][m]), ci95=interval(boot[key][:,m]), unit='ratio') for key in METRIC_KEYS},
            future_FP=int(totals[m,1:,0,1].sum()), future_FN=int(totals[m,1:,1,0].sum()))
    for left, right in contrasts:
        i, j = names.index(left), names.index(right); metrics = {}; horizons = []
        for key in METRIC_KEYS:
            delta = float(point[key][i]-point[key][j]); ci = interval(boot[key][:,i]-boot[key][:,j])
            metrics[key] = dict(delta_ratio=delta, delta_pp=100*delta, ci95_ratio=ci, ci95_pp=[100*x for x in ci])
        for h, seconds in enumerate(HORIZONS):
            ci = interval(boot['gmo_iou_by_horizon'][:,i,h]-boot['gmo_iou_by_horizon'][:,j,h])
            delta = float(point['gmo_iou_by_horizon'][i,h]-point['gmo_iou_by_horizon'][j,h])
            horizons.append(dict(horizon_seconds=seconds, gmo_delta_ratio=delta, gmo_delta_pp=100*delta,
                gmo_ci95_pp=[100*x for x in ci], delta_FP=int(totals[i,h,0,1]-totals[j,h,0,1]),
                delta_FN=int(totals[i,h,1,0]-totals[j,h,1,0])))
        main = metrics['future_macro_gmo']
        comparisons.append(dict(contrast=left+' - '+right, left=left, right=right,
            primary_contrast=(left,right) in primary, precision_only_contrast=(left,right)==('M0_fp32','M0'),
            metrics=metrics, horizons=horizons, registered_minimum_meaningful_effect_pp=minimum_effect_pp,
            point_meets_registered_minimum=None if minimum_effect_pp is None else main['delta_pp'] >= minimum_effect_pp,
            ci_lower_meets_registered_minimum=None if minimum_effect_pp is None else main['ci95_pp'][0] >= minimum_effect_pp,
            ci_lower_positive=main['ci95_pp'][0] > 0))
    return models, comparisons, receipt


def render(out, result):
    with (out/'metrics.csv').open('x', newline='') as stream:
        fields = ['model', 'horizon_seconds', 'GMO_ratio', 'GMO_percent', 'binary_mIoU_ratio',
                  'binary_mIoU_percent', 'TN', 'FP', 'FN', 'TP', 'GT_non_GMO', 'GT_GMO']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name, model in result['models'].items():
            for h in model['horizons']:
                writer.writerow(dict(model=name, horizon_seconds=h['horizon_seconds'], GMO_ratio=h['gmo_iou'],
                    GMO_percent=100*h['gmo_iou'], binary_mIoU_ratio=h['binary_miou'], binary_mIoU_percent=100*h['binary_miou'],
                    **{key: h[key] for key in fields[6:]}))
    with (out/'contrasts.csv').open('x', newline='') as stream:
        fields = ['contrast', 'primary_contrast', 'metric', 'delta_ratio', 'delta_pp', 'ci95_lower_pp', 'ci95_upper_pp']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in result['comparisons']:
            for metric, value in row['metrics'].items():
                writer.writerow(dict(contrast=row['contrast'], primary_contrast=row['primary_contrast'], metric=metric,
                    delta_ratio=value['delta_ratio'], delta_pp=value['delta_pp'],
                    ci95_lower_pp=value['ci95_pp'][0], ci95_upper_pp=value['ci95_pp'][1]))
    lines = ['# 监督分辨率与损失组成：固定开发集比较', '',
        '本结果覆盖 200 anchors、100 scenes，每场景两条。它是经过历史开发选择的单训练 seed11 筛查，不是新盲测、完整 5119 样本验证或多训练种子结果。', '',
        '主指标在每个未来时域分别池化完整 GT 混淆矩阵、计算 GMO IoU，再平均 0.5/1/1.5/2 秒。GMO 指可移动语义类，不是真实运动标注；binary mIoU 是 GMO 与 non-GMO 的两类 IoU 平均，二者不得混称。', '',
        'C/native1 是已经完成的原目标延续训练；F 在完整 GT 上优化原 12 项损失；O 在原粗 GT 上只优化 CE[1,5] 与 Lovász 共 6 项，sem/geo 仍前向计算。F/O 都从原 M0 初始化，原生单槽状态、输入和 evaluator 不变，不叠加坐标或双槽干预。', '',
        '训练参数范围、样本顺序、seed11、AdamW、学习率及最后 512 次更新固定一致，共 2048 次样本使用。F/O loss 的数值定义不同，不能横向按 loss 高低选优。', '',
        'M0 保留原生 TF32 缓存预测；M0_fp32 是 native1 优化前的零更新精度参照，仅未来 head matmul TF32 关闭，与训练臂一致；观测缓存和 cuDNN TF32 保持原设置。', '',
        '| 模型 | Future macro GMO % [95% CI] | Future macro binary mIoU % | Future FP | Future FN |',
        '|---|---:|---:|---:|---:|']
    for name in MODELS:
        model = result['models'][name]; value = model['metrics']['future_macro_gmo']
        lines.append(f"| {name} | {100*value['estimate']:.4f} [{100*value['ci95'][0]:.4f}, {100*value['ci95'][1]:.4f}] | {100*model['metrics']['future_macro_binary_miou']['estimate']:.4f} | {model['future_FP']} | {model['future_FN']} |")
    lines += ['', '| 配对比较 | 主比较 | Future macro GMO Δ pp | 95% CI pp |', '|---|---|---:|---:|']
    for row in result['comparisons']:
        value = row['metrics']['future_macro_gmo']
        lines.append(f"| {row['contrast']} | {'是' if row['primary_contrast'] else '否'} | {value['delta_pp']:+.4f} | [{value['ci95_pp'][0]:+.4f}, {value['ci95_pp'][1]:+.4f}] |")
    lines += ['', '五模型共用 seed11 的 10,000 次场景配对 bootstrap，同场景两 anchors 一起重采样。区间未作多重比较校正，只量化固定模型的场景抽样不确定性，不含训练随机性或历史模型选择。四项主对比是 F−M0_fp32、F−C、O−M0_fp32、O−C；另列 C−M0_fp32 与 M0_fp32−M0 控制。', '',
        'F/O 是两个分开的训练干预，没有 F＋O，不能估计交互。O 同时改变整体梯度尺度、裁剪和相对权重衰减；其结果不是单一因果机制的证明。监督改动与删减现有损失本身也不构成方法新颖性。', '',
        '程序只报告固定最终结果及预登记效应门槛，不选择 checkpoint、阈值或单时域，不自动追加完整验证、不宣告研究目标完成。所有整数计数、逐时域差异、来源及 CPU checkpoint 审计见 summary.json。', '']
    (out/'report.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('objective-protocol', 'protocol', 'cache', 'control-run', 'runs-root', 'out'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args(argv); started = time.monotonic(); out = Path(a.out).resolve()
    require(not out.exists(), 'Require a new output; no overwrite or retry')
    require(Path(a.control_run).resolve().name == 'native1', 'control-run must identify the fixed native1 directory')
    require(len({Path(a.control_run).resolve(), (Path(a.runs_root)/'F').resolve(), (Path(a.runs_root)/'O').resolve()}) == 3,
            'Control and candidate output directories must be distinct')
    out.mkdir(parents=True, exist_ok=False)
    try:
        parent, protocol, trainer, helper, metric, audit, evidence = load_contracts(a)
        cache, index, records, m0_hist, selection = metric.load_cache(a.cache, parent, a.protocol)
        require(sha(cache/'index.json') == protocol['cache_index_sha256']['development'], 'Wrong frozen development cache')
        require(index['future_occupancy_is_input'] is False and index['inputs_targets_separate'] is True,
                'Input/target separation differs')
        check_selection(records, selection)
        # Actual CPU control payload, not a manifest-only claim about its budget.
        control, payload, control_reuse = helper.validate_control(a, parent, protocol, {'development': records})
        del payload
        control_dir = Path(a.control_run); native_root = control_dir.parent
        native_hist, native_receipt = metric.load_arm(native_root, 'native1', records, m0_hist,
            parent, a.protocol, sha(cache/'index.json'))
        fp32_hist, fp32_receipt = metric.load_precision_reference(native_root, records, m0_hist,
            parent, a.protocol, sha(cache/'index.json'))
        native_receipt['reuse_audit'] = control_reuse
        native_receipt['training'] = audit.check_log(control_dir, helper, parent['training'])
        native_receipt['checkpoint'] = audit.check_payload(control_dir, control,
            protocol['control_files_sha256']['latest.pth'], 'native1', helper.planned_orders(parent['training']))
        _, expected_block = helper.native_training_block(parent)
        expected_block.pop('full_task_losses_unchanged')
        expected_block.update(native_sum_accumulation_backward_unchanged=True,
                              frozen_control_helper_sha256=trainer.HELPER_SHA)
        histograms = dict(M0=m0_hist, M0_fp32=fp32_hist, native1=native_hist)
        receipts = dict(native1=native_receipt, M0_fp32=fp32_receipt)
        engineering_rows = read(Path(evidence['objective_trainmode']['directory'])/'summary.json')['reports']
        engineering_adapters = {row['arm']: row['adapter'] for row in engineering_rows}
        for arm in ('F', 'O'):
            histograms[arm], receipts[arm] = load_candidate(a, arm, parent, protocol, helper, metric, audit,
                records, m0_hist, control, expected_block)
            require(read(Path(a.runs_root)/arm/'manifest.json')['engineering_evidence'] == evidence,
                    'Candidate engineering evidence differs: '+arm)
            require(receipts[arm]['objective_adapter'] == engineering_adapters[arm],
                    'Candidate adapter must match the actual engineering proof exactly: '+arm)
            left, right = receipts[arm]['checkpoint'], native_receipt['checkpoint']
            for key in ('tensor_shapes', 'optimizer_groups', 'sample_orders_sha256'):
                require(left[key] == right[key], 'Actual candidate/C checkpoint structure, optimizer or order differs: '+arm+'/'+key)
        minimum = metric._minimum_effect(protocol)
        models, comparisons, bootstrap = aggregate(histograms, [row['scene_token'] for row in records], metric.metric_arrays, minimum)
        targets = [{key: row[key] for key in ('sample_token', 'scene_token', 'official_index', 'split')} |
            dict(targets_sha256=row['files']['targets']['sha256'],
                 target_counts_0_1_255_by_frame=row['target_counts_0_1_255_by_frame']) for row in records]
        result = dict(schema='m0-objective-supervision-aggregate-v1', status='COMPLETED_DEVELOPMENT_COMPARISON',
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), script_sha256=sha(__file__),
            objective_protocol_sha256=sha(a.objective_protocol), parent_protocol_sha256=PARENT_SHA,
            config_sha256=parent['config_sha256'], selection_sha256=index['selection_sha256'],
            cache_index_sha256=sha(cache/'index.json'), cache_complete_sha256=sha(cache/'complete.json'),
            m0_checkpoint_sha256=parent['m0_sha256'], source_model_sha256=index['native_model']['source_model_sha256'],
            source_receipts=receipts, engineering_evidence=evidence, ordered_identities_and_GT=targets,
            audit=dict(status='PASS', all_five_models_same_full_GT_row_counts=True, all_exactly_200_same_order=True,
                scenes=100, anchors_per_scene=2, source_runtime_files_rehashed=True,
                actual_three_final_checkpoints_rehashed_and_payload_checked_on_CPU=True,
                actual_512_updates_2048_examples_LR_and_orders_checked=True,
                same_weight_structure_and_M0_initialization=True,
                target_identity_basis='SHA-bound common native cache and original target SHA plus all five-horizon GT row counts; old evaluation rows do not contain target-file SHA',
                target_voxel_files_reread=False, checkpoint_selection='fixed final; no development/checkpoint/threshold selection in this run'),
            samples=len(records), scenes=bootstrap['scene_count'], models=models, comparisons=comparisons, bootstrap=bootstrap,
            primary_metric='four-future-horizon arithmetic mean of separately pooled GMO IoU',
            secondary_metric='four-future pooled GMO IoU', registered_primary_contrasts=[list(x) for x in PRIMARY],
            registered_minimum_meaningful_effect_pp=minimum, goal_complete_decision='NOT_MADE_BY_AGGREGATOR',
            semantics=dict(matrix='GT rows, prediction columns', class0='non-GMO',
                class1='GMO: movable semantic classes, not measured true motion', binary_miou='mean of GMO and non-GMO IoU',
                score_units='ratio; pp=100*ratio', historical_val_exposure=True, development_screen_not_new_blind_test=True,
                full_validation_result=False, training_seeds=1, training_seed_robustness_claim=False,
                threshold_fitting=False, R4_calibration_reused=False, frozen_references=['M0','M0_fp32'],
                trained_arms=['native1','F','O'], F='full GT with original 12 loss terms',
                O='original coarse GT with original CE[1,5] plus Lovasz; 6 terms',
                F_plus_O_not_run=True, factorial_interaction_not_identified=True,
                loss_magnitudes_comparable_across_objectives=False, multiple_comparison_adjustment=False,
                precision=parent['numerical_policy']), seconds=time.monotonic()-started)
        write(out/'summary.json', result); render(out, result)
        files = {name: sha(out/name) for name in ('summary.json', 'report.md', 'metrics.csv', 'contrasts.csv')}
        write(out/'complete.json', dict(status='COMPLETE', schema='m0-objective-supervision-aggregate-complete-v1',
            summary_sha256=files['summary.json'], files_sha256=files, objective_protocol_sha256=sha(a.objective_protocol),
            samples=len(records), scenes=100, models=list(MODELS), seconds=time.monotonic()-started))
        print(json.dumps(dict(status='COMPLETE', out=str(out), samples=len(records), models=list(MODELS))), flush=True)
    except BaseException as exc:
        write(out/'failed.json', dict(status='FAILED_NO_RETRY', error=repr(exc), seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
