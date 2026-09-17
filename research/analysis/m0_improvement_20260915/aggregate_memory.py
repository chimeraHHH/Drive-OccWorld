"""Strict paired development comparison for two frozen M0 references and three arms.

The primary score is the arithmetic mean of four future GMO IoUs, after
pooling confusion matrices over all selected anchors separately per horizon.
Bootstrap units are scenes (all anchors retained together); neural models are
fixed and seed11 is a resampling seed, not extra neural training. No threshold
fitting, probability calibration, R4 results, or checkpoint selection occurs.
M0 uses its unchanged native TF32 cache predictions. M0_fp32 changes only the
frozen future head's matmul policy; its observed t0 cache and cuDNN policy stay
unchanged. The three trained arms use the same policy as M0_fp32.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

MODELS = ('M0', 'M0_fp32', 'native1', 'persistent2', 'rolling2')
ARMS = ('native1', 'persistent2', 'rolling2')
HORIZONS = (0., .5, 1., 1.5, 2.)
M0_SHA = '0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
SOURCE_SHA = '67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5'
NATIVE_INITIAL_HEAD_SHA = '6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 11
CONTRASTS = (('persistent2', 'M0'), ('persistent2', 'native1'),
             ('persistent2', 'rolling2'), ('native1', 'M0'), ('rolling2', 'M0'),
             ('persistent2', 'M0_fp32'), ('M0_fp32', 'M0'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def hist_array(value):
    a = np.asarray(value)
    require(a.shape == (5, 2, 2), 'Expected five horizons of binary confusion matrices')
    require(np.issubdtype(a.dtype, np.integer) and np.all(a >= 0),
            'Confusion counts must be nonnegative integers, not rounded floats')
    require(np.all(a <= np.iinfo(np.int64).max), 'Count exceeds int64')
    return a.astype(np.int64)


def metric_arrays(hist):
    """Last dimensions [...,5,GTclass,predclass], scores are unitless ratios."""
    h = np.asarray(hist)
    tn, fp, fn, tp = h[..., 0, 0], h[..., 0, 1], h[..., 1, 0], h[..., 1, 1]
    foreground_union, background_union = tp+fp+fn, tn+fp+fn
    require(np.all(foreground_union > 0) and np.all(background_union > 0),
            'Undefined IoU: zero class union; do not silently omit bootstrap samples')
    gmo = tp/foreground_union
    non_gmo = tn/background_union
    miou = (gmo+non_gmo)/2
    future = h[..., 1:, :, :].sum(axis=-3)
    ft, ff, fnn, fpred = future[...,0,0], future[...,0,1], future[...,1,0], future[...,1,1]
    pooled_gmo = fpred/(fpred+ff+fnn)
    pooled_miou = .5*(pooled_gmo+ft/(ft+ff+fnn))
    return dict(gmo_iou_by_horizon=gmo, non_gmo_iou_by_horizon=non_gmo,
                binary_miou_by_horizon=miou,
                future_macro_gmo=gmo[...,1:].mean(axis=-1),
                future_macro_binary_miou=miou[...,1:].mean(axis=-1),
                future_pooled_gmo=pooled_gmo, future_pooled_binary_miou=pooled_miou)


def _ci(values):
    require(np.isfinite(values).all(), 'Nonfinite bootstrap metric')
    return [float(x) for x in np.quantile(values, [.025,.975], axis=0)]


def paired_bootstrap(histograms, scene_tokens):
    """Reuse the same 10,000 scene draws across models and all comparisons."""
    scenes = sorted(set(scene_tokens))
    require(len(scenes) >= 2, 'At least two scenes required for scene uncertainty')
    scene_index = {s:i for i,s in enumerate(scenes)}
    grouped = np.zeros((len(MODELS),len(scenes),5,2,2), dtype=np.int64)
    for m, name in enumerate(MODELS):
        require(len(histograms[name]) == len(scene_tokens), 'Model row count mismatch')
        for row, scene in zip(histograms[name], scene_tokens):
            grouped[m,scene_index[scene]] += row
    # Integer totals are exactly representable in float64 for this experiment.
    require(int(grouped.sum()) * len(scenes) < 2**53,
            'Float64 bootstrap matrix multiplication would risk count precision')
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(scenes), np.full(len(scenes),1./len(scenes)),
                              size=N_BOOTSTRAP)
    flat = grouped.transpose(1,0,2,3,4).reshape(len(scenes),-1)
    sampled = (weights.astype(np.float64) @ flat.astype(np.float64)).reshape(
        N_BOOTSTRAP,len(MODELS),5,2,2)
    scores = metric_arrays(sampled)
    return scores, scenes, hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()


def _identity(row):
    return [row[k] for k in ('sample_token','scene_token','official_index','split')]


def load_cache(cache, protocol, protocol_path):
    cache = Path(cache)
    if cache.is_file():
        cache = cache.parent
    done, index = read_json(cache/'complete.json'), read_json(cache/'index.json')
    require(done['status'] == 'COMPLETE_NATIVE_STATE_CACHE' and
            done['index_sha256'] == sha(cache/'index.json'), 'Incomplete/unbound native cache')
    require(index['schema'] == 'm0-native-state-cache-v1' and index['status'] == 'COMPLETE',
            'Wrong native cache schema/status')
    require(index['split'] == 'development', 'Aggregation requires development cache')
    require(index['native_model']['checkpoint_sha256'] == M0_SHA and
            index['native_model']['source_model_sha256'] == SOURCE_SHA,
            'Cache is not frozen native M0')
    require(index['native_model']['radar_contract'] == 'native_loader_B_without_common_source_override',
            'B_common cache is not a native M0 source')
    require(index['numerical_policy']['tf32_matmul'] is True and
            index['numerical_policy']['tf32_cudnn'] is True,
            'Original native M0 cache numerical policy changed')
    require(protocol['numerical_policy']['future_matmul_tf32'] is False and
            protocol['numerical_policy']['future_cudnn_tf32'] is True,
            'Training/reference future-head precision policy is not explicitly frozen')
    require(index['config_sha256'] == protocol['config_sha256'], 'Cache/config protocol mismatch')
    require(index['selection_sha256'] == protocol['selection_sha256'], 'Cache/selection protocol mismatch')
    expected_extractor = protocol['source_sha256']['native_state_cache.py']
    require(index['script_sha256'] == expected_extractor, 'Cache/extractor protocol mismatch')
    selection_path = Path(protocol_path).with_name('selection_v1.json')
    require(sha(selection_path) == protocol['selection_sha256'], 'Selection SHA mismatch')
    selection = read_json(selection_path)
    planned = [r for r in selection['records'] if r['split'] == 'development']
    records = index['records']
    require(len(records) == done['samples'] == protocol['training']['development_samples'],
            'Development sample count differs from completed protocol')
    require([_identity(r) for r in records] == [_identity(r) for r in planned],
            'Development identities/order differ from frozen selection')
    require(len({r['sample_token'] for r in records}) == len(records), 'Duplicate development token')
    require(not {r['scene_token'] for r in records} &
            {r['scene_token'] for r in selection['records'] if r['split'] == 'train'},
            'Training/development scene intersection')
    histograms = []
    for row in records:
        require(row['output_frame_seconds'] == list(HORIZONS), 'Cache horizon contract differs')
        h = hist_array(row['native_hist'])
        counts = np.asarray(row['target_counts_0_1_255_by_frame'],dtype=np.int64)
        require(counts.shape == (7,3) and np.array_equal(h.sum(2),counts[2:,:2]),
                'Cache native GT row sums differ from original segmentation class counts')
        require(row['parity']['status'] == 'PASS' and row['parity']['exact_native_confusion'] and
                row['parity']['bitwise_all_five_horizons_three_layers'], 'Missing native replay parity')
        histograms.append(h)
    return cache, index, records, np.stack(histograms), selection


def load_arm(runs_root, arm, records, m0_hist, protocol, protocol_path, cache_index_sha):
    directory = Path(runs_root)/arm
    done, manifest = read_json(directory/'complete.json'), read_json(directory/'manifest.json')
    training = read_json(directory/'training_complete.json')
    path = directory/'development_records.jsonl'
    require(done['status'] == 'TRAINED_AND_DEVELOPMENT_EVALUATED', 'Incomplete arm: '+arm)
    require(training['status'] == 'TRAINED_FIXED_FINAL', 'Missing fixed training endpoint: '+arm)
    require(done['evaluation']['sha256'] == sha(path), 'Evaluation JSONL SHA mismatch: '+arm)
    hp = protocol['training']
    expected_updates = hp['passes'] * hp['train_samples']//hp['accumulate']
    expected_examples = hp['passes'] * hp['train_samples']
    require(hp['train_samples'] % hp['accumulate'] == 0, 'Nonintegral protocol update count')
    for receipt in (done,training):
        require(receipt['updates'] == expected_updates and receipt['examples'] == expected_examples,
                'Wrong fixed final training budget: '+arm)
    require(done['checkpoint_sha256'] == training['checkpoint_sha256'],
            'Training/evaluation checkpoint receipts differ: '+arm)
    require(manifest['arm'] == arm and manifest['seed'] == hp['seed'] == 11,
            'Arm or training seed identity mismatch')
    require(manifest['protocol_sha256'] == sha(protocol_path) and manifest['m0_sha256'] == M0_SHA,
            'Arm protocol/M0 identity mismatch')
    require(manifest['dev_index_sha256'] == cache_index_sha, 'Arm used a different development cache')
    for name in ('memory_experiment.py','native_state_cache.py','observation_memory.py'):
        require(manifest['source_sha256'][name] == protocol['source_sha256'][name],
                'Arm source SHA differs from frozen protocol: '+name)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows) == len(records) == done['evaluation']['samples'], 'Evaluation coverage mismatch')
    matrices = []
    for i,(row,reference) in enumerate(zip(rows,records)):
        require(row['sample_token'] == reference['sample_token'] and
                row['scene_token'] == reference['scene_token'], 'Exact evaluation order/identity mismatch')
        require(row['horizon_seconds'] == list(HORIZONS), 'Prediction horizon mismatch')
        h = hist_array(row['hist_by_horizon'])
        require(np.array_equal(h.sum(2),m0_hist[i].sum(2)), 'GT row counts differ: '+arm)
        matrices.append(h)
    local_checkpoint = directory/'latest.pth'
    checkpoint_locally_rehashed = local_checkpoint.is_file()
    if checkpoint_locally_rehashed:
        require(sha(local_checkpoint) == done['checkpoint_sha256'], 'Local checkpoint SHA differs from receipt')
    receipt = dict(manifest_sha256=sha(directory/'manifest.json'), complete_sha256=sha(directory/'complete.json'),
                   training_complete_sha256=sha(directory/'training_complete.json'),
                   evaluation_jsonl_sha256=sha(path), checkpoint_sha256=done['checkpoint_sha256'],
                   checkpoint_locally_rehashed=checkpoint_locally_rehashed,
                   train_index_sha256=manifest['train_index_sha256'],
                   trainable_parameters=manifest['trainable_parameters'])
    return np.stack(matrices), receipt


def load_precision_reference(runs_root, records, m0_hist, protocol, protocol_path,
                             cache_index_sha):
    """Read native1's frozen pre-optimizer reference, never a trained checkpoint.

    Native1 completion/manifest/source bindings have already been checked by
    load_arm; repeat the identities material to this separately scored reference.
    """
    directory = Path(runs_root)/'native1'
    manifest = read_json(directory/'manifest.json')
    reference = manifest['frozen_precision_reference']
    path = directory/'frozen_native_fp32_records.jsonl'
    require(reference['sha256'] == sha(path), 'Frozen precision reference JSONL SHA mismatch')
    require(manifest['arm'] == 'native1' and manifest['m0_sha256'] == M0_SHA,
            'Precision reference is not bound to native M0 initialization')
    require(manifest['protocol_sha256'] == sha(protocol_path) and
            manifest['dev_index_sha256'] == cache_index_sha,
            'Frozen precision reference protocol/development cache differs')
    require(isinstance(manifest.get('initial_head_sha256'),str) and
            len(manifest['initial_head_sha256']) == 64,
            'Missing native1 initial head tensor identity')
    require(reference['initial_head_sha256'] == manifest['initial_head_sha256'] == NATIVE_INITIAL_HEAD_SHA,
            'Frozen reference did not retain the audited initial M0 head tensors')
    require(reference['source_m0_sha256'] == M0_SHA and reference['optimizer_updates'] == 0,
            'Precision reference must use original M0 before any optimizer update')
    require(reference['matmul_tf32'] is False and reference['cudnn_tf32'] is True,
            'Frozen reference numerical policy differs from registered FP32-matmul policy')
    for name in ('memory_experiment.py','native_state_cache.py','observation_memory.py'):
        require(manifest['source_sha256'][name] == protocol['source_sha256'][name],
                'Frozen reference source differs from protocol: '+name)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows) == len(records) == reference['samples'], 'Precision reference coverage mismatch')
    matrices = []
    for i,(row,expected) in enumerate(zip(rows,records)):
        require(row['sample_token'] == expected['sample_token'] and
                row['scene_token'] == expected['scene_token'],
                'Frozen precision reference exact order/identity mismatch')
        require(row['horizon_seconds'] == list(HORIZONS), 'Frozen reference horizon mismatch')
        h = hist_array(row['hist_by_horizon'])
        require(np.array_equal(h.sum(2),m0_hist[i].sum(2)), 'Frozen precision reference GT row counts differ')
        matrices.append(h)
    receipt = dict(manifest_sha256=sha(directory/'manifest.json'),
                   evaluation_jsonl_sha256=sha(path), samples=len(rows),
                   m0_checkpoint_sha256=M0_SHA, initial_head_sha256=manifest['initial_head_sha256'],
                   optimizer_steps=0, stored_under='native1', matmul_tf32=False, cudnn_tf32=True,
                   meaning='frozen initial future head; matmul TF32 off; native t0 cache unchanged')
    return np.stack(matrices),receipt


def aggregate(histograms, scene_tokens, minimum_effect_pp=None):
    boot, scenes, draws_sha = paired_bootstrap(histograms,scene_tokens)
    sums = np.stack([histograms[name].sum(0) for name in MODELS])
    point = metric_arrays(sums)
    models = {}
    for i,name in enumerate(MODELS):
        horizons = []
        for h,seconds in enumerate(HORIZONS):
            cm = sums[i,h]
            horizons.append(dict(horizon_seconds=seconds, confusion_GT_rows_prediction_columns=cm.tolist(),
                TN=int(cm[0,0]), FP=int(cm[0,1]), FN=int(cm[1,0]), TP=int(cm[1,1]),
                GT_non_GMO=int(cm[0].sum()), GT_GMO=int(cm[1].sum()),
                gmo_iou=float(point['gmo_iou_by_horizon'][i,h]),
                non_gmo_iou=float(point['non_gmo_iou_by_horizon'][i,h]),
                binary_miou=float(point['binary_miou_by_horizon'][i,h]),
                gmo_iou_ci95=_ci(boot['gmo_iou_by_horizon'][:,i,h]),
                binary_miou_ci95=_ci(boot['binary_miou_by_horizon'][:,i,h])))
        metrics = {}
        for key in ('future_macro_gmo','future_macro_binary_miou','future_pooled_gmo','future_pooled_binary_miou'):
            metrics[key] = dict(estimate=float(point[key][i]),ci95=_ci(boot[key][:,i]),unit='ratio')
        models[name] = dict(horizons=horizons,metrics=metrics,
                            future_FP=int(sums[i,1:,0,1].sum()),future_FN=int(sums[i,1:,1,0].sum()))
    comparisons = []
    for left,right in CONTRASTS:
        i,j = MODELS.index(left),MODELS.index(right)
        differences = {}
        for key in ('future_macro_gmo','future_macro_binary_miou','future_pooled_gmo','future_pooled_binary_miou'):
            delta = float(point[key][i]-point[key][j])
            ci = _ci(boot[key][:,i]-boot[key][:,j])
            differences[key] = dict(delta_ratio=delta,delta_pp=100*delta,ci95_ratio=ci,
                                    ci95_pp=[100*x for x in ci])
        by_horizon = []
        for h,seconds in enumerate(HORIZONS):
            delta = float(point['gmo_iou_by_horizon'][i,h]-point['gmo_iou_by_horizon'][j,h])
            ci = _ci(boot['gmo_iou_by_horizon'][:,i,h]-boot['gmo_iou_by_horizon'][:,j,h])
            dm = float(point['binary_miou_by_horizon'][i,h]-point['binary_miou_by_horizon'][j,h])
            cm = _ci(boot['binary_miou_by_horizon'][:,i,h]-boot['binary_miou_by_horizon'][:,j,h])
            by_horizon.append(dict(horizon_seconds=seconds,gmo_delta_ratio=delta,gmo_delta_pp=100*delta,
                 gmo_ci95_pp=[100*x for x in ci],binary_miou_delta_ratio=dm,
                 binary_miou_delta_pp=100*dm,binary_miou_ci95_pp=[100*x for x in cm],
                 delta_FP=int(sums[i,h,0,1]-sums[j,h,0,1]),delta_FN=int(sums[i,h,1,0]-sums[j,h,1,0])))
        primary = differences['future_macro_gmo']
        comparisons.append(dict(left=left,right=right,contrast=left+' - '+right,
            primary_contrast=left=='persistent2',metrics=differences,horizons=by_horizon,
            precision_only_contrast=(left,right)==('M0_fp32','M0'),
            registered_minimum_meaningful_effect_pp=minimum_effect_pp,
            point_meets_registered_minimum=None if minimum_effect_pp is None else primary['delta_pp']>=minimum_effect_pp,
            ci_lower_meets_registered_minimum=None if minimum_effect_pp is None else primary['ci95_pp'][0]>=minimum_effect_pp,
            ci_lower_positive=primary['ci95_pp'][0]>0))
    return models,comparisons,dict(replicates=N_BOOTSTRAP,seed=BOOTSTRAP_SEED,unit='scene',
        scene_order=scenes,scene_count=len(scenes),draws_sha256=draws_sha,paired=True,
        interval='two-sided percentile 95%',all_anchor_rows_within_each_scene_kept_together=True,
        uncertainty='fixed-trained-model scene sampling only; not training seed or model selection uncertainty',
        multiple_comparison_adjustment=False)


def _minimum_effect(protocol):
    evaluation = protocol.get('evaluation',{})
    value = evaluation.get('minimum_meaningful_effect_pp')
    if value is not None:
        require(isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) and value>=0,
                'Registered minimum meaningful effect must be nonnegative finite pp')
        return float(value)
    return None


def render_outputs(out, result):
    with (out/'metrics.csv').open('x',newline='') as f:
        fields=['model','horizon_seconds','GMO_ratio','GMO_percent','binary_mIoU_ratio','binary_mIoU_percent',
                'TN','FP','FN','TP','GT_non_GMO','GT_GMO']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,model in result['models'].items():
            for h in model['horizons']:
                writer.writerow(dict(model=name,horizon_seconds=h['horizon_seconds'],
                    GMO_ratio=h['gmo_iou'],GMO_percent=100*h['gmo_iou'],
                    binary_mIoU_ratio=h['binary_miou'],binary_mIoU_percent=100*h['binary_miou'],
                    **{k:h[k] for k in ['TN','FP','FN','TP','GT_non_GMO','GT_GMO']}))
    with (out/'contrasts.csv').open('x',newline='') as f:
        fields=['contrast','metric','delta_ratio','delta_pp','ci95_lower_pp','ci95_upper_pp']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for comparison in result['comparisons']:
            for metric,row in comparison['metrics'].items():
                writer.writerow(dict(contrast=comparison['contrast'],metric=metric,delta_ratio=row['delta_ratio'],
                    delta_pp=row['delta_pp'],ci95_lower_pp=row['ci95_pp'][0],ci95_upper_pp=row['ci95_pp'][1]))
    lines=['# 原生 M0 观测记忆实验：开发集比较','',
        f"真实开发子集：{result['samples']} anchors，{result['scenes']} scenes；不是完整验证集成绩。所有模型使用同序样本、原生 M0 雷达输入与给定未来 ego/action 条件。",'',
        '保留原生 M0 缓存预测。新增 M0_fp32 是训练前冻结的同权重未来预测器参考：仅关闭未来 head 的 matmul TF32，cuDNN TF32 与原生 t0 观测缓存保持不变。三个训练臂采用同样的 head 精度；因此 M0_fp32−M0 单独刻画这项数值策略变化，M0_fp32 不属于新增训练臂，也不代表全网络禁用 TF32。','',
        '主指标为每个未来时域先跨样本汇总混淆矩阵，计算 GMO IoU，再对0.5/1/1.5/2秒算术平均。mIoU仅指GMO/non-GMO两类均值，不是17类语义mIoU。','',
        '| 模型 | Future macro GMO % [95% CI] | Future macro binary mIoU % | Future pooled GMO % | Future FP | Future FN |',
        '|---|---:|---:|---:|---:|---:|']
    for name,model in result['models'].items():
        main=model['metrics']['future_macro_gmo'];miou=model['metrics']['future_macro_binary_miou']['estimate'];pooled=model['metrics']['future_pooled_gmo']['estimate']
        lines.append(f"| {name} | {100*main['estimate']:.4f} [{100*main['ci95'][0]:.4f}, {100*main['ci95'][1]:.4f}] | {100*miou:.4f} | {100*pooled:.4f} | {model['future_FP']} | {model['future_FN']} |")
    lines+=['','| 配对比较 | Future macro GMO Δ pp | 95% CI pp |','|---|---:|---:|']
    for comparison in result['comparisons']:
        m=comparison['metrics']['future_macro_gmo']
        lines.append(f"| {comparison['contrast']} | {m['delta_pp']:+.4f} | [{m['ci95_pp'][0]:+.4f}, {m['ci95_pp'][1]:+.4f}] |")
    minimum=result['registered_minimum_meaningful_effect_pp']
    lines+=['', '预注册最小实际效应阈值：'+('未给出；不自动判定研究目标完成。' if minimum is None else f'{minimum:g} pp；JSON分别记录点估计及区间下界是否达到阈值，不自动作投稿或总体研究成功判断。'),'',
        '区间来自固定seed11的10,000次场景配对bootstrap；同一场景的全部anchors一起抽样，五模型共享抽样。它仅反映固定模型的场景抽样不确定性，不代表多种子训练稳定性，未校正多重比较。历史验证集暴露仍需披露。',
        '逐时域GMO、binary mIoU、FP/FN和完整整数混淆矩阵见summary.json与metrics.csv；主比较和全部对照均保留。没有阈值拟合或R4校准混入。','']
    (out/'report.md').write_text('\n'.join(lines))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--runs-root',required=True)
    p.add_argument('--protocol',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    started=time.monotonic();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:
        protocol=read_json(a.protocol)
        cache,index,records,m0_hist,selection=load_cache(a.cache,protocol,a.protocol)
        histograms={'M0':m0_hist};receipts={}
        for arm in ARMS:
            histograms[arm],receipts[arm]=load_arm(a.runs_root,arm,records,m0_hist,protocol,a.protocol,sha(cache/'index.json'))
        histograms['M0_fp32'],receipts['M0_fp32']=load_precision_reference(
            a.runs_root,records,m0_hist,protocol,a.protocol,sha(cache/'index.json'))
        require(len({receipts[arm]['train_index_sha256'] for arm in ARMS})==1,
                'Arms did not use the same training cache')
        require(receipts['persistent2']['trainable_parameters']==receipts['rolling2']['trainable_parameters'],
                'Persistent and rolling capacity controls differ in parameter count')
        minimum=_minimum_effect(protocol)
        models,comparisons,bootstrap=aggregate(histograms,[r['scene_token'] for r in records],minimum)
        result=dict(schema='m0-memory-aggregate-v2',status='COMPLETED_DEVELOPMENT_COMPARISON',
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__),
            protocol_sha256=sha(a.protocol),selection_sha256=index['selection_sha256'],
            cache_index_sha256=sha(cache/'index.json'),cache_complete_sha256=sha(cache/'complete.json'),
            m0_checkpoint_sha256=M0_SHA,source_model_sha256=SOURCE_SHA,source_receipts=receipts,
            samples=len(records),scenes=len(bootstrap['scene_order']),
            ordered_identities=[{k:r[k] for k in ('sample_token','scene_token','official_index','split')} for r in records],
            primary_metric='four-future-horizon arithmetic mean of separately pooled GMO IoU',
            secondary_metric='four-future pooled GMO IoU',
            models=models,comparisons=comparisons,bootstrap=bootstrap,
            registered_minimum_meaningful_effect_pp=minimum,
            goal_complete_decision='NOT_MADE_BY_AGGREGATOR',
            semantics=dict(matrix='GT rows, prediction columns',class0='non-GMO',class1='GMO',
                binary_miou='mean of GMO and non-GMO IoU',score_units='ratio; differences additionally pp=100*ratio',
                historical_val_exposure=selection.get('historical_val_exposure',True),
                full_validation_result=False,training_seeds=1,threshold_fitting=False,
                R4_calibration_reused=False,
                frozen_references=['M0','M0_fp32'],trained_arms=list(ARMS),
                precision=dict(native_t0_cache=dict(matmul_tf32=True,cudnn_tf32=True),
                    M0_future_head=dict(matmul_tf32=True,cudnn_tf32=True),
                    M0_fp32_and_trained_future_heads=dict(matmul_tf32=False,cudnn_tf32=True))),
            seconds=time.monotonic()-started)
        write_json(out/'summary.json',result);render_outputs(out,result)
        files={name:sha(out/name) for name in ['summary.json','report.md','metrics.csv','contrasts.csv']}
        write_json(out/'complete.json',dict(status='COMPLETE',files_sha256=files,
                   summary_sha256=files['summary.json'],seconds=time.monotonic()-started))
        print(json.dumps(dict(status='COMPLETE',out=str(out),samples=len(records),scenes=result['scenes'])),flush=True)
    except BaseException as exc:
        write_json(out/'failed.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    main()
