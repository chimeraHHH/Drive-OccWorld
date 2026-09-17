"""CPU-only descriptive audit of completed train512 CRN state inputs.

No GT, Torch, model calls, live V/G results, score tuning, or altered inputs.
Recompute with Python/NumPy; only a new output directory is written.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CRN = REPO / 'code/radar-fusion-baselines/CRN'
COMPLETE_SHA = '82433bc87db02dd68d8672e094a56a38dc988390631d48e4ceab655f9460ce98'
LOADER_SHA = '55e39715fdf19e5ae1f227a943a843b194eabed08e788ccb2694a040d6f18256'
CONDITIONER_SHA = 'dad88c05105d85d9b2a6094c687630f1d661cd7a4e35be42a15a5d01b94a8f36'
Q = (0., .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.)
OFFICIAL_FILES = (
    'exps/base_exp.py', 'exps/det/CRN_r50_256x704_128x128_4key.py',
    'evaluators/det_evaluators.py', 'layers/heads/bev_depth_head_det.py',
    'mmdetection3d/mmdet3d/models/dense_heads/centerpoint_head.py',
    'mmdetection3d/mmdet3d/core/bbox/coders/centerpoint_bbox_coders.py',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def quantiles(values, weights=None):
    """Inverse empirical CDF, no interpolation; equal values share CDF mass."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {str(q): None for q in Q}
    if weights is None:
        weights = np.ones(len(values), dtype=np.float64)
    order = np.argsort(values, kind='stable')
    x = values[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    unique, first = np.unique(x, return_index=True)
    mass = np.add.reduceat(w, first)
    cdf = np.cumsum(mass)
    cdf /= cdf[-1]
    indices = np.minimum(np.searchsorted(cdf, Q, side='left'), len(unique) - 1)
    return {str(q): float(unique[i]) for q, i in zip(Q, indices)}


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    assert bool(np.isfinite(values).all())
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
                quantiles=quantiles(values))


def metric_distribution(per_frame):
    """Retain all 512 frames; missing class values stay missing, never zero."""
    assert len(per_frame) == 512
    present = [np.asarray(v, dtype=np.float64) for v in per_frame if len(v)]
    values = np.concatenate(present) if present else np.empty(0)
    # Each observed frame has original mass 1/512. The missing atom is retained
    # explicitly; numeric quantiles below condition on the observed component.
    weights = np.concatenate([np.full(len(v), 1. / (512 * len(v))) for v in present]) if present else np.empty(0)
    means = np.asarray([v.mean() for v in present])
    all_present = len(present) == 512
    return dict(
        box_weighted=describe(values),
        anchor_equal=dict(anchor_denominator=512, observed_anchors=len(present),
            missing_anchors=512 - len(present), observed_probability_mass=len(present) / 512,
            missing_probability_mass=(512 - len(present)) / 512,
            unconditional_numeric_mean=float(means.mean()) if all_present else None,
            observed_component_mean=float(means.mean()) if len(means) else None,
            observed_component_quantiles=quantiles(values, weights),
            numeric_distribution_scope='all 512 anchors' if all_present else 'conditional on this class being present; missing atom separately retained',
            frame_mean_distribution_observed_component=describe(means)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prediction-root', default=str(HERE / 'crn_train512_full_official_env_v1'))
    ap.add_argument('--out', default=str(HERE / 'crn_train512_input_quality_audit_v1'))
    args = ap.parse_args()
    root, out = Path(args.prediction_root), Path(args.out)
    assert not out.exists(), 'Use a new audit output directory'
    assert sha(root / 'complete.json') == COMPLETE_SHA
    done = read(root / 'complete.json')
    assert done['status'] == 'COMPLETE_CRN_TRAIN512_INFERENCE' and done['mode'] == 'full' and done['samples'] == 512
    for name, digest in done['files_sha256'].items():
        assert sha(root / name) == digest, name
    manifest = read(root / 'manifest.json')
    contract_path = HERE / 'crn_train512_source_contract_v1.json'
    assert sha(contract_path) == done['source_contract_sha256']
    contract = read(contract_path)
    sources = {}
    for name in OFFICIAL_FILES:
        path = CRN / name
        assert sha(path) == contract['files_sha256'][name], name
        sources[str(path.relative_to(REPO))] = sha(path)
    for name, digest in [('object_state_prediction_inputs_v1.py', LOADER_SHA),
                         ('object_state_conditioner_v2.py', CONDITIONER_SHA),
                         ('crn_train512_inference_v2.py', done['source_sha256']),
                         ('crn_box_origin_adapter_v1.py', done['origin_adapter_sha256'])]:
        assert sha(HERE / name) == digest
        sources[str((HERE / name).relative_to(REPO))] = digest
    tree = ast.parse((HERE / 'object_state_prediction_inputs_v1.py').read_text())
    gmo = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'GMO' for t in n.targets))
    selection_path = HERE.parent / 'm0_improvement_20260915/selection_v1.json'
    assert sha(selection_path) == done['selection_sha256']
    selected = [r for r in read(selection_path)['records'] if r['split'] == 'train']
    data = read(root / 'predictions.json')
    assert data['box_origin'] == 'global_geometric_center'
    records = data['records']
    assert len(records) == len(selected) == 512 and len({r['scene_token'] for r in records}) == 256
    names = ('all_GMO',) + gmo
    counts = {name: [] for name in names}
    values = {name: {'score': [], 'speed_mps': []} for name in names}
    all_scores, all_speeds, export_counts, frames, class_counts = [], [], [], [], Counter()
    native_counts = [r['native_boxes'] for r in manifest['timing']]
    for i, (record, identity) in enumerate(zip(records, selected)):
        assert record['ordinal'] == i
        assert all(record[k] == identity[k] for k in ('sample_token', 'scene_token', 'official_index', 'split'))
        boxes = record['boxes']; export_counts.append(len(boxes))
        assert len(boxes) == min(native_counts[i], 500)
        scores = np.asarray([b['detection_score'] for b in boxes], dtype=np.float64)
        velocity = np.asarray([b['velocity'] for b in boxes], dtype=np.float64)
        assert velocity.shape == (len(boxes), 2) and np.isfinite(velocity).all() and np.isfinite(scores).all()
        assert bool((scores[:-1] >= scores[1:]).all())
        speed = np.hypot(velocity[:, 0], velocity[:, 1])
        all_scores.extend(scores.tolist()); all_speeds.extend(speed.tolist())
        detected_classes = [b['detection_name'] for b in boxes]
        class_counts.update(detected_classes)
        frame = dict(ordinal=i, sample_token=record['sample_token'], scene_token=record['scene_token'],
                     exported_boxes=len(boxes), native_boxes_before_export_cap=native_counts[i], groups={})
        for name in names:
            mask = np.asarray([c in gmo if name == 'all_GMO' else c == name for c in detected_classes])
            score_values, speed_values = scores[mask], speed[mask]
            counts[name].append(int(mask.sum()))
            values[name]['score'].append(score_values)
            values[name]['speed_mps'].append(speed_values)
            frame['groups'][name] = dict(boxes=int(mask.sum()), score=describe(score_values), speed_mps=describe(speed_values))
        frames.append(frame)
    assert sum(export_counts) == manifest['selected_boxes'] == 240815
    groups = {name: dict(boxes=sum(counts[name]), count_per_anchor_all512=describe(counts[name]),
                        anchors_with_zero_boxes=counts[name].count(0),
                        score=metric_distribution(values[name]['score']),
                        speed_mps=metric_distribution(values[name]['speed_mps'])) for name in names}
    result = dict(schema='crn-train512-input-quality-audit-v1', status='COMPLETE_DESCRIPTIVE_INPUT_AUDIT',
        created_utc=datetime.now(timezone.utc).isoformat(), audit_source_sha256=sha(__file__),
        sources=dict(prediction_complete_sha256=COMPLETE_SHA, prediction_files_sha256=done['files_sha256'],
            selection_sha256=done['selection_sha256'], source_contract_sha256=done['source_contract_sha256'],
            checkpoint_sha256=done['checkpoint_sha256'], official_commit=manifest['official_commit'], code_sha256=sources),
        anchors=512, scenes=256, retained_GMO_classes=list(gmo),
        weighting=dict(box='Each retained exported box has equal weight; these are detections, not unique GT objects.',
            anchor='Every original anchor has mass 1/512; within it retained boxes share that mass. Per-class missing anchors retain a missing atom; numeric summaries are explicitly conditional on class presence.',
            counts='Counts always include all 512 anchors, including class-absent zeros.',
            quantiles='Inverse empirical CDF after combining ties; no interpolation. Anchor mixture quantiles are not quantiles of per-frame means.',
            speed='hypot(exported global vx, global vy), metres/second; no GT. Equal to the full 3D norm after an ideal rigid R rotation of [vx,vy,0], not necessarily R-frame XY norm; no claim of bit-identical FP32 packed norm.'),
        export=dict(boxes=sum(export_counts), class_counts=dict(class_counts), count_per_anchor_all512=describe(export_counts),
            anchors_at_500=sum(v == 500 for v in export_counts), native_count_before_export=describe(native_counts),
            anchors_truncated_by_export_cap=sum(v > 500 for v in native_counts),
            boxes_removed_by_export_cap=sum(native_counts)-sum(export_counts),
            score_box_weighted=describe(all_scores), speed_mps_box_weighted=describe(all_speeds),
            rules=dict(task_count=6, decoder_topk_per_task=500, original_decoder_score_threshold_gt=.01,
                decoder_post_center_range=[-61.2,-61.2,-10.,61.2,61.2,10.], nms='circle per task',
                post_nms_max_per_task=200, export='descending score across all ten classes, at most 500; no padding to 500',
                consumer='retain all exported members of the fixed eight GMO classes in original order; no additional score, velocity, top-k or ROI selection')),
        groups=groups, frames=frames,
        conditioner=dict(score='one ordinary scalar feature at index 23 passed into state_mlp',
            dynamics='append predicted p0+h*v inside v2; no GT future inputs',
            attention='32 scene latents attend to all retained valid object tokens; padding mask only, no explicit score multiplier, threshold or score attention bias',
            interpretation='A learned model can use score through its MLP. This audit does not inspect learned attention weights or establish which detections dominate.'),
        boundaries=dict(low_score_is_not_verified_noise=True, low_score_is_not_verified_velocity_error=True,
            detector_score_calibration_not_assessed=True, GT_read=False, V_G_training_or_development_results_read=False,
            score_threshold_added=False, inputs_modified=False, GPU_used=False, torch_imported=False,
            model_forward_calls=0, optimizer_updates=0, causal_or_performance_claim=False),
        runtime=dict(python=sys.version, numpy=np.__version__))
    def qstat(group, metric, q):
        return groups[group][metric]['box_weighted']['quantiles'][str(q)]
    lines = ['# CRN train512 信息入口分布审计', '',
        f'读取已完成的 512 anchors／256 scenes，共 {sum(export_counts):,} 个官方导出框；固定八类筛选后保留 {groups["all_GMO"]["boxes"]:,} 个检测框。所有 512 个 anchor 均保留，不读取 V/G 训练或开发成绩。', '',
        f'官方具体配置是每 task 解码 top500、score > 0.01 和中心范围过滤，随后 circle NMS 每 task 最多 200；六个 task 合并后，exporter 按 score 排序最多取 500。实际每帧导出 {min(export_counts)}–{max(export_counts)} 个，{sum(v == 500 for v in export_counts)}／512 帧为 500；其中 {sum(v > 500 for v in native_counts)} 帧被最终上限截断。这是上限，不是补齐到固定数量。conditioner 只排除非八类框，不新增 score／速度／ROI／top-k 过滤。', '',
        '| 保留类别 | 框数 | 该类缺失 anchor /512 | score p50 / p90 | 速度 p50 / p90 (m/s) |',
        '|---|---:|---:|---:|---:|']
    for name in names:
        group = groups[name]
        lines.append(f'| {name} | {group["boxes"]:,} | {group["anchors_with_zero_boxes"]} | {qstat(name,"score",.5):.6f} / {qstat(name,"score",.9):.6f} | {qstat(name,"speed_mps",.5):.4f} / {qstat(name,"speed_mps",.9):.4f} |')
    overall = groups['all_GMO']; score = overall['score']; speed = overall['speed_mps']
    lines += ['', f'八类保留框每帧 p10／p50／p90 为 {overall["count_per_anchor_all512"]["quantiles"]["0.1"]:g}／{overall["count_per_anchor_all512"]["quantiles"]["0.5"]:g}／{overall["count_per_anchor_all512"]["quantiles"]["0.9"]:g}。框加权 score p25／p50／p75 为 {qstat("all_GMO","score",.25):.6f}／{qstat("all_GMO","score",.5):.6f}／{qstat("all_GMO","score",.75):.6f}；这里只用分位数刻画分数分布，不把任何分位数转成筛选阈值。', '',
        f'框等权 score 均值 {score["box_weighted"]["mean"]:.6f}，512-anchor 等权均值 {score["anchor_equal"]["unconditional_numeric_mean"]:.6f}；速度均值分别为 {speed["box_weighted"]["mean"]:.4f} 和 {speed["anchor_equal"]["unconditional_numeric_mean"]:.4f} m/s。前者使多框帧贡献更多，后者先使每帧总权重相同。每类计数仍保留所有512分母；缺类帧的 score／速度为缺失，不填零。JSON 同时给出缺失概率和明确标注的“类别存在时”anchor 等权数值分布，不冒称全512无条件分布。', '',
        'score 是第23槽普通输入，经 state_mlp 编码；32个场景 latent 对所有有效保留框做注意力聚合，没有显式 score gating。普通输入仍可被模型学习使用，不能因没有显式 gating 就声称 score 没有作用，也不能由框数认定注意力被某些框主导。', '',
        '**低检测 score 不等于已证实的噪声框，更不等于速度不准。** 本审计没有 GT 匹配、速度误差或分数校准检验；不能把入口分布归因为当前学习瓶颈，不建议按这些分位数调阈值。速度是原导出 global XY 的幅值，不是预测误差。', '',
        '来源定位：官方配置 `exps/det/CRN_r50_256x704_128x128_4key.py:224–265`；decoder `centerpoint_bbox_coders.py:194–216`；逐task NMS `centerpoint_head.py:685–732`；exporter `det_evaluators.py:243–251`；消费过滤 `object_state_prediction_inputs_v1.py:47–68`；score与attention `object_state_conditioner_v2.py:79–84,141–168`。对应实际源码与预测 SHA 全部写入 summary.json。', '',
        '重算：`python3 -B analysis/o_motion_20260915/audit_crn_train512_input_quality_v1.py --out <全新审计目录>`。JSON 包含512逐帧、逐类统计，全部分位点和精确定义；无GPU/Torch/新训练/模型前向。']
    assert sha(root / 'complete.json') == COMPLETE_SHA
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    complete = dict(schema=result['schema'], status=result['status'], audit_source_sha256=sha(__file__),
                    source_complete_sha256=COMPLETE_SHA, files_sha256={name: sha(out/name) for name in ('summary.json','report.md')})
    (out / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
    print(json.dumps(dict(out=str(out.resolve()), retained_GMO=groups['all_GMO']['boxes'],
        summary_sha256=complete['files_sha256']['summary.json'], audit_source_sha256=sha(__file__))))


if __name__ == '__main__':
    main()
