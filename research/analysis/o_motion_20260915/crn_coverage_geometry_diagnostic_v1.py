"""Read-only geometry decomposition of the ORIGINAL CRN CV support.

No shifted boxes, alternative predictions, EPE, confidence threshold or training.
XY means dropping only the local-z inequality of the existing full-quaternion
box test. It is an infinite local-z prism, not a new deployment field.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SCORER_SHA = 'ef748a3efabbcdbbcbf8e1678b64f5692e2e957aef0bd3d3ddb9d24c77fe1b85'
RUNTIME_SHA = '2d9b711857e82ff1c3e88be16064a05d12b03d63f23f1d2545941c5b6fca0149'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_scorer():
    path = HERE/'predicted_object_state_cv_diagnostic_v1.py'
    require(sha(path) == SCORER_SHA, 'Original scoring source changed')
    spec = importlib.util.spec_from_file_location('_crn_original_cv_authentication', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def containment(boxes, global_points, geo, classes):
    """Unions over unmodified GMO boxes; no GT or instance identity accepted."""
    n = len(global_points)
    xy = np.zeros(n, dtype=bool)
    xyz = np.zeros(n, dtype=bool)
    above = np.zeros(n, dtype=bool)
    below = np.zeros(n, dtype=bool)
    owner = np.full(n, -1, dtype=np.int64)
    best = np.full(n, -np.inf)
    for index, box in enumerate(boxes):
        if box['detection_name'] not in classes:
            continue
        center = np.asarray(box['translation'], dtype=np.float64)
        size = np.asarray(box['size'], dtype=np.float64)
        rotation = geo.quaternion_wxyz_to_matrix(box['rotation'])
        local = (global_points-center) @ rotation
        half = size[[1, 0, 2]]/2. + 1e-9
        inside_xy = np.all(np.abs(local[:, :2]) <= half[:2], axis=1)
        inside_xyz = inside_xy & (np.abs(local[:, 2]) <= half[2])
        xy |= inside_xy
        xyz |= inside_xyz
        above |= inside_xy & (local[:, 2] > half[2])
        below |= inside_xy & (local[:, 2] < -half[2])
        choose = inside_xyz & (float(box['detection_score']) > best)
        owner[choose] = index
        best[choose] = float(box['detection_score'])
    excluded = xy & ~xyz
    require(np.array_equal(xyz, owner >= 0), 'Owner/union disagreement')
    require(np.array_equal(excluded, excluded & (above | below)), 'Z exclusion unaccounted')
    return dict(xy=xy, xyz=xyz, xy_z_excluded=excluded,
                z_above_only=excluded & above & ~below,
                z_below_only=excluded & below & ~above,
                z_mixed=excluded & above & below), owner


def count(masks, subset):
    n = int(np.count_nonzero(subset))
    row = dict(source_points=n)
    for key, values in masks.items():
        row[key+'_points'] = int(np.count_nonzero(values & subset))
    require(row['xy_points'] == row['xyz_points'] + row['xy_z_excluded_points'], 'XY partition')
    require(row['xy_z_excluded_points'] == sum(row[k+'_points'] for k in
        ('z_above_only', 'z_below_only', 'z_mixed')), 'Direction partition')
    return row


def aggregate(rows):
    keys = ('source_points', 'xy_points', 'xyz_points', 'xy_z_excluded_points',
            'z_above_only_points', 'z_below_only_points', 'z_mixed_points')
    out = {key: sum(row[key] for row in rows) for key in keys}
    out['objects'] = len(rows)
    for axis in ('xy', 'xyz'):
        key = axis+'_points'
        out[axis+'_objects_any'] = sum(row[key] > 0 for row in rows)
        out[axis+'_objects_full'] = sum(row[key] == row['source_points'] for row in rows)
        out[axis+'_objects_none'] = sum(row[key] == 0 for row in rows)
        out[axis+'_objects_partial'] = sum(0 < row[key] < row['source_points'] for row in rows)
        out[axis+'_point_coverage'] = out[key]/out['source_points'] if out['source_points'] else None
    out['z_rejected_among_xy_fraction'] = (out['xy_z_excluded_points']/out['xy_points']
        if out['xy_points'] else None)
    missing = out['source_points']-out['xyz_points']
    out['z_rejected_among_original_uncovered_fraction'] = out['xy_z_excluded_points']/missing if missing else None
    return out


def run(args):
    start = time.monotonic()
    scorer = load_scorer()
    base = Path(args.root)
    oldroot = base/'crn_object_state_cv_dev200_v1'
    oldcomplete = read(oldroot/'complete.json')
    require(oldcomplete['status'] == 'COMPLETE_READ_ONLY_DIAGNOSTIC', 'Original diagnostic incomplete')
    require(oldcomplete['script_sha256'] == SCORER_SHA, 'Original diagnostic source differs')
    for name, digest in oldcomplete['files_sha256'].items():
        require(sha(oldroot/name) == digest, 'Original completed output differs: '+name)
    old = read(oldroot/'summary.json')
    helper, _, selection, records, descriptors, rawdesc, _, _, sources = scorer.authenticate(
        base, Path(args.selection), base/'crn_state_dev200_v1')
    geo = scorer.bound('motion_geometry.py')
    oldanchors = {r['identity']['sample_token']: r for r in old['per_anchor_coverage']}
    oldobjects = {(r['sample_token'], r['instance_token'], r['horizon_seconds']): r
                  for r in old['coverage_object_records']}
    points = geo.voxel_centers_xyz(helper.SHAPE, helper.EXTENT).reshape(-1, 3)
    rows, unique_objects, anchors = [], [], []
    gt_total = 0
    for ordinal, (identity, record, desc) in enumerate(zip(selection, records, descriptors)):
        require(time.monotonic()-start < args.max_seconds, 'CPU deadline')
        token = identity['sample_token']
        rawd = rawdesc[token]
        rawpath = base/'motion_targets_v1'/rawd['file']
        require(sha(rawpath) == rawd['sha256'], 'Raw bytes changed')
        rawbytes = gzip.decompress(rawpath.read_bytes())
        require(hashlib.sha256(rawbytes).hexdigest() == rawd['uncompressed_json_sha256'], 'Raw JSON changed')
        raw = json.loads(rawbytes)
        require(raw['identity'] == identity and desc['label_source_sha256'] == rawd['sha256'], 'Raw identity')
        label = helper.load_sparse(base/'sparse_motion_v1', desc, identity)
        query = points[label['source_flat_indices']]
        G0 = geo._rigid_pose(raw['frames'][2]['lidar_to_global_column_matrix'], 'G0')
        global_points = query @ G0[:3, :3].T + G0[:3, 3]
        # Prediction geometry is completed before consulting instance GT boxes.
        masks, owner = containment(record['boxes'], global_points, geo, scorer.GMO)
        previous = oldanchors[token]
        require(len(owner) == previous['source_points'] and
            int(masks['xyz'].sum()) == previous['covered_points'] and
            hashlib.sha256(owner.tobytes()).hexdigest() == previous['owners_original_export_index_sha256'],
            'Independent containment does not reproduce original prediction ownership')
        anchors.append(dict(ordinal=ordinal, **identity, **count(masks, np.ones(len(query), dtype=bool))))
        tracks = {t['instance_token']: t for t in raw['tracks']}
        invG0 = np.linalg.inv(G0)
        for k, instance in enumerate(label['instance_tokens'].tolist()):
            own = label['object_index'] == k
            track = tracks[instance]
            require(track['valid_mask'][2], 'Sparse object has no current annotation')
            box_R = invG0 @ geo.box_to_global(track['global_centers_m'][2], track['global_rotations_wxyz'][2])
            inverse = np.linalg.inv(box_R)
            local = query[own] @ inverse[:3, :3].T + inverse[:3, 3]
            half = np.asarray(track['sizes_wlh_m'][2], dtype=np.float64)[[1, 0, 2]]/2.
            require(np.all(np.abs(local) <= half+1e-9), 'Source point outside original centered GT box')
            gt_total += int(own.sum())
            unique_objects.append(dict(sample_token=token, scene_token=identity['scene_token'],
                instance_token=instance, **count(masks, own)))
            for h, horizon in enumerate(scorer.HORIZONS):
                if not label['object_future_valid'][h, k]:
                    continue
                valid = own & label['valid'][h]
                row = dict(sample_token=token, scene_token=identity['scene_token'], instance_token=instance,
                    horizon_seconds=horizon, group=scorer.GROUPS[int(label['object_speed_group'][h, k])],
                    **count(masks, valid))
                oldrow = oldobjects[(token, instance, horizon)]
                require(row['source_points'] == oldrow['source_points'] and row['group'] == oldrow['group'] and
                    row['xyz_points'] == oldrow['covered_points'], 'Original object support/coverage differs')
                rows.append(row)
    require(len(rows) == len(oldobjects) == 16074, 'Original object horizon coverage lost')
    grouped = [dict(horizon_seconds=h, group=g, **aggregate([r for r in rows
        if r['horizon_seconds'] == h and (g == 'all' or r['group'] == g)]))
        for h in scorer.HORIZONS for g in ('all', *scorer.GROUPS)]
    runtime = base/'receipts/crn_z_origin_runtime_v1.json'
    require(sha(runtime) == RUNTIME_SHA, 'Runtime code evidence differs')
    require('torch' not in sys.modules, 'Local geometric diagnostic must remain NumPy CPU only')
    return dict(schema='crn-original-box-coverage-geometry-v1', status='COMPLETE_READ_ONLY_GEOMETRY_DIAGNOSTIC',
        elapsed_seconds=time.monotonic()-start, samples=200, scenes=100,
        original_owner_indices_exact=True, original_object_coverage_exact=True,
        original_source_points_inside_own_GT_box=gt_total,
        GT_box_origin='raw global annotation center, local half extents lwh/2',
        xy_definition='Drop only the local-z inequality of the unchanged full-quaternion box test; infinite local-z prism',
        xyz_definition='Original exported translation treated as center, unmodified sizes/quaternions/scores/classes',
        interpretation='Posthoc geometry only. No shifted-box containment, alternative field, EPE, threshold selection or new inference.',
        prediction_changed=False, GPU_used=False, model_forward_performed=False, optimizer_updates=0,
        sources=dict(authenticated_inputs=sources, original_complete_sha256=sha(oldroot/'complete.json'),
            original_summary_sha256=sha(oldroot/'summary.json'), runtime_code_receipt_sha256=RUNTIME_SHA,
            diagnostic_source_sha256=sha(__file__)),
        unique_anchor_object_coverage=aggregate(unique_objects), per_horizon_group=grouped,
        per_anchor=anchors, unique_object_records=unique_objects, object_horizon_records=rows)


def report(result):
    lines = ['# 原 CRN 框覆盖的几何分解', '',
        '仅对既有 dev200 作事后几何诊断，不改原 v1 评分。XY 是原全四元数框去掉局部 z 约束后的棱柱，不是另一个预测方案。', '',
        '| h | 组 | 对象 | 点数 | XY覆盖 | 原XYZ覆盖 | XY有但Z排除 | XY对象any/full/none | XYZ对象any/full/none |',
        '|---:|---|---:|---:|---:|---:|---:|---|---|']
    for r in result['per_horizon_group']:
        cell = lambda a: '/'.join(str(r[a+'_objects_'+k]) for k in ('any', 'full', 'none'))
        lines.append(f"| {r['horizon_seconds']} | {r['group']} | {r['objects']} | {r['source_points']} | "
            f"{r['xy_point_coverage']:.4%} | {r['xyz_point_coverage']:.4%} | {r['xy_z_excluded_points']} | {cell('xy')} | {cell('xyz')} |")
    lines += ['', '每个原始预测 owner 索引与原评分逐字节一致；全部 16,074 个对象时域的原覆盖计数与分母一致。',
        f"全部 {result['original_source_points_inside_own_GT_box']} 个去重材料点在其原 t0 中心定义 GT 框内；没有把 GT 匹配用于预测。",
        '独立 runtime 源码证据表明：CRN 中心 z 监督，经 CenterHead 解码减半高得到底面中心；eval_step 取 .tensor 后 exporter 直接当作 nuScenes Box(center)。这解释了一项坐标合同错误，但本诊断不估计修正后的覆盖或 EPE，也不把全部剩余误差归因于该错误。', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(HERE))
    p.add_argument('--selection', default=str(HERE.parent/'m0_improvement_20260915/selection_v1.json'))
    p.add_argument('--out', required=True)
    p.add_argument('--max-seconds', type=float, default=300)
    args = p.parse_args()
    require(0 < args.max_seconds <= 600, 'CPU budget must be positive and bounded')
    out = Path(args.out)
    require(not out.exists(), 'New diagnostic directory required')
    result = run(args)
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (out/'report.md').write_text(report(result))
    complete = dict(status=result['status'], schema='crn-coverage-geometry-complete-v1',
        files_sha256={f: sha(out/f) for f in ('summary.json', 'report.md')}, source_sha256=sha(__file__))
    (out/'complete.json').write_text(json.dumps(complete, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], seconds=result['elapsed_seconds'],
        moving_2s=next(r for r in result['per_horizon_group'] if r['horizon_seconds'] == 2 and r['group'] == 'moving'))))


if __name__ == '__main__':
    main()
