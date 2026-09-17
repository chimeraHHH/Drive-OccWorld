"""Independent 2 s raw-NPZ audit; no experiment/analyzer/helper imports.

Only no-radar / CRN-uncovered all and future-moving populations are audited.
The original history metadata supplies LiDAR t0 for camera-time benefits.
This program neither reads a main analysis result nor decodes images/GT boxes.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
COMPLETE_SHA = {
    'motion': '3bd429effb1fbe50ac0aeb77330bc0100ee144bf8406fbd078097fb336301a73',
    'radar': '7dcc6d321a005b4b6e7e41eb4cdceb376534eb55493a9115882cdef6ea8f976c',
    'camera': '0c0d2e640a12e3299e8dacc03189776669cd2e50036a65de823026304119b648',
    'history': '040d62242a541b9685f3f39f1534b3663b0e1142b4cd7d9737cdbfd15d94314d',
}
POSE_SOURCES = {
    'source_sha256': ('extract_history_pose_address_control_train_v1.py', 'f721ae8b4ee12b7795a7b2e5d26d6c9aa6ef6c02103af21fbf85522244e9869f'),
    'core_sha256': ('history_pose_address_control_v1.py', '7de75c862634729a804b95ac6b6d9f03b2193a0622a279102083977fa1724072'),
    'protocol_sha256': ('history_pose_address_control_train_protocol_v1.json', '8183a66c5beaea546ab479b93e2bc98ba6902fb1f76a74fa2dd5dc3faf0a2d13'),
}
METRICS = ('delta_eD_minus_eGT', 'delta_eZero_minus_eGT',
           'broken_delta_eD_minus_eGT', 'broken_delta_eZero_minus_eGT',
           'actual_minus_broken_delta_eD_minus_eGT', 'actual_minus_broken_delta_eZero_minus_eGT')
REASONS = ('valid', 'original_photo_invalid', 'current_pose_unavailable',
           'past_pose_unavailable', 'current_GT_projection_invalid', 'past_GT_projection_invalid')
CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_file(root, name):
    relative = Path(name)
    need(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe ledger path')
    path = (root / relative).resolve()
    need(root.resolve() in path.parents, 'Ledger escaped its root')
    return path


def authenticate(root, expected_sha, kind):
    need(len(expected_sha) == 64 and sha(root / 'complete.json') == expected_sha, kind + ' complete SHA')
    need(not (root / 'failed.json').exists(), kind + ' failure receipt exists')
    done = read(root / 'complete.json')
    need(done['samples'] == 512 and done['status'].startswith('COMPLETE'), kind + ' incomplete endpoint')
    middle = 'records.json' if kind == 'history' else 'index.json'
    need(set(done['files_sha256']) == {'manifest.json', middle, 'summary.json'}, kind + ' incomplete ledger')
    for name, digest in done['files_sha256'].items():
        need(sha(safe_file(root, name)) == digest, kind + ' ledger SHA ' + name)
    manifest = read(root / 'manifest.json')
    index = read(root / middle)
    need(manifest['schema'] == index['schema'] == done['schema'], kind + ' schema')
    rows = index['records']
    need(len(rows) == 512, kind + ' rows')
    if kind != 'history':
        need({p.relative_to(root).as_posix() for p in (root / 'samples').glob('*.npz')} == {r['file'] for r in rows}, kind + ' file set')
    return done, manifest, rows


def npz(root, row):
    path = safe_file(root, row['file'])
    need(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'NPZ SHA/size ' + row['file'])
    with np.load(path, allow_pickle=False) as archive:
        value = {key: archive[key] for key in archive.files}
    need(json.loads(value['identity_json'].item()) == row['identity'], 'NPZ embedded identity')
    if 'arrays' in row:
        need(set(value) == set(row['arrays']), 'NPZ descriptor keys')
        for key, array in value.items():
            need(row['arrays'][key] == {'shape': list(array.shape), 'dtype': str(array.dtype)}, 'NPZ array descriptor ' + key)
    return value


def byte_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def descending_roc_auc(score, label, weight):
    """Independent weighted ROC integration; equal-score groups enter together."""
    need(score.shape == label.shape == weight.shape and np.isfinite(score).all() and
         np.isfinite(label).all() and np.isfinite(weight).all() and np.all(weight > 0), 'AUC input')
    pos, neg = label > 0, label < 0
    p, n = float(weight[pos].sum()), float(weight[neg].sum())
    result = dict(auc=None, positive_points=int(pos.sum()), negative_points=int(neg.sum()),
                  zero_points=int((label == 0).sum()), positive_weight=p, negative_weight=n,
                  zero_weight=float(weight[label == 0].sum()))
    if not p or not n:
        return result
    binary = pos | neg
    levels, inverse = np.unique(score[binary], return_inverse=True)
    plus = np.bincount(inverse, weights=weight[binary] * pos[binary], minlength=len(levels))[::-1]
    minus = np.bincount(inverse, weights=weight[binary] * neg[binary], minlength=len(levels))[::-1]
    tp = np.r_[0., np.cumsum(plus)] / p
    fp = np.r_[0., np.cumsum(minus)] / n
    area = float(np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1]) * .5))
    need(-1e-12 <= area <= 1 + 1e-12, 'ROC integral bounds')
    result['auc'] = min(1., max(0., area))
    return result


def population_stats(value, weight, objects, object_ids):
    mass = float(weight.sum())
    gain = float(np.dot(weight, np.maximum(value, 0)))
    cost = float(np.dot(weight, np.maximum(-value, 0)))
    signed = float(np.dot(weight, value))
    return dict(points=len(value), nonempty_objects=len(np.unique(object_ids)), original_object_count=objects,
                weight_sum=mass, positive_points=int((value > 0).sum()), negative_points=int((value < 0).sum()),
                zero_points=int((value == 0).sum()), conditional_original_weight_mean_xy_m=signed / mass if mass else None,
                FULL_denominator_benefit_contribution_xy_m=signed / objects if objects else None,
                FULL_denominator_positive_gain_xy_m=gain / objects if objects else None,
                FULL_denominator_negative_cost_xy_m=cost / objects if objects else None)


def summarize(parts, original_objects):
    values = {key: np.concatenate([row[key] for row in parts], axis=0) for key in parts[0]}
    weight, reason = values['weight'], values['reason']
    common = reason == 0
    physical = population_stats(values['future_benefit'], weight, original_objects, values['object'])
    reasons = {name: population_stats(values['future_benefit'][reason == ri], weight[reason == ri],
                                      original_objects, values['object'][reason == ri]) for ri, name in enumerate(REASONS)}
    common_weight = weight[common]; mass = float(common_weight.sum())
    metric = {}
    for column, name in enumerate(METRICS):
        value = values['contrasts'][common, column]
        need(np.isfinite(value).all(), 'Common contrast missing')
        signed = float(np.dot(common_weight, value))
        metric[name] = dict(conditional_original_weight_mean=signed / mass if mass else None,
            FULL_original_object_denominator_contribution=signed / original_objects if original_objects else None,
            sign_weight_fraction={name: float(common_weight[mask].sum()) / mass if mass else None
                                  for name, mask in (('positive', value > 0), ('zero', value == 0), ('negative', value < 0))})
    correlation = {name: float(np.dot(common_weight, values['correlations'][common, column])) / mass if mass else None
                   for column, name in enumerate(('GT_reference', 'broken_GT_reference'))}
    auc = {name: descending_roc_auc(values['scores'][common, column], values['past_benefit'][common], common_weight)
           for column, name in enumerate(('score_true', 'score_broken', 'lsq_speed_xy_mps'))}
    need(sum(row['points'] for row in reasons.values()) == physical['points'], 'Reason partition')
    return dict(original_object_count=original_objects, original_future_physical_support=physical,
                original_future_physical_by_reference_reason=reasons,
                common_reference_valid=dict(points=int(common.sum()), nonempty_objects=len(np.unique(values['object'][common])),
                    weight_sum=mass, dimensionless_metrics=metric,
                    reference_correlation_conditional_original_weight_mean=correlation,
                    secondary_actual_past_benefit_auc=auc))


def run(a):
    started = time.monotonic()
    roots = {name: getattr(a, name + '_run') for name in ('motion', 'radar', 'camera', 'pose')}
    roots['history'] = a.history_inputs
    expected = dict(COMPLETE_SHA, pose=a.pose_complete_sha256)
    ledgers = {kind: authenticate(root, expected[kind], kind) for kind, root in roots.items()}
    pd, pm, pr = ledgers['pose']
    need(pd['schema'] == 'history-pose-address-control-train-v1' and
         pd['status'] == 'COMPLETE_HISTORY_POSE_ADDRESS_CONTROL_TRAIN', 'Pose endpoint kind')
    for key, (filename, digest) in POSE_SOURCES.items():
        need(sha(HERE / filename) == digest == pd[key] == pm[key], 'Pose source binding ' + key)
    for name, key in (('camera', 'camera'), ('motion', 'motion'), ('history', 'history_inputs')):
        need(pm[key + '_complete_sha256'] == expected[name] and
             pm[key + '_files_sha256'] == ledgers[name][0]['files_sha256'], 'Pose parent ledger ' + name)
    rm = ledgers['radar'][1]
    need(rm['layout'] == 'B,C,Y,X' and rm['shape'] == [1, 8, 200, 200], 'Radar axis contract')
    all_parts = {'all': [], 'moving': []}; denominators = dict(all=0, moving=0)
    source_points = valid_points = all_object_horizons = object_offset = 0
    scenes = set(); samples = set(); reference_counts = np.zeros(6, dtype=np.int64)
    for ordinal in range(512):
        rows = {name: item[2][ordinal] for name, item in ledgers.items()}
        identity = rows['motion']['identity']
        need(identity['split'] == 'train' and all(row['ordinal'] == ordinal and row['identity'] == identity for row in rows.values()), 'Original sample order/identity')
        need(all(rows[k]['file'] == rows['motion']['file'] for k in roots if k != 'history'), 'Original sample filenames')
        need(rows['pose']['camera_npz_sha256'] == rows['camera']['sha256'] and
             rows['pose']['motion_npz_sha256'] == rows['motion']['sha256'], 'Pose per-sample parent SHA')
        v = {name: npz(roots[name], rows[name]) for name in ('motion', 'radar', 'camera', 'pose')}
        m, r, c, p = (v[name] for name in ('motion', 'radar', 'camera', 'pose'))
        idx, obj = m['source_flat_indices'], m['object_index']; n = len(idx); k = len(m['instance_tokens'])
        need(idx.dtype == obj.dtype == np.int64 and idx.shape == obj.shape and
             np.array_equal(np.unique(obj), np.arange(k)) and
             (not n or (idx[0] >= 0 and idx[-1] < 640000 and np.all(np.diff(idx) > 0))), 'Source/object index layout')
        need(byte_equal(idx, c['source_flat_indices']) and byte_equal(idx, p['source_flat_indices']) and
             byte_equal(c['camera_index'], p['camera_index']) and byte_equal(c['reason_code'], p['original_reason_code']), 'Original point/view bytes')
        need(np.array_equal(m['valid'], m['object_future_valid'][:, obj]) and
             np.array_equal(m['object_speed_group'] >= 0, m['object_future_valid']) and
             m['valid'].shape == (4, n), 'Original full-object future support')
        count = np.bincount(obj, minlength=k)
        radar = r['radar_bev']; need(radar.shape == (1, 8, 200, 200) and radar.dtype == np.float32 and
                                    np.isfinite(radar).all() and np.isin(radar[:, 0], [0, 1]).all(), 'Radar carrier')
        digest = hashlib.sha256(b'torch.float32' + json.dumps(list(radar.shape)).encode() + radar.tobytes()).hexdigest()
        need(digest == rows['radar']['native_radar_tensor_sha256'], 'Native radar tensor digest')
        x, y = idx // 3200, (idx // 16) % 200
        present = radar[0, 0, y, x] > 0
        old_valid = c['valid']; reason = p['reference_reason_code']; common = reason == 0
        need(np.isin(reason, np.arange(6)).all() and np.array_equal(common, p['reference_valid']) and
             np.array_equal(old_valid, c['reason_code'] == 0) and np.array_equal(reason == 1, ~old_valid), 'Photo/reference masks')
        need(np.all(~common | old_valid) and np.all(p['pose_reason_code'][common] == 0), 'Reference common support')
        reference_counts += np.bincount(reason, minlength=6)
        for ri in range(6):
            need(int((reason == ri).sum()) == rows['pose']['reference_reason_counts'].get(str(ri), 0), 'Reference reason row count')
        d = m['D_displacement_m'].astype(np.float64)
        need(d.shape == (4, n, 3) and np.isfinite(d).all(), 'Original D field')
        velocity = (((.5 * d[0] + d[1]) + 1.5 * d[2]) + 2. * d[3]) / 7.5
        speed = np.sqrt((velocity[:, :2] ** 2).sum(axis=1))
        gt = m['target_displacement_m'][3].astype(np.float64)
        benefit = np.linalg.norm(gt[:, :2], axis=1) - np.linalg.norm(d[3, :, :2] - gt[:, :2], axis=1)
        need(np.isfinite(benefit).all(), 'Future benefit finite')
        history = rows['history']
        need(history['t0_lidar_us'] == history['lidar_sample_data']['timestamp'] and
             tuple(pair['channel'] for pair in history['cameras']) == CAMERAS, 'Original history time/order')
        times = np.asarray([[pair[t]['sample_data']['timestamp'] for t in ('current', 'past')]
                            for pair in history['cameras']], dtype=np.int64)
        selected = c['camera_index'] >= 0
        need(np.array_equal(c['timestamps_us'][selected], times[c['camera_index'][selected]]), 'Selected actual camera timestamps')
        dt = (c['timestamps_us'][common, 1] - int(history['t0_lidar_us'])).astype(np.float64) / 1e6
        need(np.all(dt < 0), 'Camera history time sign')
        hgt = p['reference_displacement_R_m'][common, 1, :2]
        past = np.full(n, np.nan)
        past[common] = np.linalg.norm(hgt, axis=1) - np.linalg.norm(hgt - dt[:, None] * velocity[common, :2], axis=1)
        need(np.isfinite(past[common]).all(), 'Past camera benefit')
        contrasts = np.full((n, 6), np.nan)
        for actual_column, (gt_key, old_key) in enumerate((('reference_correlation', 'correlation'),
                                                          ('reference_broken_correlation', 'broken_correlation'))):
            reference = p[gt_key][common]
            need(np.isfinite(reference).all() and np.all(np.abs(reference) <= 1 + 1e-12), 'GT correlation range')
            for target_column, old_hypothesis in enumerate((1, 0)):
                column = 2 * actual_column + target_column
                contrasts[common, column] = reference - c[old_key][common, old_hypothesis]
                need(np.array_equal(contrasts[common, column], p[METRICS[column]][common]), 'Stored paired delta algebra')
        contrasts[:, 4:] = contrasts[:, :2] - contrasts[:, 2:4]
        need(np.array_equal(c['score'][old_valid], c['residual'][old_valid, 0] - c['residual'][old_valid, 1]) and
             np.array_equal(c['broken_score'][old_valid], c['broken_residual'][old_valid, 0] - c['broken_residual'][old_valid, 1]), 'Original score direction')
        scores = np.column_stack([c['score'], c['broken_score'], speed])
        correlations = np.column_stack([p['reference_correlation'], p['reference_broken_correlation']])
        need(np.isfinite(scores[common]).all(), 'Same-support score finiteness')
        base = m['valid'][3] & (m['owner'] < 0) & ~present
        future_objects = m['object_future_valid'][3]
        for group in ('all', 'moving'):
            valid_objects = future_objects if group == 'all' else future_objects & (m['object_speed_group'][3] == 2)
            take = base & valid_objects[obj]
            denominators[group] += int(valid_objects.sum())
            all_parts[group].append(dict(weight=1. / count[obj[take]], object=obj[take] + object_offset,
                reason=reason[take], future_benefit=benefit[take], past_benefit=past[take],
                contrasts=contrasts[take], correlations=correlations[take], scores=scores[take]))
        source_points += n; valid_points += int(m['valid'][3].sum())
        all_object_horizons += int(m['object_future_valid'].sum()); object_offset += k
        scenes.add(identity['scene_token']); samples.add(identity['sample_token'])
    need((source_points, valid_points, all_object_horizons, denominators['all'], len(scenes), len(samples)) ==
         (1740053, 1608197, 43189, 10332, 256, 512), 'Full original support changed')
    pose_index = read(roots['pose'] / 'index.json')
    need(reference_counts.tolist() == [pose_index['reference_reason_counts'].get(str(i), 0) for i in range(6)], 'Global pose reason accounting')
    result = dict(schema='history-pose-address-control-independent-2s-audit-v1',
        status='COMPLETE_INDEPENDENT_RAW_2S_RECOMPUTATION_NOT_MAIN_RESULT_COMPARISON',
        sources={kind: {'complete_sha256': expected[kind], 'files_sha256': ledger[0]['files_sha256']}
                 for kind, ledger in ledgers.items()}, source_sha256=sha(__file__),
        samples=512, scenes=256, source_points=source_points, original_2s_valid_points=valid_points,
        original_object_horizon_rows=all_object_horizons, npz_files_SHA_verified=2048,
        horizon_seconds=2., strata={'no_radar': {group: summarize(parts, denominators[group]) for group, parts in all_parts.items()}},
        methodology={'weight': '1/n_ALL_original_object_source_points',
            'denominator': 'ALL original future-valid objects of the named group; not subset objects',
            'AUC': 'descending ROC trapezoidal integration at tied-score groups; zero label excluded only from binary AUC',
            'past_label': 'norm(GT reference past displacement R.xy) minus norm(GTdispR.xy - actual(camera_past-LiDAR_t0) * original 4h LSQ D.xy)',
            'scope': '2s no-radar CRN-uncovered all and moving only; not a full four-horizon/stratum/bin audit',
            'no_main_analysis_read': True, 'no_analysis_helper_import': True, 'no_images_or_raw_box_read': True,
            'no_geometry_recomputation': True, 'no_model_or_Torch': True},
        elapsed_seconds=time.monotonic() - started, runtime={'python': sys.version, 'numpy': np.__version__},
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    need('torch' not in sys.modules, 'Unexpected Torch import')
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'out': str(a.out), 'sha256': sha(a.out), 'seconds': result['elapsed_seconds']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion-run', 'radar-run', 'camera-run', 'pose-run', 'history-inputs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pose-complete-sha256', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
