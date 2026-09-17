"""Independent completed N/P/Z audit. Stdlib + NumPy; no producer imports."""
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np

N = Path(__file__).resolve().parent
P = N.parent / 'm0_improvement_20260915'
RUN = N / 'server_results/training/object_state_velocity_assignment_v1'
TRAIN = N / 'server_results/training/object_state_forecast_train_v1'
OUT = N / 'independent_object_state_velocity_assignment_audit_v1'
ARMS = ('N', 'P', 'Z')
H = (.5, 1., 1.5, 2.)
GMO = ('car', 'bus', 'truck', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle', 'pedestrian')
GROUPS = ('stationary', 'ambiguous', 'moving')
POS = ('speed_le_0.1', 'speed_gt_0.1_le_0.5', 'speed_gt_0.5_le_5', 'speed_gt_5',
       'annotated_future_only', 't0_missing_with_history', 'unknown_no_current_box', 'overlap_current_boxes')
EXPECTED_COMPLETE = 'ec70afff3b03a3915758a60071947864724981de2fc362abf428759f0751b395'
EXPECTED_STATE = 'd93e4905cbd896ae02d744609baec395079180973f97b354e09e3f4c6ea9fd08'


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    with path.open('rb') as f:
        result = hashlib.sha256()
        while chunk := f.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def array_sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def ledger(root, entry):
    for name, expected in entry['files_sha256'].items():
        need(sha(root / name) == expected, 'ledger: ' + str(root / name))


def finite_tree(x):
    if isinstance(x, float):
        need(math.isfinite(x), 'nonfinite JSON number')
    elif isinstance(x, dict):
        for v in x.values(): finite_tree(v)
    elif isinstance(x, list):
        for v in x: finite_tree(v)


def quat_rotation(q):
    v = np.asarray(q, np.float64)
    v = v / np.abs(v).max()
    w, x, y, z = v / np.linalg.norm(v)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def packed_from_input(boxes, G0):
    inverse = np.linalg.inv(G0)
    records, indices = [], []
    for i, b in enumerate(boxes):
        if b['detection_name'] not in GMO:
            continue
        center = inverse[:3, :3] @ b['translation'] + inverse[:3, 3]
        rotation = inverse[:3, :3] @ quat_rotation(b['rotation'])
        velocity = inverse[:3, :3] @ np.r_[b['velocity'], 0.]
        records.append(np.r_[center, b['size'], rotation.ravel(),
                             np.eye(8)[GMO.index(b['detection_name'])], b['detection_score'], velocity])
        indices.append(i)
    a = np.asarray(records, dtype=np.float32).reshape(1, len(records), 27)
    # Match the frozen CUDA scalar normalization's observed byte contract.
    # Float32 reciprocal multiplication and direct NumPy division can differ.
    a[:, :, :6] *= np.float32(1.) / np.float32(51.2)
    a[:, :, 24:] *= np.float32(1.) / np.float32(20.)
    return a, indices


def raw_counts(metric):
    occ = np.asarray([h['occupancy']['confusion'] for h in metric['horizons']], dtype=np.int64)
    trans = np.asarray([h['confusion'] for h in metric['transitions']], dtype=np.int64)
    group = np.asarray([[[h['motion_positive_attribution']['groups'][g][k] for k in ('GT', 'TP', 'FN')]
                         for g in POS] for h in metric['horizons'][1:]], dtype=np.int64)
    need(occ.shape == (5, 2, 2) and trans.shape == (4, 4, 4) and group.shape == (4, 8, 3), 'count shapes')
    need(all((a >= 0).all() for a in (occ, trans, group)), 'negative counts')
    need(np.array_equal(group[..., 0], group[..., 1] + group[..., 2]), 'group conservation')
    need(np.array_equal(group[:, :, 0].sum(1), occ[1:, 1].sum(1)), 'group positive denominator')
    need(np.array_equal(group[:, :, 1].sum(1), occ[1:, 1, 1]), 'group TP')
    for h, t in enumerate(metric['transitions']):
        d = t['domain']
        need(trans[h].sum() == d['both_valid'] and sum(d[k] for k in (
             'both_valid', 't0_valid_h_ignored', 't0_ignored_h_valid', 'both_ignored')) == d['grid_voxels'], 'transition domain')
    return occ, trans, group


def ratio(n, d):
    n, d = np.broadcast_arrays(np.asarray(n, float), np.asarray(d, float))
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d != 0)


def metrics(o, t, g):
    iou = 100 * ratio(o[..., 1, 1], o[..., 0, 1] + o[..., 1, 0] + o[..., 1, 1])
    result = {'future_GMO_IoU_percent': iou[..., 1:].mean(-1), 't0_GMO_IoU_percent': iou[..., 0],
              'future_FP': o[..., 1:, 0, 1].sum(-1), 'future_FN': o[..., 1:, 1, 0].sum(-1)}
    for i in range(4): result[f'h{i+1}_GMO_IoU_percent'] = iou[..., i+1]
    for i, name in enumerate(('00_background', '01_arrival', '10_vacating', '11_persistent')):
        v = 100 * ratio(t[..., i, i], t[..., i, :].sum(-1) + t[..., :, i].sum(-1) - t[..., i, i])
        result[f'future_{name}_IoU_percent'] = v.mean(-1)
        for h in range(4): result[f'h{h+1}_{name}_IoU_percent'] = v[..., h]
    for i, name in enumerate(POS):
        v = 100 * ratio(g[..., i, 1], g[..., i, 0])
        result[f'future_{name}_recall_percent'] = v.mean(-1)
        for h in range(4): result[f'h{h+1}_{name}_recall_percent'] = v[..., h]
    moving = g[..., 2:4, :].sum(-2)
    v = 100 * ratio(moving[..., 1], moving[..., 0])
    result['future_speed_gt_0.5_recall_percent'] = v.mean(-1)
    for h in range(4): result[f'h{h+1}_speed_gt_0.5_recall_percent'] = v[..., h]
    return result


def number_comparison(actual, expected, differences, path):
    if expected is None:
        need(not np.isfinite(actual), 'unexpected finite value: ' + path)
    else:
        need(np.isfinite(actual), 'undefined value: ' + path)
        delta = float(actual) - float(expected)
        if delta:
            differences.append({'path': path, 'independent': float(actual), 'producer': expected, 'difference': delta})


def interval(point, samples):
    finite = samples[np.isfinite(samples)]
    return dict(difference=float(point), lower95=float(np.quantile(finite, .025)) if len(finite) else None,
                upper95=float(np.quantile(finite, .975)) if len(finite) else None,
                finite_draws=len(finite), undefined_draws=len(samples)-len(finite))


def run():
    started = time.monotonic()
    need(not OUT.exists(), 'Output exists; do not overwrite an audit')
    need(sha(RUN/'complete.json') == EXPECTED_COMPLETE, 'expected complete SHA')
    need(sha(RUN/'_job/state.json') == EXPECTED_STATE, 'expected terminal state SHA')
    complete, manifest, summary = (read(RUN/n) for n in ('complete.json', 'manifest.json', 'summary.json'))
    need(complete['status'] == summary['status'] == 'COMPLETE_SAME_WEIGHT_VELOCITY_DIAGNOSTIC', 'terminal status')
    ledger(RUN, complete)
    need(sha(N/'diagnose_object_state_velocity_assignment_v1.py') == complete['source_sha256'], 'producer source')
    for name, digest in manifest['dependencies_sha256'].items():
        source = N/name if (N/name).is_file() else P/name
        need(sha(source) == digest, 'frozen dependency: ' + name)
    need(sha(TRAIN/'complete.json') == complete['training_complete_sha256'], 'formal V/G endpoint')
    need(sha(N/'object_state_forecast_training_protocol_v1.json') == manifest['training_protocol_sha256'], 'formal protocol')
    for name, digest in manifest['completed_training_metadata']['files_sha256'].items():
        need(sha(TRAIN/name) == digest, 'formal small artifact: ' + name)
    cpu = N/'object_state_final_cpu_audit_v1'
    need(sha(cpu/'complete.json') == complete['cpu_audit_complete_sha256'], 'actual CPU audit endpoint')
    ledger(cpu, read(cpu/'complete.json'))
    audit = read(cpu/'audit.json'); loaded = read(RUN/'loaded_models.json')
    need(audit['status'] == 'PASS_ACTUAL_CPU_FINAL_TENSORS' and audit['complete_sha256'] == complete['training_complete_sha256'], 'actual CPU audit identity')
    need(loaded['actual_V_components'] == audit['arms']['V']['components'], 'loaded V versus actual CPU tensor receipt')
    need(loaded['V_checkpoint_sha256'] == audit['arms']['V']['checkpoint_sha256'], 'V checkpoint identity')
    need(loaded['optimizer_updates'] == complete['optimizer_updates'] == summary['optimizer_updates'] == 0
         and loaded['optimizer_restored'] is False, 'evaluation updates')
    selection_path = P/'selection_v1.json'; selection = read(selection_path)
    chosen = [r for r in selection['records'] if r['split'] == 'development']
    need(sha(selection_path) == manifest['completed_training_metadata']['selection_sha256'], 'selection SHA')
    data = rows(RUN/'records.jsonl'); objects = rows(RUN/'physical_objects.jsonl')
    original = rows(TRAIN/'development_records.jsonl')
    original_objects = [dict(r, arm='N') for r in rows(TRAIN/'development_objects.jsonl') if r['arm'] == 'V']
    need(len(data) == len(original) == len(chosen) == 200 and len({r['sample_token'] for r in data}) == 200, '200 unique anchors')
    scenes = sorted({r['scene_token'] for r in chosen}); scene_ids = {s: i for i, s in enumerate(scenes)}
    need(len(scenes) == 100 and set(Counter(r['scene_token'] for r in data).values()) == {2}, '100 scenes, two anchors each')
    need([r for r in objects if r['arm'] == 'N'] == original_objects, 'exact original V physical rows')
    key = lambda r: (r['sample_token'], r['instance_token'], r['horizon_seconds'])
    object_maps = {a: {key(r): r for r in objects if r['arm'] == a} for a in ARMS}
    need(Counter(r['arm'] for r in objects) == {a: 16074 for a in ARMS}, 'physical row counts')
    need(all(len(m) == 16074 for m in object_maps.values()), 'duplicate physical identities')
    keys = sorted(object_maps['N'])
    need(all(set(m) == set(keys) for m in object_maps.values()), 'physical support differs')
    for k in keys:
        ref = object_maps['N'][k]
        for a in ARMS:
            need(all(object_maps[a][k][f] == ref[f] for f in ('scene_token', 'source_points', 'dt_seconds', 'group', 'ordinal',
                     'zero_epe_xy_m', 'zero_epe_3d_m')), 'physical denominator/label identity')
    finite_tree(objects); finite_tree(data); finite_tree(summary)
    prediction_root = N/'crn_state_dev200_centered_v1'; raw_root = N/'motion_targets_v1'
    for filename, field in (('complete.json', 'complete_sha256'), ('manifest.json', 'manifest_sha256'), ('predictions.json', 'predictions_sha256')):
        need(sha(prediction_root/filename) == manifest['predicted_states'][field], 'centered input: ' + filename)
    pred = read(prediction_root/'predictions.json')
    need(pred['box_origin'] == 'global_geometric_center', 'center already corrected')
    raw_manifest = read(raw_root/'manifest.json')
    need(sha(raw_root/'manifest.json') == manifest['predicted_states']['raw_manifest_sha256'], 'raw manifest')
    need(sha(raw_root/'complete.json') == manifest['predicted_states']['raw_complete_sha256'], 'raw complete')
    raw_desc = {r['identity']['sample_token']: r for r in raw_manifest['records']}
    sparse_root = N/'sparse_motion_v1'; sparse_manifest = read(sparse_root/'manifest.json')
    need(sha(sparse_root/'manifest.json') == manifest['completed_training_metadata']['sparse_manifest_sha256'], 'sparse manifest')
    sparse_desc = {r['identity']['sample_token']: r for r in sparse_manifest['records']}
    scene_arrays = {a: [np.zeros((100, 5, 2, 2), np.int64), np.zeros((100, 4, 4, 4), np.int64), np.zeros((100, 4, 8, 3), np.int64)] for a in ARMS}
    deviations = []; zero_deviations = []; assignments = []
    support_count = np.zeros((4, 3), np.int64); support_points = support_count.copy()
    for ordinal, (r, prior, selected, pr) in enumerate(zip(data, original, chosen, pred['records'])):
        token = r['sample_token']; si = scene_ids[r['scene_token']]
        need(r['ordinal'] == ordinal and all(r[k] == prior[k] == selected[k] == pr[k] for k in selected), 'selection identity/order')
        need(r['hist_by_arm']['N'] == prior['hist_by_arm']['V'] and r['metrics_by_arm']['N'] == prior['metrics_by_arm']['V'], 'N exact original V common')
        need(r['inputs_sha256'] == prior['inputs_sha256'] and r['targets_sha256'] == prior['targets_sha256'], 'native input/target identity')
        condition_counts = {}
        for a in ARMS:
            condition_counts[a] = raw_counts(r['metrics_by_arm'][a])
            need(condition_counts[a][0].tolist() == r['hist_by_arm'][a], 'native/CPU hist')
            for i, array in enumerate(condition_counts[a]): scene_arrays[a][i][si] += array
            hook = r['conditioner_hooks'][a]
            need(hook['use_velocity'] is True and hook['frames'] == [1, 2, 3, 4] and hook['completed'] and hook['hook_removed'], 'full independent rollout hook')
        for a in ('P', 'Z'):
            aa, tt, gg = condition_counts[a]; nn, nt, ng = condition_counts['N']
            need(np.array_equal(aa.sum(-1), nn.sum(-1)) and np.array_equal(tt.sum(-1), nt.sum(-1)) and np.array_equal(gg[..., 0], ng[..., 0]), 'common GT denominators')
            need(r['metrics_by_arm'][a]['t0_boundary'] == r['metrics_by_arm']['N']['t0_boundary'], 'same-V t0 boundary')
            need([x['domain'] for x in r['metrics_by_arm'][a]['transitions']] == [x['domain'] for x in r['metrics_by_arm']['N']['transitions']], 'transition support')
        rd = raw_desc[token]; carrier_path = raw_root/rd['file']
        need(sha(carrier_path) == rd['sha256'] == r['raw_label_sha256'] == r['state_source']['raw_pose_carrier_sha256'], 'raw carrier SHA')
        content = gzip.decompress(carrier_path.read_bytes())
        need(hashlib.sha256(content).hexdigest() == rd['uncompressed_json_sha256'], 'raw decompressed SHA')
        carrier = json.loads(content); frame = carrier['frames'][2]
        need(carrier['identity'] == selected and frame['relative_frame_index'] == 0 and frame['sample_token'] == token, 'current G0 identity')
        G0 = np.asarray(frame['lidar_to_global_column_matrix'], np.float64)
        need(array_sha(G0) == r['state_source']['current_G0_sha256'], 'current G0 bytes')
        packed, indices = packed_from_input(pr['boxes'], G0); count = len(indices); rec = r['assignment']
        need(indices == rec['original_box_indices'] == r['state_source']['retained_original_box_indices'], 'all eight-class boxes and original order')
        need(array_sha(packed) == r['state_source']['packed_states_sha256'] and array_sha(packed[:, :, :24]) == r['state_source']['geometry_sha256'], 'independent packed/geometry bytes')
        order = sorted(indices, key=lambda i: (hashlib.sha256(f'velocity-assignment-v1|{token}|{i}'.encode()).digest(), i))
        donor_dict = dict(zip(order, order[1:] + order[:1])) if count > 1 else dict(zip(order, order))
        donor = [donor_dict[i] for i in indices]; reverse = {v: i for i, v in enumerate(indices)}
        need(order == rec['hash_sorted_original_indices'] and donor == rec['donor_original_box_indices'], 'hash-cycle donor map')
        need(rec['reassigned_object_indices'] == (count if count > 1 else 0) and rec['valid_objects'] == count, 'reassignment counts')
        variants = {a: packed.copy() for a in ARMS}
        variants['P'][0, :, 24:] = packed[0, [reverse[i] for i in donor], 24:]
        variants['Z'][0, :, 24:] = 0.
        multiset = lambda v: sorted(x.tobytes() for x in v[0, :, 24:])
        need(multiset(variants['P']) == multiset(packed), 'independent full-vector byte multiset')
        item = {'ordinal': ordinal, 'sample_token': token, 'objects': count, 'conditions': {}}
        for a, v in variants.items():
            need(v[:, :, :24].tobytes() == packed[:, :, :24].tobytes(), 'score/geometry/class changed')
            velocity = v[0, :, 24:]; before = packed[0, :, 24:]; delta = (velocity.astype(float)-before)*20
            change = np.linalg.norm(delta, axis=1)
            measured = dict(actual_numeric_velocity_changes=int(np.any(velocity != before, axis=1).sum()),
                            byte_changed_velocities=sum(x.tobytes() != y.tobytes() for x, y in zip(velocity, before)),
                            max_velocity_change_mps=float(change.max()) if count else 0.,
                            mean_velocity_change_mps=float(change.mean()) if count else 0., packed_states_sha256=array_sha(v))
            need(measured == rec['conditions'][a], 'actual packed intervention differs: ' + token + '/' + a)
            item['conditions'][a] = measured
        assignments.append(item)
        sd = sparse_desc[token]; need(sha(sparse_root/sd['file']) == sd['sha256'] == r['sparse_label_sha256'], 'sparse NPZ SHA')
        with np.load(sparse_root/sd['file'], allow_pickle=False) as sp:
            need(str(sp['sample_token']) == token and str(sp['scene_token']) == r['scene_token'], 'NPZ identity')
            for hi, horizon in enumerate(H):
                nobjects, npoints = 0, 0
                for oi, instance in enumerate(sp['instance_tokens']):
                    mask = sp['valid'][hi] & (sp['object_index'] == oi); size = int(mask.sum())
                    if size == 0: continue
                    nobjects += 1; npoints += size; gi = int(sp['object_speed_group'][hi, oi])
                    support_count[hi, gi] += 1; support_points[hi, gi] += size
                    obj = object_maps['N'][(token, str(instance), horizon)]
                    need(obj['source_points'] == size and obj['group'] == GROUPS[gi] and obj['dt_seconds'] == float(sp['dt_future_seconds'][hi]), 'actual NPZ object support')
                    target = sp['target_displacement_m'][hi, mask].astype(float)
                    for suffix, values in (('xy', target[:, :2]), ('3d', target)):
                        actual = float(np.linalg.norm(values, axis=1).mean())
                        number_comparison(actual, obj['zero_epe_'+suffix+'_m'], zero_deviations, token+f'/{horizon}/{instance}/{suffix}')
                need(nobjects == sd['counts']['valid_objects_by_horizon'][hi] and npoints == sd['counts']['valid_points_by_horizon'][hi], 'manifest valid counts')
        if (ordinal+1) % 50 == 0: print('Verified input/support anchors', ordinal+1, flush=True)
    assignment_totals = dict(anchors_N0=sum(x['objects'] == 0 for x in assignments), anchors_N1=sum(x['objects'] == 1 for x in assignments),
                            reassigned_objects=sum(x['objects'] if x['objects'] > 1 else 0 for x in assignments),
                            actual_velocity_changes={a: sum(x['conditions'][a]['actual_numeric_velocity_changes'] for x in assignments) for a in ARMS})
    need(assignment_totals == summary['assignments'], 'summary assignment totals')
    draws = np.random.RandomState(11).randint(100, size=(10000, 100))
    weights = np.asarray([np.bincount(x, minlength=100) for x in draws], dtype=np.int64)
    need(array_sha(weights) == summary['common']['bootstrap_weights_sha256'] == summary['physical']['bootstrap_weights_sha256'], 'bootstrap exact draw weights')
    points, boot = {}, {}
    for a in ARMS:
        totals = [x.sum(0) for x in scene_arrays[a]]
        for key_name, total in zip(('occupancy_5x2x2', 'transition_4x4x4', 'groups_4x8x3_GT_TP_FN'), totals):
            need(total.tolist() == summary['common']['pooled_counts'][a][key_name], 'pooled integer counts')
        points[a] = metrics(*totals)
        sampled = [(weights @ x.reshape(100, -1)).reshape((10000,) + x.shape[1:]) for x in scene_arrays[a]]
        boot[a] = metrics(*sampled)
        for k, value in points[a].items(): number_comparison(value, summary['common']['values'][a][k], deviations, 'common/'+a+'/'+k)
    common_comparisons = {}
    for a in ('P', 'Z'):
        pair = a+'-minus-N'; common_comparisons[pair] = {}
        for k in points[a]:
            v = interval(points[a][k]-points['N'][k], boot[a][k]-boot['N'][k]); expected = summary['common']['comparisons'][pair][k]
            for f in ('difference', 'lower95', 'upper95'): number_comparison(np.nan if v[f] is None else v[f], expected[f], deviations, 'common/'+pair+'/'+k+'/'+f)
            need(v['finite_draws'] == expected['finite_bootstrap_repetitions'] and v['undefined_draws'] == expected['undefined_bootstrap_repetitions'], 'common defined draws')
            common_comparisons[pair][k] = v
    physical = []; phy_expected = {(x['horizon_seconds'], x['group']): x for x in summary['physical']['physical']}
    for horizon in H:
        for group in ('all', *GROUPS):
            selected = [k for k in keys if k[2] == horizon and (group == 'all' or object_maps['N'][k]['group'] == group)]
            ids = np.array([scene_ids[object_maps['N'][k]['scene_token']] for k in selected]); counts = np.bincount(ids, minlength=100)
            item = dict(horizon_seconds=horizon, group=group, object_anchor_pairs=len(selected),
                        source_point_occurrences=sum(object_maps['N'][k]['source_points'] for k in selected), scenes_with_support=int((counts>0).sum()), metrics={})
            expected = phy_expected[(horizon, group)]
            need(all(item[f] == expected[f] for f in item if f != 'metrics'), 'physical group support')
            for metric in ('epe_xy_m', 'epe_3d_m'):
                values = {a: np.array([object_maps[a][k][metric] for k in selected]) for a in ARMS}
                values['zero'] = np.array([object_maps['N'][k]['zero_'+metric] for k in selected])
                descriptions = {}; sampled = {}
                for a, value in values.items():
                    descriptions[a] = dict(count=len(value), mean=float(value.mean()), median=float(np.median(value)), p90=float(np.quantile(value, .9)))
                    scene_sum = np.bincount(ids, weights=value, minlength=100)
                    # Independent draw-index summation (not producer matrix multiplication).
                    sampled[a] = scene_sum[draws].sum(1) / counts[draws].sum(1)
                    need(descriptions[a]['count'] == expected['metrics'][metric]['values'][a]['count'], 'physical object count')
                    for f in ('mean', 'median', 'p90'): number_comparison(descriptions[a][f], expected['metrics'][metric]['values'][a][f], deviations, f'physical/{horizon}/{group}/{metric}/{a}/{f}')
                comparisons = {}
                for a in ('P', 'Z'):
                    pair=a+'-minus-N'; v=interval(descriptions[a]['mean']-descriptions['N']['mean'], sampled[a]-sampled['N'])
                    e=expected['metrics'][metric]['comparisons'][pair]
                    for f in ('difference', 'lower95', 'upper95'): number_comparison(v[f], e[f], deviations, f'physical/{horizon}/{group}/{metric}/{pair}/{f}')
                    need(v['finite_draws'] == e['defined_draws'] and v['undefined_draws'] == e['undefined_draws'], 'physical defined draws')
                    comparisons[pair]=v
                item['metrics'][metric]=dict(values=descriptions, comparisons=comparisons)
            physical.append(item)
    dense = {}
    for pair in ('P-minus-N', 'Z-minus-N'):
        dense[pair] = {}
        for family, nh, expected_size in (('native_logits_by_horizon',5,3840000), ('terminal_features_by_horizon',5,10240000), ('physical_field_by_future_horizon',4,1920000)):
            pooled=[]
            for hi in range(nh):
                entries=[r['tensor_differences'][pair][family][hi] for r in data]
                for e in entries:
                    need(e['index']==hi and e['elements']==expected_size and 0<=e['numeric_changed']<=e['elements'], 'dense receipt dimensions')
                    need(0<=e['mean_abs']<=e['max_abs'], 'dense magnitude consistency')
                    need((e['numeric_changed']==0)==(e['max_abs']==0), 'dense numeric count consistency')
                    if e['byte_equal']: need(e['numeric_changed']==0, 'byte equality/numeric inequality')
                    if hi==0 and nh==5: need(e['byte_equal'] and e['numeric_changed']==0, 't0 logits/features changed')
                total=sum(e['elements'] for e in entries); changed=sum(e['numeric_changed'] for e in entries)
                pooled.append(dict(horizon_seconds=(hi if nh==5 else hi+1)*.5, elements=total, numeric_changed=changed,
                                   changed_fraction=changed/total, byte_equal_anchors=sum(e['byte_equal'] for e in entries),
                                   max_abs=max(e['max_abs'] for e in entries),
                                   element_weighted_mean_abs=math.fsum(e['elements']*e['mean_abs'] for e in entries)/total))
            dense[pair][family]=pooled
    need('torch' not in sys.modules, 'Torch imported')
    # No numerical acceptance tolerance: expose each measured reduction difference.
    result=dict(schema='independent-object-state-velocity-assignment-audit-v1', status='COMPLETED_LOCAL_CPU_AUDIT',
        bindings=dict(complete_sha256=EXPECTED_COMPLETE, state_sha256=EXPECTED_STATE, producer_sha256=complete['source_sha256'],
                      files_sha256=complete['files_sha256'], training_complete_sha256=complete['training_complete_sha256'],
                      cpu_audit_complete_sha256=complete['cpu_audit_complete_sha256'], audit_source_sha256=sha(Path(__file__))),
        checks=dict(all_six_output_files_rehashed=True, frozen_dependencies_rehashed=len(manifest['dependencies_sha256']),
                    native_input_and_label_identity=True, samples=200, scenes=100, physical_rows_per_arm=16074,
                    N_original_V_common_and_objects_exact=True, all_600_packed_states_sha_exact=True,
                    all_200_geometry_sha_exact=True, all_200_G0_sha_exact=True, all_200_sparse_npz_sha_verified=True,
                    full_support_from_actual_npz=True, P_donor_map_and_vector_multiset_exact=True,
                    all_pooled_counts_exact=True, all_point_estimates_recomputed=True, paired_scene_draws_sha_exact=True,
                    independent_10000_scene_intervals=True, dense_receipt_consistency=True, t0_same_V_bytes_receipts=True),
        assignments=dict(totals=assignment_totals, per_anchor=assignments),
        sparse_support=dict(groups=list(GROUPS), horizons=list(H), object_counts=support_count.tolist(), point_counts=support_points.tolist()),
        common=dict(values={a:{k:None if not np.isfinite(v) else float(v) for k,v in points[a].items()} for a in ARMS}, comparisons=common_comparisons),
        physical=physical, dense_receipt_aggregation=dense,
        numeric_comparison=dict(nonidentical_values=deviations, max_absolute_difference=max([abs(x['difference']) for x in deviations],default=0),
                                zero_EPE_from_NPZ_nonidentical_values=zero_deviations,
                                zero_EPE_max_absolute_difference=max([abs(x['difference']) for x in zero_deviations],default=0),
                                no_acceptance_tolerance_added=True,
                                packing_arithmetic='Float64 geometry -> float32 -> float32 reciprocal multiply for /51.2 and /20; all packed hashes exact; direct NumPy division is not byte-equivalent.'),
        execution=dict(python=sys.version,numpy=np.__version__,seconds=time.monotonic()-started,torch_imported=False,
                       model_calls=0,optimizer_updates=0,GPU_used=False,SSH_used=False,remote_PID_check_performed=False),
        limitations=['Dense native tensors and sparse predicted point vectors were not stored; this audit reaggregates authenticated tensor-difference receipts and per-object errors, not raw model outputs.',
                     'Final model tensors are cross-linked to the prior actual CPU audit; this audit does not load checkpoints.',
                     'Remote process disappearance is root-reported; only the downloaded terminal state SHA was checked here.',
                     'P may be OOD; Z is same final V with zero velocities, not trained G. No permutation or training-seed uncertainty in scene intervals.',
                     'No threshold, filtering, new model, new performance gate or training change.'])
    finite_tree(result)
    OUT.mkdir()
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    m=next(x for x in physical if x['horizon_seconds']==2 and x['group']=='moving')['metrics']['epe_xy_m']
    lines=['# N/P/Z 完整终态独立 CPU 审查','',
           '只读已完成本地镜像；未调用 Torch、GPU、SSH、模型或优化器。未改冻结源、指标、标签、阈值和支持。',
           '', '**来源与支持通过。** 六个终点文件和 28 份依赖逐个重算 SHA；正式 V/G 及实际 final CPU tensor audit 相互绑定。200 anchors／100 scenes、每臂 16074 个同身份对象—时域记录；读取 200 份原 sparse NPZ 重建支持。N 的逐样本完整 common/hist 与逐对象记录精确复现正式 V。',
           '', '**输入干预复现。** 从 centered CRN 输入和原当前 G0 独立重建 200 份 27 维 packed states；全部 N/P/Z 共 600 份 SHA 与 200 份 geometry SHA 精确相同。没有再次 origin 转换。归一化使用 float32 倒数乘法以复现冻结 GPU 的标量算术；直接 NumPy 除法不是字节等价，未放宽容差。',
           f"P 重分配 {assignment_totals['reassigned_objects']} 个框 index，实际数值改变 {assignment_totals['actual_velocity_changes']['P']}；Z 改变 {assignment_totals['actual_velocity_changes']['Z']}。P 的完整 R 三维速度向量多重集逐帧精确保留，geometry/class/score 不变；无 0/1 框 anchor。", '',
           '| 条件 | future GMO (%) | 2 s moving XY EPE (m) |','|---|---:|---:|']
    for a in ARMS:lines.append(f"| {a} | {float(points[a]['future_GMO_IoU_percent']):.12f} | {m['values'][a]['mean']:.12f} |")
    lines += ['', '| 对比 | future GMO 差 (pp)，95% CI | 2 s moving 差 (m)，95% CI |','|---|---:|---:|']
    for a in ('P','Z'):
        pair=a+'-minus-N';c=common_comparisons[pair]['future_GMO_IoU_percent'];q=m['comparisons'][pair]
        lines.append(f"| {pair} | {c['difference']:+.12g} [{c['lower95']:+.12g}, {c['upper95']:+.12g}] | {q['difference']:+.12g} [{q['lower95']:+.12g}, {q['upper95']:+.12g}] |")
    lines += ['', '从原整数独立池化各时域 confusion，再平均四未来 IoU；物理先保留原每对象点平均误差，再按 anchor-instance 等权。固定 10,000 次 seed11 配对整场景抽样，权重字节精确相同。全部分组/四时域/XY、XYZ mean、median、p90 与所有 common 点估计均复算；未把场景等权误当对象等权。',
              f"独立与 producer 数值非逐位相同项 {len(deviations)} 个，最大绝对差 {result['numeric_comparison']['max_absolute_difference']:.17g}；逐项保存在 audit.json。物理 bootstrap 使用独立 draw-index 求和，producer 使用矩阵乘法；保留归约差，不加数值容差。原 NPZ 重算 zero EPE 最大差 {result['numeric_comparison']['zero_EPE_max_absolute_difference']:.17g} m。", '',
              '**密集影响的边界。** 已核 t0 logits/features 的全部零差与 byte-equal 收据，并按元素数汇总未来 logits、features、physical field 的改变数量、最大差及平均绝对差。原密集张量及逐点预测向量未保存，不能声称本次独立复算了密集张量差或原始逐点预测误差；本次独立复算的是经过 SHA 绑定的收据和每对象统计。', '',
              'P/Z 是同权重事后敏感性/效用诊断，P 同时破坏位置/类别与速度的联合关系，可能分布外；Z 不是独立训练 G。张量变化不等于预测更准，极小但非零变化不得写成严格为零。场景 CI 不包含其他置换或训练 seed 方差。本报告不新增性能晋级门。', '',
              '远端 runner/child 消失由根任务确认；本审查只核下载 terminal state SHA。checkpoint 张量仅对齐此前真实 CPU 审计，不在本次重复载入。', '',
              f'complete SHA: `{EXPECTED_COMPLETE}`', f'audit JSON SHA: `{sha(OUT/"audit.json")}`', f'auditor SHA: `{sha(Path(__file__))}`']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'out':str(OUT),'audit_sha256':sha(OUT/'audit.json'),'report_sha256':sha(OUT/'report.md'),
                      'numeric_nonidentical':len(deviations),'max_abs_difference':result['numeric_comparison']['max_absolute_difference']}),flush=True)


if __name__ == '__main__':
    run()
