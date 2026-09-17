"""CPU-only privileged history-pose address diagnostic; QA never opens data.

The frozen parent supplies all original geometry, future-valid support, object
weights and denominators. New correlation differences are dimensionless. The
only new AUC target is signed displacement benefit at the actual past-camera
timestamp. No fitting, threshold selection, bootstrap, images or raw GT reads.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = 'history-pose-address-control-train-analysis-v1'
POSE_SCHEMA = 'history-pose-address-control-train-v1'
PARENT_ANALYZER_SHA = '69ad87766e8964412fef11bd3e0b7810b4c109a5e2bc7ce1f94475e580df68c0'
RULES_PATH = HERE / 'history_pose_address_control_train_analysis_rules_v1.json'
RULES_SHA = '0294946c9a60c4cd7278c783e62c86ceee0674ca38d4967612d71b2832ca740a'
GROUPS = ('all', 'stationary', 'ambiguous', 'moving')
STRATA = ('all', 'no_radar', 'radar_present')
HORIZONS = (.5, 1., 1.5, 2.)
EDGES = np.asarray([0., .1, .5, 1., 2., 5., 10., np.inf])
METRICS = ('delta_eD_minus_eGT', 'delta_eZero_minus_eGT',
           'broken_delta_eD_minus_eGT', 'broken_delta_eZero_minus_eGT',
           'actual_minus_broken_delta_eD_minus_eGT', 'actual_minus_broken_delta_eZero_minus_eGT')
AUC_FEATURES = ('score_true', 'score_broken', 'lsq_speed_xy_mps')
DESCRIPTORS = ('actual_past_GT_displacement_xy_m', 'actual_past_D_to_GT_error_xy_m',
               'past_reference_departure_from_zero_px', 'past_reference_departure_from_D_px')
REASONS = ('valid', 'original_photo_invalid', 'current_pose_unavailable',
           'past_pose_unavailable', 'current_GT_projection_invalid', 'past_GT_projection_invalid')
POSE_REASONS = ('available', 'time_outside', 'missing_box', 'broken_link', 'not_evaluated')
ENDPOINTS = {
    'parent_core_sha256': 'c7469179d1628979b30eb256e695fdd6479ffd6e2df047d380d890512f57a5ea',
    'camera_complete_sha256': '0c0d2e640a12e3299e8dacc03189776669cd2e50036a65de823026304119b648',
    'history_inputs_complete_sha256': '040d62242a541b9685f3f39f1534b3663b0e1142b4cd7d9737cdbfd15d94314d',
    'motion_complete_sha256': '3bd429effb1fbe50ac0aeb77330bc0100ee144bf8406fbd078097fb336301a73',
    'raw_manifest_sha256': '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71',
    'raw_complete_sha256': 'e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69',
    'points_file_sha256': '91bcc80882e41ffd5234cecce4b517ef4081839c89dbb6a048c4226b723ca6fd',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_module(name, digest):
    path = HERE / (name + '.py')
    require(sha(path) == digest, 'Frozen source changed: ' + name)
    spec = importlib.util.spec_from_file_location('_pose_analysis_' + name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def dependencies():
    require(sha(RULES_PATH) == RULES_SHA, 'Frozen analysis rules changed')
    rules = read(RULES_PATH)
    require(rules['schema'] == SCHEMA and rules['metrics'] == list(METRICS) and
            rules['auc_features'] == list(AUC_FEATURES) and rules['directions'] == [1, 1, 1] and
            rules['reference_reason_codes'] == {str(i): name for i, name in enumerate(REASONS)} and
            rules['pose_reason_codes'] == {str(i): name for i, name in enumerate(POSE_REASONS)} and
            rules['parent_analyzer_sha256'] == PARENT_ANALYZER_SHA and rules['source_endpoints'] == ENDPOINTS and
            rules['speed_edges_mps'] == [0., .1, .5, 1., 2., 5., 10., None], 'Analysis rules differ')
    parent = frozen_module('analyze_history_camera_evidence_train_v1', PARENT_ANALYZER_SHA)
    parent_rules, radar, helper = parent.dependencies()
    return rules, parent, parent_rules, radar, helper


def authenticate_pose(a, rules, radar, cr, mr, sources):
    for key in ('pose_extractor_sha256', 'pose_core_sha256', 'pose_protocol_sha256'):
        value = rules[key]
        require(len(value) == 64 and all(ch in '0123456789abcdef' for ch in value), 'Unfrozen binding: ' + key)
    protocol = read(a.pose_protocol)
    require(sha(a.pose_protocol) == rules['pose_protocol_sha256'] and protocol['schema'] == POSE_SCHEMA and
            protocol['status'] == 'FROZEN' and protocol['resources'] == dict(max_seconds=600, cpu_threads=2),
            'Pose protocol is not the frozen protocol')
    require(protocol['extractor_sha256'] == rules['pose_extractor_sha256'] ==
            sha(HERE / 'extract_history_pose_address_control_train_v1.py') and
            protocol['core_sha256'] == rules['pose_core_sha256'] == sha(HERE / 'history_pose_address_control_v1.py'),
            'Pose collector/core source differs')
    require({key: protocol[key] for key in ENDPOINTS} == ENDPOINTS and
            protocol['reference_reason_codes'] == rules['reference_reason_codes'] and
            protocol['pose_reason_codes'] == rules['pose_reason_codes'], 'Pose protocol endpoints/reasons differ')
    done, manifest, index, summary = radar.ledger(a.pose_run, POSE_SCHEMA, 'COMPLETE_HISTORY_POSE_ADDRESS_CONTROL_TRAIN')
    require(done['source_sha256'] == manifest['source_sha256'] == rules['pose_extractor_sha256'] and
            done['core_sha256'] == manifest['core_sha256'] == rules['pose_core_sha256'] and
            done['protocol_sha256'] == manifest['protocol_sha256'] == rules['pose_protocol_sha256'] and
            {key: manifest[key] for key in ENDPOINTS} == ENDPOINTS and
            manifest['scope'] == protocol['scope'], 'Pose ledger/source/protocol differs')
    require(manifest['camera_files_sha256'] == sources['camera_files_sha256'] and
            manifest['motion_files_sha256'] == sources['motion_files_sha256'] and
            manifest['history_inputs_files_sha256'] == sources['history_inputs_files_sha256'], 'Pose parent input ledgers differ')
    require(manifest['schema'] == index['schema'] == summary['schema'] == POSE_SCHEMA and
            summary['status'] == done['status'] and summary['samples'] == 512 and summary['scenes'] == 256 and
            summary['source_points'] == index['source_points'] == 1740053 and summary['images_decoded'] == 6144 and
            0 < summary['seconds'] < 600 and summary['optimizer_updates'] == 0 and
            all(summary[key] is True for key in ('raw_GT_used', 'raw_container_includes_future_frames',
                'original_cached_arrays_unchanged', 'all_saved_arrays_roundtrip_exact')) and
            all(summary[key] is False for key in ('future_slots_passed_to_pose_core', 'future_GT_used_in_reference',
                'fitting', 'threshold_selection', 'model_forward', 'is_deployable')), 'Pose full extraction contract differs')
    audit = summary['first_anchor_original_correlation_check']
    require(audit['anchor_ordinal'] == 0 and audit['compared_correlations'] > 0 and audit['maximum_absolute_error'] == 0 and
            audit['first_eight_valid_points_per_camera'] is True and audit['all512_images_recomputed_for_new_reference'] is True,
            'Original correlation equivalence check differs')
    rows = index['records']
    require(len(rows) == len(cr) == len(mr) == 512 and
            {p.relative_to(a.pose_run).as_posix() for p in (a.pose_run / 'samples').glob('*.npz')} == {r['file'] for r in rows},
            'Pose source file set differs')
    for ordinal, (row, camera, motion) in enumerate(zip(rows, cr, mr)):
        require(row['ordinal'] == camera['ordinal'] == motion['ordinal'] == ordinal and
                row['identity'] == camera['identity'] == motion['identity'] and
                row['file'] == camera['file'] == motion['file'] and
                row['source_points'] == camera['source_points'] == motion['source_points'] and
                row['camera_npz_sha256'] == camera['sha256'] and row['motion_npz_sha256'] == motion['sha256'],
                'Pose/camera/motion row binding differs')
        require(len(row['raw_label_sha256']) == 64, 'Missing raw history label provenance')
    for field, nreason, multiple in (('reference_reason_counts', 6, 1), ('pose_reason_counts', 5, 2)):
        totals = {str(i): sum(row[field].get(str(i), 0) for row in rows) for i in range(nreason)}
        require(set(index[field]).issubset(totals) and set(summary[field]).issubset(totals) and
                totals == {str(i): index[field].get(str(i), 0) for i in range(nreason)} ==
                {str(i): summary[field].get(str(i), 0) for i in range(nreason)} and
                sum(totals.values()) == multiple * 1740053, 'Pose reason totals differ: ' + field)
    sources.update(parent_camera_analysis_source_sha256=PARENT_ANALYZER_SHA,
                   pose_complete_sha256=sha(a.pose_run / 'complete.json'), pose_files_sha256=done['files_sha256'],
                   pose_extractor_sha256=rules['pose_extractor_sha256'], pose_core_sha256=rules['pose_core_sha256'],
                   pose_protocol_sha256=rules['pose_protocol_sha256'])
    return rows


def same_bytes(first, second):
    return first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()


def pose_arrays(v, row, old, motion, t0_us):
    """Validate storage and exact score algebra without reopening raw labels/images."""
    n = len(motion['source_flat_indices'])
    shapes = dict(camera_index=(n,), original_reason_code=(n,), reference_reason_code=(n,), reference_valid=(n,),
                  pose_reason_code=(n, 2), reference_uv=(n, 2, 2), reference_depth_m=(n, 2),
                  reference_bracket_indices=(n, 2, 2), reference_alpha=(n, 2), reference_displacement_R_m=(n, 2, 3),
                  reference_patch_std=(n, 2), reference_pixel_departure_from_D=(n, 2),
                  reference_pixel_departure_from_zero=(n, 2))
    score_keys = ('reference_correlation', 'reference_broken_correlation', 'reference_residual',
                  'reference_broken_residual', 'delta_eD_minus_eGT', 'delta_eZero_minus_eGT',
                  'broken_delta_eD_minus_eGT', 'broken_delta_eZero_minus_eGT', 'reference_broken_past_patch_std')
    shapes.update({key: (n,) for key in score_keys})
    require(set(v) == set(row['arrays']) == set(shapes) | {'identity_json', 'source_flat_indices'}, 'Pose NPZ field set differs')
    require(same_bytes(v['source_flat_indices'], motion['source_flat_indices']) and
            same_bytes(v['source_flat_indices'], old['source_flat_indices']) and
            same_bytes(v['identity_json'], old['identity_json']), 'Pose original source indices/identity changed')
    for key, value in v.items():
        require(row['arrays'][key] == dict(shape=list(value.shape), dtype=str(value.dtype)), 'Pose descriptor differs: ' + key)
    for key, shape in shapes.items():
        dtype = {'camera_index': 'int8', 'original_reason_code': 'uint8', 'reference_reason_code': 'uint8',
                 'reference_valid': 'bool', 'pose_reason_code': 'uint8', 'reference_bracket_indices': 'int8'}.get(key, 'float64')
        require(v[key].shape == shape and str(v[key].dtype) == dtype, 'Pose array shape/dtype: ' + key)
    require(same_bytes(v['camera_index'], old['camera_index']) and same_bytes(v['original_reason_code'], old['reason_code']),
            'Original camera view/reason bytes changed')
    reason, valid, poses = v['reference_reason_code'], v['reference_valid'], v['pose_reason_code']
    require(np.isin(reason, np.arange(6)).all() and np.array_equal(valid, reason == 0) and
            np.array_equal(reason == 1, ~old['valid']) and np.all(~valid | old['valid']) and
            np.isin(poses, np.arange(5)).all() and np.all(poses[reason == 1] == 4) and
            np.all(poses[reason != 1] != 4), 'Pose reference support/reasons differ')
    require(np.array_equal(reason == 2, old['valid'] & (poses[:, 0] != 0)) and
            np.array_equal(reason == 3, old['valid'] & (poses[:, 0] == 0) & (poses[:, 1] != 0)) and
            np.all(poses[np.isin(reason, [0, 4, 5])] == 0), 'Pose reason priority differs')
    for field, array, count in (('reference_reason_counts', reason, 6), ('pose_reason_counts', poses, 5)):
        require(set(row[field]).issubset({str(i) for i in range(count)}) and
                {str(i): row[field].get(str(i), 0) for i in range(count)} ==
                {str(i): int((array == i).sum()) for i in range(count)}, 'Per-sample pose reason counts differ')
    bracket, alpha = v['reference_bracket_indices'], v['reference_alpha']
    evaluated = np.isin(poses, [0, 2, 3])
    require(np.all((bracket[evaluated] >= 0) & (bracket[evaluated] <= 2)) and
            np.all(bracket[evaluated][:, 1] == bracket[evaluated][:, 0] + 1) and
            np.isfinite(alpha[evaluated]).all() and np.all((alpha[evaluated] >= 0) & (alpha[evaluated] <= 1)) and
            np.all(bracket[~evaluated] == -1) and np.isnan(alpha[~evaluated]).all(), 'Historical bracket slots/alpha differ')
    for key in score_keys:
        require(np.isfinite(v[key][valid]).all() and np.isnan(v[key][~valid]).all(), 'Missing score was imputed: ' + key)
    require(np.isfinite(v['reference_uv'][valid]).all() and np.isfinite(v['reference_displacement_R_m'][valid]).all() and
            np.isfinite(v['reference_depth_m'][valid]).all() and np.all(v['reference_depth_m'][valid] > 0) and
            np.isfinite(v['reference_patch_std'][valid]).all() and np.all(v['reference_patch_std'][valid] >= 0) and
            np.all(v['reference_broken_past_patch_std'][valid] >= 0), 'Invalid reference geometry/texture')
    for prefix in ('', 'broken_'):
        correlation = v['reference_' + prefix + 'correlation'][valid]
        require(np.all(np.abs(correlation) <= 1. + 1e-12) and
                np.array_equal(v['reference_' + prefix + 'residual'][valid], 1. - correlation), 'Reference residual algebra differs')
        for name, hi in (('D', 1), ('Zero', 0)):
            require(np.array_equal(v[prefix + 'delta_e' + name + '_minus_eGT'][valid],
                                   correlation - old[prefix + 'correlation'][valid, hi]), 'Reference delta algebra/sign differs')
    for suffix, hi in (('zero', 0), ('D', 1)):
        value = v['reference_pixel_departure_from_' + suffix]
        require(np.array_equal(value[valid], np.linalg.norm(v['reference_uv'][valid] - old['uv'][valid, hi], axis=-1)) and
                np.isnan(value[~valid]).all(), 'Reference pixel departure differs')
    metrics = np.stack([v[key] for key in METRICS[:4]] +
                       [v[METRICS[0]] - v[METRICS[2]], v[METRICS[1]] - v[METRICS[3]]], axis=1)
    h = np.asarray(HORIZONS)
    velocity = np.sum(h[:, None, None] * motion['D_displacement_m'].astype(np.float64), axis=0) / np.sum(h*h)
    past_benefit = np.full(n, np.nan)
    dt = (old['timestamps_us'][valid, 1] - int(t0_us)).astype(np.float64) / 1e6
    require(np.all(dt < 0), 'Past camera timestamp is not historical')
    gt = v['reference_displacement_R_m'][valid, 1, :2]
    magnitude = np.linalg.norm(gt, axis=1)
    error = np.linalg.norm(gt - dt[:, None] * velocity[valid, :2], axis=1)
    past_benefit[valid] = magnitude - error
    require(np.isfinite(past_benefit[valid]).all(), 'Actual past benefit nonfinite')
    descriptors = np.full((n, 4), np.nan)
    descriptors[valid] = np.stack([magnitude, error, v['reference_pixel_departure_from_zero'][valid, 1],
                                  v['reference_pixel_departure_from_D'][valid, 1]], axis=1)
    return metrics, past_benefit, descriptors


def collect(a, parent, radar, rr, mr, nr, cr, pr, scenes, prior, core, times):
    arrays, counts, coverage = parent.collect(a, radar, rr, mr, nr, cr, scenes, prior, core, times)
    offsets = [0] * 4
    histories = read(a.history_inputs / 'records.json')['records']
    for pi, (row, camera_row, motion_row) in enumerate(zip(pr, cr, mr)):
        values = radar.load_npz(a.pose_run, row)
        camera = radar.load_npz(a.camera_run, camera_row)
        motion = radar.load_npz(a.motion_run, motion_row)
        metrics, past_benefit, descriptors = pose_arrays(values, row, camera, motion, histories[pi]['t0_lidar_us'])
        for hi, e in enumerate(arrays):
            if 'pose_metrics' not in e:
                count = len(e['benefit'])
                e.update(pose_metrics=np.full((count, 6), np.nan), reference_reason=np.empty(count, dtype=np.uint8),
                         actual_past_benefit_xy_m=np.full(count, np.nan), reference_correlations=np.full((count, 2), np.nan),
                         reference_descriptors=np.full((count, 4), np.nan))
            take = motion['valid'][hi] & (motion['owner'] < 0)
            count = int(take.sum()); sl = slice(offsets[hi], offsets[hi] + count)
            require(same_bytes(e['reason'][sl], values['original_reason_code'][take]) and
                    np.all(e['scene'][sl] == scenes.index(row['identity']['scene_token'])), 'Pose parent ordering/mask differs')
            e['pose_metrics'][sl] = metrics[take]
            e['reference_reason'][sl] = values['reference_reason_code'][take]
            e['actual_past_benefit_xy_m'][sl] = past_benefit[take]
            e['reference_descriptors'][sl] = descriptors[take]
            e['reference_correlations'][sl] = np.stack([values['reference_correlation'][take], values['reference_broken_correlation'][take]], axis=1)
            offsets[hi] += count
    require(offsets == [len(e['benefit']) for e in arrays], 'Pose join incomplete')
    return arrays, counts, coverage


def correlation_summary(value, weight, original_objects):
    """Dimensionless correlation differences; never call these physical EPE."""
    require(value.shape == weight.shape and np.isfinite(value).all() and np.isfinite(weight).all() and
            np.all(weight > 0), 'Invalid dimensionless metric support')
    mass = float(weight.sum()); signed = float(np.dot(value, weight))
    fractions = {name: float(weight[mask].sum()) / mass if mass else None
                 for name, mask in (('positive', value > 0), ('zero', value == 0), ('negative', value < 0))}
    require(not mass or math.isclose(sum(fractions.values()), 1., abs_tol=1e-12), 'Metric sign shares differ')
    return dict(units='dimensionless_correlation_difference', points=len(value), weight_sum=mass,
                original_object_count=original_objects, conditional_original_weight_mean=signed/mass if mass else None,
                sign_weight_fraction=fractions, FULL_original_object_denominator_contribution=signed/original_objects if original_objects else None)


def common_summary(helper, e, mask, original_objects):
    require(np.all(e['reason'][mask] == 0) and np.all(e['reference_reason'][mask] == 0), 'Reference population is not common valid')
    weight = e['weight'][mask]
    require(np.isfinite(e['pose_metrics'][mask]).all() and np.isfinite(e['reference_correlations'][mask]).all() and
            np.isfinite(e['reference_descriptors'][mask]).all() and
            np.isfinite(e['camera_features'][mask, :3]).all(), 'Nonfinite common population')
    metrics = {name: correlation_summary(e['pose_metrics'][mask, mi], weight, original_objects) for mi, name in enumerate(METRICS)}
    auc = {name: helper.weighted_auc(e['camera_features'][mask, fi], e['actual_past_benefit_xy_m'][mask], weight)
           for fi, name in enumerate(AUC_FEATURES)}
    mass = float(weight.sum())
    correlations = {name: float(np.dot(weight, e['reference_correlations'][mask, ci]))/mass if mass else None
                    for ci, name in enumerate(('GT_reference', 'broken_GT_reference'))}
    descriptors = {name: float(np.dot(weight, e['reference_descriptors'][mask, ci]))/mass if mass else None
                   for ci, name in enumerate(DESCRIPTORS)}
    return dict(points=int(mask.sum()), nonempty_objects=int(len(np.unique(e['object'][mask]))), weight_sum=mass,
                original_object_count=original_objects, dimensionless_metrics=metrics,
                reference_correlation_conditional_original_weight_mean=correlations,
                descriptive_common_population_conditional_original_weight_mean=descriptors,
                secondary_actual_past_benefit_auc=auc,
                secondary_actual_past_zero_benefit_points=int((e['actual_past_benefit_xy_m'][mask] == 0).sum()))


def summarize(helper, radar, arrays, counts, prior):
    output = []
    for hi, horizon in enumerate(HORIZONS):
        e = arrays[hi]; bins = np.searchsorted(EDGES, e['camera_features'][:, 2], side='right') - 1
        require(np.isin(bins, np.arange(7)).all(), 'Original speed bins changed')
        masks = radar.strata_masks(e)
        item = dict(horizon_seconds=horizon, strata={})
        for group in GROUPS:
            gm = radar.group_mask(e, group)
            denom = int(counts[hi].sum() if group == 'all' else counts[hi, GROUPS.index(group)-1])
            previous = next(row for row in prior['physical'] if row['horizon_seconds'] == horizon and row['group'] == group)
            now = helper.continuous(e, gm, denom)
            require(denom == previous['original_objects'] and now['points'] == previous['uncovered_points'], 'Original full support differs')
            for key, value in now.items():
                radar.close(value, previous['uncovered_D_versus_zero_benefit_xy'][key], 'Parent physical ' + key)
        for stratum in STRATA:
            groups = {}; item['strata'][stratum] = groups
            for group in GROUPS:
                denom = int(counts[hi].sum() if group == 'all' else counts[hi, GROUPS.index(group)-1])
                base = masks[stratum] & radar.group_mask(e, group)
                records = []
                selected_bins = [-1] + (list(range(7)) if stratum == 'no_radar' and group == 'all' else [])
                for bi in selected_bins:
                    selected = base if bi == -1 else base & (bins == bi)
                    support = helper.continuous(e, selected, denom)
                    by_reason = {reason: helper.continuous(e, selected & (e['reference_reason'] == ri), denom)
                                 for ri, reason in enumerate(REASONS)}
                    require(sum(row['points'] for row in by_reason.values()) == support['points'], 'Reason point support lost')
                    for key in ('weight_sum', 'FULL_denominator_benefit_contribution_xy_m',
                                'FULL_denominator_positive_gain_xy_m', 'FULL_denominator_negative_cost_xy_m'):
                        if support[key] is not None:
                            radar.close(sum(row[key] for row in by_reason.values()), support[key], 'Six-reason physical partition ' + key)
                    common = selected & (e['reference_reason'] == 0)
                    records.append(dict(bin_index=None if bi == -1 else bi, all_speeds=bi == -1,
                                        original_future_physical_support=support, original_future_physical_by_reference_reason=by_reason,
                                        common_reference_valid=common_summary(helper, e, common, denom)))
                groups[group] = dict(original_object_count=denom, bins=records)
        output.append(item)
    return output


def synthetic_array_qa():
    """Exercise the actual NPZ validator using a fully synthetic eight-point carrier."""
    n = 8
    reason = np.array([0, 0, 0, 1, 2, 3, 4, 5], dtype=np.uint8)
    original_reason = np.array([0, 0, 0, 4, 0, 0, 0, 0], dtype=np.uint8)
    valid = reason == 0
    identity = np.asarray(json.dumps({'sample_token': 'synthetic_only'}))
    idx = np.arange(n, dtype=np.int64)
    old = dict(identity_json=identity, source_flat_indices=idx, camera_index=np.zeros(n, dtype=np.int8),
               reason_code=original_reason, valid=original_reason == 0, uv=np.zeros((n, 2, 2, 2)),
               correlation=np.tile([.1, .2], (n, 1)), broken_correlation=np.tile([.3, .4], (n, 1)),
               timestamps_us=np.tile(np.array([1000000, 0], dtype=np.int64), (n, 1)))
    velocity = np.zeros((n, 3)); velocity[0, 0] = 1.; velocity[1, 0] = -1.
    motion = dict(source_flat_indices=idx,
                  D_displacement_m=(np.asarray(HORIZONS)[:, None, None] * velocity[None, :, :]).astype(np.float32))
    poses = np.zeros((n, 2), dtype=np.uint8)
    poses[3] = 4; poses[4, 0] = 1; poses[5, 1] = 2
    evaluated = np.isin(poses, [0, 2, 3])
    bracket = np.full((n, 2, 2), -1, dtype=np.int8); bracket[evaluated] = [0, 1]
    alpha = np.full((n, 2), np.nan); alpha[evaluated] = .5
    v = dict(identity_json=identity.copy(), source_flat_indices=idx.copy(), camera_index=old['camera_index'].copy(),
             original_reason_code=original_reason.copy(), reference_reason_code=reason, reference_valid=valid,
             pose_reason_code=poses, reference_bracket_indices=bracket, reference_alpha=alpha,
             reference_uv=np.full((n, 2, 2), np.nan), reference_depth_m=np.full((n, 2), np.nan),
             reference_displacement_R_m=np.full((n, 2, 3), np.nan), reference_patch_std=np.full((n, 2), np.nan),
             reference_pixel_departure_from_D=np.full((n, 2), np.nan),
             reference_pixel_departure_from_zero=np.full((n, 2), np.nan),
             reference_broken_past_patch_std=np.full(n, np.nan))
    v['reference_uv'][valid] = [[[1., 1.], [2., 1.]]]*3
    v['reference_depth_m'][valid] = 1.; v['reference_patch_std'][valid] = .2
    v['reference_broken_past_patch_std'][valid] = .2
    v['reference_displacement_R_m'][valid] = 0.
    v['reference_displacement_R_m'][:2, 1, 0] = -1.
    for prefix in ('', 'broken_'):
        correlation = np.full(n, np.nan)
        correlation[valid] = [.8, .2, .5] if not prefix else [.1, .3, .5]
        v['reference_' + prefix + 'correlation'] = correlation
        v['reference_' + prefix + 'residual'] = 1. - correlation
        for name, hi in (('D', 1), ('Zero', 0)):
            delta = np.full(n, np.nan)
            delta[valid] = correlation[valid] - old[prefix + 'correlation'][valid, hi]
            v[prefix + 'delta_e' + name + '_minus_eGT'] = delta
    for suffix, hi in (('zero', 0), ('D', 1)):
        v['reference_pixel_departure_from_' + suffix][valid] = np.linalg.norm(v['reference_uv'][valid] - old['uv'][valid, hi], axis=-1)
    row = dict(arrays={k: dict(shape=list(x.shape), dtype=str(x.dtype)) for k, x in v.items()},
               reference_reason_counts={str(i): int((reason == i).sum()) for i in range(6)},
               pose_reason_counts={str(i): int((poses == i).sum()) for i in range(5)})
    metrics, benefit, descriptors = pose_arrays(v, row, old, motion, 1000000)
    require(np.array_equal(benefit[valid], [1., -1., 0.]) and np.isnan(benefit[~valid]).all() and
            np.array_equal(descriptors[valid, 0], [1., 1., 0.]) and
            np.array_equal(descriptors[valid, 1], [0., 2., 0.]) and
            np.array_equal(metrics[valid, 4], metrics[valid, 0] - metrics[valid, 2]), 'QA actual NPZ validator and camera-time formula')
    for name, location, value in (
            ('camera_index', 0, 1), ('original_reason_code', 3, 0), ('reference_alpha', (0, 0), 1.01),
            ('reference_bracket_indices', (0, 0, 1), 3), ('delta_eD_minus_eGT', 0, -.6),
            ('broken_delta_eZero_minus_eGT', 3, 0.)):
        bad = {k: x.copy() for k, x in v.items()}; bad[name][location] = value
        try:
            pose_arrays(bad, row, old, motion, 1000000)
        except ValueError:
            pass
        else:
            raise ValueError('QA malformed NPZ accepted: ' + name)


def analytic_qa(helper, radar):
    synthetic_array_qa()
    # All six reason partitions retain physical support and the denominator of 8
    # original objects, including objects not represented in this selected subset.
    reason = np.array([0, 0, 0, 1, 2, 3, 4, 5], dtype=np.uint8)
    weight = np.array([.25, .75, .5, .25, .25, .5, .5, 1.])
    metrics = np.full((8, 6), np.nan)
    metrics[:3, :4] = [[2., 1., -1., -.5], [-1., -2., .5, 1.], [0., 0., 0., 0.]]
    metrics[:3, 4:] = metrics[:3, :2] - metrics[:3, 2:4]
    e = dict(benefit=np.array([2., -3., 0., -1., 4., -2., 0., -6.]), weight=weight,
             object=np.array([0, 0, 1, 2, 2, 3, 4, 5]), reason=np.array([0, 0, 0, 4, 0, 0, 0, 0], dtype=np.uint8),
             reference_reason=reason, pose_metrics=metrics,
             camera_features=np.array([[2., -2., 2.], [1., -1., 1.], [0., 0., 0.]] + [[np.nan]*3]*5),
             actual_past_benefit_xy_m=np.array([1., -1., 0.] + [np.nan]*5),
             reference_correlations=np.array([[.8, .1], [.2, .3], [.5, .5]] + [[np.nan]*2]*5),
             reference_descriptors=np.array([[1., 0., 2., 1.], [1., 2., 2., 1.], [0., 0., 0., 0.]] + [[np.nan]*4]*5))
    common = reason == 0; summary = common_summary(helper, e, common, 8)
    d = summary['dimensionless_metrics'][METRICS[0]]
    require(summary['points'] == 3 and summary['weight_sum'] == 1.5 and
            math.isclose(d['conditional_original_weight_mean'], -1./6.) and
            d['FULL_original_object_denominator_contribution'] == -.25/8. and
            d['sign_weight_fraction'] == dict(positive=1./6., zero=1./3., negative=.5), 'QA original weight and denominator')
    require(summary['dimensionless_metrics'][METRICS[4]]['FULL_original_object_denominator_contribution'] == -.375/8.,
            'QA paired actual-minus-broken is computed per point')
    whole = helper.continuous(e, np.ones(8, dtype=bool), 8)
    parts = [helper.continuous(e, reason == ri, 8) for ri in range(6)]
    require(sum(p['points'] for p in parts) == 8 and all(p['original_object_count'] == 8 for p in parts), 'QA six reasons keep all support')
    for key in ('FULL_denominator_positive_gain_xy_m', 'FULL_denominator_negative_cost_xy_m', 'FULL_denominator_benefit_contribution_xy_m'):
        require(math.isclose(sum(p[key] for p in parts), whole[key], abs_tol=1e-12), 'QA missing reference physical cost/gain retained')
    auc = summary['secondary_actual_past_benefit_auc']
    require(auc['score_true']['auc'] == auc['lsq_speed_xy_mps']['auc'] == 1. and auc['score_broken']['auc'] == 0. and
            summary['secondary_actual_past_zero_benefit_points'] == 1, 'QA fixed AUC direction/zero class')
    # A signed benefit can be negative, zero or positive; a nonnegative GT-D
    # error is never passed as its label. A perfect backward D is positive.
    gt = np.array([[-1., 0.], [-1., 0.], [0., 0.]])
    velocity = np.array([[1., 0.], [-1., 0.], [0., 0.]])
    benefit = np.linalg.norm(gt, axis=1) - np.linalg.norm(gt - (-1.)*velocity, axis=1)
    require(np.array_equal(benefit, [1., -1., 0.]), 'QA signed actual-time backward displacement label')
    require(helper.weighted_auc(np.array([2., 2.]), np.array([1., -1.]), np.array([.25, .75]))['auc'] == .5,
            'QA ties retain half credit')
    for label in (np.ones(2), -np.ones(2), np.zeros(2), np.empty(0)):
        require(helper.weighted_auc(np.ones(len(label)), label, np.ones(len(label)))['auc'] is None, 'QA absent AUC class is null')
    empty = common_summary(helper, e, np.zeros(8, dtype=bool), 8)
    require(empty['dimensionless_metrics'][METRICS[0]]['conditional_original_weight_mean'] is None and
            empty['dimensionless_metrics'][METRICS[0]]['FULL_original_object_denominator_contribution'] == 0 and
            all(v is None for v in empty['dimensionless_metrics'][METRICS[0]]['sign_weight_fraction'].values()), 'QA empty subset semantics')
    try:
        common_summary(helper, e, np.ones(8, dtype=bool), 8)
    except ValueError:
        pass
    else:
        raise ValueError('QA non-common support accepted')
    e['camera_features'][3:, 2] = [.1, .5, 5., 10., 11.]
    e['present'] = np.array([False, False, False, True, False, True, False, False])
    e['clip'] = np.zeros(8, dtype=bool)
    e['group'] = np.array([0, 0, 1, 0, 1, 1, 2, 2], dtype=np.int8)
    counts = np.tile(np.array([3, 2, 3]), (4, 1))
    previous = []
    for horizon in HORIZONS:
        for group in GROUPS:
            denom = 8 if group == 'all' else int(counts[0, GROUPS.index(group)-1])
            physical = helper.continuous(e, radar.group_mask(e, group), denom)
            previous.append(dict(horizon_seconds=horizon, group=group, original_objects=denom,
                                 uncovered_points=physical['points'], uncovered_D_versus_zero_benefit_xy=physical))
    integrated = summarize(helper, radar, [e]*4, counts, dict(physical=previous))
    for horizon in integrated:
        require(set(horizon['strata']) == set(STRATA), 'QA exact three-stratum scope')
        for stratum in STRATA:
            for group in GROUPS:
                records = horizon['strata'][stratum][group]['bins']
                expected = 8 if stratum == 'no_radar' and group == 'all' else 1
                require(len(records) == expected, 'QA speed-bin scope expanded')
                require(all(len(record['original_future_physical_by_reference_reason']) == 6 for record in records),
                        'QA every cell retains all six reasons')
    return dict(status='PASS_SYNTHETIC_ANALYTIC_QA_ONLY', tests=[
        'synthetic NPZ schema and exact score algebra; view/mask, alpha, future-slot, sign and zero-imputation corruptions rejected',
        'same reference-valid population for all six dimensionless metrics and three AUC scores',
        'original point weights and complete original object denominator, including absent objects',
        'six reference reasons partition original future physical gain/cost without loss',
        'per-point actual-minus-broken and positive/zero/negative weight shares',
        'signed benefit at actual past-camera time; never GT-D error as a positive-only label',
        'fixed AUC directions, zero labels, tie credit and absent-class null',
        'empty subset null conditional mean and zero full-denominator contribution',
        'non-common population rejected',
        'end-to-end synthetic summary: three radar strata and four groups, seven individual bins only for no_radar/all'],
        real_data_opened=False, model_forward=False, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths = ('pose-run', 'pose-protocol', 'camera-run', 'camera-protocol', 'history-inputs',
             'radar-run', 'motion-run', 'motion-analysis', 'train-index')
    for key in paths:
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qa-only', action='store_true')
    a = parser.parse_args()
    rules, parent, parent_rules, radar, helper = dependencies()
    if a.qa_only:
        result = analytic_qa(helper, radar)
    else:
        require(all(getattr(a, key.replace('-', '_')) is not None for key in paths), 'All nine input paths required')
        # Reject an unfrozen pose protocol before any parent real-data arrays open.
        for key in ('pose_extractor_sha256', 'pose_core_sha256', 'pose_protocol_sha256'):
            require(len(rules[key]) == 64 and all(ch in '0123456789abcdef' for ch in rules[key]), 'Unfrozen binding: ' + key)
        rr, mr, nr, cr, scenes, prior, sources, core, times = parent.authenticate(a, parent_rules, radar)
        pr = authenticate_pose(a, rules, radar, cr, mr, sources)
        sources.update(analysis_source_sha256=sha(__file__), rules_sha256=sha(RULES_PATH))
        arrays, counts, coverage = collect(a, parent, radar, rr, mr, nr, cr, pr, scenes, prior, core, times)
        result = dict(schema=SCHEMA, status='COMPLETE_CPU_HISTORY_POSE_ADDRESS_CONTROL_TRAIN_ANALYSIS',
                      sources=sources, rules=rules, samples=512, scenes=256, source_points=1740053, object_horizon_rows=43189,
                      scene_tokens=scenes, sampling_coverage=coverage, horizons=summarize(helper, radar, arrays, counts, prior),
                      unchanged_full_physical_reference=dict(source_sha256=radar.MOTION_ANALYSIS_SHA,
                          physical=prior['physical'], recomputed_by_this_analysis=False),
                      verification=dict(original_support_join=True, all_sample_npz_SHA_checked=True,
                          original_camera_view_and_reason_bytes_unchanged=True, exact_reference_score_algebra_checked=True,
                          original_future_denominators_complete=True, all_reference_reason_physical_support_retained=True,
                          alpha_and_history_slot_bounds_checked=True, raw_images_reopened=False, raw_GT_reopened=False,
                          fitting=False, threshold_selection=False, bootstrap=False, model_forward=False, optimizer_updates=0))
    require('torch' not in sys.modules, 'CPU NumPy-only analysis')
    result.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  analysis_source_sha256=sha(__file__), rules_sha256=sha(RULES_PATH))
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(out=str(a.out), sha256=sha(a.out), status=result['status'])))


if __name__ == '__main__':
    main()
