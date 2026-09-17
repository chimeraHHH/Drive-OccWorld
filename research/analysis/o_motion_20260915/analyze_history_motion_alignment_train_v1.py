"""Fixed-support historical label alignment diagnostic, CPU/NumPy only.

Historical GT is scoring-only, never an inference feature, view selector, or
mathematical upper bound. No camera-time interpolation or image/model rerun.
Use --qa-only for synthetic analytic checks; real analysis requires all inputs.
"""
import argparse
import datetime
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = 'history-motion-alignment-train-analysis-v1'
RULES_FILE = HERE / 'history_motion_alignment_train_analysis_rules_v1.json'
RULES_SHA = '15f0f65522fffb2d6b042c7f237e5a6512a68792b05b822809c47d9e1c8ada39'
HORIZONS = (.5, 1., 1.5, 2.)
EDGES = np.asarray([0., .1, .5, 1., 2., 5., 10., np.inf])
HISTORY_REASONS = ('available_previous_box_and_link', 'missing_previous_box',
                   'incomplete_previous_annotation_link')
STRATA = ('all', 'no_radar', 'radar_present')
GROUPS = ('all', 'stationary', 'ambiguous', 'moving')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name, expected):
    path = HERE / (name + '.py')
    require(sha(path) == expected, 'Frozen dependency differs: ' + name)
    spec = importlib.util.spec_from_file_location('_alignment_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    require(sha(RULES_FILE) == RULES_SHA, 'Prospective rules changed')
    rules = read(RULES_FILE)
    require(rules['schema'] == SCHEMA and rules['horizons_seconds'] == list(HORIZONS) and
            rules['history_reason_codes'] == {str(i): v for i, v in enumerate(HISTORY_REASONS)} and
            rules['auc_tasks']['all_directions'] == 1 and
            rules['populations']['radar_strata'] == list(STRATA) and
            rules['populations']['GT_motion_groups'] == list(GROUPS) and
            rules['resources'] == dict(CPU_only=True, max_seconds=300), 'Fixed definitions differ')
    source = rules['sources']
    require(sha(HERE / 'history_camera_evidence_train_analysis_rules_v1.json') ==
            source['history_camera_rules_sha256'], 'Parent camera analysis rules changed')
    history = load_module('analyze_history_camera_evidence_train_v1', source['history_camera_analyzer_sha256'])
    camera_rules, radar, helper = history.dependencies()
    geometry = load_module('motion_geometry', source['geometry_sha256'])
    require(sha(HERE / 'build_sparse_motion_supervision_v1.py') == source['sparse_builder_sha256'],
            'Original target builder changed')
    return rules, history, camera_rules, radar, helper, geometry


def past_link_reason(track):
    """Missing previous box/link is missing evidence, with no -2 fallback."""
    require(track['valid_mask'][2] and track['annotation_tokens'][2] is not None,
            'Original source object absent at t0')
    if not track['valid_mask'][1]:
        return 1
    previous, current = track['annotation_tokens'][1:3]
    if (previous is None or current is None or
            track['annotation_next_tokens'][1] != current or
            track['annotation_prev_tokens'][2] != previous):
        return 2
    return 0


def past_benefit(points, g0, a0, a_previous, velocity, signed_dt, geometry):
    """Historical material displacement and extrapolated D, all in fixed R."""
    require(np.isfinite(signed_dt) and signed_dt < 0., 'Previous sample dt must be negative')
    historical = geometry.rigid_displacement(points, g0, a0, a_previous)
    predicted = signed_dt * np.asarray(velocity, dtype=np.float64)
    require(predicted.shape == historical.shape and np.isfinite(predicted).all(), 'Historical D layout')
    benefit = np.linalg.norm(historical[:, :2], axis=1) - np.linalg.norm(
        predicted[:, :2] - historical[:, :2], axis=1)
    return benefit, historical, predicted


def history_labels(raw, motion, points, geometry):
    """Read only current/previous track slots when defining historical scores."""
    frames = raw['frames']
    require([f['relative_frame_index'] for f in frames] == list(range(-2, 5)), 'Seven-frame contract')
    signed_dt = (frames[1]['timestamp_us'] - frames[2]['timestamp_us']) / 1e6
    require(signed_dt < 0. and signed_dt == frames[1]['dt_seconds'], 'Actual previous sample interval differs')
    g0 = np.asarray(frames[2]['lidar_to_global_column_matrix'], dtype=np.float64)
    tracks = {t['instance_token']: t for t in raw['tracks']}
    require(len(tracks) == len(raw['tracks']), 'Duplicate raw instance')
    obj = motion['object_index']; idx = motion['source_flat_indices']
    d = motion['D_displacement_m'].astype(np.float64); h = np.asarray(HORIZONS)
    velocity = np.einsum('h,hnc->nc', h, d) / np.dot(h, h)
    reason = np.full(len(idx), 1, dtype=np.uint8)
    benefit = np.full(len(idx), np.nan)
    object_reasons = []
    for k, token in enumerate(motion['instance_tokens'].tolist()):
        require(token in tracks, 'Original sparse instance missing from raw union')
        track = tracks[token]; status = past_link_reason(track)
        selected = np.flatnonzero(obj == k); require(len(selected) > 0, 'Empty original object')
        reason[selected] = status; object_reasons.append(status)
        if status:
            continue
        a0 = geometry.box_to_global(track['global_centers_m'][2], track['global_rotations_wxyz'][2])
        ap = geometry.box_to_global(track['global_centers_m'][1], track['global_rotations_wxyz'][1])
        b, _, _ = past_benefit(points[idx[selected]], g0, a0, ap, velocity[selected], signed_dt, geometry)
        benefit[selected] = b
    require(np.isfinite(benefit[reason == 0]).all() and np.isnan(benefit[reason != 0]).all(), 'History missing imputation')
    return reason, benefit, signed_dt, np.asarray(object_reasons, dtype=np.uint8)


def forward_2s_gate(raw, motion, points, geometry):
    """Separate geometry validation only; future labels cannot define Bpast."""
    tracks = {t['instance_token']: t for t in raw['tracks']}
    g0 = np.asarray(raw['frames'][2]['lidar_to_global_column_matrix'], dtype=np.float64)
    checked = exact_points = 0; max_error = 0.
    for k, token in enumerate(motion['instance_tokens'].tolist()):
        track = tracks[token]
        require(bool(track['valid_mask'][6]) == bool(motion['object_future_valid'][3, k]), 'Original 2s validity differs')
        if not track['valid_mask'][6]:
            continue
        selected = np.flatnonzero(motion['object_index'] == k)
        require(np.all(motion['valid'][3, selected]), 'Partial future validity within object')
        a0 = geometry.box_to_global(track['global_centers_m'][2], track['global_rotations_wxyz'][2])
        a2 = geometry.box_to_global(track['global_centers_m'][6], track['global_rotations_wxyz'][6])
        expected = geometry.rigid_displacement(points[motion['source_flat_indices'][selected]], g0, a0, a2)
        saved = motion['target_displacement_m'][3, selected]
        error = np.abs(expected - saved.astype(np.float64))
        # Exactly the previously frozen independent_material_audit tolerance.
        require(np.all(error <= 1e-6 + 2e-7 * np.abs(expected)), 'Original 2s material-point reconstruction failed')
        max_error = max(max_error, float(error.max(initial=0.)))
        checked += len(selected)
        exact_points += int(np.all(expected.astype(np.float32).view(np.uint32) == saved.view(np.uint32), axis=1).sum())
    return dict(checked_points=checked, float32_bitwise_equal_points=exact_points,
                maximum_absolute_error_m=max_error, tolerance_absolute_m=1e-6, tolerance_relative=2e-7)


def raw_contract(a, rules, nr):
    source = rules['sources']; raw = a.raw_targets; sparse = a.sparse_labels
    for path, key in ((raw / 'manifest.json', 'raw_manifest_sha256'), (raw / 'complete.json', 'raw_complete_sha256'),
                      (sparse / 'manifest.json', 'sparse_manifest_sha256'), (sparse / 'complete.json', 'sparse_complete_sha256')):
        require(sha(path) == source[key], 'Original raw/sparse contract changed: ' + key)
    rm, rc = read(raw / 'manifest.json'), read(raw / 'complete.json')
    sm, sc = read(sparse / 'manifest.json'), read(sparse / 'complete.json')
    require(rc['status'] == sc['status'] == 'COMPLETE' and rc['manifest_sha256'] == source['raw_manifest_sha256'] and
            sc['manifest_sha256'] == source['sparse_manifest_sha256'] and sc['raw_manifest_sha256'] == source['raw_manifest_sha256'] and
            sc['raw_complete_sha256'] == source['raw_complete_sha256'] and sc['geometry_sha256'] == source['geometry_sha256'] and
            sc['script_sha256'] == source['sparse_builder_sha256'], 'Raw/sparse completion linkage')
    rd = [r for r in rm['records'] if r['identity']['split'] == 'train']
    sd = [r for r in sm['records'] if r['identity']['split'] == 'train']
    expected = [{k: n[k] for k in ('sample_token', 'scene_token', 'official_index', 'split')} for n in nr]
    require(len(rd) == len(sd) == 512 and [r['identity'] for r in rd] == [r['identity'] for r in sd] == expected,
            'Original train-only raw/sparse selection differs')
    require(sha(a.points_file) == source['points_file_sha256'], 'Authenticated original grid differs')
    points = np.load(a.points_file, allow_pickle=False)
    require(points.shape == (640000, 3) and points.dtype == np.float64 and np.isfinite(points).all(), 'Original grid layout')
    return rd, sd, points


def empty_total():
    return dict(points=0, nonempty_objects=0, weight_sum=0., positive_points=0, negative_points=0,
                zero_points=0, weighted_gain_sum_xy_m=0., weighted_cost_sum_xy_m=0.)


def accumulate(total, benefit, weights, obj, mask):
    b, w = benefit[mask], weights[mask]
    total['points'] += int(mask.sum()); total['nonempty_objects'] += len(np.unique(obj[mask]))
    total['weight_sum'] += float(w.sum()); total['positive_points'] += int((b > 0).sum())
    total['negative_points'] += int((b < 0).sum()); total['zero_points'] += int((b == 0).sum())
    total['weighted_gain_sum_xy_m'] += float(np.dot(w, np.maximum(b, 0)))
    total['weighted_cost_sum_xy_m'] += float(np.dot(w, np.maximum(-b, 0)))


def finish_total(total, denominator):
    t = dict(total); gain, cost = t.pop('weighted_gain_sum_xy_m'), t.pop('weighted_cost_sum_xy_m')
    return dict(t, original_object_count=denominator,
                conditional_original_weight_mean_benefit_xy_m=(gain - cost) / t['weight_sum'] if t['weight_sum'] else None,
                FULL_denominator_positive_gain_xy_m=gain / denominator if denominator else None,
                FULL_denominator_negative_cost_xy_m=cost / denominator if denominator else None,
                FULL_denominator_benefit_contribution_xy_m=(gain - cost) / denominator if denominator else None)


def collect_history(a, rules, history, radar, geometry, arrays, counts, mr, nr, cr, times):
    rd, sd, points = raw_contract(a, rules, nr)
    input_rows = read(a.history_inputs / 'records.json')['records']
    offsets = [0] * 4; audits = []
    full = [[[empty_total() for _ in HISTORY_REASONS] for _ in GROUPS] for _ in HORIZONS]
    for ordinal, (rawrow, sparserow, m, c, hr) in enumerate(zip(rd, sd, mr, cr, input_rows)):
        require(rawrow['ordinal'] == sparserow['ordinal'] == ordinal and m['sparse_sha256'] == sparserow['sha256'] and
                sparserow['label_source_sha256'] == rawrow['sha256'], 'Motion-to-raw label source chain')
        expected_file = 'train/' + m['identity']['sample_token'] + '.json.gz'
        require(rawrow['file'] == expected_file, 'Unexpected raw training path')
        blob = (a.raw_targets / expected_file).read_bytes()
        require(len(blob) == rawrow['bytes'] and hashlib.sha256(blob).hexdigest() == rawrow['sha256'], 'Raw compressed label changed')
        decoded = gzip.decompress(blob)
        require(hashlib.sha256(decoded).hexdigest() == rawrow['uncompressed_json_sha256'], 'Raw uncompressed label changed')
        raw = json.loads(decoded); motion = radar.load_npz(a.motion_run, m); camera = radar.load_npz(a.camera_run, c)
        require(raw['schema'] == 'raw-nuscenes-motion-target-v1' and raw['ordinal'] == ordinal and
                raw['identity'] == m['identity'] and raw['selection_sha256'] == radar.EXPECTED['selection'], 'Raw identity/selection')
        f = raw['frames']
        require(f[2]['sample_token'] == m['identity']['sample_token'] and f[1]['sample_token'] == hr['previous_sample_token'] and
                all(x['scene_token'] == m['identity']['scene_token'] for x in f) and
                f[2]['lidar_sample_data_token'] == nr[ordinal]['lidar_token'] and
                f[2]['lidar_timestamp_us'] == hr['t0_lidar_us'] and
                np.allclose(f[2]['lidar_to_global_column_matrix'], hr['lidar_to_global'], atol=1e-9, rtol=0.),
                'History labels and authenticated camera-reference frame differ')
        status, past, dt, object_status = history_labels(raw, motion, points, geometry)
        gate = forward_2s_gate(raw, motion, points, geometry)
        require(len(motion['source_flat_indices']) == sparserow['counts']['source_points'], 'Original source point count')
        obj = motion['object_index']; weights = 1. / np.bincount(obj, minlength=len(object_status))[obj]
        for hi, e in enumerate(arrays):
            if 'past_benefit' not in e:
                e['past_benefit'] = np.full(len(e['benefit']), np.nan)
                e['past_reason'] = np.full(len(e['benefit']), 255, dtype=np.uint8)
            take = motion['valid'][hi] & (motion['owner'] < 0); size = int(take.sum())
            sl = slice(offsets[hi], offsets[hi] + size)
            require(np.array_equal(e['reason'][sl], camera['reason_code'][take]), 'Parent source/camera ordering differs')
            e['past_benefit'][sl] = past[take]; e['past_reason'][sl] = status[take]; offsets[hi] += size
            target = motion['target_displacement_m'][hi].astype(np.float64)
            d = motion['D_displacement_m'][hi].astype(np.float64)
            b = np.linalg.norm(target[:, :2], axis=1) - np.linalg.norm(d[:, :2] - target[:, :2], axis=1)
            for gi, group in enumerate(GROUPS):
                gm = np.ones(len(obj), dtype=bool) if group == 'all' else motion['object_speed_group'][hi, obj] == gi - 1
                for ri in range(3):
                    accumulate(full[hi][gi][ri], b, weights, obj, motion['valid'][hi] & gm & (status == ri))
        sample_time = np.asarray([f[2]['timestamp_us'], f[1]['timestamp_us']])
        delta = (times[ordinal] - sample_time[None, :]) / 1e6
        audits.append(dict(ordinal=ordinal, identity=m['identity'], raw_file=expected_file,
                           raw_sha256=rawrow['sha256'], sparse_sha256=sparserow['sha256'], motion_npz_sha256=m['sha256'],
                           signed_previous_sample_dt_seconds=dt, camera_minus_box_sample_time_seconds=delta.tolist(),
                           lidar_minus_sample_seconds=[x['lidar_sample_timestamp_delta_seconds'] for x in f[1:3]],
                           source_points=len(status), objects=len(object_status),
                           history_point_counts={name: int((status == i).sum()) for i, name in enumerate(HISTORY_REASONS)},
                           history_object_counts={name: int((object_status == i).sum()) for i, name in enumerate(HISTORY_REASONS)},
                           forward_2s_reconstruction=gate))
    require(offsets == [len(e['benefit']) for e in arrays] and sum(a['source_points'] for a in audits) == 1740053,
            'Incomplete full training history join')
    require(sum(a['forward_2s_reconstruction']['checked_points'] for a in audits) == 1608197,
            'Incomplete original 2s geometry check')
    for e in arrays:
        require(np.isin(e['past_reason'], np.arange(3)).all() and
                np.isfinite(e['past_benefit'][e['past_reason'] == 0]).all() and
                np.isnan(e['past_benefit'][e['past_reason'] != 0]).all(), 'History support not fully accounted')
    for hi in range(4):
        require(sum(v['points'] for v in full[hi][0]) == [1709467, 1675202, 1639983, 1608197][hi],
                'History ledger omitted original future points')
        for gi in range(4):
            expected_objects = int(counts[hi].sum() if gi == 0 else counts[hi, gi - 1])
            require(sum(v['nonempty_objects'] for v in full[hi][gi]) == expected_objects,
                    'History ledger omitted original future objects')
            require(np.isclose(sum(v['weight_sum'] for v in full[hi][gi]), expected_objects, rtol=1e-12, atol=1e-12),
                    'Original object denominator was reweighted')
    return audits, [{group: {name: finish_total(full[h][gi][i], int(counts[h].sum() if gi == 0 else counts[h, gi - 1]))
                            for i, name in enumerate(HISTORY_REASONS)}
                     for gi, group in enumerate(GROUPS)} for h in range(4)]


def aucs(helper, e, mask):
    require(np.all((e['past_reason'][mask] == 0) & (e['reason'][mask] == 0)), 'AUC common history/camera support')
    past = e['past_benefit'][mask]; future = e['benefit'][mask]; w = e['weight'][mask]
    fields = {name: e['camera_features'][mask, i] for i, name in enumerate(('score_true', 'score_broken', 'lsq_speed_xy_mps'))}
    label_counts = {name: dict(positive_points=int((value > 0).sum()), negative_points=int((value < 0).sum()),
                              zero_points=int((value == 0).sum()), zero_weight=float(w[value == 0].sum()))
                    for name, value in (('past', past), ('future', future))}
    return dict(common_label_counts=label_counts,
                past_benefit_positive={name: helper.weighted_auc(value, past, w) for name, value in fields.items()},
                future_benefit_positive={name: helper.weighted_auc(value, future, w)
                                        for name, value in dict(Bpast_oracle_reference=past, **fields).items()})


def sign_table(helper, e, mask, denominator):
    rows = {}
    for pi, pn in ((-1, 'negative'), (0, 'zero'), (1, 'positive')):
        rows[pn] = {}
        for fi, fn in ((-1, 'negative'), (0, 'zero'), (1, 'positive')):
            selected = mask & (np.sign(e['past_benefit']) == pi) & (np.sign(e['benefit']) == fi)
            rows[pn][fn] = helper.continuous(e, selected, denominator)
    require(sum(v['points'] for row in rows.values() for v in row.values()) == int(mask.sum()), 'Sign table lost zero or nonzero cases')
    return rows


def summarize(helper, radar, arrays, counts, full):
    output = []
    for hi, horizon in enumerate(HORIZONS):
        e = arrays[hi]; masks = radar.strata_masks(e)
        bins = np.searchsorted(EDGES, e['camera_features'][:, 2], side='right') - 1
        require(np.isin(bins, np.arange(7)).all(), 'Fixed speed bin support')
        item = dict(horizon_seconds=horizon, original_future_object_count=int(counts[hi].sum()),
                    ALL_original_future_support_by_history_reason=full[hi], uncovered_strata={})
        def cell(mask, denom):
            by_history = {}
            for ri, name in enumerate(HISTORY_REASONS):
                hm = mask & (e['past_reason'] == ri)
                by_history[name] = dict(support=helper.continuous(e, hm, denom),
                    by_camera_reason={str(ci): helper.continuous(e, hm & (e['reason'] == ci), denom) for ci in range(5)})
            historical = mask & (e['past_reason'] == 0)
            photo = mask & (e['reason'] == 0)
            common = historical & photo
            summary = helper.continuous(e, mask, denom)
            for key in ('points', 'weight_sum', 'FULL_denominator_positive_gain_xy_m', 'FULL_denominator_negative_cost_xy_m'):
                radar.close(sum(v['support'][key] for v in by_history.values()), summary[key], 'History missing conservation ' + key)
            return dict(full_uncovered_support=summary, by_history_reason=by_history,
                        history_available_support=helper.continuous(e, historical, denom),
                        camera_available_support=helper.continuous(e, photo, denom),
                        common_history_camera_support=helper.continuous(e, common, denom),
                        AUC_same_common_population=aucs(helper, e, common),
                        past_rows_future_columns_sign_table=sign_table(helper, e, common, denom))
        for stratum in STRATA:
            group_rows = {}
            for gi, group in enumerate(GROUPS):
                denom = int(counts[hi].sum() if gi == 0 else counts[hi, gi - 1])
                group_rows[group] = cell(masks[stratum] & radar.group_mask(e, group), denom)
            item['uncovered_strata'][stratum] = dict(all_speed_GT_groups=group_rows)
        records = [dict(bin_index=bi, **cell(masks['no_radar'] & (bins == bi), int(counts[hi].sum()))) for bi in range(7)]
        all_no_radar = item['uncovered_strata']['no_radar']['all_speed_GT_groups']['all']
        for key in ('points', 'weight_sum', 'FULL_denominator_positive_gain_xy_m', 'FULL_denominator_negative_cost_xy_m'):
            radar.close(sum(r['full_uncovered_support'][key] for r in records),
                        all_no_radar['full_uncovered_support'][key], 'Seven bins conserve ' + key)
        item['uncovered_strata']['no_radar']['all_GT_groups_pooled_fixed_speed_bins'] = records
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('camera-run', 'camera-protocol', 'history-inputs', 'radar-run', 'motion-run', 'motion-analysis',
                'train-index', 'raw-targets', 'sparse-labels', 'points-file'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qa-only', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=300)
    args = parser.parse_args(); require(not args.out.exists(), 'Fresh output required')
    require(args.max_seconds == 300, 'Fixed CPU-only time ceiling is 300 seconds')
    def timeout(signum, frame):
        raise TimeoutError('Fixed CPU time ceiling or termination signal: ' + str(signum))
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, timeout)
    signal.alarm(args.max_seconds)
    started = time.monotonic(); rules, history, camera_rules, radar, helper, geometry = dependencies()
    if args.qa_only:
        qa = load_module('test_history_motion_alignment_train_v1', sha(HERE / 'test_history_motion_alignment_train_v1.py'))
        result = qa.run_analytic_qa(sys.modules[__name__], geometry, helper)
    else:
        required = ('camera_run', 'camera_protocol', 'history_inputs', 'radar_run', 'motion_run', 'motion_analysis',
                    'train_index', 'raw_targets', 'sparse_labels', 'points_file')
        require(all(getattr(args, key) is not None for key in required), 'All authenticated inputs required')
        require(sha(args.camera_run / 'complete.json') == rules['sources']['camera_complete_sha256'], 'Wrong completed camera evidence')
        rr, mr, nr, cr, scenes, prior, sources, core, times = history.authenticate(args, camera_rules, radar)
        arrays, counts, coverage = history.collect(args, radar, rr, mr, nr, cr, scenes, prior, core, times)
        audits, full = collect_history(args, rules, history, radar, geometry, arrays, counts, mr, nr, cr, times)
        result = dict(schema=SCHEMA, status='COMPLETE_CPU_TRAIN_HISTORY_LABEL_ALIGNMENT',
            samples=512, scenes=256, source_points=1740053, original_object_horizon_rows=int(counts.sum()),
            sources=dict(parent_authentication=sources, new_frozen_sources=rules['sources']), rules=rules,
            sampling_coverage=coverage, per_anchor_label_and_time_audit=audits,
            horizons=summarize(helper, radar, arrays, counts, full),
            verification=dict(raw_train512_compressed_and_uncompressed_SHA=True, original_2s_material_geometry_checked=True,
                labels_used_only_for_diagnostic_scoring=True, actual_images_reopened=False,
                no_new_camera_or_source_selection=True, model_forward=False, optimizer_updates=0,
                bootstrap=False, development_data_files_opened=False, oracle_reference_is_mathematical_upper_bound=False))
    require('torch' not in sys.modules, 'CPU-only analysis must not import Torch')
    result.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                  analyzer_sha256=sha(__file__), rules_sha256=sha(RULES_FILE),
                  runtime=dict(python=sys.executable, version=sys.version, numpy=np.__version__))
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(out=str(args.out), status=result['status'], sha256=sha(args.out))))


if __name__ == '__main__':
    try:
        main()
    finally:
        signal.alarm(0)
