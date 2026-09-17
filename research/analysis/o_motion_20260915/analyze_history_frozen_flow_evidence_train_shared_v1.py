"""CPU summaries of frozen-flow evidence on unchanged train512 source support.

Primary photo-valid and low-texture extension populations are kept separate.
Future benefit and privileged actual-past benefit are different AUC targets.
No flow model, images, raw GT, fit, threshold search or bootstrap is run here.
Shared-GPU operational binding only; the original statistical functions are unchanged.
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
SCHEMA = 'history-frozen-flow-evidence-train-analysis-v1'
FLOW_SCHEMA = 'history-frozen-flow-evidence-train-v1'
PARENT_ANALYZER_SHA = '69ad87766e8964412fef11bd3e0b7810b4c109a5e2bc7ce1f94475e580df68c0'
POSE_ANALYZER_SHA = '610fec1c9eb95c482fc53d51a04214e65ea5f596f0f0e513051ede0cd4465e04'
POSE_COMPLETE_SHA = '1f759d6050f003d90cc692d490421d179ec406f02266ace41d59ff37654165b0'
POSE_PROTOCOL_PATH = HERE / 'history_pose_address_control_train_protocol_v1.json'
POSE_PROTOCOL_SHA = '8183a66c5beaea546ab479b93e2bc98ba6902fb1f76a74fa2dd5dc3faf0a2d13'
ASSET_MANIFEST_PATH = HERE / 'external' / 'raft_official_v1' / 'manifest.json'
FLOW_ENDPOINTS = dict(
    asset_manifest_sha256='db378bcf49a348a8968d571f834a33e3f434582e7b0d551b805f49cefb320cdd',
    checkpoint_sha256='fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1',
    probe_source_sha256='60491da9cc3b0dcfabcf9c09d15ca6ac1b2f2e8175b657870c6d54fb6e65789e',
    history_inputs_complete_sha256='040d62242a541b9685f3f39f1534b3663b0e1142b4cd7d9737cdbfd15d94314d',
    camera_complete_sha256='0c0d2e640a12e3299e8dacc03189776669cd2e50036a65de823026304119b648')
RULES_PATH = HERE / 'history_frozen_flow_evidence_train_shared_analysis_rules_v1.json'
RULES_SHA = '38e35d66ffb91e8d80082feaeaa7f6f7291a96171651fd2b6b0754ec78c0ed1f'
OPERATIONAL_AMENDMENT = {'kind': 'shared_gpu_admission_and_allocator_cap_only',
 'parent_protocol_sha256': 'bed2332cff08b4f857e46eec22f456344e1f1b5d1b1f0f36fd864d432b0b46c5',
 'parent_collector_sha256': '17ecf5be37edcee747f4b2b6e2a760655b697c94142a5ec539acf90a3f298e93',
 'physical_index': 0,
 'gpu_uuid': 'GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe',
 'minimum_free_bytes': 17179869184,
 'allow_existing_compute_processes': True,
 'allow_nonzero_utilization': True,
 'allocator_cap_bytes': 12884901888,
 'scientific_settings_unchanged': True}
HORIZONS = (.5, 1., 1.5, 2.)
GROUPS = ('all', 'stationary', 'ambiguous', 'moving')
STRATA = ('all', 'no_radar', 'radar_present')
EDGES = np.asarray([0., .1, .5, 1., 2., 5., 10., np.inf])
PRIMARY_FEATURES = ('flow_score_px', 'broken_flow_score_px', 'score_true', 'score_broken',
                    'lsq_speed_xy_mps', 'hypothesis_flow_separation_px')
EXTENSION_FEATURES = ('flow_score_px', 'broken_flow_score_px', 'lsq_speed_xy_mps',
                      'hypothesis_flow_separation_px')
PHOTO_POPULATIONS = ('primary_original_reason_0', 'low_texture_extension_original_reason_4', 'other_original_reasons')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_module(name, digest):
    path = HERE / (name + '.py')
    require(sha(path) == digest, 'Frozen dependency changed: ' + name)
    spec = importlib.util.spec_from_file_location('_flow_analysis_' + name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def dependencies():
    require(sha(RULES_PATH) == RULES_SHA, 'Prespecified flow analysis rules changed')
    rules = read(RULES_PATH)
    require(rules.get('operational_amendment') == OPERATIONAL_AMENDMENT, 'Shared execution analysis binding differs')
    require(rules['schema'] == SCHEMA and rules['primary_features'] == list(PRIMARY_FEATURES) and
            rules['extension_features'] == list(EXTENSION_FEATURES) and rules['directions'] == [1]*6 and
            rules['parent_analyzer_sha256'] == PARENT_ANALYZER_SHA and
            rules['pose_analyzer_sha256'] == POSE_ANALYZER_SHA and rules['pose_complete_sha256'] == POSE_COMPLETE_SHA and
            rules['frozen_flow_endpoints'] == FLOW_ENDPOINTS and rules['extension_directions'] == [1]*4 and
            rules['speed_edges_mps'] == [0., .1, .5, 1., 2., 5., 10., None], 'Prespecified flow analysis semantics differ')
    parent = frozen_module('analyze_history_camera_evidence_train_v1', PARENT_ANALYZER_SHA)
    parent_rules, radar, helper = parent.dependencies()
    pose = frozen_module('analyze_history_pose_address_control_train_v1', POSE_ANALYZER_SHA)
    pose_rules, _, _, _, _ = pose.dependencies()
    require(sha(POSE_PROTOCOL_PATH) == POSE_PROTOCOL_SHA, 'Historical pose protocol changed')
    return rules, parent, parent_rules, radar, helper, pose, pose_rules


def require_frozen_bindings(rules):
    require(rules['status'] == 'FROZEN_PRESPECIFIED_NO_REAL_FLOW_RESULT_READ', 'Flow analysis is not frozen')
    for key in ('flow_extractor_sha256', 'flow_core_sha256', 'flow_protocol_sha256'):
        value = rules[key]
        require(len(value) == 64 and all(ch in '0123456789abcdef' for ch in value), 'Unfrozen flow binding: ' + key)


def authenticate_flow(a, rules, radar, cr, mr, sources):
    """Authenticate the full extraction and its fixed official inference contract."""
    require_frozen_bindings(rules)
    protocol = read(a.flow_protocol)
    require(sha(a.flow_protocol) == rules['flow_protocol_sha256'] and protocol['schema'] == FLOW_SCHEMA and
            protocol['status'] == 'FROZEN' and protocol['resources'] == dict(max_seconds=3600, cpu_threads=2, max_allocated_bytes=12*1024**3),
            'Frozen flow protocol differs')
    require(protocol.get('operational_amendment') == OPERATIONAL_AMENDMENT, 'Shared operational protocol differs')
    require(protocol['settings'] == rules['frozen_inference_settings'] and
            protocol['prespecified_analysis'] == rules['extraction_protocol_analysis_contract'], 'Flow inference/analysis contract differs')
    require(protocol['source_sha256'] == rules['flow_extractor_sha256'] == sha(HERE / 'extract_history_frozen_flow_evidence_train_shared_v1.py') and
            protocol['module_sha256'] == rules['flow_core_sha256'] == sha(HERE / 'history_frozen_flow_evidence_v1.py') and
            {key: protocol[key] for key in FLOW_ENDPOINTS} == FLOW_ENDPOINTS and
            sha(HERE / 'run_raft_official_pair_probe_v1.py') == FLOW_ENDPOINTS['probe_source_sha256'], 'Flow source/endpoints differ')
    require(sha(ASSET_MANIFEST_PATH) == FLOW_ENDPOINTS['asset_manifest_sha256'], 'Official asset metadata changed')
    assets = read(ASSET_MANIFEST_PATH)
    done, manifest, index, summary = radar.ledger(a.flow_run, FLOW_SCHEMA, 'COMPLETE_HISTORY_FROZEN_FLOW_TRAIN_EVIDENCE')
    require(done['source_sha256'] == manifest['source_sha256'] == rules['flow_extractor_sha256'] and
            done['module_sha256'] == manifest['module_sha256'] == rules['flow_core_sha256'] and
            done['protocol_sha256'] == manifest['protocol_sha256'] == rules['flow_protocol_sha256'] and
            {key: manifest[key] for key in FLOW_ENDPOINTS} == FLOW_ENDPOINTS and
            manifest['camera_files_sha256'] == sources['camera_files_sha256'] and
            manifest['history_inputs_files_sha256'] == sources['history_inputs_files_sha256'] and
            manifest['scope'] == protocol['scope'] and manifest['settings'] == protocol['settings'], 'Flow ledger/source/input binding differs')
    require(manifest.get('operational_amendment') == OPERATIONAL_AMENDMENT,
            'Shared manifest operational binding differs')
    gpu = manifest['gpu']; contract = OPERATIONAL_AMENDMENT
    require(gpu['index'] == contract['physical_index'] and gpu['uuid'] == contract['gpu_uuid'] and
            gpu['shared_execution'] is True and gpu['admission_is_observation_not_memory_reservation'] is True and
            gpu['minimum_free_bytes'] == contract['minimum_free_bytes'] and
            gpu['prelaunch_memory_free_MiB'] * 1024**2 >= contract['minimum_free_bytes'] and
            0 <= gpu['prelaunch_memory_used_MiB'] <= gpu['prelaunch_memory_total_MiB'] and
            0 <= gpu['prelaunch_utilization_percent'] <= 100 and
            isinstance(gpu['existing_compute_pids'], list) and
            all(type(pid) is int and pid > 0 for pid in gpu['existing_compute_pids']),
            'Shared GPU admission receipt differs')
    require(gpu['allocator_cap_bytes'] == contract['allocator_cap_bytes'] and
            gpu['allocator_total_memory_bytes'] >= contract['minimum_free_bytes'] and
            gpu['allocator_cap_installed_before_model_to_cuda'] is True and
            math.isclose(gpu['allocator_fraction'], gpu['allocator_cap_bytes']/gpu['allocator_total_memory_bytes'],
                         rel_tol=1e-15, abs_tol=0.), 'Allocator cap receipt differs')
    require(manifest['official_commit'] == assets['commit'] == '2888e15a51fa41140771d3f498ed8023cff098d1' and
            manifest['checkpoint_sha256'] == assets['checkpoint']['sha256'] and
            manifest['verified_consumed_assets'] == {key: value for key, value in assets['files'].items()
                                                    if key.startswith('source/') or key == 'weights/raft-things.pth'},
            'Official consumed source/checkpoint descriptors differ')
    require(manifest['schema'] == index['schema'] == summary['schema'] == FLOW_SCHEMA and
            summary['status'] == done['status'] and (summary['samples'], summary['scenes'], summary['source_points'],
                summary['images_decoded'], summary['model_forwards']) == (512, 256, 1740053, 6144, 9216) and
            index['source_points'] == 1740053 and 0 < summary['seconds'] < 3600 and
            0 < summary['peak_allocated_bytes'] <= 12*1024**3 and summary['optimizer_updates'] == 0 and
            summary['checkpoint_sha256'] == FLOW_ENDPOINTS['checkpoint_sha256'] and
            summary['model_state_sha256'] == manifest['initial_model_state_sha256'], 'Full frozen-flow extraction/resource support differs')
    require(all(summary[key] is True for key in ('frozen_model_state_unchanged', 'all_original_cached_arrays_unchanged',
                    'all_query_arrays_roundtrip_exact', 'original_query_population_is_GT_defined')) and
            all(summary[key] is False for key in ('GT_target_arrays_read', 'image_flow_uses_GT', 'O_or_D_model_loaded',
                    'training', 'threshold_selection', 'fitting', 'development_used', 'flow_quality_claim', 'batch_accuracy_statistics_computed')),
            'Flow inference/diagnostic scope differs')
    rows = index['records']
    require(len(rows) == len(cr) == len(mr) == 512 and
            {p.relative_to(a.flow_run).as_posix() for p in (a.flow_run / 'samples').glob('*.npz')} == {r['file'] for r in rows}, 'Flow sample file set differs')
    count_fields = ('primary_valid', 'low_texture_extension', 'flow_valid', 'fb_zero_valid', 'fb_D_valid')
    total_counts = {key: 0 for key in count_fields}; total_reasons = {str(i): 0 for i in range(5)}
    camera_names = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
    flow_keys = ('current_to_past', 'past_to_current', 'current_to_broken_past')
    for ordinal, (row, camera, motion) in enumerate(zip(rows, cr, mr)):
        require(row['ordinal'] == camera['ordinal'] == motion['ordinal'] == ordinal and
                row['identity'] == camera['identity'] == motion['identity'] and row['file'] == camera['file'] == motion['file'] and
                row['source_points'] == camera['source_points'] == motion['source_points'] and row['camera_npz_sha256'] == camera['sha256'] and
                row['reason_counts'] == camera['reason_counts'] and set(row['counts']) == set(count_fields), 'Flow parent row binding differs')
        require(set(row['flow_array_sha256_HWC']) == set(row['forward_seconds']) == set(camera_names), 'Per-camera inference accounting differs')
        for name in camera_names:
            digests = row['flow_array_sha256_HWC'][name]; timing = row['forward_seconds'][name]
            require(set(digests) == set(timing) == set(flow_keys) and all(len(v) == 64 for v in digests.values()) and
                    all(math.isfinite(timing[key]) and timing[key] > 0 for key in flow_keys), 'Three independent inference calls missing')
        for key in count_fields:
            total_counts[key] += row['counts'][key]
        for key in total_reasons:
            total_reasons[key] += row['reason_counts'][key]
    require(total_counts == index['counts'] == summary['counts'] and
            total_reasons == index['reason_counts'] == summary['reason_counts'] and sum(total_reasons.values()) == 1740053 and
            total_counts['primary_valid'] == total_reasons['0'] == 1397509 and
            total_counts['low_texture_extension'] == total_reasons['4'] == 261702 and
            total_counts['flow_valid'] == 1659211, 'Flow totals differ from original full source support')
    require(set(summary['forward_seconds']) == set(flow_keys) and
            all(summary['forward_seconds'][key]['calls'] == 3072 for key in flow_keys), 'Incomplete official forward-call count')
    sources.update(flow_complete_sha256=sha(a.flow_run / 'complete.json'), flow_files_sha256=done['files_sha256'],
                   flow_extractor_sha256=rules['flow_extractor_sha256'], flow_core_sha256=rules['flow_core_sha256'],
                   flow_protocol_sha256=rules['flow_protocol_sha256'], frozen_raft_endpoints=FLOW_ENDPOINTS,
                   flow_operational_amendment=protocol['operational_amendment'],
                   frozen_raft_initial_state_sha256=manifest['initial_model_state_sha256'])
    return rows


def same_bytes(first, second):
    return first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()


def flow_arrays(values, row, old):
    """Verify query support, residual algebra and FB domain without model inference."""
    n = len(old['source_flat_indices'])
    shapes = dict(source_flat_indices=(n,), camera_index=(n,), reason_code=(n,), flow_valid=(n,), primary_valid=(n,),
                  residual=(n, 2), broken_residual=(n, 2), score=(n,), broken_score=(n,),
                  hypothesis_flow_separation_px=(n,), sampled_flow=(n, 2, 2), sampled_broken_flow=(n, 2, 2),
                  predicted_past_uv=(n, 2, 2), fb_valid=(n, 2), fb_residual=(n, 2))
    require(set(values) == set(row['arrays']) == set(shapes) | {'identity_json'}, 'Flow NPZ fields differ')
    for name, value in values.items():
        require(row['arrays'][name] == dict(shape=list(value.shape), dtype=str(value.dtype)), 'Flow descriptor differs: ' + name)
    for name, shape in shapes.items():
        dtype = dict(source_flat_indices='int64', camera_index='int8', reason_code='uint8',
                     flow_valid='bool', primary_valid='bool', fb_valid='bool').get(name, 'float64')
        require(values[name].shape == shape and str(values[name].dtype) == dtype, 'Flow array shape/dtype differs: ' + name)
    for name in ('identity_json', 'source_flat_indices', 'camera_index', 'reason_code'):
        require(same_bytes(values[name], old[name]), 'Original camera support bytes changed: ' + name)
    eligible = np.isin(old['reason_code'], [0, 4])
    require(np.array_equal(values['flow_valid'], eligible) and
            np.array_equal(values['primary_valid'], old['reason_code'] == 0), 'Flow changed original primary/extension population')
    require(row['reason_counts'] == {str(i): int((old['reason_code'] == i).sum()) for i in range(5)} and
            row['counts'] == dict(primary_valid=int(values['primary_valid'].sum()),
                low_texture_extension=int((old['reason_code'] == 4).sum()), flow_valid=int(eligible.sum()),
                fb_zero_valid=int(values['fb_valid'][:, 0].sum()), fb_D_valid=int(values['fb_valid'][:, 1].sum())),
            'Flow NPZ count ledger differs')
    finite_fields = ('residual', 'broken_residual', 'score', 'broken_score', 'hypothesis_flow_separation_px',
                     'sampled_flow', 'sampled_broken_flow', 'predicted_past_uv')
    for name in finite_fields:
        require(np.isfinite(values[name][eligible]).all() and np.isnan(values[name][~eligible]).all(),
                'Invalid flow score imputation/support: ' + name)
    current, past = old['uv'][eligible, :, 0], old['uv'][eligible, :, 1]
    endpoint = current + values['sampled_flow'][eligible]
    require(np.array_equal(values['predicted_past_uv'][eligible], endpoint) and
            np.array_equal(values['residual'][eligible], np.linalg.norm(endpoint - past, axis=-1)) and
            np.array_equal(values['broken_residual'][eligible],
                           np.linalg.norm(current + values['sampled_broken_flow'][eligible] - past, axis=-1)),
            'Flow residual algebra differs; broken targets must remain unshifted')
    displacement = past - current
    require(np.array_equal(values['hypothesis_flow_separation_px'][eligible],
                           np.linalg.norm(displacement[:, 1] - displacement[:, 0], axis=-1)), 'Flow geometry control differs')
    for prefix in ('', 'broken_'):
        require(np.array_equal(values[prefix + 'score'][eligible],
                               values[prefix + 'residual'][eligible, 0] - values[prefix + 'residual'][eligible, 1]),
                'Flow score sign/algebra differs')
    endpoint = values['predicted_past_uv']
    inside = (eligible[:, None] & np.isfinite(endpoint).all(axis=-1) &
              (endpoint[:, :, 0] >= 0.) & (endpoint[:, :, 0] <= 1599.) &
              (endpoint[:, :, 1] >= 0.) & (endpoint[:, :, 1] <= 899.))
    require(np.array_equal(values['fb_valid'], inside) and np.isfinite(values['fb_residual'][inside]).all() and
            np.all(values['fb_residual'][inside] >= 0) and np.isnan(values['fb_residual'][~inside]).all(),
            'FB availability differs from closed original-image domain')
    return eligible[:, None] & ~inside


def collect_flow(a, parent, radar, pose, rr, mr, nr, cr, fr, pr, scenes, prior, core, times):
    arrays, counts, coverage = parent.collect(a, radar, rr, mr, nr, cr, scenes, prior, core, times)
    histories = read(a.history_inputs / 'records.json')['records']
    offsets = [0]*4
    for ordinal, (flow_row, photo_row, motion_row, pose_row) in enumerate(zip(fr, cr, mr, pr)):
        values = radar.load_npz(a.flow_run, flow_row)
        camera = radar.load_npz(a.camera_run, photo_row)
        motion = radar.load_npz(a.motion_run, motion_row)
        reference = radar.load_npz(a.pose_run, pose_row)
        endpoint_oob = flow_arrays(values, flow_row, camera)
        _, past_benefit, _ = pose.pose_arrays(reference, pose_row, camera, motion, histories[ordinal]['t0_lidar_us'])
        require(not np.any(reference['reference_valid'] & (camera['reason_code'] == 4)), 'Existing pose reference must exclude extension')
        for hi, e in enumerate(arrays):
            if 'flow_scores' not in e:
                n = len(e['benefit'])
                e.update(flow_scores=np.full((n, 2), np.nan), flow_geometry_separation_px=np.full(n, np.nan),
                         fb_valid=np.zeros((n, 2), dtype=bool), flow_endpoint_out_of_bounds=np.zeros((n, 2), dtype=bool),
                         fb_residual_px=np.full((n, 2), np.nan), pose_reference_valid=np.zeros(n, dtype=bool),
                         actual_past_benefit_xy_m=np.full(n, np.nan))
            take = motion['valid'][hi] & (motion['owner'] < 0)
            n = int(take.sum()); sl = slice(offsets[hi], offsets[hi] + n)
            require(same_bytes(e['reason'][sl], values['reason_code'][take]) and
                    np.all(e['scene'][sl] == scenes.index(flow_row['identity']['scene_token'])), 'Flow/parent evidence join differs')
            e['flow_scores'][sl] = np.stack([values['score'][take], values['broken_score'][take]], axis=1)
            e['flow_geometry_separation_px'][sl] = values['hypothesis_flow_separation_px'][take]
            e['fb_valid'][sl] = values['fb_valid'][take]
            e['fb_residual_px'][sl] = values['fb_residual'][take]
            e['flow_endpoint_out_of_bounds'][sl] = endpoint_oob[take]
            e['pose_reference_valid'][sl] = reference['reference_valid'][take]
            e['actual_past_benefit_xy_m'][sl] = past_benefit[take]
            offsets[hi] += n
    require(offsets == [len(e['benefit']) for e in arrays], 'Flow evidence join incomplete')
    return arrays, counts, coverage


def population_masks(e):
    reason = e['reason']
    require(np.isin(reason, np.arange(5)).all(), 'Original photo reason outside fixed support')
    return dict(zip(PHOTO_POPULATIONS, (reason == 0, reason == 4, (reason != 0) & (reason != 4))))


def feature_values(e, extension=False):
    # Parent columns supply ZNCC and old sum-reduction LSQ speed; geometry is the
    # new two-hypothesis displacement separation, not just past-UV separation.
    values = [e['flow_scores'][:, 0], e['flow_scores'][:, 1]]
    if not extension:
        values.extend([e['camera_features'][:, 0], e['camera_features'][:, 1]])
    values.extend([e['camera_features'][:, 2], e['flow_geometry_separation_px']])
    return np.stack(values, axis=1)


def auc_scores(helper, e, mask, target, extension=False):
    expected_reason = 4 if extension else 0
    require(np.all(e['reason'][mask] == expected_reason), 'Primary and extension supports mixed')
    features = EXTENSION_FEATURES if extension else PRIMARY_FEATURES
    values = feature_values(e, extension)[mask]
    benefit, weight = target[mask], e['weight'][mask]
    require(np.isfinite(values).all() and np.isfinite(benefit).all(), 'Missing scores or labels in common AUC population')
    result = {name: helper.weighted_auc(values[:, fi], benefit, weight) for fi, name in enumerate(features)}
    first = result[features[0]]
    support = ('positive_points', 'negative_points', 'positive_weight', 'negative_weight')
    require(all({key: row[key] for key in support} == {key: first[key] for key in support}
                for row in result.values()), 'AUC feature populations differ')
    return dict(points=int(mask.sum()), nonempty_objects=int(len(np.unique(e['object'][mask]))),
                original_weight_sum=float(weight.sum()), zero_benefit_points=int((benefit == 0).sum()),
                features_same_population=result)


def fb_description(helper, radar, e, mask, denom):
    """True-flow forward/backward residuals only; never used as a score mask."""
    require(np.all(np.isin(e['reason'][mask], [0, 4])), 'FB descriptor outside queried population')
    output = {}
    for hi, hypothesis in enumerate(('zero', 'D')):
        available = mask & e['fb_valid'][:, hi]
        outside = mask & e['flow_endpoint_out_of_bounds'][:, hi]
        require(np.array_equal(available | outside, mask) and not np.any(available & outside),
                'FB availability and endpoint-outside do not partition queried support')
        residual, weights = e['fb_residual_px'][available, hi], e['weight'][available]
        require(np.isfinite(residual).all() and np.all(residual >= 0) and
                np.isnan(e['fb_residual_px'][outside, hi]).all(), 'FB missing residual imputed or invalid')
        full = helper.continuous(e, mask, denom)
        part = dict(available=helper.continuous(e, available, denom),
                    endpoint_out_of_bounds=helper.continuous(e, outside, denom))
        for key in ('points', 'weight_sum', 'FULL_denominator_positive_gain_xy_m', 'FULL_denominator_negative_cost_xy_m'):
            if full[key] is not None:
                radar.close(sum(row[key] for row in part.values()), full[key], 'FB physical support partition ' + key)
        mass = float(e['weight'][mask].sum()); available_mass = float(weights.sum())
        output[hypothesis] = dict(
            conditional_original_weight_mean_fb_residual_px=float(np.dot(weights, residual))/available_mass if available_mass else None,
            endpoint_out_of_bounds_point_fraction=float(outside.sum())/int(mask.sum()) if np.any(mask) else None,
            endpoint_out_of_bounds_original_weight_fraction=float(e['weight'][outside].sum())/mass if mass else None,
            original_future_physical_by_FB_availability=part)
    return output


def population_summary(helper, radar, e, mask, denom, extension=False):
    require(np.all(e['reason'][mask] == (4 if extension else 0)), 'Wrong named scoring population')
    future = auc_scores(helper, e, mask, e['benefit'], extension)
    past_valid = mask & e['pose_reference_valid']
    require(not extension or not np.any(past_valid), 'Low-texture pose labels unexpectedly available; no new pose extraction authorized')
    require(np.all(~e['pose_reference_valid'] | (e['reason'] == 0)), 'Pose reference expanded original photo-valid support')
    past = auc_scores(helper, e, past_valid, e['actual_past_benefit_xy_m'], extension)
    past.update(reference_available_points=int(past_valid.sum()), reference_missing_points=int((mask & ~past_valid).sum()),
                reference_missing_original_weight=float(e['weight'][mask & ~past_valid].sum()),
                missing_reason='Existing GT-reference requires original photo-valid support; all reason-4 extension references are missing'
                               if extension else 'Outside existing historical GT-reference-valid support')
    return dict(original_object_count=denom, primary_original_future_benefit_auc=future,
                secondary_actual_past_camera_benefit_auc=past,
                true_flow_forward_backward_descriptive=fb_description(helper, radar, e, mask, denom))


def summarize(helper, radar, arrays, counts, prior):
    output = []
    for hi, horizon in enumerate(HORIZONS):
        e = arrays[hi]
        bins = np.searchsorted(EDGES, e['camera_features'][:, 2], side='right') - 1
        require(np.isin(bins, np.arange(7)).all(), 'Original D-speed bin changed')
        masks, populations = radar.strata_masks(e), population_masks(e)
        for group in GROUPS:
            group_mask = radar.group_mask(e, group)
            denom = int(counts[hi].sum() if group == 'all' else counts[hi, GROUPS.index(group)-1])
            previous = next(row for row in prior['physical'] if row['horizon_seconds'] == horizon and row['group'] == group)
            current = helper.continuous(e, group_mask, denom)
            require(current['points'] == previous['uncovered_points'] and denom == previous['original_objects'], 'Original future support differs')
            for key, value in current.items():
                radar.close(value, previous['uncovered_D_versus_zero_benefit_xy'][key], 'Original physical ' + key)
        item = dict(horizon_seconds=horizon, strata={})
        for stratum in STRATA:
            groups = {}; item['strata'][stratum] = groups
            for group in GROUPS:
                denom = int(counts[hi].sum() if group == 'all' else counts[hi, GROUPS.index(group)-1])
                base = masks[stratum] & radar.group_mask(e, group)
                records = []
                for bi in [-1] + (list(range(7)) if stratum == 'no_radar' and group == 'all' else []):
                    selected = base if bi == -1 else base & (bins == bi)
                    full = helper.continuous(e, selected, denom)
                    by_photo = {name: helper.continuous(e, selected & population, denom) for name, population in populations.items()}
                    require(sum(row['points'] for row in by_photo.values()) == full['points'], 'Original photo reason support lost')
                    for key in ('weight_sum', 'FULL_denominator_benefit_contribution_xy_m', 'FULL_denominator_positive_gain_xy_m',
                                'FULL_denominator_negative_cost_xy_m'):
                        if full[key] is not None:
                            radar.close(sum(row[key] for row in by_photo.values()), full[key], 'Photo physical partition ' + key)
                    records.append(dict(bin_index=None if bi == -1 else bi, all_speeds=bi == -1,
                        original_future_physical_support=full, original_future_physical_by_photo_population=by_photo,
                        primary_photo_valid=population_summary(helper, radar, e, selected & populations[PHOTO_POPULATIONS[0]], denom),
                        low_texture_extension=population_summary(helper, radar, e, selected & populations[PHOTO_POPULATIONS[1]], denom, extension=True)))
                groups[group] = dict(original_object_count=denom, bins=records)
        output.append(item)
    return output


def synthetic_array_qa():
    n = 4
    reason = np.array([0, 4, 0, 3], dtype=np.uint8); eligible = np.isin(reason, [0, 4])
    uv = np.full((n, 2, 2, 2), np.nan)
    uv[eligible] = np.array([[[100., 100.], [101., 100.]], [[101., 100.], [104., 100.]]])
    old = dict(identity_json=np.asarray('{"sample_token":"synthetic_only"}'), source_flat_indices=np.arange(n, dtype=np.int64),
               camera_index=np.zeros(n, dtype=np.int8), reason_code=reason, uv=uv)
    values = {key: old[key].copy() for key in ('identity_json', 'source_flat_indices', 'camera_index', 'reason_code')}
    values.update(flow_valid=eligible.copy(), primary_valid=reason == 0, fb_valid=np.zeros((n, 2), dtype=bool))
    for name, shape in (('residual', (n, 2)), ('broken_residual', (n, 2)), ('score', (n,)), ('broken_score', (n,)),
                        ('hypothesis_flow_separation_px', (n,)), ('sampled_flow', (n, 2, 2)),
                        ('sampled_broken_flow', (n, 2, 2)), ('predicted_past_uv', (n, 2, 2)), ('fb_residual', (n, 2))):
        values[name] = np.full(shape, np.nan)
    values['sampled_flow'][eligible] = [2., 0.]
    values['sampled_flow'][2] = [[1500., 0.], [1498., 799.]]  # zero OOB; D exactly on last valid corner.
    values['sampled_broken_flow'][eligible] = [7., 0.]
    current, past = uv[eligible, :, 0], uv[eligible, :, 1]
    values['predicted_past_uv'][eligible] = current + values['sampled_flow'][eligible]
    values['residual'][eligible] = np.linalg.norm(values['predicted_past_uv'][eligible] - past, axis=-1)
    values['broken_residual'][eligible] = np.linalg.norm(current + values['sampled_broken_flow'][eligible] - past, axis=-1)
    values['score'][eligible] = values['residual'][eligible, 0] - values['residual'][eligible, 1]
    values['broken_score'][eligible] = values['broken_residual'][eligible, 0] - values['broken_residual'][eligible, 1]
    displacement = past - current
    values['hypothesis_flow_separation_px'][eligible] = np.linalg.norm(displacement[:, 1] - displacement[:, 0], axis=-1)
    values['fb_valid'][eligible] = True; values['fb_valid'][2, 0] = False
    values['fb_residual'][values['fb_valid']] = .25
    row = dict(arrays={key: dict(shape=list(x.shape), dtype=str(x.dtype)) for key, x in values.items()},
               reason_counts={str(i): int((reason == i).sum()) for i in range(5)},
               counts=dict(primary_valid=2, low_texture_extension=1, flow_valid=3, fb_zero_valid=2, fb_D_valid=3))
    outside = flow_arrays(values, row, old)
    require(outside[2, 0] and not outside[2, 1] and values['flow_valid'][2] and
            np.isfinite(values['score'][2]) and np.isfinite(values['score'][1]), 'QA OOB keeps score and extension supports score')
    for name, location, changed in (('reason_code', 1, 0), ('score', 2, 0.),
                                     ('hypothesis_flow_separation_px', 0, 99.), ('broken_score', 3, 0.)):
        bad = {key: x.copy() for key, x in values.items()}; bad[name][location] = changed
        try:
            flow_arrays(bad, row, old)
        except ValueError:
            pass
        else:
            raise ValueError('QA invalid flow NPZ accepted: ' + name)


def analytic_qa(helper, radar):
    synthetic_array_qa()
    reason = np.array([0, 0, 0, 4, 4, 1, 2, 3], dtype=np.uint8)
    n = len(reason); eligible = np.isin(reason, [0, 4])
    flow_scores = np.full((n, 2), np.nan)
    flow_scores[:5] = [[2., -2.], [1., -1.], [0., 0.], [2., -2.], [1., -1.]]
    camera = np.full((n, 5), np.nan)
    camera[:3, :2] = [[2., -2.], [1., -1.], [0., 0.]]
    camera[:, 2] = [2., 1., 0., 2., 1., .1, .5, 11.]
    camera[:5, 4] = [2., 1., 0., 2., 1.]
    fb_valid = np.zeros((n, 2), dtype=bool); fb_valid[:5] = True; fb_valid[0, 0] = False
    endpoint_oob = np.zeros((n, 2), dtype=bool); endpoint_oob[0, 0] = True
    fb_residual = np.full((n, 2), np.nan); fb_residual[fb_valid] = .25
    e = dict(reason=reason, flow_scores=flow_scores, camera_features=camera, flow_geometry_separation_px=camera[:, 4].copy(), fb_valid=fb_valid,
             flow_endpoint_out_of_bounds=endpoint_oob, fb_residual_px=fb_residual,
             pose_reference_valid=np.array([True, True, False, False, False, False, False, False]),
             actual_past_benefit_xy_m=np.array([-1., 1.] + [np.nan]*6),
             benefit=np.array([2., -3., 0., 1., -2., 3., -4., -5.]),
             weight=np.array([.25, .75, .5, .25, .75, .5, .5, 1.]),
             object=np.array([0, 0, 1, 2, 2, 3, 4, 5]), group=np.array([0, 0, 1, 0, 1, 1, 2, 2], dtype=np.int8),
             present=np.array([False, False, False, False, False, True, False, True]), clip=np.zeros(n, dtype=bool))
    primary = population_summary(helper, radar, e, reason == 0, 8)
    extension = population_summary(helper, radar, e, reason == 4, 8, extension=True)
    require(primary['primary_original_future_benefit_auc']['points'] == 3 and
            primary['primary_original_future_benefit_auc']['features_same_population']['flow_score_px']['auc'] == 1. and
            primary['primary_original_future_benefit_auc']['features_same_population']['broken_flow_score_px']['auc'] == 0.,
            'QA score directions and FB-outside points remain in primary AUC')
    require(primary['secondary_actual_past_camera_benefit_auc']['points'] == 2 and
            primary['secondary_actual_past_camera_benefit_auc']['reference_missing_points'] == 1 and
            primary['secondary_actual_past_camera_benefit_auc']['features_same_population']['flow_score_px']['auc'] == 0.,
            'QA actual-past intersection/target differs from future target')
    require(set(extension['primary_original_future_benefit_auc']['features_same_population']) == set(EXTENSION_FEATURES) and
            extension['primary_original_future_benefit_auc']['points'] == 2 and
            extension['secondary_actual_past_camera_benefit_auc']['points'] == 0 and
            all(row['auc'] is None for row in extension['secondary_actual_past_camera_benefit_auc']['features_same_population'].values()),
            'QA extension has no ZNCC or invented past label')
    fb = primary['true_flow_forward_backward_descriptive']['zero']
    require(math.isclose(fb['endpoint_out_of_bounds_original_weight_fraction'], 1./6.) and
            fb['original_future_physical_by_FB_availability']['endpoint_out_of_bounds']['original_object_count'] == 8,
            'QA FB uses unchanged full object denominator')
    for label in (np.ones(2), -np.ones(2), np.zeros(2), np.empty(0)):
        require(helper.weighted_auc(np.ones(len(label)), label, np.ones(len(label)))['auc'] is None, 'QA empty/one-sided AUC null')
    require(helper.weighted_auc(np.array([1., 1.]), np.array([1., -1.]), np.array([.25, .75]))['auc'] == .5, 'QA weighted tie credit')
    counts = np.tile(np.array([3, 2, 3]), (4, 1)); previous = []
    for horizon in HORIZONS:
        for group in GROUPS:
            denom = 8 if group == 'all' else int(counts[0, GROUPS.index(group)-1])
            physical = helper.continuous(e, radar.group_mask(e, group), denom)
            previous.append(dict(horizon_seconds=horizon, group=group, original_objects=denom, uncovered_points=physical['points'],
                                 uncovered_D_versus_zero_benefit_xy=physical))
    result = summarize(helper, radar, [e]*4, counts, dict(physical=previous))
    for horizon in result:
        for stratum in STRATA:
            for group in GROUPS:
                rows = horizon['strata'][stratum][group]['bins']
                require(len(rows) == (8 if stratum == 'no_radar' and group == 'all' else 1), 'QA prespecified bin scope')
                require(all(len(row['original_future_physical_by_photo_population']) == 3 for row in rows), 'QA photo missingness partitions')
    try:
        auc_scores(helper, e, eligible, e['benefit'])
    except ValueError:
        pass
    else:
        raise ValueError('QA primary/extension mixing accepted')
    return dict(status='PASS_SYNTHETIC_ANALYTIC_QA_ONLY', tests=[
        'synthetic NPZ algebra, unchanged support and final-pixel FB domain; altered score/geometry/missing values rejected',
        'fixed six-score primary and four-score extension directions on identical populations',
        'low-texture extension has no ZNCC imputation or invented actual-past labels',
        'FB-unavailable/OOB support remains in primary AUC and full-object physical denominator',
        'actual-past target uses reference-valid intersection',
        'empty and one-sided AUC null; zero-benefit excluded; ties receive half credit',
        'three photo populations and FB availability preserve original future physical gain/cost',
        'three radar strata and four groups; seven speed bins only for no_radar/all',
        'primary/extension mixing rejected'], real_data_opened=False, model_forward=False, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths = ('flow-run', 'flow-protocol', 'pose-run', 'camera-run', 'camera-protocol', 'history-inputs',
             'radar-run', 'motion-run', 'motion-analysis', 'train-index')
    for key in paths:
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qa-only', action='store_true')
    a = parser.parse_args()
    rules, parent, parent_rules, radar, helper, pose, pose_rules = dependencies()
    if a.qa_only:
        result = analytic_qa(helper, radar)
    else:
        require_frozen_bindings(rules)
        require(all(getattr(a, key.replace('-', '_')) is not None for key in paths), 'All ten input paths required')
        require(sha(a.pose_run / 'complete.json') == POSE_COMPLETE_SHA, 'Wrong completed historical pose reference')
        a.pose_protocol = POSE_PROTOCOL_PATH
        rr, mr, nr, cr, scenes, prior, sources, core, times = parent.authenticate(a, parent_rules, radar)
        pr = pose.authenticate_pose(a, pose_rules, radar, cr, mr, sources)
        fr = authenticate_flow(a, rules, radar, cr, mr, sources)
        arrays, counts, coverage = collect_flow(a, parent, radar, pose, rr, mr, nr, cr, fr, pr, scenes, prior, core, times)
        sources.update(analysis_source_sha256=sha(__file__), rules_sha256=sha(RULES_PATH))
        result = dict(schema=SCHEMA, status='COMPLETE_CPU_HISTORY_FROZEN_FLOW_TRAIN_EVIDENCE_ANALYSIS',
            sources=sources, rules=rules, samples=512, scenes=256, source_points=1740053, object_horizon_rows=43189,
            scene_tokens=scenes, sampling_coverage=coverage, horizons=summarize(helper, radar, arrays, counts, prior),
            unchanged_full_physical_reference=dict(source_sha256=radar.MOTION_ANALYSIS_SHA, physical=prior['physical'], recomputed_by_this_analysis=False),
            verification=dict(original_support_join=True, original_future_denominators_complete=True,
                photo_valid_and_low_texture_reported_separately=True, FB_does_not_filter_scores=True,
                existing_pose_reference_only=True, raw_images_reopened=False, raw_GT_reopened=False,
                fitting=False, threshold_selection=False, bootstrap=False, model_forward=False, optimizer_updates=0))
    require('torch' not in sys.modules, 'CPU NumPy-only analysis')
    result.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  analysis_source_sha256=sha(__file__), rules_sha256=sha(RULES_PATH))
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(out=str(a.out), sha256=sha(a.out), status=result['status'])))


if __name__ == '__main__':
    main()
