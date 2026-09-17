"""Strict full5119 merge; frozen scientific summaries, independent label coverage.

No training, GPU, threshold selection, candidate promotion or partial-result
summary. Every completed shard and every physical label must authenticate
before the unchanged scientific functions are invoked.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

SCHEMA = 'full-connected-motion-merge-v1'
EVALUATOR_SHA = '6f8656a249973bd2a45cb781086170eb8bbc72821ecff0556c576728bc82c1a2'
TRAINING_PROTOCOL_SHA = 'e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
SPARSE_MANIFEST_SHA = '59535fe4f21e73182da5090e6b12ec60a0dfb40382564f0bb16409565699fc4e'
SPARSE_COMPLETE_SHA = '68fc77c6fe474fe5ed722e07be0341731a15a3adbb215dc5ecba453e06fec545'
PINNED = {
    'native_full_o_stream_v1.py': '3a8d338de78caae5d73e11012cd9cec38a47621913ddbb0bbda9cfae6e510bef',
    'summarize_connected_motion_v1.py': '3b0822c7d4c9218211dfec7c7d6854fd67b28f9b43640134dfe2228690a2c40e',
    'summarize_common_change_v2.py': '318dee1a19fbd5b0863f50d2e0346723d7050329b55170cb8efac60929b962ae',
    'common_occupancy_change_metrics_v1.py': '0ca592ca0b41ecf3524a0e49d8414e35518f4f9e50ccbb50937af8fded666951',
    'train_source_motion_v1.py': 'e34efdfd6dca7b60a8a4ddb3359417bbe5ba9e7b04a4b150e8b6428b81dd582e',
}
PHYSICAL_ARMS = ('P', 'J', 'D')
OCCUPANCY_ARMS = ('O', 'J', 'D')
GROUPS = ('stationary', 'ambiguous', 'moving')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def read(path, expected=None):
    if expected is not None:
        require(sha(path) == expected, 'File SHA mismatch: ' + str(path))
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def json_rows(path):
    with Path(path).open() as f:
        for number, line in enumerate(f, 1):
            require(bool(line.strip()), 'Blank JSONL row: '+str(path)+':'+str(number))
            row = json.loads(line, parse_constant=reject_constant)
            require(type(row) is dict, 'JSONL row must be an object')
            yield row


def source_path(name):
    require(Path(name).name == name and name.endswith('.py'), 'Source must be a Python basename')
    for path in (Path(__file__).with_name(name), Path(__file__).parent.parent/'m0_improvement_20260915'/name):
        if path.is_file():
            return path
    raise FileNotFoundError(name)


def import_bound(name, digest):
    path = source_path(name + '.py')
    require(sha(path) == digest, 'Frozen source changed: '+name)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve() == path.resolve(), 'Conflicting module alias: '+name)
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_files(root, completion, names):
    require(set(completion['files_sha256']) == set(names), 'Incomplete/unexpected completion file set: '+str(root))
    for name in names:
        require(Path(name).name == name and sha(root/name) == completion['files_sha256'][name], 'Completed artifact changed: '+str(root/name))
    return dict(completion['files_sha256'])


def sparse_contract(root, selection):
    root = Path(root).resolve()
    manifest = read(root/'manifest.json', SPARSE_MANIFEST_SHA)
    done = read(root/'complete.json', SPARSE_COMPLETE_SHA)
    require(manifest['schema'] == 'sparse-rigid-box-motion-manifest-v1'
            and done['schema'] == 'sparse-rigid-box-motion-complete-v1'
            and manifest['status'] == done['status'] == 'COMPLETE'
            and manifest['samples'] == done['samples'] == 5119
            and manifest['scenes'] == done['scenes'] == 150
            and done['manifest_sha256'] == SPARSE_MANIFEST_SHA
            and done['selection_sha256'] == '60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391'
            and done['all5119_NPZ_byte_roundtrip_verified'] is True
            and done['optimizer_steps'] == 0 and done['model_or_prediction_read'] is False,
            'Incomplete/full sparse label contract')
    for name, key in (('statistics.json','statistics_sha256'),('material_point_audit.json','material_point_audit_sha256'),
                      ('development_intersection_audit.json','development_intersection_audit_sha256')):
        require(sha(root/name) == done[key], 'Sparse auxiliary receipt changed: '+name)
    descriptors = manifest['records']
    require(len(descriptors) == 5119 and [r['ordinal'] for r in descriptors] == list(range(5119)), 'Sparse official ordinal order')
    for i, (row, wanted) in enumerate(zip(descriptors, selection['records'])):
        require(row['identity'] == wanted and row['file'] == 'validation/'+wanted['sample_token']+'.npz', 'Sparse identity/path changed: '+str(i))
    return root, descriptors


def expected_support(label):
    """Reconstruct physical row support from masks/assignment, not result rows."""
    indices, valid = label['object_index'], label['valid']
    tokens, groups, dt = label['instance_tokens'], label['object_speed_group'], label['dt_future_seconds']
    expected = {}
    for h in range(4):
        counts = np.bincount(indices[valid[h]], minlength=len(tokens))
        for obj in np.flatnonzero(counts):
            group = int(groups[h, obj])
            require(0 <= group < 3, 'Valid point has no physical speed group')
            expected[(str(tokens[obj]), .5*(h+1))] = dict(group=GROUPS[group],
                source_points=int(counts[obj]), dt_seconds=float(dt[h]))
    return expected


def validate_object_anchor(rows, record, expected):
    """Reject shared omissions, extra/duplicate rows, wrong groups/support/dt."""
    seen = {arm:set() for arm in PHYSICAL_ARMS}
    for row in rows:
        require(row['arm'] in seen and row['sample_token'] == record['sample_token']
                and row['scene_token'] == record['scene_token'], 'Physical arm/anchor/scene mismatch')
        key = (row['instance_token'], row['horizon_seconds'])
        require(key in expected and key not in seen[row['arm']], 'Unexpected/duplicate physical instance-horizon')
        label = expected[key]
        require(type(row['source_points']) is int and row['source_points'] == label['source_points']
                and row['group'] == label['group'] and row['dt_seconds'] == label['dt_seconds'], 'Physical source support/group/actual dt differs')
        for metric in ('epe_xy_m','epe_3d_m','zero_epe_xy_m','zero_epe_3d_m'):
            require(np.isfinite(row[metric]) and row[metric] >= 0, 'Invalid physical error: '+metric)
        seen[row['arm']].add(key)
        # Equal bytes/values already checked; reuse identity strings to limit
        # full1.15M-row memory without changing a scientific value or row order.
        row['sample_token'] = record['sample_token']; row['scene_token'] = record['scene_token']
        row['group'] = label['group']
    require(all(keys == set(expected) for keys in seen.values()), 'Incomplete physical coverage, including shared omissions')
    return dict(objects_horizons_per_arm=len(expected), physical_rows=len(rows),
                source_point_horizons_per_arm=sum(v['source_points'] for v in expected.values()),
                empty_anchor=not expected)


class ObjectCursor:
    """Consume one sample's contiguous object rows without loading shards twice."""
    def __init__(self, path):
        self.iterator = iter(json_rows(path)); self.next = next(self.iterator, None)

    def take(self, token):
        rows = []
        while self.next is not None and self.next['sample_token'] == token:
            rows.append(self.next); self.next = next(self.iterator, None)
        return rows

    def finish(self):
        require(self.next is None, 'Unexpected/unordered physical object rows after final anchor')


def validate_loaded(loaded, manifest, training_done, training_manifest, arm_completions, protocol, stream):
    require(loaded['optimizer_updates'] == 0 and loaded['optimizer_restored'] is False
            and set(loaded['arms']) == {'J','D'}, 'Evaluation restored optimizer or missing final arms')
    native = loaded['native_O']
    require(native['O_checkpoint_sha256'] == stream.O_SHA and native['O_head_state_sha256'] == stream.O_HEAD_SHA
            and native['training'] is False and native['optimizer_created'] is False
            and native['observation_matmul_tf32'] is True and native['future_matmul_tf32'] is False
            and native['cudnn_tf32'] is True
            and loaded['O_full_helper_list_shape_state_sha256'] == training_manifest['frozen_O_state_sha256'], 'Loaded O source/state/precision differs')
    require(loaded['P'] == dict(checkpoint_sha256=protocol['motion']['checkpoint_sha256'],
            actual_motion_state_sha256=protocol['motion']['head_state_sha256'],
            complete_sha256=protocol['motion']['complete_sha256'], optimizer_restored=False), 'Loaded P differs')
    for arm in ('J','D'):
        actual = loaded['arms'][arm]; final = training_done['final_checkpoints'][arm]
        require(actual == dict(schema='connected-motion-training-v2',arm=arm,
            checkpoint_sha256=final['sha256'], actual_motion_state_sha256=final['motion_state_sha256'],
            actual_gate_state_sha256=final['gate_state_sha256'],
            manifest_sha256=arm_completions[arm]['files_sha256']['manifest.json'],
            common_manifest_sha256=manifest['training_manifest_sha256'],
            top_complete_sha256=manifest['training_complete_sha256'],
            arm_complete_sha256=training_done['arm_complete_sha256'][arm],
            protocol_sha256=TRAINING_PROTOCOL_SHA, fixed_final_update=512, fixed_examples=2048,
            fixed_development_samples=200, actual_optimizer_parameter_steps_all512=True,
            optimizer_restored=False, optimizer_updates_in_this_evaluation=0), 'Actual strict final loader receipt differs: '+arm)


def validate_record(row, ordinal, reference, sparse_descriptor, previous, summarizer):
    identity = reference['selection']['records'][ordinal]
    require(row['schema'] == 'full-connected-motion-evaluation-v1'
            and row['ordinal'] == row['selection_ordinal'] == row['official_index'] == ordinal
            and all(row[k] == identity[k] for k in identity)
            and set(row['metrics_by_arm']) == set(row['hist_by_arm']) == set(OCCUPANCY_ARMS)
            and set(row['physical_object_rows_by_arm']) == set(PHYSICAL_ARMS), 'Full record identity/arms/order mismatch')
    require(row['raw_label_sha256'] == sparse_descriptor['label_source_sha256']
            and row['sparse_label_sha256'] == sparse_descriptor['sha256'], 'Evaluation label source differs')
    for key in ('original_full_O_hist_exact','original_full_O_all_layer_logits_exact','native_CPU_hist_exact',
                't0_all_models_exact','GT_never_used_as_prediction_input'):
        require(row[key] is True, 'Missing full row engineering proof: '+key)
    require(row['optimizer_updates'] == 0, 'Evaluation optimization update')
    audit = row['native_stream_audit']; ledger = reference['ledger'][ordinal]
    require(audit['schema'] == 'native-full-o-stream-v1' and audit['identity'] == ledger['identity']
            and audit['targets_array_sha256'] == ledger['targets_array_sha256']
            and audit['observed_bev_tensor_sha256'] == ledger['observed_bev_tensor_sha256']
            and audit['native_radar_tensor_sha256'] == ledger['native_radar_tensor_sha256']
            and audit['O_logits_array_sha256'] == ledger['logits_array_sha256']['O']
            and audit['historical_full_input_tree_sha256'] == ledger['input_tree_sha256']
            and audit['historical_full_input_hash_equal'] == (audit['input_tree_sha256'] == ledger['input_tree_sha256'])
            and audit['historical_full_active_tree_comparison_performed'] is False
            and audit['original_full_O_hist_exact'] is True
            and audit['original_full_O_all5h_3layer_logit_bytes_exact'] is True
            and audit['planning_guard'] == {'O':dict(turn_on_plan=False,all_modules_eval=True)}
            and audit['optimizer_steps'] == 0 and audit['inputs_gt_separate'] is True
            and audit['GT_loaded_by_original_dataset_before_prediction'] is True, 'Native full parity/source audit differs')
    require(audit['target_counts_0_1_255_by_frame'] == reference['counts'][ordinal].tolist()
            and audit['hist_by_horizon'] == row['hist_by_arm']['O'] == reference['hist'][ordinal].tolist(), 'Original full O histogram or GT is not exact')
    for arm in OCCUPANCY_ARMS:
        hist = np.asarray(row['hist_by_arm'][arm])
        require(hist.shape == (5,2,2) and hist.dtype.kind in 'iu' and np.all(hist >= 0)
                and np.array_equal(hist.sum(-1), reference['counts'][ordinal,2:,:2])
                and row['hist_by_arm'][arm][0] == row['hist_by_arm']['O'][0], 'GT confusion orientation/t0 differs')
        vector = summarizer.count_vector(row['metrics_by_arm'][arm], row)
        require(summarizer.unpack(vector)[0].tolist() == hist.tolist(), 'Common CPU/native original histogram differs')
    has_previous = row['sample_token'] in previous
    require(row['is_training_development_intersection'] is has_previous
            and row['training_final_hist_exact'] is (True if has_previous else None), 'Historical dev intersection flag differs')
    if has_previous:
        require(row['hist_by_arm'] == previous[row['sample_token']]['hist_by_arm'], 'Final development intersection histogram changed')


def validate_partition_headers(headers):
    require(len(headers) in (1,2), 'Require one or two completed shards')
    count = len(headers)
    require(sorted(h['shard_index'] for h in headers) == list(range(count)), 'Duplicate or missing shard index')
    for h in headers:
        i = h['shard_index']; start, stop = 5119*i//count, 5119*(i+1)//count
        require(h['mode'] == 'full' and h['status'] == 'COMPLETE_FULL_CONNECTED_MOTION_SHARD'
                and h['shard_count'] == count and h['ordinal_start'] == start
                and h['ordinal_stop_exclusive'] == stop and h['planned_ordinals'] == list(range(start,stop))
                and h['processed_samples'] == stop-start, 'Partial/wrong full continuous partition')


def collect(a):
    """Only authenticate/read completed inputs here; no scientific summaries yet."""
    stream = import_bound('native_full_o_stream_v1', PINNED['native_full_o_stream_v1.py'])
    for name, digest in PINNED.items():
        require(sha(source_path(name)) == digest, 'Changed merge mathematical dependency: '+name)
    import_bound('common_occupancy_change_metrics_v1', PINNED['common_occupancy_change_metrics_v1.py'])
    math_common = import_bound('summarize_common_change_v2', PINNED['summarize_common_change_v2.py'])
    math_motion = import_bound('summarize_connected_motion_v1', PINNED['summarize_connected_motion_v1.py'])
    helper = import_bound('train_source_motion_v1', PINNED['train_source_motion_v1.py'])
    # The evaluator SHA is supplied explicitly by the caller's approved source
    # receipt, then matched to every shard and the actual imported file.
    require(a.evaluator_sha256 == EVALUATOR_SHA, 'Actual frozen evaluator SHA required')
    evaluator = import_bound('full_connected_motion_evaluation_v1', a.evaluator_sha256)
    protocol = read(a.protocol, TRAINING_PROTOCOL_SHA)
    require(protocol['status'] == 'FROZEN' and protocol['schema'] == 'connected-motion-training-v2', 'Wrong training protocol')
    _, _, modules, evaluation_sources = evaluator.imports(protocol)
    engineering = evaluator.stream_engineering(a.stream_engineering, stream)
    reference = stream.load_full_reference(a.reference_dir, a.selection)
    sparse_root, sparse_descriptors = sparse_contract(a.sparse_labels, reference['selection'])
    training_root, training_done, training_manifest, previous = evaluator.training_metadata(
        a.training_run, a.training_complete_sha256, protocol, modules['train_connected_motion_v2'], reference['selection'])
    require(training_done['actual_optimizer_updates_by_arm'] == {arm:dict(motion=512,gate=512) for arm in ('J','D')}, 'Unmatched final optimization budgets')
    arm_completions = {}
    for arm in ('J','D'):
        path = training_root/'runs'/arm
        require(not (path/'failed.json').exists(), 'Failed training arm')
        completion = read(path/'complete.json', training_done['arm_complete_sha256'][arm])
        require(completion['schema'] == 'connected-motion-training-v2' and completion['mode'] == 'train'
                and completion['arm'] == arm and completion['status'] == training_done['status']
                and completion['updates'] == 512 and completion['examples'] == 2048 and completion['evaluated_samples'] == 200
                and completion['optimizer_updates'] == dict(motion=512,gate=512), 'Incomplete final arm receipt')
        # Actual tensor/payload loading was independently done by each evaluator;
        # the CPU merger verifies its exact loader receipt and small source chain.
        for name in ('manifest.json','training.jsonl','development_records.jsonl'):
            require(sha(path/name) == completion['files_sha256'][name], 'Training small artifact changed: '+arm+'/'+name)
        final = training_done['final_checkpoints'][arm]
        require(final == dict(file='runs/'+arm+'/final.pth',sha256=completion['files_sha256']['final.pth'],
            motion_state_sha256=completion['final_motion_sha256'],gate_state_sha256=completion['final_gate_sha256']), 'Training final pointer differs')
        arm_completions[arm] = completion
    shards = [Path(p).resolve() for p in a.shards]
    require(len(shards) in (1,2) and len(set(shards)) == len(shards), 'Require one or two distinct complete shards')
    require(all(not (p/'failed.json').exists() for p in shards), 'Failed shard cannot merge')
    validate_partition_headers([read(p/'complete.json') for p in shards])
    shards = sorted(shards, key=lambda p:read(p/'complete.json')['shard_index'])
    records = []; objects = []; shard_bindings = []; coverage_ledger = []
    common_manifest = None; common_loaded = None
    object_totals = {arm:0 for arm in PHYSICAL_ARMS}; histogram_totals = {arm:np.zeros((5,2,2),dtype=np.int64) for arm in OCCUPANCY_ARMS}
    for shard_index, directory in enumerate(shards):
        done = read(directory/'complete.json')
        start, stop = 5119*shard_index//len(shards), 5119*(shard_index+1)//len(shards)
        wanted_ordinals = list(range(start,stop))
        require(done['schema'] == evaluator.SCHEMA == 'full-connected-motion-evaluation-v1'
                and done['status'] == 'COMPLETE_FULL_CONNECTED_MOTION_SHARD' and done['mode'] == 'full'
                and done['shard_count'] == len(shards) and done['shard_index'] == shard_index
                and done['ordinal_start'] == start and done['ordinal_stop_exclusive'] == stop
                and done['planned_ordinals'] == wanted_ordinals and done['processed_samples'] == stop-start
                and done['full_samples'] == 5119 and done['full_scenes'] == 150
                and done['arms'] == list(OCCUPANCY_ARMS) and done['physical_arms'] == list(PHYSICAL_ARMS)
                and done['optimizer_updates'] == 0 and done['protocol_sha256'] == TRAINING_PROTOCOL_SHA
                and done['training_complete_sha256'] == a.training_complete_sha256
                and done['source_sha256'] == a.evaluator_sha256 and done['stream_sha256'] == PINNED['native_full_o_stream_v1.py']
                and done['full_selection_sha256'] == stream.FULL_SELECTION_SHA, 'Shard completion/partition/source contract differs')
        for key in ('all_original_full_O_hist_exact','all_original_full_O_all_layer_logits_exact','all_native_CPU_hist_exact','all_t0_models_exact'):
            require(done[key] is True, 'Shard missing parity completion: '+key)
        files = checked_files(directory, done, ('manifest.json','loaded_models.json','records.jsonl','physical_objects.jsonl','summary.json'))
        manifest = read(directory/'manifest.json'); loaded = read(directory/'loaded_models.json'); summary = read(directory/'summary.json')
        for key in ('schema','mode','source_sha256','stream_sha256','protocol_sha256','training_complete_sha256',
                    'full_selection_sha256','shard_index','shard_count','ordinal_start','ordinal_stop_exclusive','planned_ordinals','full_samples','full_scenes','arms','physical_arms','optimizer_updates'):
            require(manifest[key] == done[key], 'Manifest/complete differs: '+key)
        require(manifest['sources_sha256'] == evaluation_sources and manifest['planned_samples'] == stop-start
                and manifest['stream_engineering'] == engineering
                and manifest['training_manifest_sha256'] == training_done['files_sha256']['manifest.json']
                and manifest['training_development_records_sha256'] == training_done['files_sha256']['development_records.jsonl']
                and manifest['fixed_final_receipts'] == training_done['final_checkpoints']
                and manifest['P_binding'] == protocol['motion']
                and manifest['original_full_reference_files_sha256'] == reference['verified_files_sha256']
                and manifest['sparse_labels_manifest_sha256'] == SPARSE_MANIFEST_SHA
                and manifest['sparse_labels_complete_sha256'] == SPARSE_COMPLETE_SHA
                and manifest['raw_labels_manifest_sha256'] == evaluator.RAW_MANIFEST_SHA
                and manifest['raw_labels_complete_sha256'] == evaluator.RAW_COMPLETE_SHA
                and manifest['horizon_seconds'] == [0.,.5,1.,1.5,2.]
                and manifest['numerical_policy'] == protocol['numerical_policy']
                and manifest['observation_matmul_tf32'] is True and manifest['training'] is False
                and manifest['seed'] == 11 and manifest['no_threshold_selection'] is True
                and manifest['GT_never_used_as_prediction_input'] is True
                and manifest['GT_loaded_by_original_dataset_before_prediction'] is True
                and manifest['raw_and_sparse_label_contents_loaded_after_predictions'] is True
                and manifest['dense_sample_cache_written'] is False and manifest['historical_validation_exposure'] is True
                and manifest['pilot_selection_sha256'] is None, 'Manifest scientific source/label/precision contract differs')
        validate_loaded(loaded, manifest, training_done, training_manifest, arm_completions, protocol, stream)
        canonical_manifest = {k:v for k,v in manifest.items() if k not in ('shard_index','ordinal_start','ordinal_stop_exclusive','planned_ordinals','planned_samples')}
        # Device/init-time fields vary between cards; exact model identities must not.
        canonical_loaded = dict(O=loaded['O_full_helper_list_shape_state_sha256'], P=loaded['P'], arms=loaded['arms'])
        require(common_manifest is None or canonical_manifest == common_manifest, 'Shards used different scientific/resource contracts')
        require(common_loaded is None or canonical_loaded == common_loaded, 'Shards loaded different final tensors')
        common_manifest, common_loaded = canonical_manifest, canonical_loaded
        cursor = ObjectCursor(directory/'physical_objects.jsonl')
        counts = {arm:0 for arm in PHYSICAL_ARMS}; hists = {arm:np.zeros((5,2,2),dtype=np.int64) for arm in OCCUPANCY_ARMS}
        scenes = set(); actual_ordinals = []; intersections = 0
        for row in json_rows(directory/'records.jsonl'):
            ordinal = start+len(actual_ordinals)
            require(ordinal < stop, 'Shard has excess anchors')
            descriptor = sparse_descriptors[ordinal]
            validate_record(row, ordinal, reference, descriptor, previous, math_common)
            label = helper.load_sparse(sparse_root, descriptor, descriptor['identity'])
            expected = expected_support(label)
            physical = cursor.take(row['sample_token'])
            audit = validate_object_anchor(physical, row, expected)
            require(row['physical_object_rows_by_arm'] == {arm:len(expected) for arm in PHYSICAL_ARMS}, 'Reported physical row counts do not equal actual labels')
            for physical_row in physical:
                require(physical_row['ordinal'] == physical_row['selection_ordinal'] == physical_row['official_index'] == ordinal
                        and physical_row['split'] == 'validation', 'Physical ordinal/split differs')
            coverage_ledger.append(dict(ordinal=ordinal,sample_token=row['sample_token'],scene_token=row['scene_token'],
                NPZ_sha256=descriptor['sha256'], **audit))
            objects.extend(physical); records.append(row); actual_ordinals.append(ordinal); scenes.add(row['scene_token'])
            intersections += row['is_training_development_intersection']
            for arm in PHYSICAL_ARMS: counts[arm] += len(expected)
            for arm in OCCUPANCY_ARMS: hists[arm] += np.asarray(row['hist_by_arm'][arm],dtype=np.int64)
            del label, expected, physical
        cursor.finish()
        require(actual_ordinals == wanted_ordinals and len(scenes) == done['scenes'], 'Incomplete shard rows/scenes')
        require(summary['schema'] == evaluator.SCHEMA and summary['mode'] == 'full' and summary['status'] == done['status']
                and summary['processed_samples'] == stop-start and summary['scenes'] == len(scenes)
                and summary['physical_object_rows_by_arm'] == counts
                and summary['histogram_sums_by_arm'] == {k:v.tolist() for k,v in hists.items()}
                and summary['training_development_intersections_exact'] == intersections
                and summary['stream_final']['completed_samples'] == stop-start
                and summary['stream_final']['model_unchanged'] is True and summary['stream_final']['optimizer_steps'] == 0
                and summary['optimizer_updates'] == 0 and summary['bootstrap_performed'] is False
                and summary['model_promotion_decided'] is False, 'Shard summary not equal to exact completed records')
        for arm in PHYSICAL_ARMS: object_totals[arm] += counts[arm]
        for arm in OCCUPANCY_ARMS: histogram_totals[arm] += hists[arm]
        shard_bindings.append(dict(directory=str(directory), complete_sha256=sha(directory/'complete.json'),
            files_sha256=files, shard_index=shard_index, ordinal_start=start, ordinal_stop_exclusive=stop,
            samples=stop-start, scenes=len(scenes)))
    require([r['official_index'] for r in records] == list(range(5119))
            and len({r['sample_token'] for r in records}) == 5119
            and len({r['scene_token'] for r in records}) == 150
            and sum(r['is_training_development_intersection'] for r in records) == 200, 'Full official population/200 intersection incomplete')
    return dict(records=records, objects=objects, math_common=math_common, math_motion=math_motion,
        coverage=coverage_ledger, bindings=dict(schema=SCHEMA, evaluation_source_sha256=a.evaluator_sha256,
            stream_source_sha256=PINNED['native_full_o_stream_v1.py'], stream_engineering=engineering, protocol_sha256=TRAINING_PROTOCOL_SHA,
            training_complete_sha256=a.training_complete_sha256, training_manifest_sha256=sha(training_root/'manifest.json'),
            fixed_final_receipts=training_done['final_checkpoints'], P_binding=protocol['motion'],
            mathematical_sources_sha256=PINNED, evaluation_sources_sha256=evaluation_sources,
            full_selection_sha256=stream.FULL_SELECTION_SHA, reference_files_sha256=reference['verified_files_sha256'],
            sparse_manifest_sha256=SPARSE_MANIFEST_SHA,sparse_complete_sha256=SPARSE_COMPLETE_SHA,
            shards=shard_bindings, physical_object_rows_by_arm=object_totals,
            histogram_sums_by_arm={k:v.tolist() for k,v in histogram_totals.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', nargs='+', required=True)
    for name in ('evaluator-sha256','protocol','stream-engineering','training-run','training-complete-sha256','selection','reference-dir','sparse-labels','out'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    out = Path(a.out); require(not out.exists(), 'Do not overwrite an analysis directory')
    started = time.monotonic()
    # Do not create a results directory or calculate a partial scientific result
    # when any completion, row, label, identity, or source gate fails.
    inputs = collect(a)
    joined = inputs['math_motion'].summary(inputs['records'],inputs['objects'],repetitions=10000,seed=11)
    common = inputs['math_common'].summarize(inputs['records'],bootstrap_repetitions=10000,seed=11)
    require(joined['anchors'] == common['samples'] == 5119 and joined['scenes'] == common['scenes'] == 150,
            'Frozen scientific summaries changed population')
    for arm in OCCUPANCY_ARMS:
        for key, value in joined['original_occupancy'][arm].items():
            require(value == common['values'][arm][key], 'The two frozen summaries disagree on original occupancy')
    for pair, values in joined['occupancy_comparisons'].items():
        for key, item in values.items():
            other = common['comparisons'][pair][key]
            require(all(item[k] == other[k] for k in ('difference','unit','lower95','upper95')), 'Frozen paired-scene occupancy contrast differs')
    summary = dict(schema=SCHEMA,status='COMPLETE_FULL_CONNECTED_MOTION_MERGE',samples=5119,scenes=150,
        joined_occupancy_physical=joined,common_output=common,physical_coverage=dict(anchors=5119,
            physical_rows_per_arm=joined['physical_rows_per_arm'],empty_anchors=sum(r['empty_anchor'] for r in inputs['coverage']),
            every_P_J_D_instance_horizon_checked_against_actual_NPZ=True,
            group_source_points_actual_dt_exact=True,shared_omissions_rejected=True),
        source_sha256=sha(__file__),optimizer_updates=0,model_selection_performed=False,
        single_training_seed=11,historical_validation_exposure=True,new_blind_test=False,
        confidence_interval='10000 paired whole-scene draws seed11; percentile95; no training-seed uncertainty or multiplicity correction',
        interpretation='Original GMO, common occupancy/change and physical source-head EPE are separate claims; O has no native flow.',
        seconds=time.monotonic()-started)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'input_bindings.json',inputs['bindings'])
    write(out/'physical_coverage.json',dict(schema=SCHEMA,records=inputs['coverage']))
    write(out/'summary.json',summary)
    (out/'report.md').write_text('# Full O/J/D 合并统计\n\n'
        '完整 5119 anchors / 150 官方 validation scenes；所有分片、原 O 五时域混淆与标签覆盖核验通过。\n\n'
        '原生 GMO 先按时域汇总混淆再计算 IoU，主统计为四未来时域算术均值；t0、FP/FN 保留。'
        '物理 P/J/D 对象点均值 EPE 与共同占据变化/速度分组召回分开报告；P 不是 O 原生 flow。\n\n'
        '固定单训练 seed11；10000 次同场景配对 bootstrap；无多重比较校正。validation 已有历史曝光，'
        '不是新盲测，也不覆盖训练种子不确定性。本程序不选择或晋级模型。\n')
    write(out/'complete.json',dict(schema=SCHEMA,status=summary['status'],samples=5119,scenes=150,
        source_sha256=sha(__file__),protocol_sha256=TRAINING_PROTOCOL_SHA,
        training_complete_sha256=a.training_complete_sha256,optimizer_updates=0,
        files_sha256={name:sha(out/name) for name in ('summary.json','report.md','input_bindings.json','physical_coverage.json')},
        seconds=time.monotonic()-started))
    print(json.dumps(dict(status=summary['status'],samples=5119,scenes=150,out=str(out),seconds=time.monotonic()-started)))


if __name__ == '__main__':
    main()
