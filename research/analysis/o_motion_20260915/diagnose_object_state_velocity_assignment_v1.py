"""Unexecuted same-final-V N/P/Z assignment diagnostic; check is the default.

No training, detector inference, support filtering, threshold or checkpoint
selection. Run requires an actual final CPU audit PASS and explicit resources.
Every condition reruns the full native future rollout from the same inputs.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = 'object-state-velocity-assignment-diagnostic-v1'
CONDITIONS = ('N', 'P', 'Z')
PAIRS = (('P', 'N'), ('Z', 'N'))
VERIFIER_SHA = 'a17905b6a61ae3f32573cecb48ab5dbc67f1545f55be0221dfcd7c1694877098'
PROTOCOL_SHA = '93bd80fa938739f40d46a13339a888a4bf1e377385e5cf269ce644f60982ccc9'
EXTRA = {
    'verify_object_state_final_checkpoints_v1.py': VERIFIER_SHA,
    'summarize_future_state_common_v1.py': '23f6fc0a6d7a9c63a1e77f90ca94a2aedc72b68b82b201df4f34c6d3f3123e8a',
    'summarize_future_state_physical_v1.py': '920f4f96e0bd5df247b3d096bdf1709d80fed593622b70373093ea1160db2de3',
    'summarize_common_change_v3.py': '338f9c0bb28b800711e6527d9208c3046132a5c652805a92333e4c209c529918',
    'summarize_connected_motion_v1.py': '3b0822c7d4c9218211dfec7c7d6854fd67b28f9b43640134dfe2228690a2c40e',
}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def jsonl(path): return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def append(path, rows):
    with Path(path).open('a') as stream:
        for row in rows: stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def source_path(root, name):
    require(Path(name).name == name, 'Dependency must be a basename')
    for path in (Path(root) / name, Path(root).parent / 'm0_improvement_20260915' / name):
        if path.is_file(): return path
    raise FileNotFoundError(name)


def module(root, name, digest):
    path = source_path(root, name + '.py')
    require(sha(path) == digest, 'Frozen dependency changed: ' + name)
    if name in sys.modules:
        result = sys.modules[name]
        require(Path(result.__file__).resolve() == path.resolve(), 'Imported dependency path differs: ' + name)
        return result
    spec = importlib.util.spec_from_file_location(name, path); result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result; spec.loader.exec_module(result); return result


def assignment_numpy(states, valid, sample_token, original_indices):
    """Only the last three slots change; hash-cycle uses no RNG or labels."""
    require(states.dtype == np.float32 and states.ndim == 3 and states.shape[0] == 1 and states.shape[2] == 27 and
            valid.dtype == np.bool_ and valid.shape == states.shape[:2], 'Packed state boundary differs')
    positions = np.flatnonzero(valid[0]).tolist(); count = len(positions)
    require(len(original_indices) == count and len(set(original_indices)) == count and
            all(type(i) is int and i >= 0 for i in original_indices), 'Original exported box indices differ')
    require(np.isfinite(states).all(), 'Nonfinite packed state')
    order = sorted(range(count), key=lambda i: (hashlib.sha256(
        ('velocity-assignment-v1|' + sample_token + '|' + str(original_indices[i])).encode()).digest(), original_indices[i]))
    donor = list(range(count))
    if count >= 2:
        for j, recipient in enumerate(order): donor[recipient] = order[(j + 1) % count]
        require(all(i != j for i, j in enumerate(donor)), 'Cycle unexpectedly has a fixed index')
    variants = {name: states.copy() for name in CONDITIONS}
    for i, position in enumerate(positions):
        variants['P'][0, position, 24:27] = states[0, positions[donor[i]], 24:27]
        variants['Z'][0, position, 24:27] = 0.
    original_velocity = states[0, positions, 24:27]
    multiset = lambda x: sorted(row.tobytes() for row in x)
    require(multiset(variants['P'][0, positions, 24:27]) == multiset(original_velocity), 'P velocity multiset changed')
    stats = {}
    for name, value in variants.items():
        require(value[..., :24].tobytes() == states[..., :24].tobytes() and
                value[~valid].tobytes() == states[~valid].tobytes(), 'Geometry or padding changed')
        velocity = value[0, positions, 24:27]; delta = (velocity.astype(np.float64) - original_velocity) * 20.
        magnitudes = np.linalg.norm(delta, axis=1)
        stats[name] = dict(actual_numeric_velocity_changes=int(np.any(velocity != original_velocity, axis=1).sum()),
            byte_changed_velocities=sum(a.tobytes() != b.tobytes() for a, b in zip(velocity, original_velocity)),
            max_velocity_change_mps=float(magnitudes.max()) if count else 0.,
            mean_velocity_change_mps=float(magnitudes.mean()) if count else 0.,
            packed_states_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return variants, dict(valid_objects=count, zero_or_one_object_no_permutation=count < 2,
        reassigned_object_indices=count if count >= 2 else 0, original_box_indices=original_indices,
        hash_sorted_original_indices=[original_indices[i] for i in order],
        donor_original_box_indices=[original_indices[i] for i in donor],
        geometry_and_validity_unchanged=True, P_velocity_multiset_byte_exact=True,
        all_three_R_velocity_components_preserved_by_permutation=True, conditions=stats)


def authenticate(a):
    require(sha(a.protocol) == PROTOCOL_SHA, 'Only the fixed completed V/G protocol is allowed')
    root = Path(a.training_run); p = read(a.protocol)
    require(sha(root / 'complete.json') == a.complete_sha256, 'Explicit final completion SHA differs')
    bindings = dict(p['sources_sha256'], **p['analysis_sources_sha256'], **EXTRA)
    for name, digest in bindings.items(): require(sha(source_path(a.source_root, name)) == digest, 'Source changed: ' + name)
    auth = module(a.source_root, 'summarize_object_state_common_v1', bindings['summarize_object_state_common_v1.py'])
    rows, metadata = auth.authenticate_training(root, a.protocol, a.selection, a.o_reference,
        Path(a.sparse_labels) / 'manifest.json', Path(a.raw_labels) / 'manifest.json')
    audit_root = Path(a.cpu_audit); require(sha(audit_root / 'complete.json') == a.cpu_audit_complete_sha256,
                                         'Explicit real CPU audit completion SHA differs')
    done = read(audit_root / 'complete.json')
    require(done['schema'] == 'object-state-final-checkpoint-cpu-audit-v1' and
            done['status'] == 'PASS_ACTUAL_CPU_FINAL_TENSORS' and set(done['files_sha256']) == {'audit.json', 'report.md'} and
            not (audit_root / 'failed.json').exists(), 'A successful final tensor CPU audit is mandatory')
    for name, digest in done['files_sha256'].items(): require(sha(audit_root / name) == digest, 'CPU audit artifact changed')
    audit = read(audit_root / 'audit.json'); complete = read(root / 'complete.json'); manifest = read(root / 'manifest.json')
    require(audit['status'] == done['status'] and audit['source_sha256'] == VERIFIER_SHA and
            audit['protocol_sha256'] == PROTOCOL_SHA and audit['complete_sha256'] == a.complete_sha256 and
            audit['optimizer_updates'] == 0 and audit['cuda_initialized'] is False, 'CPU tensor audit belongs to another endpoint')
    for arm in ('V', 'G'):
        require(audit['arms'][arm]['checkpoint_sha256'] == complete['final_checkpoints'][arm]['sha256'], 'Audited final checkpoint differs')
    original_objects = [r for r in jsonl(root / 'development_objects.jsonl') if r['arm'] == 'V']
    require(len(original_objects) == 16074, 'Original V must have the complete 16074 object-horizon support')
    per_token = {}
    for row in original_objects: per_token.setdefault(row['sample_token'], []).append(row)
    evidence = dict(training_complete_sha256=a.complete_sha256, training_protocol_sha256=PROTOCOL_SHA,
        cpu_audit_complete_sha256=a.cpu_audit_complete_sha256, cpu_audit_sha256=sha(audit_root / 'audit.json'),
        source_sha256=sha(__file__), dependencies_sha256=bindings, completed_training_metadata=metadata,
        condition_definitions={'N': 'same final V, original state/velocity assignment',
            'P': 'same V, sample-token/original-box-index hash cycle reassigns only packed R velocity slots 24:27',
            'Z': 'same V, only packed R velocity slots 24:27 zero; not separately trained G'},
        full_native_rollout_per_condition=True, all_use_velocity=True, fixed_nominal_horizon_seconds=[.5, 1., 1.5, 2.],
        support='all fixed dev200/100 scenes, all 16074 original valid sparse object-horizon rows per condition',
        training=False, new_performance_gate=False, permutation_seed_search=False)
    return p, bindings, manifest, audit, rows, per_token, evidence


def load_final(a, p, bindings, manifest, audit, torch):
    get = lambda name: module(a.source_root, name, bindings[name + '.py'])
    helper, native, oracle = (get(n) for n in ('train_source_motion_v1', 'native_state_cache', 'oracle_transport_probe'))
    trainer, common, connected = (get(n) for n in ('object_state_forecast_train_v1', 'common_change_evaluation_v2', 'train_connected_motion_v2'))
    require(sha(a.config) == oracle.CONFIG_SHA and sha(a.checkpoint) == oracle.M0_SHA and
            sha(a.o_checkpoint) == oracle.O_SHA and sha(a.runtime_contract) == p['runtime_source_contract_sha256'] and
            sha(a.connected_protocol) == p['connected_protocol_sha256'], 'Native model or D assets differ')
    for path, digest in read(a.runtime_contract)['runtime_source_sha256'].items(): require(sha(path) == digest, 'Native runtime source changed')
    verifier = get('verify_object_state_final_checkpoints_v1')
    model = native.build_native_model(a.config, a.checkpoint, device=a.device, repo=a.repo)
    original = verifier.load_cpu(a.o_checkpoint, oracle.O_SHA, torch)
    model.future_pred_head.load_state_dict(original['future_pred_head'], strict=True); del original
    require(common.historical_O_head_digest(model.future_pred_head) == common.O_HEAD_SHA and
            helper.state_digest(model) == manifest['initial_O_state_sha256'] and
            trainer.pre.non_head_digest(model) == manifest['frozen_non_head_state_sha256'], 'Actual initial O template differs')
    core, condition_core = get('future_state_motion_v1'), get('object_state_conditioner_v2')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(11); readout = core.FutureStateMotionReadout(); conditioner = condition_core.ObjectStateConditioner()
    require(helper.state_digest(readout) == manifest['initial_readout_sha256'] and
            helper.state_digest(conditioner) == manifest['initial_conditioner_sha256'], 'Fresh module schema/initialization differs')
    final = read(Path(a.training_run) / 'complete.json')['final_checkpoints']['V']
    payload = verifier.load_cpu(Path(a.training_run) / 'runs/V/final.pth', final['sha256'], torch)
    require(payload['arm'] == 'V' and payload['status'] == 'FIXED_FINAL_512' and payload['update'] == 512 and
            payload['examples'] == 2048 and payload['sources'] == manifest['sources'] and
            payload['common_manifest_sha256'] == sha(Path(a.training_run) / 'manifest.json') and
            payload['manifest_sha256'] == sha(Path(a.training_run) / 'runs/V/manifest.json'), 'Final V payload provenance differs')
    for key, network in (('future_pred_head', model.future_pred_head), ('motion_readout', readout), ('conditioner', conditioner)):
        # CPU references preserve native parameter shapes; inspect_state also hashes every actual tensor.
        expected = {n: v.detach().cpu() for n, v in network.state_dict().items()}
        result = verifier.inspect_state(payload[key], expected, [n for n, _ in network.named_parameters()], torch)
        require(result == audit['arms']['V']['components'][key], 'Actual V tensor differs from completed CPU audit: ' + key)
        network.load_state_dict(payload[key], strict=True)
    del payload
    D, gate, d_receipt = connected.load_completed_arm(Path(a.connected_training_run) / 'runs/D', Path(a.connected_protocol), helper, a.device)
    del gate
    require(d_receipt == read(Path(a.training_run) / 'loaded_models.json')['D'], 'Actual frozen D receipt differs')
    for network in (model, readout, conditioner, D):
        for parameter in network.parameters(): parameter.requires_grad_(False)
        network.to(a.device).eval()
    require(trainer.pre.non_head_digest(model) == manifest['frozen_non_head_state_sha256'], 'Loading V changed frozen observer')
    loaded = dict(V_checkpoint_sha256=final['sha256'], actual_V_components=audit['arms']['V']['components'], D=d_receipt,
        actual_V_full_state_sha256=helper.state_digest(model), actual_readout_state_sha256=helper.state_digest(readout),
        actual_conditioner_state_sha256=helper.state_digest(conditioner), actual_D_state_sha256=helper.state_digest(D),
        frozen_non_head_O_state_sha256=trainer.pre.non_head_digest(model), optimizer_restored=False, optimizer_updates=0)
    return model, readout, conditioner, D, loaded


def differences(reference, current, torch, tree):
    require(reference.shape == current.shape and reference.dtype == current.dtype, 'Comparison tensor boundary differs')
    records = []
    for i, (left, right) in enumerate(zip(reference, current)):
        delta = (right - left).abs()
        records.append(dict(index=i, elements=left.numel(), numeric_changed=int((left != right).sum()),
            max_abs=float(delta.max()), mean_abs=float(delta.double().mean()),
            byte_equal=tree.tree_digest(left) == tree.tree_digest(right)))
    return records


def summarize(rows, objects, a, bindings):
    cm = module(a.source_root, 'summarize_future_state_common_v1', EXTRA['summarize_future_state_common_v1.py'])
    pm = module(a.source_root, 'summarize_future_state_physical_v1', EXTRA['summarize_future_state_physical_v1.py'])
    counts = cm.prior_math(); scenes = sorted({r['scene_token'] for r in rows}); lookup = {s: i for i, s in enumerate(scenes)}
    length = counts.N_OCC + counts.N_CHANGE + counts.N_GROUP
    sums = {name: np.zeros((len(scenes), length), dtype=np.int64) for name in CONDITIONS}
    for row in rows:
        for name in CONDITIONS: sums[name][lookup[row['scene_token']]] += counts.count_vector(row['metrics_by_arm'][name], row)
    common = cm.aggregate_counts(CONDITIONS, scenes, sums, PAIRS, repetitions=10000, seed=11)
    grouped, keys = pm.physical_rows(objects, {r['sample_token']: r['scene_token'] for r in rows}, CONDITIONS)
    require(len(keys) == 16074, 'Complete common physical support required')
    # check_manifest names its reference slot D; alias N only for this GT-count validator, never as a prediction result.
    support = pm.check_manifest(rows, {'D': grouped['N']}, Path(a.sparse_labels) / 'manifest.json')
    physical = pm.aggregate_physical(grouped, keys, CONDITIONS, scenes, PAIRS, repetitions=10000, seed=11)
    return dict(common=common, physical=physical, support=support, samples=200, scenes=100, physical_rows_per_condition=16074,
        comparisons=['P-minus-N', 'Z-minus-N'], bootstrap=dict(repetitions=10000, seed=11, unit='paired whole scene',
            interval='percentile95', includes_permutation_uncertainty=False, includes_training_seed_uncertainty=False),
        limitations=['same checkpoint posthoc sensitivity/value diagnostic; no new candidate selection',
            'P breaks joint velocity-position-class relations and may be OOD; not randomized causal identification',
            'Z is same final V with zero input velocities, not the separately trained G',
            'physical labels are original rigid-box material-point proxies; no native O flow claim'])


def run(a, context, out, started):
    p, bindings, manifest, audit, references, original_objects, evidence = context
    require(all(getattr(a, name) for name in ('config', 'checkpoint', 'o_checkpoint', 'repo', 'runtime_contract',
        'connected_protocol', 'connected_training_run', 'dev_cache', 'development_predictions', 'device')),
        'Explicit actual model/cache/prediction/runtime paths and device are required for run')
    require(a.max_seconds is not None and math.isfinite(a.max_seconds) and 0 < a.max_seconds <= 42600 and
            a.max_allocated_gib is not None and math.isfinite(a.max_allocated_gib) and 0 < a.max_allocated_gib <= 32,
            'Explicit finite wall/allocated-memory ceilings required; outer detached runner is separately required')
    get = lambda name: module(a.source_root, name, bindings[name + '.py'])
    helper, native, common, oracle, tree, trainer = (get(n) for n in ('train_source_motion_v1', 'native_state_cache',
        'common_change_evaluation_v2', 'oracle_transport_probe', 'joint_native_evaluation', 'object_state_forecast_train_v1'))
    dev_root, dev, cache_receipt = helper.cache_index(a.dev_cache, 'development', p)
    label_root, labels = helper.labels_manifest(a.sparse_labels, p)
    require(all(common.identity(r) == common.identity(t) for r, t in zip(dev, references)), 'Fixed dev identities differ')
    for record in dev: helper.check_identity(record, labels[record['sample_token']])
    prediction_inputs = get('object_state_forecast_development_inputs_v1').DevelopmentPredictionInputs(
        a.development_predictions, a.raw_labels, a.selection, dev, p['development_predictions'])
    raw_root = Path(a.raw_labels); raw_desc = {r['identity']['sample_token']: r for r in read(raw_root / 'manifest.json')['records']}
    import torch
    require(a.device.startswith('cuda') and torch.cuda.is_available(), 'Native run requires the explicitly selected CUDA device')
    torch.cuda.set_device(a.device); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(min(1., a.max_allocated_gib * 2**30 / torch.cuda.get_device_properties(a.device).total_memory), a.device)
    torch.cuda.reset_peak_memory_stats(a.device); stopped = []
    signal.signal(signal.SIGTERM, lambda s, f: stopped.append(s)); signal.signal(signal.SIGINT, lambda s, f: stopped.append(s))
    def check():
        require(not stopped and time.monotonic() - started < a.max_seconds, 'Signal or wall-clock ceiling; no retry')
        require(torch.cuda.max_memory_allocated(a.device) <= a.max_allocated_gib * 2**30, 'Allocated-memory ceiling')
    check(); model, readout, conditioner, D, loaded = load_final(a, p, bindings, manifest, audit, torch); check()
    core, condition_core, metric = get('future_state_motion_v1'), get('object_state_conditioner_v2'), get('common_occupancy_change_metrics_v1')
    write(out / 'loaded_models.json', loaded)
    write(out / 'manifest.json', dict(schema=SCHEMA, **evidence, dev_cache=cache_receipt, predicted_states=prediction_inputs.receipt,
        resources=dict(max_seconds=a.max_seconds, max_allocated_gib=a.max_allocated_gib),
        current_pose_carrier_read_before_predictions=True, annotations_or_future_poses_not_model_inputs=True,
        t0_same_final_V_self_parity=True, original_O_t0_equality_required=False, status='DIAGNOSTIC_ONLY'))
    rows, objects, assignment_rows = [], [], []; trainer.pre.seed_all(torch)
    with torch.no_grad():
        for ordinal, record in enumerate(dev):
            check(); tick = time.monotonic(); sample = common.input_only(native, dev_root, record, a.device)
            input_digest = tree.tree_digest(sample['inputs']); base = D(sample['inputs']['prev_bev_input'][:, -1])
            states, valid, state_receipt = prediction_inputs.get(ordinal, condition_core, a.device)
            state_digest, valid_digest = tree.tree_digest(states), tree.tree_digest(valid)
            variants, assignment = assignment_numpy(states.cpu().numpy(), valid.cpu().numpy(), record['sample_token'], state_receipt['retained_original_box_indices'])
            assignment.update(ordinal=ordinal, sample_token=record['sample_token'], scene_token=record['scene_token'])
            paired = trainer.pre.rng_state(torch); before_rng = trainer.pre.rng_digest(paired)
            predictions, flows, hooks, tensor_differences = {}, {}, {}, {}; reference_features = None
            for name in CONDITIONS:
                check(); trainer.pre.restore_rng(torch, paired)
                changed = torch.as_tensor(variants[name], device=a.device)
                with condition_core.condition_future_head(model.future_pred_head, conditioner, changed, valid, use_velocity=True) as hooks[name]:
                    predictions[name], features = core.capture_native_terminal(model, sample, native, training=False)
                flows[name] = readout(features, base, detach_features=False)
                require(torch.isfinite(predictions[name]).all() and torch.isfinite(features).all() and torch.isfinite(flows[name]).all(), 'Nonfinite prediction')
                require(trainer.pre.rng_digest(trainer.pre.rng_state(torch)) == before_rng, 'Eval replay unexpectedly consumed RNG')
                if name == 'N': reference_features = features
                else:
                    require(tree.tree_digest(predictions[name][0]) == tree.tree_digest(predictions['N'][0]), 'Same-weight t0 native prediction changed')
                    tensor_differences[name + '-minus-N'] = dict(
                        native_logits_by_horizon=differences(predictions['N'], predictions[name], torch, tree),
                        terminal_features_by_horizon=differences(reference_features, features, torch, tree),
                        physical_field_by_future_horizon=differences(flows['N'][0], flows[name][0], torch, tree))
                del features, changed
            del reference_features
            require(tree.tree_digest(sample['inputs']) == input_digest and tree.tree_digest(states) == state_digest and
                    tree.tree_digest(valid) == valid_digest, 'Shared inputs mutated')
            # All three complete predictions exist before loading any scoring GT or sparse labels.
            gt = common.load_targets_after_predictions(native, dev_root, record, sample, a.device)
            descriptor = labels[record['sample_token']]; label = helper.load_sparse(label_root, descriptor, record)
            rd = raw_desc[record['sample_token']]; raw_path = raw_root / rd['file']
            require(sha(raw_path) == rd['sha256'], 'Raw scoring label changed')
            raw_bytes = gzip.decompress(raw_path.read_bytes()); require(hashlib.sha256(raw_bytes).hexdigest() == rd['uncompressed_json_sha256'], 'Raw label decoding changed')
            raw = json.loads(raw_bytes); hist, metrics, nobjects = {}, {}, {}
            for name in CONDITIONS:
                check(); hist[name] = common.native_hist(model, predictions[name], sample, record)
                binary = common.fine_binary(predictions[name], oracle, check)
                metrics[name] = metric.evaluate_common_occupancy_change(binary, gt, oracle.EXTENT, raw); del binary
                require([h['occupancy']['confusion'] for h in metrics[name]['horizons']] == hist[name], 'CPU/native full-GT confusion differs')
                values = [dict(r, arm=name, ordinal=ordinal) for r in helper.epe_records(helper.gather_sparse(flows[name], label), label, record)]
                if name == 'N':
                    require(hist[name] == references[ordinal]['hist_by_arm']['V'] and metrics[name] == references[ordinal]['metrics_by_arm']['V'],
                            'N failed exact original final V occupancy/common reproduction: ' + record['sample_token'])
                    expected = [dict(r, arm='N') for r in original_objects.get(record['sample_token'], [])]
                    require(values == expected, 'N failed exact original final V per-object reproduction: ' + record['sample_token'])
                nobjects[name] = len(values); append(out / 'physical_objects.jsonl', values); objects.extend(values)
            require(len(set(nobjects.values())) == 1, 'Physical object support differs across conditions')
            trainer.verify_common_denominators({'O': metrics['N'], 'V': metrics['P'], 'G': metrics['Z']})
            row = dict(ordinal=ordinal, **common.identity(record), hist_by_arm=hist, metrics_by_arm=metrics,
                inputs_sha256=record['files']['inputs']['sha256'], targets_sha256=record['files']['targets']['sha256'],
                raw_label_sha256=rd['sha256'], sparse_label_sha256=descriptor['sha256'], physical_object_rows_by_arm=nobjects,
                N_original_V_hist_and_objects_exact=True, native_CPU_hist_exact=True, GT_common_denominators_exact=True,
                t0_prediction_policy='same_final_V_self_parity', conditioner_hooks=hooks, state_source=state_receipt,
                assignment=assignment, tensor_differences=tensor_differences, seconds=time.monotonic() - tick)
            append(out / 'records.jsonl', [row]); rows.append(row); assignment_rows.append(assignment)
            print(json.dumps(dict(samples=ordinal + 1, seconds=time.monotonic() - started)), flush=True)
            del sample, base, states, valid, variants, predictions, flows, gt, label, raw, raw_bytes
    require(len(rows) == 200 and all(sum(r['physical_object_rows_by_arm'][n] for r in rows) == 16074 for n in CONDITIONS), 'Incomplete fixed support')
    for network, key in ((model, 'actual_V_full_state_sha256'), (readout, 'actual_readout_state_sha256'),
                         (conditioner, 'actual_conditioner_state_sha256'), (D, 'actual_D_state_sha256')):
        require(helper.state_digest(network) == loaded[key], 'Evaluation changed a fixed model tensor')
    check(); summary = summarize(rows, objects, a, bindings); check()
    summary.update(schema=SCHEMA, status='COMPLETE_SAME_WEIGHT_VELOCITY_DIAGNOSTIC',
        assignments=dict(anchors_N0=sum(r['valid_objects'] == 0 for r in assignment_rows), anchors_N1=sum(r['valid_objects'] == 1 for r in assignment_rows),
            reassigned_objects=sum(r['reassigned_object_indices'] for r in assignment_rows),
            actual_velocity_changes={n: sum(r['conditions'][n]['actual_numeric_velocity_changes'] for r in assignment_rows) for n in CONDITIONS}),
        optimizer_updates=0, resources=dict(seconds=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
                                           peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device)))
    write(out / 'summary.json', summary)
    report = '# 同一最终 V 的速度分配诊断\n\n完整 dev200/100 scenes、每条件原 16074 对象—时域支持；N 原最终 V 精确重现。'
    report += '\n\nP−N、Z−N 的完整占据变化/分组与物理 EPE 见 summary.json，10,000 次 seed11 场景配对区间。'
    report += '\n\nP 保留帧内速度多重集但破坏联合关系，可能 OOD；Z 不是独立训练的 G。敏感性不是准确性或唯一因果归因。无训练、调参或新增晋级门。\n'
    (out / 'report.md').write_text(report)
    for name, digest in bindings.items(): require(sha(source_path(a.source_root, name)) == digest, 'Source changed during run')
    require(sha(Path(a.training_run) / 'complete.json') == a.complete_sha256, 'Training endpoint changed')
    check(); files = ('manifest.json', 'loaded_models.json', 'records.jsonl', 'physical_objects.jsonl', 'summary.json', 'report.md')
    write(out / 'complete.json', dict(schema=SCHEMA, status=summary['status'], samples=200, scenes=100,
        physical_rows_per_condition=16074, optimizer_updates=0, training_complete_sha256=a.complete_sha256,
        cpu_audit_complete_sha256=a.cpu_audit_complete_sha256, source_sha256=sha(__file__), files_sha256={n: sha(out / n) for n in files}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'run'), default='check')
    for name in ('source-root', 'training-run', 'complete-sha256', 'protocol', 'cpu-audit', 'cpu-audit-complete-sha256',
                 'selection', 'o-reference', 'sparse-labels', 'raw-labels'):
        parser.add_argument('--' + name, required=True)
    for name in ('config', 'checkpoint', 'o-checkpoint', 'repo', 'runtime-contract', 'connected-protocol',
                 'connected-training-run', 'dev-cache', 'development-predictions', 'device', 'out'):
        parser.add_argument('--' + name)
    parser.add_argument('--max-seconds', type=float); parser.add_argument('--max-allocated-gib', type=float)
    a = parser.parse_args(); started = time.monotonic(); context = authenticate(a)
    if a.mode == 'check':
        print(json.dumps(dict(status='SOURCE_AND_COMPLETED_ENDPOINT_ONLY', sources=context[-1],
            Torch_imported='torch' in sys.modules, GPU_called=False, diagnostic_run_performed=False), indent=2)); return
    require(a.out, 'Explicit new output required'); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    try: run(a, context, out, started)
    except Exception as error:
        write(out / 'failed.json', dict(schema=SCHEMA, status='FAIL_NO_RETRY', error=repr(error), seconds=time.monotonic() - started)); raise


if __name__ == '__main__': main()
