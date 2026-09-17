"""Final cached dev200 evaluation of frozen O and completed J/D models.

No training or model selection. Reuses the authenticated common-change v2
input-only, historical O digest, target loading, fine argmax, native evaluator,
and descriptive summary helpers. GT/raw-box contents are read only after all
three predictions have been computed. Exact previous final J/D confusion is
a required gate; CUDA scatter differences are reported, never tolerated away.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np

SCHEMA = 'common-connected-motion-evaluation-v1'
ARMS = ('O', 'J', 'D')
PINNED = {
    'common_change_evaluation_v2.py': 'b73a616c16a1765aeb71e2922c8ba512972b38566f74363a391b0f2e57fc2059',
    'common_occupancy_change_metrics_v1.py': '0ca592ca0b41ecf3524a0e49d8414e35518f4f9e50ccbb50937af8fded666951',
    'motion_geometry.py': 'e9d232c3f7aaacd073cfda645868e357afad9e5a2684d07b2f3f2394bd796a31',
    'supported_motion_fusion.py': '60b1a248ae3237f6f55686cf11349bd85d7c1b98495c714ca55f828da38d140c',
    'train_supported_fusion_v2.py': '61cb7c96fc8127f4b3b5ef34fff380e0c2076210463cff29e4cee5d3a5d87d7c',
    'transport_ops.py': '18708c7b33de30c31f684d7a3d7d39cf754c9b54e91ff87da894d9d14140406c',
    'train_source_motion_v1.py': 'e34efdfd6dca7b60a8a4ddb3359417bbe5ba9e7b04a4b150e8b6428b81dd582e',
    'native_state_cache.py': '41953909ddfdfd08d51d98385a450640e5314b775ab01458cb55ff1b18e4bfb0',
    'oracle_transport_probe.py': '2ca37ca1a3cbfd335890fc315c9e1f058f91f00325f72d29eda62dd66396d95b',
}
RUNTIME_SHA = '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c'
LABEL_MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
LABEL_COMPLETE_SHA = 'e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69'
DEV_REFERENCE_SHA = '3f426891f6811a595542a1e9694783521ae89cf18d6385896012e5f52a8e9b9a'
O_HEAD_SHA = '1816cc040b3068b6e5ed3591aec6b8b919d2b3bd8b54b2186e4311ed2fa284fb'
IDENTITY = ('sample_token', 'scene_token', 'split', 'official_index')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and set(value) <= set('0123456789abcdef') and len(set(value)) > 1)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def identity(record):
    return {k: record[k] for k in IDENTITY}


def source_path(name):
    require(Path(name).name == name and name.endswith('.py'), 'Source must be Python basename')
    for path in (Path(__file__).parent / name,
                 Path(__file__).parent.parent / 'm0_improvement_20260915' / name):
        if path.is_file():
            return path
    raise FileNotFoundError(name)


def import_bound(name, sources):
    path = source_path(name + '.py')
    require(sha(path) == sources[path.name], 'Import source changed: ' + name)
    if name in sys.modules:
        existing = sys.modules[name]
        require(Path(existing.__file__).resolve() == path.resolve(), 'Module alias uses other source: ' + name)
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_sources(protocol):
    bindings = dict(protocol['sources_sha256'])
    require('train_connected_motion_v1.py' in bindings, 'Protocol must bind actual J/D trainer')
    for name, expected in PINNED.items():
        require(name not in bindings or bindings[name] == expected, 'Frozen scoring/source contract changed: ' + name)
        bindings[name] = expected
    # Evaluator is independently versioned; its SHA is retained even if the
    # training protocol froze before this read-only evaluator was completed.
    if Path(__file__).name in bindings:
        require(bindings[Path(__file__).name] == sha(__file__), 'Evaluator protocol binding changed')
    bindings[Path(__file__).name] = sha(__file__)
    for name, expected in bindings.items():
        require(valid_sha(expected) and sha(source_path(name)) == expected, 'Source changed: ' + name)
    return bindings


def label_descriptors(root, selected, common):
    root = Path(root).resolve()
    require(sha(root / 'manifest.json') == LABEL_MANIFEST_SHA
            and sha(root / 'complete.json') == LABEL_COMPLETE_SHA, 'Frozen raw labels changed')
    done = read(root / 'complete.json'); manifest = read(root / 'manifest.json')
    require(done['schema'] == 'raw-nuscenes-motion-target-complete-v1'
            and done['status'] == 'COMPLETE' and done['samples'] == 712
            and done['manifest_sha256'] == LABEL_MANIFEST_SHA
            and done['optimizer_steps'] == 0 and not done['model_or_prediction_read']
            and manifest['train_dev_scenes_disjoint'] and manifest['inputs_and_labels_physically_separate'], 'Raw label isolation/completion')
    index = {row['identity']['sample_token']: row for row in manifest['records']}
    require(len(index) == len(manifest['records']) == 712, 'Raw label identity duplicates')
    result = []
    for record in selected:
        descriptor = index[record['sample_token']]
        require(descriptor['identity'] == common.identity(record)
                and descriptor['file'] == 'development/' + record['sample_token'] + '.json.gz', 'Wrong raw label identity')
        result.append(descriptor)
    return root, result


def reject_hist_difference(out, ordinal, token, arm, actual, expected):
    """Small exact difference receipt before fail-closed exit; never adjust output."""
    if actual != expected:
        a = np.asarray(actual, dtype=np.int64); b = np.asarray(expected, dtype=np.int64)
        write(out / 'hist_parity_failure.json', dict(schema=SCHEMA,
            status='FAILED_EXACT_TRAINING_FINAL_HISTOGRAM', ordinal=ordinal,
            sample_token=token, arm=arm, actual=actual, expected=expected,
            delta=(a-b).tolist(), optimizer_updates=0,
            note='Possible scatter reduction difference is not hidden or automatically retried.'))
        raise ValueError('Exact prior final histogram differs: ' + arm + '/' + token)


def validate_training(root, complete_sha, p, protocol_sha, trainer, selected, train_receipt, dev_receipt):
    root = Path(root).resolve()
    require(sha(root / 'complete.json') == complete_sha, 'Require actual approved training completion SHA')
    require(not (root / 'failed.json').exists(), 'Training failed')
    done = read(root / 'complete.json'); manifest = read(root / 'manifest.json')
    require(done['schema'] == trainer.SCHEMA == 'connected-motion-training-v1'
            and done['mode'] == 'train' and done['status'] == 'COMPLETE_CONNECTED_MOTION_TRAINING'
            and done['updates'] == 512 and done['examples'] == 2048 and done['evaluated_samples'] == 200,
            'Require fully completed final512 plus final dev200, not preflight or interrupted weights')
    require(set(done['files_sha256']) == {'manifest.json', 'engineering.json', 'summary.json',
            'development_records.jsonl', 'development_objects.jsonl'}
            and set(done['arm_complete_sha256']) == set(done['final_checkpoints']) == {'J', 'D'},
            'Incomplete top-level training chain')
    require(done['actual_optimizer_updates_by_arm'] == {arm: dict(motion=512, gate=512) for arm in ('J', 'D')},
            'Both final optimizers must have 512 actual updates')
    for name, digest in done['files_sha256'].items():
        require(sha(root / name) == digest, 'Training artifact changed: ' + name)
    require(manifest['schema'] == trainer.SCHEMA and manifest['mode'] == 'train'
            and manifest['arms'] == ['J', 'D'] and manifest['training'] == trainer.TRAINING == p['training']
            and manifest['numerical_policy'] == p['numerical_policy']
            and manifest['sources']['protocol_sha256'] == protocol_sha
            and manifest['sources']['sources_sha256'] == p['sources_sha256']
            and manifest['sources']['train_cache'] == train_receipt
            and manifest['sources']['development_cache'] == dev_receipt
            and manifest['sources']['O_development_records_sha256'] == DEV_REFERENCE_SHA
            and manifest['sources']['motion'] == p['motion'] and manifest['sources']['labels'] == p['labels']
            and manifest['planned_updates'] == 512 and manifest['single_training_seed'] == 11
            and manifest['same_initial_parameters'] and manifest['independent_optimizers'],
            'Final training sources/recipe/cache differ')
    for arm in ('J', 'D'):
        armroot = root / 'runs' / arm
        require(not (armroot / 'failed.json').exists()
                and sha(armroot / 'complete.json') == done['arm_complete_sha256'][arm], 'Arm completion binding')
        complete = read(armroot / 'complete.json')
        require(complete['schema'] == trainer.SCHEMA and complete['arm'] == arm
                and complete['status'] == done['status'] and complete['mode'] == 'train'
                and complete['updates'] == 512 and complete['examples'] == 2048
                and complete['evaluated_samples'] == 200
                and complete['optimizer_updates'] == dict(motion=512, gate=512), 'Not fixed final arm')
        # load_completed_arm authenticates every actual file/payload/state.
        require(valid_sha(done['final_checkpoints'][arm]['sha256'])
                and complete['files_sha256']['final.pth'] == done['final_checkpoints'][arm]['sha256']
                and complete['final_motion_sha256'] == done['final_checkpoints'][arm]['motion_state_sha256']
                and complete['final_gate_sha256'] == done['final_checkpoints'][arm]['gate_state_sha256'],
                'Final checkpoint receipt mismatch')
    summary = read(root / 'summary.json')
    require(summary['mode'] == 'train' and summary['updates'] == 512 and summary['examples'] == 2048
            and summary['evaluated_samples'] == 200 and summary['weights_persisted'] is True
            and summary['all_gradients_finite'] and summary['O_and_pretrained_parameters_buffers_unchanged']
            and summary['actual_optimizer_updates_by_arm'] == done['actual_optimizer_updates_by_arm'], 'Training summary incomplete')
    rows = [json.loads(line) for line in (root / 'development_records.jsonl').read_text().splitlines()]
    require(len(rows) == len(selected) == 200, 'Need all final development records')
    for i, (row, record) in enumerate(zip(rows, selected)):
        require(row['ordinal'] == i and identity(row) == identity(record)
                and set(row['hist_by_arm']) == set(ARMS)
                and row['inputs_sha256'] == record['files']['inputs']['sha256']
                and row['targets_sha256'] == record['files']['targets']['sha256'], 'Final development identity/input mismatch')
        for hist in row['hist_by_arm'].values():
            array = np.asarray(hist)
            require(array.shape == (5, 2, 2) and array.dtype.kind in 'iu' and np.all(array >= 0), 'Invalid final confusion')
    return root, done, manifest, rows


def run(a, out, started, stopping):
    require(sha(a.protocol) == a.protocol_sha256, 'Actual frozen protocol bytes differ')
    p = read(a.protocol)
    require(p['status'] == 'FROZEN' and p['schema'] == 'connected-motion-training-v1', 'Frozen J/D protocol required')
    sources = check_sources(p)
    common = import_bound('common_change_evaluation_v2', sources)
    import_bound('motion_geometry', sources)
    metric = import_bound('common_occupancy_change_metrics_v1', sources)
    trainer = import_bound('train_connected_motion_v1', sources)
    helper = import_bound('train_source_motion_v1', sources)
    native = import_bound('native_state_cache', sources)
    oracle = import_bound('oracle_transport_probe', sources)
    fusion = import_bound('supported_motion_fusion', sources)
    composer = import_bound('train_supported_fusion_v2', sources)
    require(p['training'] == trainer.TRAINING and p['numerical_policy'] == trainer.NUMERICAL
            and p['training']['seed'] == 11 and p['training']['updates'] == 512, 'Scientific recipe differs')
    require(sha(a.runtime_contract) == RUNTIME_SHA, 'Runtime contract differs')
    runtime = read(a.runtime_contract)
    for path, digest in runtime['runtime_source_sha256'].items():
        require(sha(path) == digest, 'Frozen native runtime source changed: ' + path)
    require(sha(a.config) == oracle.CONFIG_SHA and sha(a.checkpoint) == oracle.M0_SHA
            and sha(a.o_checkpoint) == oracle.O_SHA, 'O/M0/config differs')
    train_root, train, train_receipt = helper.cache_index(a.train_cache, 'train', p)
    cache, selected, dev_receipt = helper.cache_index(a.dev_cache, 'development', p)
    require(not ({r['scene_token'] for r in train} & {r['scene_token'] for r in selected})
            and len({r['scene_token'] for r in selected}) == 100, 'Wrong/overlapping fixed scene selection')
    require(sha(a.o_development_records) == DEV_REFERENCE_SHA == p['o_development_records_sha256'], 'Frozen O reference differs')
    o_reference = [json.loads(line) for line in Path(a.o_development_records).read_text().splitlines()]
    require(len(o_reference) == 200, 'Require complete O dev200 reference')
    root, train_done, train_manifest, prior_rows = validate_training(
        a.training_run, a.training_complete_sha256, p, a.protocol_sha256, trainer,
        selected, train_receipt, dev_receipt)
    for record, old, prior in zip(selected, o_reference, prior_rows):
        require(all(record[k] == old[k] for k in ('sample_token', 'scene_token'))
                and old['hist_by_horizon'] == prior['hist_by_arm']['O'], 'Prior O identity/parity mismatch')
    labels, descriptors = label_descriptors(a.labels, selected, common)
    manifest = dict(schema=SCHEMA, mode='final_dev200', status='RUNNING', arms=list(ARMS),
        protocol_sha256=a.protocol_sha256, sources_sha256=sources,
        runtime_source_contract_sha256=RUNTIME_SHA, runtime_source_sha256=runtime['runtime_source_sha256'],
        train_cache=train_receipt, sample_cache=dev_receipt,
        anchors=[identity(r) for r in selected],
        training_complete_sha256=a.training_complete_sha256,
        training_manifest_sha256=sha(root / 'manifest.json'),
        training_development_records_sha256=sha(root / 'development_records.jsonl'),
        fixed_final_receipts=train_done['final_checkpoints'],
        O_reference_sha256=DEV_REFERENCE_SHA, O_checkpoint_sha256=oracle.O_SHA,
        M0_checkpoint_sha256=oracle.M0_SHA, config_sha256=oracle.CONFIG_SHA,
        raw_labels_manifest_sha256=LABEL_MANIFEST_SHA, raw_labels_complete_sha256=LABEL_COMPLETE_SHA,
        raw_label_descriptors=descriptors, numerical_policy=p['numerical_policy'],
        resources=dict(max_seconds=a.max_seconds, max_allocated_gib=a.max_allocated_gib),
        optimizer_updates=0, seed=11, training=False, oracle_inference=False,
        target_and_raw_box_load_after_all_model_predictions=True,
        prediction_readout='Unchanged native XYZ full512x512x40 trilinear align_corners=False then argmax',
        no_threshold_selection=True, flow_EPE_or_identity_accuracy_claim=False,
        exposure='historically exposed development200; not unseen or full5119',
        final_training_histogram_parity='EXACT; mismatch receipt then fail, no relaxed tolerance or retry')
    write(out / 'manifest.json', manifest)
    import torch
    require(str(a.device).startswith('cuda'), 'Native O requires CUDA')
    torch.cuda.set_device(a.device); torch.set_num_threads(2)
    random.seed(11); np.random.seed(11); torch.manual_seed(11); torch.cuda.manual_seed_all(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = False
    memory = torch.cuda.get_device_properties(a.device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., a.max_allocated_gib * 2**30 / memory), a.device)
    torch.cuda.reset_peak_memory_stats(a.device)

    def check():
        require(not stopping, 'Signal received; no retry')
        require(time.monotonic() - started < a.max_seconds, 'Evaluation time ceiling')
        require(torch.cuda.max_memory_allocated(a.device) <= a.max_allocated_gib * 2**30, 'Evaluation allocation ceiling')

    check()
    model = native.build_native_model(a.config, a.checkpoint, device=a.device, repo=a.repo)
    payload = torch.load(a.o_checkpoint, map_location='cpu', weights_only=False)
    model.future_pred_head.load_state_dict(payload['future_pred_head'], strict=True); del payload
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.eval()
    require(common.historical_O_head_digest(model.future_pred_head) == O_HEAD_SHA, 'Historical O head digest differs')
    original_sha = helper.state_digest(model)
    require(original_sha == train_manifest['frozen_O_state_sha256'], 'Actual full O state differs from training')
    motions = {}; gates = {}; arm_receipts = {}
    for arm in ('J', 'D'):
        check()
        motion, gate, receipt = trainer.load_completed_arm(root / 'runs' / arm, Path(a.protocol), helper, a.device)
        for module in (motion, gate):
            module.eval()
            for parameter in module.parameters(): parameter.requires_grad_(False)
        # Preserve actual digest independently of producer receipt field names.
        actual_motion = helper.state_digest(motion); actual_gate = helper.state_digest(gate)
        require(actual_motion == train_done['final_checkpoints'][arm]['motion_state_sha256']
                and actual_gate == train_done['final_checkpoints'][arm]['gate_state_sha256'],
                'Actual final motion/gate state does not match completed training')
        motions[arm] = motion; gates[arm] = gate
        arm_receipts[arm] = dict(training_loader_receipt=receipt,
            actual_motion_state_sha256=actual_motion, actual_gate_state_sha256=actual_gate,
            checkpoint_sha256=sha(root / 'runs' / arm / 'final.pth'),
            final_update=512, optimizer_updates_in_this_evaluation=0, optimizer_restored=False)
        require(arm_receipts[arm]['checkpoint_sha256'] == train_done['final_checkpoints'][arm]['sha256'], 'Loaded checkpoint changed')
    write(out / 'loaded_models.json', dict(native=model._native_state_provenance,
        O_head_state_sha256=O_HEAD_SHA, O_head_digest_format='memory_experiment.state_digest: tuple shape',
        O_head_motion_helper_sha256=helper.state_digest(model.future_pred_head), O_full_state_sha256=original_sha,
        arms=arm_receipts, torch=torch.__version__, numpy=np.__version__, optimizer_restored=False, optimizer_updates=0))
    rows = []; initialization_seconds = time.monotonic() - started
    with torch.no_grad(), (out / 'records.jsonl').open('x') as stream:
        for ordinal, (record, descriptor) in enumerate(zip(selected, descriptors)):
            check(); tick = time.monotonic()
            sample = common.input_only(native, cache, record, a.device)
            measured = sample['inputs']['prev_bev_input']
            original = native.replay(model, sample, training=False)[0]
            require(tuple(original.shape) == (5, 3, 1, 1, 40000, 16, 2), 'Original prediction layout differs')
            predictions = {'O': original}
            base = oracle.predictions_to_xyz(original)
            probability = torch.softmax(base[0:1], dim=1)[:, 1:2]
            foreground = base[0:1, 1:2] > base[0:1, 0:1]
            future = (base[1:, 1] - base[1:, 0])[None, :, None]
            for arm in ('J', 'D'):
                # D uses its own learned displacement too. detach affects only
                # training derivatives; it is not a zero-flow inference arm.
                displacement = motions[arm](measured[:, -1])
                field = fusion.supported_transport(probability, foreground, displacement, oracle.EXTENT)
                fused = gates[arm](future, field['probability_mass'], field['support_weight'])
                predictions[arm] = composer.compose_prediction(original, base, fused, oracle)
                del displacement, field, fused
            del probability, foreground, future, base
            require(native._tensor_digest(measured) == record['reference_bev_tensor_sha256'], 'Measured input mutated')
            require(all(composer.exact32(value[0], original[0]) for value in predictions.values()), 't0 logits differ')
            torch.cuda.synchronize(a.device); prediction_seconds = time.monotonic() - tick
            # All O/J/D inference above is complete before first target/box read.
            gt = common.load_targets_after_predictions(native, cache, record, sample, a.device)
            label_path = labels / descriptor['file']
            require(label_path.stat().st_size == descriptor['bytes']
                    and sha(label_path) == descriptor['sha256'], 'Raw label bytes changed')
            label_bytes = gzip.decompress(label_path.read_bytes())
            require(hashlib.sha256(label_bytes).hexdigest() == descriptor['uncompressed_json_sha256'], 'Raw label decoded bytes changed')
            label = json.loads(label_bytes); del label_bytes
            require(label['identity'] == identity(record), 'Raw label identity mismatch')
            results = {}; histograms = {}
            for arm in ARMS:
                check()
                hist = common.native_hist(model, predictions[arm], sample, record)
                if arm == 'O':
                    require(hist == o_reference[ordinal]['hist_by_horizon'], 'O original dev200 native histogram differs')
                reject_hist_difference(out, ordinal, record['sample_token'], arm,
                    hist, prior_rows[ordinal]['hist_by_arm'][arm])
                binary = common.fine_binary(predictions[arm], oracle, check)
                result = metric.evaluate_common_occupancy_change(binary, gt, oracle.EXTENT, label)
                require([h['occupancy']['confusion'] for h in result['horizons']] == hist,
                        'CPU fine decode differs from original evaluator')
                if arm != 'O':
                    require(result['t0_boundary'] == results['O']['t0_boundary'], 'Fine t0 boundary differs')
                    require(all([x['occupancy']['classes'][str(c)]['GT'] for x in result['horizons']]
                            == [x['occupancy']['classes'][str(c)]['GT'] for x in results['O']['horizons']]
                            for c in (0, 1)), 'GT support differs across arms')
                results[arm] = result; histograms[arm] = hist
                del binary
            torch.cuda.synchronize(a.device); check()
            row = dict(ordinal=ordinal, **identity(record), metrics_by_arm=results, hist_by_arm=histograms,
                input_sha256=record['files']['inputs']['sha256'], GT_cache_sha256=record['files']['targets']['sha256'],
                raw_label_sha256=descriptor['sha256'], O_reference_hist_exact=True,
                training_final_hist_exact=True, native_CPU_hist_exact=True, t0_all_models_exact=True,
                GT_raw_labels_first_loaded_after_all_predictions=True,
                prediction_seconds=prediction_seconds, seconds=time.monotonic()-tick,
                peak_allocated_gib=torch.cuda.max_memory_allocated(a.device)/2**30)
            stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
            rows.append(row)
            del sample, measured, original, predictions, gt, label, results
            write(out / 'progress.json', dict(completed_samples=len(rows), planned_samples=200,
                last_token=record['sample_token'], seconds=time.monotonic()-started, optimizer_updates=0))
            print(json.dumps(dict(event='COMMON_CONNECTED_SAMPLE', ordinal=ordinal,
                sample_token=record['sample_token'], seconds=row['seconds'])), flush=True)
    check()
    require(len(rows) == 200 and helper.state_digest(model) == original_sha, 'Incomplete evaluation or O mutated')
    for arm in ('J', 'D'):
        require(helper.state_digest(motions[arm]) == arm_receipts[arm]['actual_motion_state_sha256']
                and helper.state_digest(gates[arm]) == arm_receipts[arm]['actual_gate_state_sha256'], 'Final model state mutated')
    require(sha(a.protocol) == a.protocol_sha256
            and sha(root / 'complete.json') == a.training_complete_sha256
            and check_sources(p) == sources, 'Source/completion changed during evaluation')
    summary = dict(schema=SCHEMA, mode='final_dev200', status='COMPLETE_FINAL_DEVELOPMENT_EVALUATION',
        samples=200, scenes=100, arms=list(ARMS), optimizer_updates=0,
        scores=common.summarize(rows, ARMS, metric), all_O_reference_hist_exact=True,
        all_training_final_hist_exact=True, all_native_CPU_hist_exact=True, all_t0_models_exact=True,
        initialization_seconds=initialization_seconds, seconds=time.monotonic()-started,
        max_peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
        no_flow_EPE_or_identity_claim=True, bootstrap_performed=False,
        description='Same original occupancy/change and GT-only motion-positive risk; historical dev200, not full validation.')
    write(out / 'summary.json', summary)
    (out / 'report.md').write_text('# O/J/D 共同输出评测\n\n'
        '固定开发集 200 anchors / 100 scenes；0 次训练或优化更新。\n\n'
        '原 O 与本轮训练最终 J/D 的全部原生 confusion 精确重现；CPU 原指标一致；t0 严格一致。\n\n'
        '完整 01/10、GT 速度分组与唯一一份全负类 FP/TN 保留；没有阈值选择、GT 裁剪或组件修补。\n\n'
        '不是 flow EPE、身份能力、未见数据或完整 5119 验证。本文件不做 bootstrap 或模型选择。\n')
    check()
    write(out / 'complete.json', dict(schema=SCHEMA, mode='final_dev200', status=summary['status'],
        samples=200, scenes=100, arms=list(ARMS), optimizer_updates=0,
        source_sha256=sha(__file__), protocol_sha256=a.protocol_sha256,
        training_complete_sha256=a.training_complete_sha256,
        files_sha256={name:sha(out / name) for name in
            ('manifest.json', 'loaded_models.json', 'records.jsonl', 'summary.json', 'report.md')},
        seconds=time.monotonic()-started))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('final_dev200',), default='final_dev200')
    for name in ('protocol', 'protocol-sha256', 'training-run', 'training-complete-sha256',
                 'config', 'checkpoint', 'o-checkpoint', 'repo', 'runtime-contract',
                 'train-cache', 'dev-cache', 'o-development-records', 'labels', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--max-allocated-gib', type=float, required=True)
    a = parser.parse_args(argv)
    require(valid_sha(a.protocol_sha256) and valid_sha(a.training_complete_sha256), 'Real protocol and final completion SHA required')
    require(math.isfinite(a.max_seconds) and 0 < a.max_seconds <= 1800
            and math.isfinite(a.max_allocated_gib) and 0 < a.max_allocated_gib <= 32,
            'Evaluation resource cap is 1800 seconds / 32 GiB')
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); stopping = []
    signal.signal(signal.SIGTERM, lambda sig, frame: stopping.append(sig))
    signal.signal(signal.SIGINT, lambda sig, frame: stopping.append(sig))
    try:
        run(a, out, started, stopping)
    except BaseException as exc:
        write(out / 'failed.json', dict(schema=SCHEMA, mode='final_dev200', status='FAILED_NO_RETRY',
            exception=type(exc).__name__, message=str(exc), optimizer_updates=0,
            seconds=time.monotonic()-started, source_sha256=sha(__file__)))
        raise


if __name__ == '__main__':
    main()
