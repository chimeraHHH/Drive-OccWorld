"""Original validation streaming for fixed final O/R/R0 common outputs.

No training, physical-flow scoring, model selection, new coordinate convention
or dense cache. NativeOStream owns the original loader, TF32-on observation
capture, TF32-off O replay, full GT and exact historical O all-logit/hist gates.
The same fresh sample is replayed once more only to capture terminal features;
its O logits must exactly match the stream. This does not repeat the backbone.
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
import traceback

import numpy as np

SCHEMA = 'full-future-feature-residual-evaluation-v1'
ARMS = ('O', 'R', 'R0')
COMMON_SHA = '57cf304bafb050fac05b06abe68ce4600a37ca020d9d10d0673ec0ab1edb3bc1'
STREAM_SHA = '3a8d338de78caae5d73e11012cd9cec38a47621913ddbb0bbda9cfae6e510bef'
STREAM_ENGINEERING_SHA = 'c35bbe651fe0f28f7d8bbd2c7cf02f50dae893b2bba356cf4931beb26dcebe43'
OLD_FULL_SHA = '6f8656a249973bd2a45cb781086170eb8bbc72821ecff0556c576728bc82c1a2'
PROTOCOL_SHA = '45048560948fae975945685e5151db158933b81b94b53b543551213b29a79a56'
CONNECTED_PROTOCOL_SHA = 'e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
LOADER_SHA = 'b97d660db64b37f2e1196896df35128d33e4f331757a46004197bbe12e132e86'
FULL_SELECTION_SHA = '60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391'
RAW_MANIFEST_SHA = '48ac12f1ce24ed1639c17aed0fcdc8ad2eef2cceff10c181b2a0f82f81fc90bb'
RAW_COMPLETE_SHA = 'de10ddd9013d7efd68249bced50448fb083c6d059546f8cfa0560be02b702252'
HORIZONS = [0., .5, 1., 1.5, 2.]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path, expected=None):
    if expected is not None:
        require(sha(path) == expected, 'File SHA mismatch: '+str(path))
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(str(path)+'.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def source_path(name):
    require(Path(name).name == name and name.endswith('.py'), 'Source must be a Python basename')
    for path in (Path(__file__).with_name(name),
                 Path(__file__).parent.parent/'m0_improvement_20260915'/name):
        if path.is_file():
            return path
    raise FileNotFoundError(name)


def old_full_helper():
    path = source_path('full_connected_motion_evaluation_v1.py')
    require(sha(path) == OLD_FULL_SHA, 'Original schedule/stream receipt helper changed')
    name = 'full_connected_motion_evaluation_v1'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    require(Path(sys.modules[name].__file__).resolve() == path.resolve(), 'Wrong original helper alias')
    return sys.modules[name]


def schedule(mode, shard_index, shard_count, selection, pilot=None):
    return old_full_helper().schedule(mode, shard_index, shard_count, selection, pilot)


def stream_engineering(root, stream):
    return old_full_helper().stream_engineering(root, stream)


def imports(protocol):
    path = source_path('common_connected_motion_evaluation_v2.py')
    require(sha(path) == COMMON_SHA, 'Original common input/metric loader changed')
    spec = importlib.util.spec_from_file_location('common_connected_motion_evaluation_v2', path)
    contract = importlib.util.module_from_spec(spec); sys.modules[spec.name] = contract
    spec.loader.exec_module(contract)
    sources = contract.check_sources(protocol)
    require(isinstance(LOADER_SHA, str) and len(LOADER_SHA) == 64, 'Final loader source not yet frozen')
    for name, digest in {'native_full_o_stream_v1.py': STREAM_SHA,
                         'full_connected_motion_evaluation_v1.py': OLD_FULL_SHA,
                         'load_future_feature_residual_v1.py': LOADER_SHA}.items():
        require(name not in sources or sources[name] == digest, 'Conflicting evaluation source: '+name)
        sources[name] = digest
    stream = contract.import_bound('native_full_o_stream_v1', sources)
    for name, digest in stream.SOURCES.items():
        require(name not in sources or sources[name] == digest, 'Conflicting native source: '+name)
        sources[name] = digest
    sources[Path(__file__).name] = sha(__file__)
    for name, digest in sources.items():
        require(sha(source_path(name)) == digest, 'Bound source changed: '+name)
    contract.import_bound('motion_geometry', sources)
    modules = {name: contract.import_bound(name, sources) for name in (
        'train_future_feature_residual_v1', 'train_source_motion_v1', 'native_state_cache',
        'common_change_evaluation_v2', 'common_occupancy_change_metrics_v1',
        'oracle_transport_probe', 'load_future_feature_residual_v1')}
    return contract, stream, modules, sources


def labels_contract(rawroot, selection):
    root = Path(rawroot).resolve()
    manifest = read(root/'manifest.json', RAW_MANIFEST_SHA)
    complete = read(root/'complete.json', RAW_COMPLETE_SHA)
    require(complete['status'] == 'COMPLETE' and complete['samples'] == 5119
            and complete['scenes'] == 150 and complete['manifest_sha256'] == RAW_MANIFEST_SHA
            and manifest['selection']['sha256'] == FULL_SELECTION_SHA, 'Incomplete original full raw labels')
    require(len(manifest['records']) == len(selection['records']) == 5119, 'Wrong full raw-label count')
    for i, (descriptor, identity) in enumerate(zip(manifest['records'], selection['records'])):
        require(descriptor['ordinal'] == i and descriptor['identity'] == identity
                and descriptor['file'] == 'validation/'+identity['sample_token']+'.json.gz',
                'Full raw-label identity/path/order differs')
    return root, manifest['records']


def training_metadata(root, complete_sha, protocol, trainer, selection):
    """CPU small-artifact chain; the strict loader independently verifies weights."""
    root = Path(root).resolve(); done = read(root/'complete.json', complete_sha)
    require(not (root/'failed.json').exists(), 'Training has failure marker')
    require(done['schema'] == trainer.SCHEMA == 'future-feature-residual-training-v1'
            and done['mode'] == 'train' and done['status'] == 'COMPLETE_FUTURE_FEATURE_RESIDUAL_TRAINING'
            and done['updates'] == 512 and done['examples'] == 2048 and done['evaluated_samples'] == 200,
            'Require actual completed fixed-final512/dev200 residual training')
    require(set(done['files_sha256']) == {'manifest.json','loaded_models.json','summary.json','development_records.jsonl'}
            and set(done['arm_complete_sha256']) == set(done['final_checkpoints']) == {'R','R0'},
            'Missing matched residual final artifacts')
    for name, digest in done['files_sha256'].items():
        require(sha(root/name) == digest, 'Training artifact changed: '+name)
    manifest = read(root/'manifest.json')
    require(manifest['sources']['protocol_sha256'] == PROTOCOL_SHA
            and manifest['sources']['sources_sha256'] == protocol['sources_sha256']
            and manifest['training'] == protocol['training'] == trainer.TRAINING
            and manifest['numerical_policy'] == protocol['numerical_policy'] == trainer.NUMERICAL
            and manifest['arms'] == ['R','R0'] and manifest['mode'] == 'train', 'Final training contract differs')
    rows = [json.loads(line) for line in (root/'development_records.jsonl').read_text().splitlines()]
    previous = {row['sample_token']: row for row in rows}
    full = {row['sample_token']: row for row in selection['records']}
    require(len(rows) == len(previous) == 200 and len({row['scene_token'] for row in rows}) == 100,
            'Incomplete residual development identity set')
    for i, row in enumerate(rows):
        require(row['ordinal'] == i and row['split'] == 'development' and row['sample_token'] in full
                and all(row[k] == full[row['sample_token']][k] for k in ('scene_token','official_index'))
                and set(row['hist_by_arm']) == set(row['metrics_by_arm']) == set(ARMS),
                'Final development identity/arms differ')
    return root, done, manifest, previous


def run(a, out, started, stopping):
    require(sha(a.protocol) == a.protocol_sha256 == PROTOCOL_SHA, 'Actual frozen residual protocol required')
    require(sha(a.connected_protocol) == CONNECTED_PROTOCOL_SHA, 'Actual connected D protocol required')
    require(a.stream_sha256 == STREAM_SHA, 'Approved native stream source required')
    p = read(a.protocol)
    require(p['schema'] == 'future-feature-residual-training-v1' and p['status'] == 'FROZEN', 'Wrong residual protocol')
    contract, stream_module, modules, sources = imports(p)
    trainer = modules['train_future_feature_residual_v1']; helper = modules['train_source_motion_v1']
    native = modules['native_state_cache']; common = modules['common_change_evaluation_v2']
    metric = modules['common_occupancy_change_metrics_v1']; oracle = modules['oracle_transport_probe']
    loader = modules['load_future_feature_residual_v1']
    engineering = stream_engineering(a.stream_engineering, stream_module)
    selection = read(a.selection, FULL_SELECTION_SHA)
    pilot = read(a.pilot_selection, stream_module.PILOT_SELECTION_SHA) if a.mode == 'pilot' else None
    ordinals, start, stop = schedule(a.mode, a.shard_index, a.shard_count, selection, pilot)
    rawroot, descriptors = labels_contract(a.raw_labels, selection)
    root, done, training_manifest, previous = training_metadata(a.training_run, a.training_complete_sha256, p, trainer, selection)
    reference = stream_module.load_full_reference(a.reference_dir, a.selection)
    for row in previous.values():
        require(row['hist_by_arm']['O'] == reference['hist'][row['official_index']].tolist(), 'Old O full/dev reference differs')
    manifest = dict(schema=SCHEMA, mode=a.mode, source_sha256=sha(__file__), sources_sha256=sources,
        protocol_sha256=PROTOCOL_SHA, connected_protocol_sha256=CONNECTED_PROTOCOL_SHA,
        stream_sha256=STREAM_SHA, stream_engineering=engineering, loader_sha256=LOADER_SHA,
        training_complete_sha256=a.training_complete_sha256, training_manifest_sha256=sha(root/'manifest.json'),
        training_development_records_sha256=sha(root/'development_records.jsonl'), fixed_final_receipts=done['final_checkpoints'],
        D_checkpoint_sha256=p['D_checkpoint_sha256'], D_training_complete_sha256=p['connected_complete_sha256'],
        full_selection_sha256=FULL_SELECTION_SHA, pilot_selection_sha256=stream_module.PILOT_SELECTION_SHA if pilot else None,
        shard_index=a.shard_index, shard_count=a.shard_count, ordinal_start=start, ordinal_stop_exclusive=stop,
        planned_ordinals=ordinals, planned_samples=len(ordinals), full_samples=5119, full_scenes=150,
        arms=list(ARMS), horizon_seconds=HORIZONS, original_full_reference_files_sha256=reference['verified_files_sha256'],
        raw_labels_manifest_sha256=RAW_MANIFEST_SHA, raw_labels_complete_sha256=RAW_COMPLETE_SHA,
        numerical_policy=p['numerical_policy'], observation_matmul_tf32=True,
        resources=dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib),
        training=False, optimizer_updates=0, seed=11, no_threshold_selection=True,
        GT_loaded_by_original_dataset_before_prediction=True, GT_never_used_as_prediction_input=True,
        raw_label_contents_loaded_after_all_predictions=True, dense_sample_cache_written=False,
        historical_validation_exposure=True, no_flow_EPE_or_instance_identity_claim=True,
        one_original_raw_observation_per_anchor=True, additional_future_only_replay_for_terminal_capture=True,
        residual_can_change_uncovered_future_destinations=True, no_local_W_influence_bound_claim=True)
    write(out/'manifest.json', manifest)
    torch_module = None
    def check():
        require(not stopping, 'Signal received; no retry')
        require(time.monotonic()-started < a.max_seconds, 'Evaluation deadline reached')
        require(not any(out.parent.glob('shard_*/failed.json')), 'A shard failed; stop without retry')
        if torch_module is not None:
            require(torch_module.cuda.max_memory_allocated(a.device) <= a.max_allocated_gib*2**30, 'CUDA allocation ceiling')
    check()
    stream = stream_module.NativeOStream(config=a.config, checkpoint=a.checkpoint, o_checkpoint=a.o_checkpoint,
        repo=a.repo, runtime_contract=a.runtime_contract, selection=a.selection, reference_dir=a.reference_dir,
        device=a.device, ordinals=ordinals, max_allocated_gib=a.max_allocated_gib, check=check)
    import torch
    torch_module = torch
    core = contract.import_bound('future_feature_residual_v1', sources)
    require(helper.state_digest(stream.model) == training_manifest['frozen_O_state_sha256'], 'O differs from residual training')
    motion, D_receipt = loader.load_frozen_D(a.connected_training_run, Path(a.connected_protocol), root,
                                            Path(a.protocol), helper, a.device)
    require(helper.state_digest(motion) == training_manifest['frozen_D_motion_state_sha256'], 'Actual D motion differs')
    heads, loaded = {}, {}
    for arm in ('R','R0'):
        check()
        heads[arm], loaded[arm] = loader.load_completed_arm(root, arm, Path(a.protocol), helper, a.device)
        require(loaded[arm]['top_complete_sha256'] == a.training_complete_sha256
                and helper.state_digest(heads[arm]) == done['final_checkpoints'][arm]['module_state_sha256']
                and loaded[arm]['checkpoint_sha256'] == done['final_checkpoints'][arm]['sha256'], 'Loaded final residual identity differs')
    write(out/'loaded_models.json', dict(native_O=stream.loaded,
        O_full_helper_list_shape_state_sha256=helper.state_digest(stream.model), D=D_receipt, arms=loaded,
        optimizer_updates=0, optimizer_restored=False))
    initialization_seconds = time.monotonic()-started
    processed = intersections = 0; scenes = set(); maximum_sample_seconds = 0.
    hist_sums = {arm:np.zeros((5,2,2), np.int64) for arm in ARMS}
    with torch.no_grad(), (out/'records.jsonl').open('x') as f:
        for ordinal in ordinals:
            check(); tick = time.monotonic(); frame = stream.fetch(ordinal)
            sample, original = frame['sample'], frame['O_prediction']; identity = sample['record']
            observed, features = core.capture_O_terminal(stream.model, sample, native)
            require(core.exact32(observed, original), 'Terminal capture replay changed original O full logits')
            require(core.exact32(features[0], frame['t0_bev']), 'Captured t0 context differs from measured BEV')
            del observed
            snapshot = features.clone()
            base = oracle.predictions_to_xyz(original)
            foreground = (base[0,1] > base[0,0])[None,None]
            future = (base[1:,1]-base[1:,0])[None,:,None]
            flow = motion(frame['t0_bev']); flow_snapshot = flow.clone()
            require(flow.shape == (1,4,3,200,200,16) and flow.dtype == torch.float32
                    and bool(torch.isfinite(flow).all()), 'Invalid actual D source displacement')
            predictions = {'O':original}; diagnostics = {}
            for arm in ('R','R0'):
                check()
                require(not any(m.training for m in heads[arm].modules())
                        and not any(v.requires_grad for v in heads[arm].parameters()), 'Residual model became trainable')
                result = heads[arm](features, foreground, flow if arm == 'R' else torch.zeros_like(flow), future)
                predictions[arm] = core.compose_prediction(original, base, result['residual'], oracle)
                diagnostics[arm] = trainer.output_stats(result)
                del result
            require(core.exact32(features, snapshot) and core.exact32(flow, flow_snapshot), 'Frozen context or D displacement mutated')
            features_sha = native._tensor_digest(features); flow_sha = native._tensor_digest(flow)
            del snapshot, flow_snapshot, features, flow, base, foreground, future
            # The original dataset supplied GT earlier, but it is absent from
            # every prediction input; raw-box contents enter only this metric.
            descriptor = descriptors[ordinal]; payload = (rawroot/descriptor['file']).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == descriptor['sha256'], 'Raw labels changed')
            decoded = gzip.decompress(payload)
            require(hashlib.sha256(decoded).hexdigest() == descriptor['uncompressed_json_sha256'], 'Decoded raw labels changed')
            raw = json.loads(decoded); del payload, decoded
            require(raw['identity'] == selection['records'][ordinal] and raw['ordinal'] == ordinal
                    and raw['selection_sha256'] == FULL_SELECTION_SHA, 'Raw label identity/order changed')
            gt = sample['targets'][0,2:].to(torch.uint8).cpu().numpy()
            metrics, histograms = {}, {}
            for arm in ARMS:
                check()
                hist = frame['audit']['hist_by_horizon'] if arm == 'O' else common.native_hist(stream.model, predictions[arm], sample, identity)
                if identity['sample_token'] in previous:
                    contract.reject_hist_difference(out, ordinal, identity['sample_token'], arm,
                        hist, previous[identity['sample_token']]['hist_by_arm'][arm])
                binary = common.fine_binary(predictions[arm], oracle, check)
                value = metric.evaluate_common_occupancy_change(binary, gt, oracle.EXTENT, raw)
                require([h['occupancy']['confusion'] for h in value['horizons']] == hist, 'Native/CPU full fine histogram differs')
                if arm != 'O':
                    require(value['t0_boundary'] == metrics['O']['t0_boundary'], 'Full fine t0 differs between arms')
                metrics[arm] = value; histograms[arm] = hist; hist_sums[arm] += np.asarray(hist,np.int64)
                del binary
            stream.verify_frame(frame); torch.cuda.synchronize(a.device); check()
            has_previous = identity['sample_token'] in previous
            row = dict(schema=SCHEMA, ordinal=ordinal, selection_ordinal=ordinal, official_index=ordinal,
                sample_token=identity['sample_token'], scene_token=identity['scene_token'], split='validation',
                historical_partition=selection['records'][ordinal]['historical_partition'],
                metrics_by_arm=metrics, hist_by_arm=histograms, native_stream_audit=frame['audit'], diagnostics=diagnostics,
                raw_label_sha256=descriptor['sha256'], captured_terminal_features_tensor_sha256=features_sha,
                D_displacement_tensor_sha256=flow_sha, captured_O_replay_all_layer_logits_exact=True,
                captured_t0_features_equal_measured=True, captured_context_and_D_displacement_unchanged=True,
                original_full_O_hist_exact=True, original_full_O_all_layer_logits_exact=True,
                native_CPU_hist_exact=True, t0_all_models_exact=True,
                is_training_development_intersection=has_previous, training_final_hist_exact=True if has_previous else None,
                GT_never_used_as_prediction_input=True, raw_labels_read_after_predictions=True, optimizer_updates=0,
                seconds=time.monotonic()-tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
            f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
            processed += 1; intersections += has_previous; scenes.add(identity['scene_token'])
            maximum_sample_seconds = max(maximum_sample_seconds, row['seconds'])
            write(out/'progress.json', dict(processed_samples=processed,planned_samples=len(ordinals),
                last_official_index=ordinal,seconds=time.monotonic()-started,optimizer_updates=0))
            print(json.dumps(dict(event='FULL_FEATURE_RESIDUAL_SAMPLE',official_index=ordinal,
                processed_samples=processed,seconds=row['seconds'])),flush=True)
            del frame,sample,original,identity,predictions,diagnostics,raw,gt,metrics,histograms,row
    stream_final = stream.finish(); check()
    require(helper.state_digest(motion) == training_manifest['frozen_D_motion_state_sha256'], 'Frozen D state changed')
    for arm in ('R','R0'):
        require(helper.state_digest(heads[arm]) == done['final_checkpoints'][arm]['module_state_sha256'], 'Residual state changed')
    require(processed == len(ordinals) and sha(a.protocol) == PROTOCOL_SHA
            and sha(root/'complete.json') == a.training_complete_sha256, 'Completion or training source changed')
    for name, digest in sources.items():
        require(sha(source_path(name)) == digest, 'Source changed during evaluation')
    labels_contract(rawroot, selection)
    status = 'COMPLETE_FULL_FUTURE_FEATURE_RESIDUAL_SHARD' if a.mode == 'full' else 'PASS_FULL_FUTURE_FEATURE_RESIDUAL_PILOT'
    summary = dict(schema=SCHEMA,mode=a.mode,status=status,processed_samples=processed,scenes=len(scenes),
        histogram_sums_by_arm={k:v.tolist() for k,v in hist_sums.items()},
        original_full_O_hist_exact=True,original_full_O_all_layer_logits_exact=True,all_native_CPU_hist_exact=True,
        all_t0_models_exact=True,all_captured_O_replay_all_layer_logits_exact=True,
        training_development_intersections_exact=intersections,initialization_seconds=initialization_seconds,
        seconds=time.monotonic()-started,max_sample_seconds=maximum_sample_seconds,stream_final=stream_final,
        optimizer_updates=0,bootstrap_performed=False,model_promotion_decided=False,
        scope='This explicit shard only; population statistics require validated official-order merge',
        no_flow_EPE_or_instance_identity_claim=True)
    write(out/'summary.json',summary); check()
    write(out/'complete.json',dict(schema=SCHEMA,mode=a.mode,status=status,processed_samples=processed,scenes=len(scenes),
        full_samples=5119,full_scenes=150,shard_index=a.shard_index,shard_count=a.shard_count,
        ordinal_start=start,ordinal_stop_exclusive=stop,planned_ordinals=ordinals,
        protocol_sha256=PROTOCOL_SHA,connected_protocol_sha256=CONNECTED_PROTOCOL_SHA,
        training_complete_sha256=a.training_complete_sha256,source_sha256=sha(__file__),
        stream_sha256=STREAM_SHA,loader_sha256=LOADER_SHA,full_selection_sha256=FULL_SELECTION_SHA,
        arms=list(ARMS),optimizer_updates=0,all_original_full_O_hist_exact=True,
        all_original_full_O_all_layer_logits_exact=True,all_native_CPU_hist_exact=True,all_t0_models_exact=True,
        all_captured_O_replay_all_layer_logits_exact=True,
        files_sha256={n:sha(out/n) for n in ('manifest.json','loaded_models.json','records.jsonl','summary.json')},
        seconds=time.monotonic()-started))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('pilot','full'),required=True)
    for name in ('protocol','protocol-sha256','connected-protocol','connected-training-run','stream-sha256',
                 'training-run','training-complete-sha256','config','checkpoint','o-checkpoint','repo',
                 'runtime-contract','selection','reference-dir','stream-engineering','raw-labels','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--pilot-selection')
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('--max-allocated-gib',type=float,required=True)
    a = parser.parse_args(argv)
    require(a.mode != 'pilot' or a.pilot_selection is not None, 'Pilot needs frozen two-anchor selection')
    require(all(len(s)==64 and all(c in '0123456789abcdef' for c in s)
                for s in (a.protocol_sha256,a.stream_sha256,a.training_complete_sha256)), 'Explicit actual SHA arguments required')
    require(math.isfinite(a.max_seconds) and a.max_seconds>0 and math.isfinite(a.max_allocated_gib)
            and 0<a.max_allocated_gib<=32, 'Positive explicit time and <=32GiB allocation required')
    require(a.mode != 'pilot' or a.max_seconds<=600, 'Fixed two-anchor pilot capped at 600 seconds')
    require(a.shard_count in (1,2) and 0<=a.shard_index<a.shard_count, 'Invalid continuous shard')
    parent = Path(a.out).resolve(); parent.mkdir(parents=True,exist_ok=True)
    out = parent/('shard_%02d'%a.shard_index); out.mkdir(exist_ok=False)
    started = time.monotonic(); stopping = []
    signal.signal(signal.SIGTERM,lambda s,f:stopping.append(s))
    signal.signal(signal.SIGINT,lambda s,f:stopping.append(s))
    try:
        run(a,out,started,stopping)
    except BaseException as exc:
        write(out/'failed.json',dict(schema=SCHEMA,mode=a.mode,status='FAILED_NO_RETRY',error=repr(exc),
            traceback=traceback.format_exc(),processed_complete=False,source_sha256=sha(__file__),
            seconds=time.monotonic()-started,optimizer_updates=0))
        raise


if __name__ == '__main__':
    main()
