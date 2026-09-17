"""Stream fixed final O/J/D over original validation; no cache or training.

NativeOStream owns the unchanged original loader and O evaluator. The original
loader reads GT before prediction, but GT never enters prediction inputs. Raw
box and sparse displacement labels are read only after O/P/J/D predictions.
All geometry, fine occupancy/change and rigid-box EPE definitions are reused.
P is the original pretrained displacement head, not an O flow readout.
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

SCHEMA = 'full-connected-motion-evaluation-v1'
ARMS = ('O', 'J', 'D')
PHYSICAL_ARMS = ('P', 'J', 'D')
COMMON_SHA = '57cf304bafb050fac05b06abe68ce4600a37ca020d9d10d0673ec0ab1edb3bc1'
STREAM_SHA = '3a8d338de78caae5d73e11012cd9cec38a47621913ddbb0bbda9cfae6e510bef'
STREAM_ENGINEERING_SHA = 'c35bbe651fe0f28f7d8bbd2c7cf02f50dae893b2bba356cf4931beb26dcebe43'
PROTOCOL_SHA = 'e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
FULL_SELECTION_SHA = '60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391'
RAW_MANIFEST_SHA = '48ac12f1ce24ed1639c17aed0fcdc8ad2eef2cceff10c181b2a0f82f81fc90bb'
RAW_COMPLETE_SHA = 'de10ddd9013d7efd68249bced50448fb083c6d059546f8cfa0560be02b702252'
SPARSE_MANIFEST_SHA = '59535fe4f21e73182da5090e6b12ec60a0dfb40382564f0bb16409565699fc4e'
SPARSE_COMPLETE_SHA = '68fc77c6fe474fe5ed722e07be0341731a15a3adbb215dc5ecba453e06fec545'
HORIZONS = [0., .5, 1., 1.5, 2.]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path, expected=None):
    if expected is not None:
        require(sha(path) == expected, 'File SHA mismatch: '+str(path))
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(str(path)+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def emit(stream, row):
    stream.write(json.dumps(row, allow_nan=False)+'\n')


def schedule(mode, shard_index, shard_count, selection, pilot=None):
    require(shard_count in (1, 2) and 0 <= shard_index < shard_count, 'Only one or two continuous shards')
    rows = selection['records']
    require(len(rows) == selection['samples'] == 5119 and selection['scenes'] == 150
            and [r['official_index'] for r in rows] == list(range(5119))
            and len({r['sample_token'] for r in rows}) == 5119
            and len({r['scene_token'] for r in rows}) == 150
            and all(r['split'] == 'validation' for r in rows), 'Wrong full selection')
    if mode == 'full':
        start, stop = 5119*shard_index//shard_count, 5119*(shard_index+1)//shard_count
        return list(range(start, stop)), start, stop
    require(mode == 'pilot' and shard_index == 0 and shard_count == 1, 'Pilot must be a single fixed two-anchor shard')
    require(pilot is not None and pilot['samples'] == pilot['scenes'] == 2, 'Missing fixed two-anchor pilot')
    ordinals = [r['official_index'] for r in pilot['records']]
    require(ordinals == sorted(set(ordinals)) and len(ordinals) == 2, 'Wrong pilot ordering')
    for r in pilot['records']:
        require(all(r[k] == rows[r['official_index']][k] for k in ('sample_token','scene_token','official_index')), 'Pilot identity differs')
    return ordinals, None, None


def imports(protocol):
    path = Path(__file__).with_name('common_connected_motion_evaluation_v2.py')
    require(sha(path) == COMMON_SHA, 'Changed common evaluator source')
    spec = importlib.util.spec_from_file_location('common_connected_motion_evaluation_v2', path)
    common_contract = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common_contract; spec.loader.exec_module(common_contract)
    sources = common_contract.check_sources(protocol)
    sources['native_full_o_stream_v1.py'] = STREAM_SHA
    stream = common_contract.import_bound('native_full_o_stream_v1', sources)
    for name, digest in stream.SOURCES.items():
        require(name not in sources or sources[name] == digest, 'Conflicting native source binding')
        sources[name] = digest
    sources[Path(__file__).name] = sha(__file__)
    for name, digest in sources.items():
        require(sha(common_contract.source_path(name)) == digest, 'Changed bound source: '+name)
    common_contract.import_bound('motion_geometry', sources)
    modules = {name:common_contract.import_bound(name, sources) for name in
        ('train_connected_motion_v2','train_source_motion_v1','common_change_evaluation_v2',
         'common_occupancy_change_metrics_v1','oracle_transport_probe','train_supported_fusion_v2',
         'learned_transport_probe_v1')}
    return common_contract, stream, modules, sources


def stream_engineering(root, stream):
    root = Path(root)
    complete = read(root/'complete.json', STREAM_ENGINEERING_SHA)
    require(not (root/'failed.json').exists() and complete['schema'] == stream.SCHEMA
            and complete['status'] == 'PASS_NATIVE_FULL_O_STREAM_ENGINEERING'
            and complete['mode'] == 'pilot' and complete['samples'] == 2
            and complete['optimizer_steps'] == 0, 'Native stream engineering not complete')
    require(set(complete['files_sha256']) == {'manifest.json','loaded_model.json','records.jsonl','summary.json'}, 'Missing stream engineering artifact')
    for name, digest in complete['files_sha256'].items(): require(sha(root/name) == digest, 'Changed stream engineering artifact')
    manifest = read(root/'manifest.json')
    require(manifest['script_sha256'] == STREAM_SHA and manifest['sources_sha256'] == stream.SOURCES
            and manifest['full_selection_sha256'] == FULL_SELECTION_SHA
            and manifest['config_sha256'] == stream.CONFIG_SHA, 'Stream engineering source/selection mismatch')
    return dict(complete_sha256=STREAM_ENGINEERING_SHA, files_sha256=complete['files_sha256'])


def labels_contract(raw_root, sparse_root, selection):
    raw_root, sparse_root = Path(raw_root).resolve(), Path(sparse_root).resolve()
    raw = read(raw_root/'manifest.json', RAW_MANIFEST_SHA)
    rd = read(raw_root/'complete.json', RAW_COMPLETE_SHA)
    sparse = read(sparse_root/'manifest.json', SPARSE_MANIFEST_SHA)
    sd = read(sparse_root/'complete.json', SPARSE_COMPLETE_SHA)
    require(rd['status'] == sd['status'] == 'COMPLETE' and rd['samples'] == sd['samples'] == 5119
            and rd['scenes'] == sd['scenes'] == 150
            and rd['manifest_sha256'] == RAW_MANIFEST_SHA and sd['manifest_sha256'] == SPARSE_MANIFEST_SHA,
            'Incomplete full label sets')
    require(raw['selection']['sha256'] == sd['selection_sha256'] == FULL_SELECTION_SHA
            and sparse['label_source']['manifest_sha256'] == RAW_MANIFEST_SHA
            and sd['raw_complete_sha256'] == RAW_COMPLETE_SHA
            and sd['development200_intersection_dtype_shape_bytes_exact'] is True, 'Full labels or dev200 bytes changed')
    for name, key in [('statistics.json','statistics_sha256'),('material_point_audit.json','material_point_audit_sha256'),
                      ('development_intersection_audit.json','development_intersection_audit_sha256')]:
        require(sha(sparse_root/name) == sd[key], 'Sparse audit changed: '+name)
    intersection = read(sparse_root/'development_intersection_audit.json')
    require(intersection['status'] == 'PASS' and intersection['samples'] == 200
            and len(intersection['records']) == 200 and all(r['dtype_shape_bytes_exact'] and r['support_counts_exact']
            for r in intersection['records']), 'Original dev200 sparse equality is unverified')
    require(len(raw['records']) == len(sparse['records']) == 5119, 'Wrong label record count')
    for i, (row, r, s) in enumerate(zip(selection['records'], raw['records'], sparse['records'])):
        require(r['ordinal'] == s['ordinal'] == i and r['identity'] == s['identity'] == row,
                'Full raw/sparse identity mismatch')
        require(r['file'] == 'validation/'+row['sample_token']+'.json.gz'
                and s['file'] == 'validation/'+row['sample_token']+'.npz'
                and s['label_source_sha256'] == r['sha256'], 'Full label path/source mismatch')
    return raw_root, raw['records'], sparse_root, sparse['records']


def training_metadata(root, complete_sha, protocol, trainer, selection):
    root = Path(root).resolve()
    done = read(root/'complete.json', complete_sha)
    require(not (root/'failed.json').exists(), 'Training failure marker')
    require(done['schema'] == trainer.SCHEMA == 'connected-motion-training-v2'
            and done['mode'] == 'train' and done['status'] == 'COMPLETE_CONNECTED_MOTION_TRAINING'
            and done['updates'] == 512 and done['examples'] == 2048 and done['evaluated_samples'] == 200,
            'Actual completed final512/dev200 required')
    expected = {'manifest.json','engineering.json','summary.json','development_records.jsonl','development_objects.jsonl'}
    require(set(done['files_sha256']) == expected and set(done['final_checkpoints']) == {'J','D'}, 'Incomplete training chain')
    for name, digest in done['files_sha256'].items(): require(sha(root/name) == digest, 'Training artifact changed: '+name)
    manifest = read(root/'manifest.json')
    require(manifest['sources']['protocol_sha256'] == PROTOCOL_SHA
            and manifest['sources']['sources_sha256'] == protocol['sources_sha256']
            and manifest['training'] == trainer.TRAINING == protocol['training']
            and manifest['numerical_policy'] == trainer.NUMERICAL == protocol['numerical_policy'], 'Training contract differs')
    rows = [json.loads(line) for line in (root/'development_records.jsonl').read_text().splitlines()]
    lookup = {r['sample_token']:r for r in rows}
    full = {r['sample_token']:r for r in selection['records']}
    require(len(rows) == len(lookup) == 200 and len({r['scene_token'] for r in rows}) == 100, 'Incomplete training dev identity set')
    for i, row in enumerate(rows):
        require(row['ordinal'] == i and row['split'] == 'development' and row['sample_token'] in full
                and all(row[k] == full[row['sample_token']][k] for k in ('scene_token','official_index'))
                and set(row['hist_by_arm']) == set(ARMS), 'Final development identity differs')
    return root, done, manifest, lookup


def pretrained_metadata(root, protocol_path, protocol, helper):
    root, protocol_path = Path(root).resolve(), Path(protocol_path)
    tp = read(protocol_path, protocol['motion']['protocol_sha256'])
    done = read(root/'complete.json', protocol['motion']['complete_sha256'])
    require(tp['schema'] == helper.SCHEMA and tp['status'] == 'FROZEN' and tp['training'] == helper.TRAINING
            and tp['labels'] == protocol['labels'] and tp['cache_index_sha256'] == protocol['cache_index_sha256'], 'Wrong P pretraining')
    require(not (root/'failed.json').exists() and done['schema'] == helper.SCHEMA
            and done['status'] == 'COMPLETE_SOURCE_MOTION_TRAINING' and done['mode'] == 'train'
            and done['updates'] == 512 and done['examples'] == 2048 and done['evaluated_samples'] == 200,
            'P is not complete fixed final512')
    require(set(done['files_sha256']) == {'manifest.json','training.jsonl','summary.json','final.pth',
            'development_objects.jsonl','development_records.jsonl'}, 'Incomplete P artifact chain')
    for name, digest in done['files_sha256'].items(): require(sha(root/name) == digest, 'P artifact changed: '+name)
    require(done['final_checkpoint']['sha256'] == done['files_sha256']['final.pth'] == protocol['motion']['checkpoint_sha256']
            and done['final_checkpoint']['head_state_sha256'] == protocol['motion']['head_state_sha256'], 'Wrong P final checkpoint')
    manifest = read(root/'manifest.json')
    require(manifest['source_receipt']['protocol_sha256'] == protocol['motion']['protocol_sha256']
            and manifest['source_receipt']['labels'] == protocol['labels'] and manifest['training'] == tp['training'], 'P manifest differs')
    for name, digest in tp['sources_sha256'].items():
        require(protocol['sources_sha256'][name] == digest, 'P source contract differs')
    return root/'final.pth', manifest, tp


def run(a, out, started, stopping):
    require(sha(a.protocol) == a.protocol_sha256 == PROTOCOL_SHA, 'Actual frozen training protocol required')
    require(a.stream_sha256 == STREAM_SHA, 'Actual approved native stream source SHA required')
    p = read(a.protocol)
    require(p['status'] == 'FROZEN' and p['schema'] == 'connected-motion-training-v2', 'Wrong training protocol')
    contract, stream_module, modules, sources = imports(p)
    trainer = modules['train_connected_motion_v2']; helper = modules['train_source_motion_v1']
    common = modules['common_change_evaluation_v2']; metric = modules['common_occupancy_change_metrics_v1']
    oracle = modules['oracle_transport_probe']; composer = modules['train_supported_fusion_v2']
    pretrained = modules['learned_transport_probe_v1']
    engineering = stream_engineering(a.stream_engineering, stream_module)
    selection = read(a.selection, FULL_SELECTION_SHA)
    pilot = read(a.pilot_selection, stream_module.PILOT_SELECTION_SHA) if a.mode == 'pilot' else None
    ordinals, start, stop = schedule(a.mode, a.shard_index, a.shard_count, selection, pilot)
    raw_root, raw_descriptors, sparse_root, sparse_descriptors = labels_contract(a.raw_labels, a.sparse_labels, selection)
    training_root, done, training_manifest, previous = training_metadata(
        a.training_run, a.training_complete_sha256, p, trainer, selection)
    motion_path, motion_manifest, motion_protocol = pretrained_metadata(a.motion_run, a.motion_training_protocol, p, helper)
    reference = stream_module.load_full_reference(a.reference_dir, a.selection)
    for token, row in previous.items():
        require(row['hist_by_arm']['O'] == reference['hist'][row['official_index']].tolist(), 'Original O full/dev reference mismatch')
    manifest = dict(schema=SCHEMA, mode=a.mode, sources_sha256=sources,
        source_sha256=sha(__file__), stream_sha256=STREAM_SHA, protocol_sha256=PROTOCOL_SHA,
        stream_engineering=engineering,
        training_complete_sha256=a.training_complete_sha256, training_manifest_sha256=sha(training_root/'manifest.json'),
        training_development_records_sha256=sha(training_root/'development_records.jsonl'),
        fixed_final_receipts=done['final_checkpoints'], P_binding=p['motion'],
        full_selection_sha256=FULL_SELECTION_SHA,
        pilot_selection_sha256=stream_module.PILOT_SELECTION_SHA if pilot is not None else None,
        shard_index=a.shard_index, shard_count=a.shard_count, ordinal_start=start, ordinal_stop_exclusive=stop,
        planned_ordinals=ordinals, planned_samples=len(ordinals), full_samples=5119, full_scenes=150,
        arms=list(ARMS), physical_arms=list(PHYSICAL_ARMS), horizon_seconds=HORIZONS,
        original_full_reference_files_sha256=reference['verified_files_sha256'],
        raw_labels_manifest_sha256=RAW_MANIFEST_SHA, raw_labels_complete_sha256=RAW_COMPLETE_SHA,
        sparse_labels_manifest_sha256=SPARSE_MANIFEST_SHA, sparse_labels_complete_sha256=SPARSE_COMPLETE_SHA,
        resources=dict(max_seconds=a.max_seconds, max_allocated_gib=a.max_allocated_gib),
        numerical_policy=p['numerical_policy'], observation_matmul_tf32=True,
        training=False, optimizer_updates=0, seed=11, no_threshold_selection=True,
        GT_loaded_by_original_dataset_before_prediction=True, GT_never_used_as_prediction_input=True,
        raw_and_sparse_label_contents_loaded_after_predictions=True,
        dense_sample_cache_written=False, historical_validation_exposure=True,
        physical_metric='Original equal anchor-instance mean-point rigid-box source displacement EPE; not O flow or measured scene flow')
    write(out/'manifest.json', manifest)
    torch_module = None
    def check():
        require(not stopping, 'Signal received; no retry')
        require(time.monotonic()-started < a.max_seconds, 'Streaming evaluation deadline')
        require(not any(x.exists() for x in out.parent.glob('shard_*/failed.json')), 'Peer shard failed; stop without retry')
        if torch_module is not None:
            require(torch_module.cuda.max_memory_allocated(a.device) <= a.max_allocated_gib*2**30, 'CUDA memory ceiling')
    check()
    stream = stream_module.NativeOStream(config=a.config, checkpoint=a.checkpoint,
        o_checkpoint=a.o_checkpoint, repo=a.repo, runtime_contract=a.runtime_contract,
        selection=a.selection, reference_dir=a.reference_dir, device=a.device,
        ordinals=ordinals, max_allocated_gib=a.max_allocated_gib, check=check)
    import torch
    torch_module = torch
    require(helper.state_digest(stream.model) == training_manifest['frozen_O_state_sha256'], 'Actual O state differs from J/D training')
    fixed_motion = pretrained.load_motion(motion_path, motion_manifest,
        dict(sources_sha256=p['sources_sha256'], training=p['motion']), motion_protocol, helper, a.device)
    require(helper.state_digest(fixed_motion) == p['motion']['head_state_sha256'], 'Actual P state differs')
    fusion = contract.import_bound('supported_motion_fusion', sources)
    motions = {'P':fixed_motion}; gates = {}; loaded = {}
    for arm in ('J','D'):
        check()
        motion, gate, receipt = trainer.load_completed_arm(training_root/'runs'/arm, Path(a.protocol), helper, a.device)
        require(receipt['top_complete_sha256'] == a.training_complete_sha256
                and helper.state_digest(motion) == done['final_checkpoints'][arm]['motion_state_sha256']
                and helper.state_digest(gate) == done['final_checkpoints'][arm]['gate_state_sha256'], 'Actual J/D final loading changed')
        motions[arm], gates[arm], loaded[arm] = motion, gate, receipt
    write(out/'loaded_models.json', dict(native_O=stream.loaded,
        O_full_helper_list_shape_state_sha256=helper.state_digest(stream.model),
        P=dict(checkpoint_sha256=sha(motion_path), actual_motion_state_sha256=helper.state_digest(fixed_motion),
               complete_sha256=p['motion']['complete_sha256'], optimizer_restored=False),
        arms=loaded, optimizer_updates=0, optimizer_restored=False))
    initial_seconds = time.monotonic()-started
    processed = 0; physical_counts = {arm:0 for arm in PHYSICAL_ARMS}; histogram_sums = {arm:np.zeros((5,2,2),np.int64) for arm in ARMS}
    intersections = 0; evaluated_scenes = set(); max_sample_seconds = 0.
    with torch.no_grad(), (out/'records.jsonl').open('x') as records, (out/'physical_objects.jsonl').open('x') as objects:
        for ordinal in ordinals:
            check(); tick = time.monotonic(); frame = stream.fetch(ordinal)
            sample, original, tokens = frame['sample'], frame['O_prediction'], frame['t0_bev']
            identity = sample['record']; base = oracle.predictions_to_xyz(original)
            probability = torch.softmax(base[0:1], dim=1)[:,1:2]
            foreground = base[0:1,1:2] > base[0:1,0:1]
            future = (base[1:,1]-base[1:,0])[None,:,None]
            predictions = {'O':original}; displacement = {}
            for arm in PHYSICAL_ARMS:
                displacement[arm] = motions[arm](tokens)
                require(tuple(displacement[arm].shape) == (1,4,3,200,200,16)
                        and bool(torch.isfinite(displacement[arm]).all()), 'Invalid physical prediction')
                if arm != 'P':
                    fields = fusion.supported_transport(probability, foreground, displacement[arm], oracle.EXTENT)
                    fused = gates[arm](future, fields['probability_mass'], fields['support_weight'])
                    predictions[arm] = composer.compose_prediction(original, base, fused, oracle)
                    del fields, fused
            require(all(composer.exact32(value[0], original[0]) for value in predictions.values()), 't0 logits changed')
            del base, probability, foreground, future
            # Targets were loaded by the original dataset; no label content above
            # is used to choose sources, displacements, gates, predictions or ROI.
            label = helper.load_sparse(sparse_root, sparse_descriptors[ordinal], identity)
            raw_descriptor = raw_descriptors[ordinal]; payload = (raw_root/raw_descriptor['file']).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == raw_descriptor['sha256'], 'Raw label SHA changed')
            unpacked = gzip.decompress(payload)
            require(hashlib.sha256(unpacked).hexdigest() == raw_descriptor['uncompressed_json_sha256'], 'Raw JSON SHA changed')
            raw = json.loads(unpacked); del payload, unpacked
            require(raw['identity'] == selection['records'][ordinal] and raw['ordinal'] == ordinal
                    and raw['selection_sha256'] == FULL_SELECTION_SHA, 'Raw label identity differs')
            physical_keys = None; count_by_arm = {}
            for arm in PHYSICAL_ARMS:
                physical = helper.epe_records(helper.gather_sparse(displacement[arm], label), label, identity)
                keys = [(r['instance_token'],r['horizon_seconds'],r['group'],r['source_points'],r['dt_seconds']) for r in physical]
                require(physical_keys is None or keys == physical_keys, 'Physical support differs across P/J/D')
                physical_keys = keys; count_by_arm[arm] = len(physical)
                for row in physical:
                    emit(objects, dict(row, arm=arm, ordinal=ordinal, selection_ordinal=ordinal,
                                       official_index=ordinal, split='validation'))
                physical_counts[arm] += len(physical)
            del displacement, physical, keys
            gt = sample['targets'][0,2:].to(torch.uint8).cpu().numpy()
            metrics = {}; histograms = {}
            for arm in ARMS:
                check()
                hist = frame['audit']['hist_by_horizon'] if arm == 'O' else common.native_hist(stream.model, predictions[arm], sample, identity)
                if identity['sample_token'] in previous:
                    contract.reject_hist_difference(out, ordinal, identity['sample_token'], arm,
                        hist, previous[identity['sample_token']]['hist_by_arm'][arm])
                binary = common.fine_binary(predictions[arm], oracle, check)
                value = metric.evaluate_common_occupancy_change(binary, gt, oracle.EXTENT, raw)
                require([h['occupancy']['confusion'] for h in value['horizons']] == hist, 'CPU/native fine histogram differs')
                if arm != 'O': require(value['t0_boundary'] == metrics['O']['t0_boundary'], 'Full fine t0 differs')
                metrics[arm] = value; histograms[arm] = hist; histogram_sums[arm] += np.asarray(hist,np.int64)
                del binary
            stream.verify_frame(frame); torch.cuda.synchronize(a.device); check()
            has_previous = identity['sample_token'] in previous
            row = dict(schema=SCHEMA, ordinal=ordinal, selection_ordinal=ordinal, official_index=ordinal,
                sample_token=identity['sample_token'], scene_token=identity['scene_token'], split='validation',
                historical_partition=selection['records'][ordinal]['historical_partition'],
                metrics_by_arm=metrics, hist_by_arm=histograms, native_stream_audit=frame['audit'],
                raw_label_sha256=raw_descriptor['sha256'], sparse_label_sha256=sparse_descriptors[ordinal]['sha256'],
                physical_object_rows_by_arm=count_by_arm,
                original_full_O_hist_exact=True, original_full_O_all_layer_logits_exact=True,
                native_CPU_hist_exact=True, t0_all_models_exact=True,
                is_training_development_intersection=has_previous,
                training_final_hist_exact=True if has_previous else None,
                GT_never_used_as_prediction_input=True, optimizer_updates=0,
                seconds=time.monotonic()-tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
            emit(records,row)
            max_sample_seconds = max(max_sample_seconds, row['seconds'])
            for f in (records, objects): f.flush(); os.fsync(f.fileno())
            processed += 1; intersections += has_previous; evaluated_scenes.add(identity['scene_token'])
            write(out/'progress.json',dict(processed_samples=processed,planned_samples=len(ordinals),
                last_official_index=ordinal,seconds=time.monotonic()-started,optimizer_updates=0))
            print(json.dumps(dict(event='FULL_CONNECTED_SAMPLE',official_index=ordinal,
                processed_samples=processed,seconds=row['seconds'])),flush=True)
            del frame,sample,original,tokens,predictions,label,raw,gt,metrics,histograms,row,physical_keys
    final_stream = stream.finish(); check()
    for arm in PHYSICAL_ARMS:
        expected = p['motion']['head_state_sha256'] if arm == 'P' else loaded[arm]['actual_motion_state_sha256']
        require(helper.state_digest(motions[arm]) == expected, 'Physical head mutated')
        if arm != 'P': require(helper.state_digest(gates[arm]) == loaded[arm]['actual_gate_state_sha256'], 'Gate mutated')
    require(processed == len(ordinals) and sha(a.protocol) == PROTOCOL_SHA
            and sha(training_root/'complete.json') == a.training_complete_sha256, 'Completion or training changed')
    for name, digest in sources.items(): require(sha(contract.source_path(name)) == digest, 'Source changed during evaluation')
    labels_contract(raw_root,sparse_root,selection)
    status = 'COMPLETE_FULL_CONNECTED_MOTION_SHARD' if a.mode == 'full' else 'PASS_FULL_CONNECTED_MOTION_PILOT'
    summary = dict(schema=SCHEMA,mode=a.mode,status=status,processed_samples=processed,scenes=len(evaluated_scenes),
        physical_object_rows_by_arm=physical_counts,histogram_sums_by_arm={k:v.tolist() for k,v in histogram_sums.items()},
        original_full_O_hist_exact=True,original_full_O_all_layer_logits_exact=True,all_native_CPU_hist_exact=True,
        all_t0_models_exact=True,training_development_intersections_exact=intersections,
        initialization_seconds=initial_seconds,seconds=time.monotonic()-started,stream_final=final_stream,
        max_sample_seconds=max_sample_seconds,
        optimizer_updates=0,bootstrap_performed=False,model_promotion_decided=False,
        scope='Only this explicit shard; full population statistics require validated merge of all official ordinals',
        physical_metric='Existing rigid-box source displacement proxy on original label support; no O EPE or measured flow claim')
    write(out/'summary.json',summary); check()
    write(out/'complete.json',dict(schema=SCHEMA,mode=a.mode,status=status,
        processed_samples=processed,scenes=len(evaluated_scenes),full_samples=5119,full_scenes=150,
        shard_index=a.shard_index,shard_count=a.shard_count,ordinal_start=start,ordinal_stop_exclusive=stop,
        planned_ordinals=ordinals,protocol_sha256=PROTOCOL_SHA,training_complete_sha256=a.training_complete_sha256,
        source_sha256=sha(__file__),stream_sha256=STREAM_SHA,full_selection_sha256=FULL_SELECTION_SHA,
        arms=list(ARMS),physical_arms=list(PHYSICAL_ARMS),optimizer_updates=0,
        all_original_full_O_hist_exact=True,all_original_full_O_all_layer_logits_exact=True,
        all_native_CPU_hist_exact=True,all_t0_models_exact=True,
        files_sha256={n:sha(out/n) for n in ('manifest.json','loaded_models.json','records.jsonl','physical_objects.jsonl','summary.json')},
        seconds=time.monotonic()-started))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('pilot','full'),required=True)
    for name in ('protocol','protocol-sha256','stream-sha256','training-run','training-complete-sha256',
                 'config','checkpoint','o-checkpoint','repo','runtime-contract','selection','reference-dir',
                 'stream-engineering','motion-run','motion-training-protocol','raw-labels','sparse-labels','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--pilot-selection')
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('--max-allocated-gib',type=float,required=True)
    a=parser.parse_args(argv)
    require(a.mode!='pilot' or a.pilot_selection is not None,'Pilot needs frozen selection')
    require(all(len(s)==64 and all(c in '0123456789abcdef' for c in s) for s in
                (a.protocol_sha256,a.stream_sha256,a.training_complete_sha256)),'Actual SHA arguments required')
    require(math.isfinite(a.max_seconds) and 0<a.max_seconds<=42600
            and math.isfinite(a.max_allocated_gib) and 0<a.max_allocated_gib<=32,'Positive explicit time / <=32GiB resource ceiling')
    require(a.mode != 'pilot' or a.max_seconds <= 600, 'Fixed two-anchor pilot is capped at 600 seconds')
    require(a.shard_count in (1,2) and 0<=a.shard_index<a.shard_count,'Invalid shard')
    parent=Path(a.out).resolve();parent.mkdir(parents=True,exist_ok=True)
    out=parent/('shard_%02d'%a.shard_index);out.mkdir(exist_ok=False)
    started=time.monotonic();stopping=[]
    signal.signal(signal.SIGTERM,lambda s,f:stopping.append(s))
    signal.signal(signal.SIGINT,lambda s,f:stopping.append(s))
    try:run(a,out,started,stopping)
    except BaseException as exc:
        write(out/'failed.json',dict(schema=SCHEMA,mode=a.mode,status='FAILED_NO_RETRY',
            error=repr(exc),traceback=traceback.format_exc(),processed_complete=False,
            source_sha256=sha(__file__),seconds=time.monotonic()-started,optimizer_updates=0))
        raise


if __name__=='__main__':
    main()
