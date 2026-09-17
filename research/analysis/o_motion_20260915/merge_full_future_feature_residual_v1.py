"""Strict complete original5119 O/R/R0 merge using unchanged common metrics.

No physical EPE is inferred from occupancy. Frozen D is common to the two
new heads. Complete coverage and original O/native hist identities precede
all pooled-score and scene-bootstrap computations.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

SCHEMA = 'full-future-feature-residual-merge-v1'
TRAINING_PROTOCOL_SHA = '45048560948fae975945685e5151db158933b81b94b53b543551213b29a79a56'
PINNED = {
    'native_full_o_stream_v1.py': '3a8d338de78caae5d73e11012cd9cec38a47621913ddbb0bbda9cfae6e510bef',
    'summarize_common_change_v3.py': '338f9c0bb28b800711e6527d9208c3046132a5c652805a92333e4c209c529918',
    'common_occupancy_change_metrics_v1.py': '0ca592ca0b41ecf3524a0e49d8414e35518f4f9e50ccbb50937af8fded666951',
}
OCCUPANCY_ARMS = ('O', 'R', 'R0')


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



def validate_record(row, ordinal, reference, raw_descriptor, previous, summarizer):
    identity = reference['selection']['records'][ordinal]
    require(row['schema'] == 'full-future-feature-residual-evaluation-v1'
            and row['ordinal'] == row['selection_ordinal'] == row['official_index'] == ordinal
            and all(row[k] == identity[k] for k in identity)
            and set(row['metrics_by_arm']) == set(row['hist_by_arm']) == set(OCCUPANCY_ARMS), 'Full record identity/arms/order mismatch')
    require(row['raw_label_sha256'] == raw_descriptor['sha256'], 'Evaluation label source differs')
    for key in ('original_full_O_hist_exact','original_full_O_all_layer_logits_exact','native_CPU_hist_exact',
                't0_all_models_exact','GT_never_used_as_prediction_input','captured_O_replay_all_layer_logits_exact','captured_t0_features_equal_measured'):
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
        require(h['mode'] == 'full' and h['status'] == 'COMPLETE_FULL_FUTURE_FEATURE_RESIDUAL_SHARD'
                and h['shard_count'] == count and h['ordinal_start'] == start
                and h['ordinal_stop_exclusive'] == stop and h['planned_ordinals'] == list(range(start,stop))
                and h['processed_samples'] == stop-start, 'Partial/wrong full continuous partition')



def validate_loaded(loaded, manifest, training_done, training_manifest, arm_completions, protocol, stream):
    require(loaded['optimizer_updates'] == 0 and loaded['optimizer_restored'] is False
            and set(loaded['arms']) == {'R','R0'}, 'Evaluation restored optimizer or has wrong final arms')
    native = loaded['native_O']
    require(native['O_checkpoint_sha256'] == stream.O_SHA and native['O_head_state_sha256'] == stream.O_HEAD_SHA
            and native['training'] is False and native['optimizer_created'] is False
            and native['observation_matmul_tf32'] is True and native['future_matmul_tf32'] is False
            and native['cudnn_tf32'] is True
            and loaded['O_full_helper_list_shape_state_sha256'] == training_manifest['frozen_O_state_sha256'],
            'Loaded O source/state/precision differs')
    d = loaded['D']
    require(d['arm'] == 'D' and d['protocol_sha256'] == protocol['connected_protocol_sha256']
            and d['top_complete_sha256'] == protocol['connected_complete_sha256']
            and d['checkpoint_sha256'] == protocol['D_checkpoint_sha256']
            and d['actual_motion_state_sha256'] == training_manifest['frozen_D_motion_state_sha256']
            and d['fixed_final_update'] == 512 and d['fixed_examples'] == 2048
            and d['actual_optimizer_parameter_steps_all512'] is True
            and d['optimizer_restored'] is False and d['optimizer_updates_in_this_evaluation'] == 0,
            'Loaded actual D motion differs')
    for arm in ('R','R0'):
        actual = loaded['arms'][arm]; final = training_done['final_checkpoints'][arm]
        require(actual['arm'] == arm and actual['checkpoint_sha256'] == final['sha256']
                and actual['actual_module_state_sha256'] == final['module_state_sha256']
                and actual['initial_module_sha256'] == training_manifest['initial_module_sha256']
                and actual['manifest_sha256'] == arm_completions[arm]['files_sha256']['manifest.json']
                and actual['common_manifest_sha256'] == manifest['training_manifest_sha256']
                and actual['top_complete_sha256'] == manifest['training_complete_sha256']
                and actual['arm_complete_sha256'] == training_done['arm_complete_sha256'][arm]
                and actual['protocol_sha256'] == TRAINING_PROTOCOL_SHA
                and actual['fixed_final_update'] == 512 and actual['fixed_examples'] == 2048
                and actual['fixed_development_samples'] == 200
                and actual['actual_optimizer_parameter_steps_all512'] is True
                and actual['optimizer_restored'] is False,
                'Actual strict final residual loader receipt differs: '+arm)


def collect(a):
    """Read complete, immutable source/record chains before any summary."""
    # Caller binds the reviewed evaluator; every shard must record this same
    # actual source. This is recorded again in immutable merged input bindings.
    require(len(a.evaluator_sha256) == 64 and all(c in '0123456789abcdef' for c in a.evaluator_sha256),
            'Explicit reviewed evaluator SHA required')
    stream = import_bound('native_full_o_stream_v1', PINNED['native_full_o_stream_v1.py'])
    for name,digest in PINNED.items(): require(sha(source_path(name)) == digest, 'Changed common mathematics: '+name)
    import_bound('common_occupancy_change_metrics_v1', PINNED['common_occupancy_change_metrics_v1.py'])
    summarizer = import_bound('summarize_common_change_v3', PINNED['summarize_common_change_v3.py'])
    evaluator = import_bound('full_future_feature_residual_evaluation_v1', a.evaluator_sha256)
    protocol = read(a.protocol, TRAINING_PROTOCOL_SHA)
    require(protocol['status'] == 'FROZEN' and protocol['schema'] == 'future-feature-residual-training-v1', 'Wrong final training protocol')
    _, _, modules, evaluation_sources = evaluator.imports(protocol)
    engineering = evaluator.stream_engineering(a.stream_engineering, stream)
    reference = stream.load_full_reference(a.reference_dir, a.selection)
    raw_root, raw_descriptors = evaluator.labels_contract(a.raw_labels, reference['selection'])
    training_root, training_done, training_manifest, previous = evaluator.training_metadata(
        a.training_run, a.training_complete_sha256, protocol, modules['train_future_feature_residual_v1'], reference['selection'])
    arm_completions = {}
    for arm in ('R','R0'):
        path = training_root/'runs'/arm
        require(not (path/'failed.json').exists(), 'Failed training arm')
        completion = read(path/'complete.json', training_done['arm_complete_sha256'][arm])
        require(completion['schema'] == 'future-feature-residual-training-v1' and completion['mode'] == 'train'
                and completion['arm'] == arm and completion['status'] == training_done['status']
                and completion['updates'] == 512 and completion['examples'] == 2048 and completion['evaluated_samples'] == 200,
                'Incomplete final arm receipt')
        for name in ('manifest.json','training.jsonl','development_records.jsonl'):
            require(sha(path/name) == completion['files_sha256'][name], 'Training small artifact changed: '+arm+'/'+name)
        final = training_done['final_checkpoints'][arm]
        require(final == dict(file='runs/'+arm+'/final.pth',sha256=completion['files_sha256']['final.pth'],
            module_state_sha256=completion['final_module_sha256']), 'Training final pointer differs')
        arm_completions[arm] = completion
    shards = [Path(p).resolve() for p in a.shards]
    require(len(shards) in (1,2) and len(set(shards)) == len(shards), 'Require one or two distinct complete shards')
    require(all(not (p/'failed.json').exists() for p in shards), 'Failed shard cannot merge')
    validate_partition_headers([read(p/'complete.json') for p in shards])
    shards = sorted(shards,key=lambda p:read(p/'complete.json')['shard_index'])
    records=[];bindings=[];common_manifest=common_loaded=None
    histogram_totals={arm:np.zeros((5,2,2),dtype=np.int64) for arm in OCCUPANCY_ARMS}
    for shard_index,directory in enumerate(shards):
        done = read(directory/'complete.json')
        start,stop=5119*shard_index//len(shards),5119*(shard_index+1)//len(shards)
        wanted=list(range(start,stop))
        require(done['schema'] == evaluator.SCHEMA == 'full-future-feature-residual-evaluation-v1'
                and done['status'] == 'COMPLETE_FULL_FUTURE_FEATURE_RESIDUAL_SHARD' and done['mode'] == 'full'
                and done['shard_count'] == len(shards) and done['shard_index'] == shard_index
                and done['ordinal_start'] == start and done['ordinal_stop_exclusive'] == stop
                and done['planned_ordinals'] == wanted and done['processed_samples'] == stop-start
                and done['full_samples'] == 5119 and done['full_scenes'] == 150
                and done['arms'] == list(OCCUPANCY_ARMS) and done['optimizer_updates'] == 0
                and done['protocol_sha256'] == TRAINING_PROTOCOL_SHA
                and done['training_complete_sha256'] == a.training_complete_sha256
                and done['source_sha256'] == a.evaluator_sha256 and done['stream_sha256'] == PINNED['native_full_o_stream_v1.py']
                and done['full_selection_sha256'] == stream.FULL_SELECTION_SHA, 'Shard completion/source/partition differs')
        for key in ('all_original_full_O_hist_exact','all_original_full_O_all_layer_logits_exact','all_native_CPU_hist_exact','all_t0_models_exact'):
            require(done[key] is True, 'Missing shard parity proof: '+key)
        files=checked_files(directory,done,('manifest.json','loaded_models.json','records.jsonl','summary.json'))
        manifest=read(directory/'manifest.json');loaded=read(directory/'loaded_models.json');summary=read(directory/'summary.json')
        for key in ('schema','mode','source_sha256','stream_sha256','protocol_sha256','training_complete_sha256',
                    'full_selection_sha256','shard_index','shard_count','ordinal_start','ordinal_stop_exclusive','planned_ordinals','full_samples','full_scenes','arms','optimizer_updates'):
            require(manifest[key] == done[key], 'Manifest/complete differs: '+key)
        require(manifest['sources_sha256'] == evaluation_sources and manifest['planned_samples'] == stop-start
                and manifest['stream_engineering'] == engineering
                and manifest['training_manifest_sha256'] == training_done['files_sha256']['manifest.json']
                and manifest['training_development_records_sha256'] == training_done['files_sha256']['development_records.jsonl']
                and manifest['fixed_final_receipts'] == training_done['final_checkpoints']
                and manifest['original_full_reference_files_sha256'] == reference['verified_files_sha256']
                and manifest['raw_labels_manifest_sha256'] == evaluator.RAW_MANIFEST_SHA
                and manifest['raw_labels_complete_sha256'] == evaluator.RAW_COMPLETE_SHA
                and manifest['horizon_seconds'] == [0.,.5,1.,1.5,2.]
                and manifest['numerical_policy'] == protocol['numerical_policy']
                and manifest['observation_matmul_tf32'] is True and manifest['training'] is False
                and manifest['seed'] == 11 and manifest['no_threshold_selection'] is True
                and manifest['GT_never_used_as_prediction_input'] is True
                and manifest['GT_loaded_by_original_dataset_before_prediction'] is True
                and manifest['dense_sample_cache_written'] is False and manifest['historical_validation_exposure'] is True
                and manifest['pilot_selection_sha256'] is None, 'Manifest scientific source/label/precision differs')
        validate_loaded(loaded,manifest,training_done,training_manifest,arm_completions,protocol,stream)
        canonical_manifest={k:v for k,v in manifest.items() if k not in ('shard_index','ordinal_start','ordinal_stop_exclusive','planned_ordinals','planned_samples')}
        canonical_loaded=dict(O=loaded['O_full_helper_list_shape_state_sha256'],D=loaded['D'],arms=loaded['arms'])
        require(common_manifest is None or canonical_manifest == common_manifest, 'Shards have different scientific contracts')
        require(common_loaded is None or canonical_loaded == common_loaded, 'Shards loaded different final tensors')
        common_manifest,common_loaded=canonical_manifest,canonical_loaded
        hists={arm:np.zeros((5,2,2),dtype=np.int64) for arm in OCCUPANCY_ARMS};scenes=set();ordinals=[];intersections=0
        for row in json_rows(directory/'records.jsonl'):
            ordinal=start+len(ordinals);require(ordinal<stop,'Excess shard rows')
            validate_record(row,ordinal,reference,raw_descriptors[ordinal],previous,summarizer)
            ordinals.append(ordinal);records.append(row);scenes.add(row['scene_token'])
            intersections+=row['is_training_development_intersection']
            for arm in OCCUPANCY_ARMS:hists[arm]+=np.asarray(row['hist_by_arm'][arm],dtype=np.int64)
        require(ordinals == wanted and len(scenes) == done['scenes'], 'Incomplete actual shard rows/scenes')
        require(summary['schema'] == evaluator.SCHEMA and summary['mode'] == 'full' and summary['status'] == done['status']
                and summary['processed_samples'] == stop-start and summary['scenes'] == len(scenes)
                and summary['histogram_sums_by_arm'] == {k:v.tolist() for k,v in hists.items()}
                and summary['training_development_intersections_exact'] == intersections
                and summary['stream_final']['completed_samples'] == stop-start
                and summary['stream_final']['model_unchanged'] is True and summary['stream_final']['optimizer_steps'] == 0
                and summary['optimizer_updates'] == 0 and summary['bootstrap_performed'] is False
                and summary['model_promotion_decided'] is False, 'Shard summary differs from actual completed rows')
        for arm in OCCUPANCY_ARMS:histogram_totals[arm]+=hists[arm]
        bindings.append(dict(directory=str(directory),complete_sha256=sha(directory/'complete.json'),files_sha256=files,
                             shard_index=shard_index,ordinal_start=start,ordinal_stop_exclusive=stop,samples=stop-start,scenes=len(scenes)))
    require([r['official_index'] for r in records] == list(range(5119)) and len({r['sample_token'] for r in records}) == 5119
            and len({r['scene_token'] for r in records}) == 150 and sum(r['is_training_development_intersection'] for r in records) == 200,
            'Full official population/200 intersection incomplete')
    # Check common GT denominators, group partitions and all cross-arm masks
    # once over the actual complete rows before any bootstrap is calculated.
    arms,scenes,_=summarizer.collect(records)
    require(set(arms)==set(OCCUPANCY_ARMS) and len(scenes)==150,'Common population differs')
    return dict(records=records,math_common=summarizer,bindings=dict(schema=SCHEMA,
        evaluation_source_sha256=a.evaluator_sha256,stream_source_sha256=PINNED['native_full_o_stream_v1.py'],
        stream_engineering=engineering,protocol_sha256=TRAINING_PROTOCOL_SHA,training_complete_sha256=a.training_complete_sha256,
        training_manifest_sha256=sha(training_root/'manifest.json'),fixed_final_receipts=training_done['final_checkpoints'],
        mathematical_sources_sha256=PINNED,evaluation_sources_sha256=evaluation_sources,full_selection_sha256=stream.FULL_SELECTION_SHA,
        reference_files_sha256=reference['verified_files_sha256'],raw_manifest_sha256=evaluator.RAW_MANIFEST_SHA,
        raw_complete_sha256=evaluator.RAW_COMPLETE_SHA,shards=bindings,histogram_sums_by_arm={k:v.tolist() for k,v in histogram_totals.items()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards',nargs='+',required=True)
    for name in ('evaluator-sha256','protocol','stream-engineering','training-run','training-complete-sha256','selection','reference-dir','raw-labels','out'):
        parser.add_argument('--'+name,required=True)
    a=parser.parse_args();out=Path(a.out);require(not out.exists(),'Do not overwrite prior full analysis')
    started=time.monotonic();inputs=collect(a)
    common=inputs['math_common'].summarize(inputs['records'],bootstrap_repetitions=10000,seed=11)
    require(common['samples']==5119 and common['scenes']==150,'Common full population changed')
    summary=dict(schema=SCHEMA,status='COMPLETE_FULL_FUTURE_FEATURE_RESIDUAL_MERGE',samples=5119,scenes=150,common_output=common,
        source_sha256=sha(__file__),optimizer_updates=0,model_selection_performed=False,single_training_seed=11,
        historical_validation_exposure=True,new_blind_test=False,physical_motion_head_unchanged=True,
        confidence_interval='10000 paired whole-scene draws seed11; percentile95; no training-seed uncertainty or multiplicity correction',
        interpretation='Original future GMO and speed-attributed occupancy coverage are separate from physical displacement EPE; no new flow accuracy claim.',
        seconds=time.monotonic()-started)
    out.mkdir(parents=True,exist_ok=False);write(out/'input_bindings.json',inputs['bindings']);write(out/'summary.json',summary)
    (out/'report.md').write_text('# Full O/R/R0 共同输出统计\n\n'
        '完整5119 anchors / 150原validation场景，原O输出和全部共同GT检查通过。'
        'GMO按时域池化后取四未来均值；全部速度组召回、占据变化和FP/FN保留。\n\n'
        '固定单seed11、10000次场景配对bootstrap；历史validation非盲测。'
        'D物理运动头未改变，本报告不提供新的EPE改进或自动宣布研究目标完成。\n')
    write(out/'complete.json',dict(schema=SCHEMA,status=summary['status'],samples=5119,scenes=150,source_sha256=sha(__file__),
        protocol_sha256=TRAINING_PROTOCOL_SHA,training_complete_sha256=a.training_complete_sha256,optimizer_updates=0,
        files_sha256={f:sha(out/f) for f in ('summary.json','report.md','input_bindings.json')},seconds=time.monotonic()-started))
    print(json.dumps(dict(status=summary['status'],samples=5119,scenes=150,out=str(out),seconds=time.monotonic()-started)))


if __name__=='__main__':main()
