"""Fixed final512 Cpl/Fix and original O on all original dev200 anchors.

No training, threshold choice, best checkpoint, detector inference or D model.
Current predicted geometry defines the complete source field before target
loading. Physical support keeps missed detections/uncovered points. Original
D records are authenticated references; O has no native physical-flow output.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

SCHEMA = 'shared-rigid-fixed-development-v1'
STATUS = 'COMPLETE_SHARED_RIGID_FIXED_DEV200'
ARMS = ('Cpl','Fix')
ENDPOINT_SHA = '549112b0a8de1fe5dd705da5376aff3622f48a087a67b2f0bef2d1a357191917'
EXTRA = {
    'shared_rigid_final_endpoint_v1.py': ENDPOINT_SHA,
    'summarize_object_state_common_v1.py': '4a03aeda837e78b97ff18d27140d08602619ee96502b7f62fa82f591ca08f584',
    'summarize_object_state_physical_v1.py': 'bd2c652a1682c8ea3e16c0d35ec69ba75957da7886cd9f20a1bb700af3c011b6',
    'summarize_future_state_common_v1.py': '23f6fc0a6d7a9c63a1e77f90ca94a2aedc72b68b82b201df4f34c6d3f3123e8a',
    'summarize_future_state_physical_v1.py': '920f4f96e0bd5df247b3d096bdf1709d80fed593622b70373093ea1160db2de3',
    'crn_box_origin_adapter_v1.py': 'aa05e4cd8d2c9ffbf41e751ebc0bda7114040753e404c0560975ee134e23a101',
    'history_state_predictability_diagnostic_v1.py': '2b14c95bfa027416304288eec1282d1cc001984be42261cf656ee2cb6d6c96da',
}
EVALUATION = dict(anchors=200,scenes=100,final_updates=512,training_seed=11,
    native_arms=['O','Cpl','Fix'],physical_arms=['D','CRN_CV','Cpl','Fix'],
    selection='original development200 in original order; fixed final only',
    native='original five-horizon native evaluation and unchanged full GT argmax',
    physical='all original valid virtual rigid material points; fixed current predicted ownership; uncovered zero',
    primary='four-future mean of per-horizon pooled GMO IoU; not semantic mIoU',
    risk='original full FP/FN, actual-moving and high-speed recall, arrival/vacating transitions',
    t0='each model own native t0; common GT support; no cross-model t0 equality gate',
    physical_comparison='Cpl/Fix versus same-support CRN_CV, D, zero; O has no flow',
    O_reference='all 200 original per-horizon confusion matrices exact',
    historical_CV_float_check=dict(rel_tol=1e-12,abs_tol_m=1e-10,scope='CPU platform/reduction rounding only'),
    checkpoint_selection=False,threshold_fitting=False,optimizer_updates=0)


def endpoint_module():
    path=Path(__file__).with_name('shared_rigid_final_endpoint_v1.py')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ENDPOINT_SHA:raise ValueError('Final endpoint validator changed')
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


endpoint=endpoint_module();trainer=endpoint.trainer;pre=trainer.pre
require,sha,read=pre.require,pre.sha,pre.read
write,append=trainer.u.write,trainer.u.append


def expected_sources(training_protocol):
    result=dict(training_protocol['sources_sha256']);contract=pre._contract_loader()
    result.update(EXTRA)
    # Exact transitive mathematics/reference dependencies, fixed before scores.
    for name in ('summarize_common_change_v3.py','summarize_common_change_v2.py',
                 'summarize_connected_motion_v1.py','predicted_object_state_cv_diagnostic_v2.py'):
        result[name]=sha(contract.source_path(name))
    result[Path(__file__).name]=sha(__file__)
    for name,digest in result.items():require(sha(contract.source_path(name))==digest,'Evaluation source changed: '+name)
    return result


def protocol_template():
    return dict(schema=SCHEMA,status='REVIEW_REQUIRED',evaluation=copy.deepcopy(EVALUATION),
        numerical_policy=dict(pre.NUMERICAL),training_protocol_sha256=None,sources_sha256={},resources=None,
        analysis_sources_sha256={},
        final_checkpoint_rule='actual successful fixed512 training completion; supplied SHA binds that endpoint only')


def gt_only(value):
    if isinstance(value,dict):return {k:gt_only(v) for k,v in value.items() if k not in ('TP','FN','recall')}
    if isinstance(value,list):return [gt_only(v) for v in value]
    return value


def verify_common_denominators(metrics):
    reference=metrics['O']
    for arm in ARMS:
        item=metrics[arm]
        for key in ('identity','shape_hxyz','extent_xyz_m'):require(item[key]==reference[key],'Common domain differs')
        for key in ('gt_valid_mask_sha256','valid_voxels'):
            require(item['t0_boundary'][key]==reference['t0_boundary'][key],'Common t0 GT differs')
        for left,right in zip(item['horizons'],reference['horizons']):
            require(np.array_equal(np.asarray(left['occupancy']['confusion']).sum(1),np.asarray(right['occupancy']['confusion']).sum(1))
                    and left['ignored_voxels']==right['ignored_voxels'],'Full-GT denominators differ')
            if left['horizon_index']:
                require(gt_only(left['motion_positive_attribution'])==gt_only(right['motion_positive_attribution']),
                        'Motion-group GT support differs')
        for left,right in zip(item['transitions'],reference['transitions']):
            require(left['domain']==right['domain'] and np.array_equal(np.asarray(left['confusion']).sum(1),
                    np.asarray(right['confusion']).sum(1)),'Transition GT support differs')


def run(a,out,started,stop):
    p=read(a.protocol);tp=read(a.training_protocol)
    require(p['schema']==SCHEMA and p['status']=='FROZEN' and p['evaluation']==EVALUATION and
            p['numerical_policy']==pre.NUMERICAL and p['training_protocol_sha256']==sha(a.training_protocol),
            'Exact frozen final evaluation required')
    require(math.isfinite(a.max_seconds) and a.max_seconds>0 and math.isfinite(a.max_allocated_gib) and a.max_allocated_gib>0
            and p['resources']==dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib),'Resource contract differs')
    require(p['sources_sha256']==expected_sources(tp),'Evaluation source closure differs')
    require(set(p['analysis_sources_sha256'])=={'summarize_shared_rigid_state_v1.py'},'Predeclared final analysis required')
    for name,digest in p['analysis_sources_sha256'].items():
        require(sha(Path(__file__).with_name(name))==digest,'Analysis source changed before evaluation')
    inherited=SimpleNamespace(**vars(a));inherited.protocol=a.training_protocol
    inherited.max_seconds=tp['resources']['max_seconds'];inherited.max_allocated_gib=tp['resources']['max_allocated_gib']
    tp,contract,mods,cache_module,engineering,_=trainer.load_contract(inherited)
    helper=mods['train_source_motion_v1'];native=mods['native_state_cache'];common=mods['common_change_evaluation_v2']
    oracle=mods['oracle_transport_probe'];tree=mods['joint_native_evaluation'];core=mods['object_state_conditioner_v2']
    metric=contract.import_bound('common_occupancy_change_metrics_v1',p['sources_sha256'])
    train_manifest,train_summary,counts,train_receipt=endpoint.authenticate(a.final_training_run,a.training_complete_sha256,
        a.training_protocol,a.selection,Path(a.sparse_labels)/'manifest.json')
    dev_root,dev,dev_receipt=helper.cache_index(a.dev_cache,'development',tp)
    _,train,_=helper.cache_index(a.train_cache,'train',tp)
    require(len({r['scene_token'] for r in dev})==100 and not ({r['scene_token'] for r in dev}&{r['scene_token'] for r in train}),
            'Development scenes/partition differ')
    require(sha(a.o_development_records)==common.DEV_REFERENCE_SHA,'Original O reference changed')
    reference=endpoint.jsonl(a.o_development_records)
    require(len(reference)==len(dev)==200 and all(all(x[k]==y[k] for k in ('sample_token','scene_token'))
            for x,y in zip(reference,dev)),'Original O record order differs')
    geometry_cache=cache_module.GeometryCache(a.geometry_cache,tp['geometry_cache_complete_sha256'],a.selection)
    label_root,labels=helper.labels_manifest(a.sparse_labels,tp)
    raw_root,descriptors=contract.label_descriptors(a.raw_metadata,dev,common)
    raw_desc={r['identity']['sample_token']:r for r in descriptors}
    for record in dev:helper.check_identity(record,labels[record['sample_token']])
    cuda_ready=False;evaluated=0
    def budget():
        require(not stop and time.monotonic()-started<a.max_seconds,'Evaluation signal/time ceiling')
        if cuda_ready:require(torch.cuda.max_memory_allocated(a.device)<=a.max_allocated_gib*2**30,'Evaluation memory ceiling')
    def progress(phase):
        budget();value=dict(schema=SCHEMA,phase=phase,samples=evaluated,optimizer_updates=0,seconds=time.monotonic()-started)
        write(out/'progress.json',value);print(json.dumps(value),flush=True)
    progress('authenticated_fixed_training')
    import torch
    require(str(a.device).startswith('cuda') and torch.cuda.is_available(),'Native CUDA required')
    torch.cuda.set_device(a.device);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(min(1.,a.max_allocated_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory),a.device)
    torch.cuda.reset_peak_memory_stats(a.device);cuda_ready=True
    rigid=contract.import_bound('rigid_object_motion_v1',p['sources_sha256'])
    object_core=contract.import_bound('joint_rigid_object_features_v1',p['sources_sha256'])
    bridge=contract.import_bound('shared_rigid_native_bridge_v1',p['sources_sha256'])
    baseline=native.build_native_model(a.config,a.checkpoint,device=a.device,repo=a.repo)
    original=torch.load(a.o_checkpoint,map_location='cpu',weights_only=False)
    baseline.future_pred_head.load_state_dict(original['future_pred_head'],strict=True);del original
    baseline.eval()
    for parameter in baseline.parameters():parameter.requires_grad_(False)
    O_sha=helper.state_digest(baseline)
    require(common.historical_O_head_digest(baseline.future_pred_head)==common.O_HEAD_SHA and
            O_sha==train_manifest['initial_O_state_sha256']==engineering['initial_O_state_sha256'],'Original full O differs')
    models={};objects={};loaded={}
    for arm in ARMS:
        checkpoint=train_summary['final_checkpoints'][arm];path=Path(a.final_training_run)/checkpoint['file']
        require(sha(path)==checkpoint['sha256'],'Actual final checkpoint bytes changed')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        require(payload['manifest_sha256']==sha(Path(a.final_training_run)/'runs'/arm/'manifest.json') and
                payload['common_manifest_sha256']==train_receipt['manifest_sha256'],'Checkpoint manifest differs')
        models[arm]=copy.deepcopy(baseline);objects[arm]=object_core.JointRigidObjectFeatures().to(a.device)
        models[arm].future_pred_head.load_state_dict(payload['future_pred_head'],strict=True)
        objects[arm].load_state_dict(payload['joint_rigid_object_features'],strict=True)
        loaded[arm]=endpoint.validate_payload(torch,payload,arm,models[arm],objects[arm],train_manifest,train_summary,counts,helper)
        loaded[arm]['checkpoint_sha256']=checkpoint['sha256'];del payload
        for module in (models[arm],objects[arm]):
            module.eval()
            for parameter in module.parameters():parameter.requires_grad_(False)
        budget()
    write(out/'loaded_models.json',dict(original_O_state_sha256=O_sha,arms=loaded,optimizer_updates=0,D_model_loaded=False))
    sources=dict(protocol_sha256=sha(a.protocol),sources_sha256=p['sources_sha256'],training=train_receipt,
        analysis_sources_sha256=p['analysis_sources_sha256'],
        original_O_records_sha256=common.DEV_REFERENCE_SHA,selection_sha256=sha(a.selection),development_cache=dev_receipt,
        geometry_cache=geometry_cache.receipt,sparse_labels=tp['labels'],raw_manifest_sha256=sha(raw_root/'manifest.json'))
    write(out/'manifest.json',dict(schema=SCHEMA,evaluation=EVALUATION,numerical_policy=pre.NUMERICAL,sources=sources,
        loaded_models_sha256=sha(out/'loaded_models.json'),training_weights_fixed=True,optimizer_updates=0))
    rows=[];physical=[];initialization_seconds=time.monotonic()-started;progress('development')
    with torch.no_grad():
        for ordinal,record in enumerate(dev):
            budget();tick=time.monotonic();sample=common.input_only(native,dev_root,record,a.device)
            forward_sample=dict(inputs=sample['inputs'],record=sample['record']);boundary=tree.tree_digest(sample['inputs'])
            geometry=geometry_cache.load('development',ordinal,common.identity(record))
            data,state_receipt=pre.device_geometry(torch,core,geometry,a.device);del geometry
            geometry_sha=tree.tree_digest(data);predictions={'O':native.replay(baseline,forward_sample,training=False)[0]}
            flows={'CRN_CV':pre.cv_reference(torch,bridge,data,oracle)};hooks={}
            for arm in ARMS:
                output=pre.object_forward(objects[arm],data,forward_sample['inputs']['prev_bev_input'][:,-1],oracle,arm=='Cpl')
                flows[arm]=pre.material_field(rigid,bridge,data,output,oracle)
                with bridge.inject_future_object_features(models[arm].future_pred_head,output['native_delta']) as audit:
                    predictions[arm]=native.replay(models[arm],forward_sample,training=False)[0]
                hooks[arm]=list(audit);del output
            pre.finite_tensors(torch,list(predictions.values())+list(flows.values()),'Nonfinite final prediction')
            require('targets' not in forward_sample,'GT crossed forward boundary')
            # All model outputs, ownership and complete fields already exist.
            gt=common.load_targets_after_predictions(native,dev_root,record,sample,a.device)
            ld=labels[record['sample_token']];label=helper.load_sparse(label_root,ld,record)
            rd=raw_desc[record['sample_token']];path=raw_root/rd['file'];require(sha(path)==rd['sha256'],'Raw label changed')
            raw_bytes=gzip.decompress(path.read_bytes());require(hashlib.sha256(raw_bytes).hexdigest()==rd['uncompressed_json_sha256'],'Raw JSON changed')
            raw=json.loads(raw_bytes);hist={};metrics={}
            for arm,prediction in predictions.items():
                hist[arm]=common.native_hist(baseline,prediction,sample,record)
                if arm=='O':require(hist[arm]==reference[ordinal]['hist_by_horizon'],'Original O confusion differs')
                binary=common.fine_binary(prediction,oracle,budget)
                metrics[arm]=metric.evaluate_common_occupancy_change(binary,gt,oracle.EXTENT,raw)
                require([h['occupancy']['confusion'] for h in metrics[arm]['horizons']]==hist[arm],'Native versus fine-GT CPU counts differ')
                del binary
            verify_common_denominators(metrics);physical_counts={}
            for arm,flow in flows.items():
                values=[dict(item,arm=arm,ordinal=ordinal) for item in helper.epe_records(helper.gather_sparse(flow,label),label,record)]
                physical_counts[arm]=len(values);physical.extend(values)
                for item in values:append(out/'objects.jsonl',item)
            require(len(set(physical_counts.values()))==1,'Physical support differs')
            physical_counts['D']=physical_counts['CRN_CV']  # Verified against authenticated D rows below.
            require(tree.tree_digest(sample['inputs'])==boundary and tree.tree_digest(data)==geometry_sha,'Evaluation mutated inputs')
            row=dict(ordinal=ordinal,**common.identity(record),hist_by_arm=hist,metrics_by_arm=metrics,
                physical_object_rows_by_arm=physical_counts,inputs_sha256=record['files']['inputs']['sha256'],
                targets_sha256=record['files']['targets']['sha256'],raw_label_sha256=rd['sha256'],sparse_label_sha256=ld['sha256'],
                O_reference_hist_exact=True,native_CPU_hist_exact=True,GT_common_denominators_exact=True,
                t0_prediction_policy='model_specific_native',state_source=state_receipt,hooks=hooks,
                GT_values_used_only_after_predictions=True,seconds=time.monotonic()-tick)
            append(out/'records.jsonl',row);rows.append(row);evaluated+=1;progress('development')
            del sample,forward_sample,data,predictions,flows,gt,label,raw_bytes,raw,hist,metrics
    require(evaluated==200 and helper.state_digest(baseline)==O_sha,'Incomplete evaluation or O changed')
    for arm in ARMS:
        require(helper.state_digest(models[arm].future_pred_head)==loaded[arm]['actual_tensor_digests']['head'] and
                helper.state_digest(objects[arm])==loaded[arm]['actual_tensor_digests']['new'] and
                trainer.u.non_head_digest(models[arm])==train_manifest['frozen_non_head_state_sha256'],'Evaluation changed model')
    # References enter only scoring, after every candidate forward is finished.
    references=contract.import_bound('summarize_object_state_physical_v1',p['sources_sha256'])
    D,CV,embedded_D,cv_summary,reference_receipt=references.reference_objects(a.d_training_run,a.crn_cv_root)
    references.exact_reference_D(embedded_D,D)
    lookup={r['sample_token']:r['ordinal'] for r in rows}
    for item in D:
        value=dict(item,ordinal=lookup[item['sample_token']]);physical.append(value);append(out/'objects.jsonl',value)
    old_math=contract.import_bound('summarize_future_state_physical_v1',p['sources_sha256'])
    grouped,keys=old_math.physical_rows(physical,{r['sample_token']:r['scene_token'] for r in rows},('D','CRN_CV','Cpl','Fix'))
    require(len(keys)==16074,'Original physical object-horizon coverage differs')
    support=old_math.check_manifest(rows,grouped,Path(a.sparse_labels)/'manifest.json')
    require(support==cv_summary['support'],'Original full physical support differs')
    historical={(r['sample_token'],r['instance_token'],r['horizon_seconds']):r for r in CV};differences=[]
    for key in keys:
        now,old=grouped['CRN_CV'][key],historical[key]
        for field in ('scene_token','source_points','dt_seconds','group','zero_epe_xy_m','zero_epe_3d_m'):
            require(now[field]==old[field],'Historical CRN label/support differs')
        for field in ('epe_xy_m','epe_3d_m'):
            differences.append(abs(now[field]-old[field]))
            require(math.isclose(now[field],old[field],rel_tol=1e-12,abs_tol=1e-10),'CRN baseline differs beyond rounding bound')
    for row in rows:
        for arm in ('D','CRN_CV','Cpl','Fix'):
            require(row['physical_object_rows_by_arm'][arm]==sum(k[0]==row['sample_token'] for k in grouped[arm]),'Physical count missing')
    write(out/'reference_checks.json',dict(reference_authentication=reference_receipt,support=support,
        CRN_CV_max_absolute_EPE_difference_m=max(differences),physical_rows_per_arm=len(keys),
        all_original_support_retained=True,D_model_loaded=False))
    for name,digest in p['sources_sha256'].items():require(sha(contract.source_path(name))==digest,'Source changed during evaluation')
    for name,digest in p['analysis_sources_sha256'].items():require(sha(Path(__file__).with_name(name))==digest,'Analysis source changed')
    require(sha(a.protocol)==sources['protocol_sha256'] and sha(Path(a.final_training_run)/'complete.json')==a.training_complete_sha256,
            'Frozen endpoint changed during evaluation')
    summary=dict(schema=SCHEMA,status=STATUS,evaluated_samples=200,scenes=100,optimizer_updates=0,
        scores=common.summarize(rows,('O','Cpl','Fix'),metric),
        physical={arm:helper.epe_summary([r for r in physical if r['arm']==arm]) for arm in ('D','CRN_CV','Cpl','Fix')},
        all_original_O_hist_exact=True,all_common_GT_support_exact=True,all_model_states_unchanged=True,
        actual_final_checkpoint_tensors_checked=True,bootstrap_performed=False,no_native_O_flow_claim=True,
        resources=dict(initialization_seconds=initialization_seconds,elapsed_seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device)))
    write(out/'summary.json',summary);budget()
    files=('manifest.json','loaded_models.json','records.jsonl','objects.jsonl','reference_checks.json','summary.json')
    write(out/'complete.json',dict(schema=SCHEMA,status=STATUS,evaluated_samples=200,scenes=100,optimizer_updates=0,
        fixed_training_complete_sha256=a.training_complete_sha256,files_sha256={name:sha(out/name) for name in files},
        elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','training-protocol','final-training-run','training-complete-sha256','preflight-protocol',
                 'engineering-preflight','connected-protocol','training-run','config','checkpoint','o-checkpoint','repo',
                 'runtime-contract','train-cache','dev-cache','sparse-labels','predictions','raw-metadata','selection',
                 'geometry-cache','o-development-records','d-training-run','crn-cv-root','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--max-seconds',required=True,type=float)
    parser.add_argument('--max-allocated-gib',required=True,type=float);a=parser.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();stop=[]
    signal.signal(signal.SIGTERM,lambda s,f:stop.append(s));signal.signal(signal.SIGINT,lambda s,f:stop.append(s))
    try:run(a,out,started,stop)
    except BaseException as exc:
        write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),
            last_progress=read(out/'progress.json') if (out/'progress.json').exists() else None,seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
