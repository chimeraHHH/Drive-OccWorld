"""Fixed train[:16] D displacement/support diagnosis; no update or selection.

O and D baselines finish before fine GT / raw boxes / sparse array contents are
read. The labelled branch is explicitly privileged: only valid[h] sparse XYZ
source indices receive rigid-box GT displacement; all other D flow bytes stay
unchanged. O source probability/mask and D gate PARAMETERS remain fixed; gate
activations are recomputed from each branch's S/W. No GT source mask is used.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np

SCHEMA = 'partial-oracle-D-train-probe-v1'
ARMS = ('O','D','partial_oracle_D')
COMMON_SHA = '57cf304bafb050fac05b06abe68ce4600a37ca020d9d10d0673ec0ab1edb3bc1'
PROTOCOL_SHA = 'e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
TRAIN_COMPLETE_SHA = 'f0bf075fa36622b38c0c5136b187a184d16d3e07ed00857dfe21a0545f1c5f56'
D_CHECKPOINT_SHA = '7e2750d56ade7fb36e334b6431e83f51aafa997357744780a507924a696951e3'
O_REFERENCE_SHA = '4685394ec5d551a831252a5b1a35d951bc5fd88026531fac63b75f862715d76b'
RUNTIME_SHA = '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c'
RAW_MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
RAW_COMPLETE_SHA = 'e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69'
PHYSICAL_GROUPS = ('stationary','ambiguous','moving')


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def read(path,expected=None):
    if expected is not None: require(sha(path)==expected,'Source SHA mismatch: '+str(path))
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def patch_displacement(prediction,label):
    """NumPy-only, exact XYZ C-order replacement; usable by tiny CPU tests."""
    source=np.asarray(prediction)
    require(source.dtype==np.float32 and source.ndim==6 and source.shape[:3]==(1,4,3)
            and np.isfinite(source).all(),'Displacement must be finite float32 [1,4,3,X,Y,Z]')
    ids=np.asarray(label['source_flat_indices']);valid=np.asarray(label['valid'])
    target=np.asarray(label['target_displacement_m']);n=len(ids);shape=source.shape[3:];size=math.prod(shape)
    require(ids.dtype==np.int64 and ids.shape==(n,) and (n==0 or (ids[0]>=0 and ids[-1]<size and np.all(np.diff(ids)>0))), 'Unique ascending XYZ flat source indices required')
    require(valid.dtype==np.bool_ and valid.shape==(4,n) and target.dtype==np.float32
            and target.shape==(4,n,3) and np.isfinite(target).all(),'Sparse valid/target shape or dtype differs')
    if 'grid_shape_xyz' in label: require(tuple(label['grid_shape_xyz'])==shape,'Sparse/native XYZ shape differs')
    result=np.array(source,copy=True,order='C');mask=np.zeros((4,size),dtype=bool);rows=[]
    for h in range(4):
        chosen=valid[h];positions=ids[chosen];mask[h,positions]=True
        original=source[0,h].reshape(3,size)
        patched=result[0,h].reshape(3,size)
        patched[:,positions]=target[h,chosen].T
        require(np.array_equal(patched[:,~mask[h]].view(np.uint32),original[:,~mask[h]].copy().view(np.uint32)), 'Unlabelled/invalid displacement bytes changed')
        require(np.array_equal(patched[:,positions].T.copy().view(np.uint32),target[h,chosen].copy().view(np.uint32)), 'Patched XYZ labels differ')
        rows.append(dict(horizon_index=h+1,valid_labelled_source_points=int(chosen.sum()),
            changed_source_points=int(np.any(patched[:,positions].view(np.uint32)!=original[:,positions].copy().view(np.uint32),axis=0).sum()),
            changed_coordinate_values=int(np.count_nonzero(patched[:,positions].view(np.uint32)!=original[:,positions].copy().view(np.uint32)))))
    require(array_sha(source)==array_sha(prediction),'Input displacement modified')
    return result,dict(input_sha256=array_sha(source),output_sha256=array_sha(result),horizons=rows,
        unlabelled_and_invalid_displacement_bytes_unchanged=True,source_frame='fixed t0 R',flat_order='XYZ C-order',
        semantics='Only a labelled valid source subset is privileged; non-source predictions do not become GT sources')


def source_support(label,foreground):
    """Unique raw-box virtual points are denominators, not whole visibility."""
    flat=np.asarray(foreground)
    require(flat.dtype==np.bool_ and flat.ndim==3,'Source mask must be boolean physical XYZ')
    ids=label['source_flat_indices'];obj=label['object_index'];current=flat.reshape(-1)[ids]
    tokens=label['instance_tokens'];rows=[]
    for h in range(4):
        groups={g:dict(valid_points=0,predicted_source_points=0,missing_source_points=0,
                      valid_objects=0,objects_with_any_predicted_source=0,objects_without_predicted_source=0) for g in PHYSICAL_GROUPS}
        for k in range(len(tokens)):
            selected=label['valid'][h] & (obj==k);n=int(selected.sum())
            if not n:continue
            group=PHYSICAL_GROUPS[int(label['object_speed_group'][h,k])];got=int(current[selected].sum())
            entry=groups[group];entry['valid_points']+=n;entry['predicted_source_points']+=got
            entry['missing_source_points']+=n-got;entry['valid_objects']+=1
            entry['objects_with_any_predicted_source']+=int(got>0);entry['objects_without_predicted_source']+=int(got==0)
        rows.append(dict(horizon_index=h+1,actual_dt_seconds=float(label['dt_future_seconds'][h]),groups=groups))
    return dict(horizons=rows,denominator='Original NPZ unique rigid-box virtual source points/retained objects, valid at each future',
        complete_object_visibility_claim=False,current_source='Frozen O coarse class1 > class0',
        unsupported_replacements_have_no_mass_contribution=True)


def positive_group_indices(gt,raw_info,h,extent,metric):
    """Same frozen geometry/group primitives; only evaluation-side positive indices.

    Match _motion_attribution's future boxes and point centres. Every group's
    GT/TP/FN is checked against the unchanged public metric for all three arms.
    """
    truth=gt==1;indices=np.argwhere(truth);frame=h+2
    active=[t for t in raw_info['tracks'] if t['valid_mask'][frame]]
    inverse=np.linalg.inv(raw_info['G0'])
    boxes=np.asarray([inverse@metric.box_to_global(t['global_centers_m'][frame],t['global_rotations_wxyz'][frame]) for t in active]).reshape(-1,4,4)
    sizes=np.asarray([t['sizes_wlh_m'][frame] for t in active]).reshape(-1,3)
    bounds=np.asarray(extent,dtype=np.float64)
    points=bounds[:3]+(indices.astype(np.float64)+.5)*((bounds[3:]-bounds[:3])/np.asarray(gt.shape,dtype=np.float64))
    owners=metric.unique_box_assignment(points,boxes,sizes)
    codes=np.empty(len(owners),dtype=np.int8)
    names=list(metric.POSITIVE_GROUPS)
    codes[owners==-1]=names.index('unknown_no_current_box');codes[owners==-2]=names.index('overlap_current_boxes')
    for k,t in enumerate(active):
        if t['valid_mask'][2]:
            delta=(np.asarray(t['global_centers_m'][frame])-np.asarray(t['global_centers_m'][2]))[:2]
            group=metric.speed_group(float(np.linalg.norm(delta)/raw_info['actual_dt_seconds'][h]))
        else:group='t0_missing_with_history' if any(t['valid_mask'][:2]) else 'annotated_future_only'
        codes[owners==k]=names.index(group)
    return tuple(indices.T),codes


def stats(values):
    values=np.asarray(values,dtype=np.float64).reshape(-1)
    require(np.isfinite(values).all(),'Nonfinite diagnostic distribution')
    return dict(count=int(len(values)),mean=None if not len(values) else float(values.mean()),
        p10=None if not len(values) else float(np.quantile(values,.1)),
        median=None if not len(values) else float(np.median(values)),
        p90=None if not len(values) else float(np.quantile(values,.9)))


def fine_influence(coarse,fine_shape):
    """Literal native fine_binary interpolation layout [N,C,X,Y,Z], no warp."""
    import torch
    require(coarse.dtype==torch.bool and coarse.ndim==3,'Coarse influence must be XYZ bool')
    fine=torch.nn.functional.interpolate(coarse[None,None].float(),size=tuple(fine_shape),mode='trilinear',align_corners=False)[0,0]
    return (fine>0).cpu().numpy()


def support_partition_report(binary,gt,raw,fields,metrics,extent,metric,check):
    import torch
    info=metric.validate_raw_label(raw);rows=[]
    for h in range(1,5):
        check();ix,codes=positive_group_indices(gt[h],info,h,extent,metric)
        values={a:binary[a][h][ix] for a in ARMS}
        groups={name:codes==i for i,name in enumerate(metric.POSITIVE_GROUPS)}
        groups['all_GT_positive']=np.ones(len(codes),dtype=bool)
        groups['speed_gt_0.5']=np.isin(codes,[2,3])
        for arm in ARMS:
            ref=metrics[arm]['horizons'][h]['motion_positive_attribution']['groups']
            for name in metric.POSITIVE_GROUPS:
                selected=groups[name];tp=int(values[arm][selected].sum());n=int(selected.sum())
                require([n,tp,n-tp]==[ref[name][k] for k in ('GT','TP','FN')],'Independent positive index groups differ from original metric')
        wd=fields['D']['support_weight'][0,h-1,0]>0
        wp=fields['partial_oracle_D']['support_weight'][0,h-1,0]>0
        domains={'O_to_D':wd,'O_to_partial_oracle_D':wp,'D_to_partial_oracle_D':wd|wp}
        comparisons={'O_to_D':('O','D'),'O_to_partial_oracle_D':('O','partial_oracle_D'),
                     'D_to_partial_oracle_D':('D','partial_oracle_D')}
        detail={}
        for name,coarse in domains.items():
            mask=fine_influence(coarse,gt.shape[1:]);within=mask[ix];first,second=comparisons[name]
            require(np.array_equal(binary[first][h][~mask],binary[second][h][~mask]),'Output changed outside conservative union/support influence domain')
            grouped={}
            for group,selected in groups.items():
                row=dict(GT=int(selected.sum()),inside_domain_GT=int((selected&within).sum()),outside_domain_GT=int((selected&~within).sum()),arms={})
                for arm in ARMS:
                    fn=selected&~values[arm]
                    row['arms'][arm]=dict(FN=int(fn.sum()),FN_inside=int((fn&within).sum()),FN_outside=int((fn&~within).sum()))
                for transition,changed in [('TP_to_FN',values[first]&~values[second]),('FN_to_TP',~values[first]&values[second])]:
                    chosen=selected&changed
                    row[transition]=dict(total=int(chosen.sum()),inside=int((chosen&within).sum()),outside=int((chosen&~within).sum()))
                grouped[group]=row
            valid=gt[h]!=255
            detail[name]=dict(first=first,second=second,coarse_support_cells=int(coarse.sum()),fine_domain_voxels=int(mask.sum()),
                fine_valid_domain_voxels=int((mask&valid).sum()),groups=grouped,outputs_outside_domain_identical=True)
            del mask
        rows.append(dict(horizon_index=h,actual_dt_seconds=info['actual_dt_seconds'][h],domains=detail))
    return dict(horizons=rows,definition='trilinear((coarse W>0).float), original XYZ fine size, align_corners=False, then >0; D-to-partial uses union W',
        scope='Conservative interpolation-dependency support bound; not physical reachability, attainable correct-IoU upper bound or visibility',
        full_GT_denominators_preserved=True,future_only_annotation_status_is_not_proven_birth_or_disocclusion=True)


def transport_report(field,fused,future):
    rows=[]
    for h in range(4):
        support=field['support_weight'][0,h,0]>0
        gate=fused['gate'][0,h,0];want=fused['transported_logodds'][0,h,0]-future[0,h,0]
        output=fused['logodds'][0,h,0]-future[0,h,0]
        rows.append(dict(horizon_index=h+1,destination_support_cells=int(support.sum()),
            source_probability_mass=float(field['source_probability_mass'][0,0]),
            source_foreground_voxels=int(field['source_foreground_voxels'][0,0]),
            retained_probability_mass=float(field['probability_mass'][0,h].sum()),
            retained_support_weight=float(field['support_weight'][0,h].sum()),
            collision_destinations=int((field['support_weight'][0,h,0]>1).sum()),
            gate_on_supported=stats(gate[support].detach().cpu().numpy()),
            gate_where_transport_increases_margin=stats(gate[support&(want>0)].detach().cpu().numpy()),
            gate_where_transport_decreases_margin=stats(gate[support&(want<0)].detach().cpu().numpy()),
            actual_mixed_margin_change_on_supported=stats(output[support].detach().cpu().numpy())))
    return dict(horizons=rows,gate_parameters_fixed=True,gate_activations_recomputed_from_each_S_W=True,
                gate_suppression_causal_claim=False,no_gate_one_or_maximum_counterfactual=True)


def modules_and_contract(a):
    path=Path(__file__).with_name('common_connected_motion_evaluation_v2.py')
    require(sha(path)==COMMON_SHA,'Frozen common loader source changed')
    spec=importlib.util.spec_from_file_location('common_connected_motion_evaluation_v2',path)
    contract=importlib.util.module_from_spec(spec);sys.modules[spec.name]=contract;spec.loader.exec_module(contract)
    p=read(a.protocol,PROTOCOL_SHA)
    require(p['schema']=='connected-motion-training-v2' and p['status']=='FROZEN','Wrong frozen training protocol')
    sources=contract.check_sources(p);sources[Path(__file__).name]=sha(__file__)
    contract.import_bound('motion_geometry',sources)
    names=('train_connected_motion_v2','train_source_motion_v1','native_state_cache','oracle_transport_probe',
           'common_change_evaluation_v2','common_occupancy_change_metrics_v1','train_supported_fusion_v2','joint_native_evaluation')
    modules={n:contract.import_bound(n,sources) for n in names}
    root=Path(a.training_run)
    done=read(root/'complete.json',TRAIN_COMPLETE_SHA)
    require(a.training_complete_sha256==TRAIN_COMPLETE_SHA and not (root/'failed.json').exists()
            and done['status']=='COMPLETE_CONNECTED_MOTION_TRAINING' and done['mode']=='train'
            and done['updates']==512 and done['examples']==2048 and done['evaluated_samples']==200
            and done['final_checkpoints']['D']['sha256']==D_CHECKPOINT_SHA,'Wrong actual fixed final D completion')
    for name,digest in done['files_sha256'].items():require(sha(root/name)==digest,'Final training artifact changed: '+name)
    training_manifest=read(root/'manifest.json')
    require(training_manifest['sources']['protocol_sha256']==PROTOCOL_SHA,'Training source protocol changed')
    helper=modules['train_source_motion_v1'];common=modules['common_change_evaluation_v2']
    cache,train,cache_receipt=helper.cache_index(a.train_cache,'train',p)
    label_root,labels=helper.labels_manifest(a.sparse_labels,p)
    raw_root=Path(a.raw_labels);raw_manifest=read(raw_root/'manifest.json',RAW_MANIFEST_SHA)
    raw_done=read(raw_root/'complete.json',RAW_COMPLETE_SHA)
    require(raw_done['status']=='COMPLETE' and raw_done['manifest_sha256']==RAW_MANIFEST_SHA
            and raw_done['samples']==712 and raw_done['optimizer_steps']==0,'Raw labels incomplete')
    raw={r['identity']['sample_token']:r for r in raw_manifest['records']}
    require(len(raw)==len(raw_manifest['records'])==712,'Raw duplicate identities')
    selected=train[:a.anchors]
    require(a.anchors in (2,16),'Only fixed original first2/16 train anchors')
    for i,r in enumerate(selected):
        token=r['sample_token'];helper.check_identity(r,labels[token])
        require(r['split']=='train' and labels[token]['ordinal']==i and raw[token]['ordinal']==i
                and raw[token]['identity']==common.identity(r)
                and labels[token]['label_source_sha256']==raw[token]['sha256'],'Raw/sparse/cache training order differs')
    runtime=read(a.runtime_contract,RUNTIME_SHA)
    for path,digest in runtime['runtime_source_sha256'].items():require(sha(path)==digest,'Native runtime source changed: '+path)
    oracle=modules['oracle_transport_probe']
    require(sha(a.config)==oracle.CONFIG_SHA and sha(a.checkpoint)==oracle.M0_SHA
            and sha(a.o_checkpoint)==oracle.O_SHA and sha(a.o_reference)==O_REFERENCE_SHA,'Fixed O/M0/config/reference changed')
    return contract,p,sources,modules,training_manifest,cache,selected,cache_receipt,label_root,labels,raw_root,raw


def run(a,out,started,stopping):
    contract,p,sources,mods,training_manifest,cache,selected,cache_receipt,label_root,labels,raw_root,raw_descriptors=modules_and_contract(a)
    helper=mods['train_source_motion_v1'];native=mods['native_state_cache'];oracle=mods['oracle_transport_probe']
    common=mods['common_change_evaluation_v2'];metric=mods['common_occupancy_change_metrics_v1']
    composer=mods['train_supported_fusion_v2'];trainer=mods['train_connected_motion_v2'];tree=mods['joint_native_evaluation']
    manifest=dict(schema=SCHEMA,mode='train_only_diagnostic',anchors=a.anchors,
        source_sha256=sha(__file__),sources_sha256=sources,protocol_sha256=PROTOCOL_SHA,
        training_complete_sha256=TRAIN_COMPLETE_SHA,D_checkpoint_sha256=D_CHECKPOINT_SHA,
        O_checkpoint_sha256=oracle.O_SHA,M0_checkpoint_sha256=oracle.M0_SHA,config_sha256=oracle.CONFIG_SHA,
        cache=cache_receipt,raw_labels_manifest_sha256=RAW_MANIFEST_SHA,raw_labels_complete_sha256=RAW_COMPLETE_SHA,
        sparse_labels=p['labels'],O_reference_sha256=O_REFERENCE_SHA,selected=[common.identity(r) for r in selected],
        sample_order='original frozen train cache records[:anchors], not permutation order',
        arms=list(ARMS),privileged_arms=['partial_oracle_D'],candidate_arms=['O','D'],
        numerical_policy=p['numerical_policy'],seed=11,optimizer_updates=0,training=False,
        gate_parameters_frozen=True,gate_activations_recomputed=True,source_mask_is_original_O_current_argmax=True,
        no_GT_source_mask=True,no_GOSPA_or_new_metric=True,no_maximum_or_gate_one_oracle=True,
        purpose='Source support / displacement replacement / fixed gate response diagnosis, not attainable upper bound or candidate promotion',
        resources=dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib))
    write(out/'manifest.json',manifest)
    import torch
    require(str(a.device).startswith('cuda') and torch.cuda.is_available(),'Native O needs CUDA')
    torch.cuda.set_device(a.device);torch.set_num_threads(2)
    random.seed(11);np.random.seed(11);torch.manual_seed(11);torch.cuda.manual_seed_all(11)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    memory=torch.cuda.get_device_properties(a.device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,a.max_allocated_gib*2**30/memory),a.device)
    torch.cuda.reset_peak_memory_stats(a.device)
    def check():
        require(not stopping,'Stop signal; no retry')
        require(time.monotonic()-started<a.max_seconds,'Diagnostic wall-clock ceiling')
        require(torch.cuda.max_memory_allocated(a.device)<=a.max_allocated_gib*2**30,'Diagnostic memory ceiling')
    check()
    model=native.build_native_model(a.config,a.checkpoint,device=a.device,repo=a.repo)
    payload=torch.load(a.o_checkpoint,map_location='cpu',weights_only=False)
    model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True);del payload
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval()
    require(common.historical_O_head_digest(model.future_pred_head)==common.O_HEAD_SHA,'Historical O head identity changed')
    O_state=helper.state_digest(model)
    require(O_state==training_manifest['frozen_O_state_sha256'],'Actual O differs from D training')
    motion,gate,receipt=trainer.load_completed_arm(Path(a.training_run)/'runs'/'D',Path(a.protocol),helper,a.device)
    require(receipt['top_complete_sha256']==TRAIN_COMPLETE_SHA and receipt['checkpoint_sha256']==D_CHECKPOINT_SHA,'Wrong loaded D')
    fusion=contract.import_bound('supported_motion_fusion',sources)
    write(out/'loaded_models.json',dict(schema=SCHEMA,O_full_state_sha256=O_state,D=receipt,
        optimizer_restored=False,optimizer_updates=0,torch_version=torch.__version__,device=a.device))
    rows=[];reference=None
    with torch.no_grad(),(out/'records.jsonl').open('x') as stream:
        for ordinal,record in enumerate(selected):
            check();tick=time.monotonic();sample=common.input_only(native,cache,record,a.device)
            input_sha=tree.tree_digest(sample['inputs']);tokens=sample['inputs']['prev_bev_input'][:,-1]
            original=native.replay(model,sample,training=False)[0]
            base=oracle.predictions_to_xyz(original)
            probability=torch.softmax(base[0:1],dim=1)[:,1:2]
            foreground=base[0:1,1:2]>base[0:1,0:1]
            future=(base[1:,1]-base[1:,0])[None,:,None]
            field=motion(tokens)
            field_D=fusion.supported_transport(probability,foreground,field,oracle.EXTENT)
            fused_D=gate(future,field_D['probability_mass'],field_D['support_weight'])
            pred_D=composer.compose_prediction(original,base,fused_D,oracle)
            # Candidate baselines are complete before first sample GT/box/NPZ content read.
            torch.cuda.synchronize(a.device);baseline_seconds=time.monotonic()-tick
            require('targets' not in sample,'GT entered baseline prediction dictionary')
            flow_array=field.detach().cpu().numpy();flow_sha=array_sha(flow_array)
            source_sha=native._tensor_digest(foreground);probability_sha=native._tensor_digest(probability)
            O_prediction_sha=native._tensor_digest(original);D_prediction_sha=native._tensor_digest(pred_D)
            label=helper.load_sparse(label_root,labels[record['sample_token']],record)
            descriptor=raw_descriptors[record['sample_token']];raw_path=raw_root/descriptor['file']
            require(raw_path.stat().st_size==descriptor['bytes'] and sha(raw_path)==descriptor['sha256'],'Raw label bytes changed')
            raw_bytes=gzip.decompress(raw_path.read_bytes())
            require(hashlib.sha256(raw_bytes).hexdigest()==descriptor['uncompressed_json_sha256'],'Raw decoded JSON changed')
            raw=json.loads(raw_bytes);del raw_bytes
            require(raw['identity']==common.identity(record),'Raw sample identity mismatch')
            gt=common.load_targets_after_predictions(native,cache,record,sample,a.device)
            if reference is None:
                reference=[json.loads(line) for line in Path(a.o_reference).read_text().splitlines()]
                require(len(reference)==16 and all(set(r)=={'ordinal','sample_token','scene_token','hist_by_arm'}
                    and set(r['hist_by_arm'])=={'O'} for r in reference),'O-only reference contract')
            patched,patch_audit=patch_displacement(flow_array,label)
            privileged_field=torch.from_numpy(patched).to(a.device)
            field_P=fusion.supported_transport(probability,foreground,privileged_field,oracle.EXTENT)
            fused_P=gate(future,field_P['probability_mass'],field_P['support_weight'])
            pred_P=composer.compose_prediction(original,base,fused_P,oracle)
            require(array_sha(field.detach().cpu().numpy())==flow_sha,'Original D displacement mutated')
            require(native._tensor_digest(foreground)==source_sha and native._tensor_digest(probability)==probability_sha,'O source mask/probability mutated')
            require(native._tensor_digest(original)==O_prediction_sha and native._tensor_digest(pred_D)==D_prediction_sha,'Original O/D baseline predictions mutated')
            predictions={'O':original,'D':pred_D,'partial_oracle_D':pred_P}
            require(all(composer.exact32(value[0],original[0]) for value in predictions.values()),'t0 original logits changed')
            metrics={};histograms={};binary={}
            for arm in ARMS:
                check();hist=common.native_hist(model,predictions[arm],sample,record)
                if arm=='O':
                    require(reference[ordinal]['ordinal']==ordinal and all(reference[ordinal][k]==record[k] for k in ('sample_token','scene_token'))
                        and hist==reference[ordinal]['hist_by_arm']['O'],'Original O train16 reference differs')
                binary[arm]=common.fine_binary(predictions[arm],oracle,check)
                value=metric.evaluate_common_occupancy_change(binary[arm],gt,oracle.EXTENT,raw)
                require([h['occupancy']['confusion'] for h in value['horizons']]==hist,'Common CPU/native complete-GT histogram differs')
                if arm!='O':require(value['t0_boundary']==metrics['O']['t0_boundary'],'Common t0 binary/GT domain differs')
                metrics[arm]=value;histograms[arm]=hist
            support=source_support(label,foreground[0,0].cpu().numpy())
            influence=support_partition_report(binary,gt,raw,{'D':field_D,'partial_oracle_D':field_P},metrics,oracle.EXTENT,metric,check)
            transports={'D':transport_report(field_D,fused_D,future),'partial_oracle_D':transport_report(field_P,fused_P,future)}
            require(tree.tree_digest(sample['inputs'])==input_sha,'Original input tree changed')
            require(tree.array_digest(gt)==tree.array_digest(sample['targets'][0,2:].to(torch.uint8).cpu().numpy()),'Evaluation targets changed')
            require(not any(m.training for m in model.modules()) and not motion.training and not gate.training,'Frozen model mode changed')
            row=dict(schema=SCHEMA,ordinal=ordinal,**common.identity(record),hist_by_arm=histograms,metrics_by_arm=metrics,
                inputs_sha256=record['files']['inputs']['sha256'],target_file_sha256=record['files']['targets']['sha256'],
                input_tree_sha256=input_sha,raw_label_sha256=descriptor['sha256'],sparse_label_sha256=labels[record['sample_token']]['sha256'],
                patch=patch_audit,source_support=support,influence=influence,transport=transports,
                baseline_predictions_completed_before_GT_and_label_contents=True,original_O_hist_exact=True,
                all_t0_exact=True,baseline_O_D_predictions_unchanged=True,original_D_flow_unchanged=True,
                original_source_mask_probability_unchanged=True,optimizer_updates=0,
                baseline_seconds=baseline_seconds,seconds=time.monotonic()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
            print(json.dumps(dict(event='PARTIAL_ORACLE_D_SAMPLE',ordinal=ordinal,sample_token=record['sample_token'],seconds=row['seconds'])),flush=True)
            del sample,tokens,original,base,probability,foreground,future,field,field_D,fused_D,pred_D,label,raw,gt,flow_array,patched,privileged_field,field_P,fused_P,pred_P,predictions,binary,metrics
    check()
    require(helper.state_digest(model)==O_state and helper.state_digest(motion)==receipt['actual_motion_state_sha256']
            and helper.state_digest(gate)==receipt['actual_gate_state_sha256'],'Frozen model state changed')
    require(sha(a.protocol)==PROTOCOL_SHA and sha(Path(a.training_run)/'complete.json')==TRAIN_COMPLETE_SHA,'Frozen completion/protocol changed')
    for name,digest in sources.items():require(sha(contract.source_path(name))==digest,'Frozen script changed: '+name)
    summary=dict(schema=SCHEMA,status='COMPLETE_FIXED_TRAIN_PARTIAL_ORACLE_DIAGNOSTIC',anchors=len(rows),
        scenes=len({r['scene_token'] for r in rows}),arms=list(ARMS),scores=common.summarize(rows,ARMS,metric),
        physical_source_support_totals={str(h):{g:{k:sum(r['source_support']['horizons'][h-1]['groups'][g][k] for r in rows)
            for k in rows[0]['source_support']['horizons'][h-1]['groups'][g]} for g in PHYSICAL_GROUPS} for h in range(1,5)},
        optimizer_updates=0,weights_changed=False,privileged_branch_not_candidate=True,
        evaluation_population='Original fixed train cache first16; first2 mode engineering only' if a.anchors==16 else 'Original fixed train cache first2 engineering only',
        intervention='Valid sparse source points receive rigid-box GT; outside-labelled/invalid points keep D; O source and D gate parameters fixed',
        influence='Conservative interpolation dependency support, not attainable performance or physical reachability upper bound',
        gate_activation_change_is_part_of_fixed_system_response=True,
        no_model_promotion_or_generalization_conclusion=True,no_bootstrap_or_threshold_selection=True,
        seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
    write(out/'summary.json',summary)
    (out/'report.md').write_text('# 固定训练锚点局部 oracle 位移诊断\n\n'
        'O/D 原预测先完成，再读取标签；partial_oracle_D 使用有效标注源点的 GT 刚体位移，是 privileged 诊断，不是候选模型。'
        '其余位移、O 源掩码/概率及 D 门参数不变；门的激活值随运输场重算。\n\n'
        '完整 GT 与原八类速度/标注状态归因保留。源点支持以原 NPZ 唯一虚拟点为分母，不代表整个对象可见性。'
        'FN 和 TP↔FN 的域内外划分仅界定当前插值依赖域；D→partial 使用两运输场支持并集。'
        '没有最大拼接、完整可达上界或“运动无用”的结论。\n')
    write(out/'complete.json',dict(schema=SCHEMA,status=summary['status'],anchors=len(rows),optimizer_updates=0,
        source_sha256=sha(__file__),protocol_sha256=PROTOCOL_SHA,training_complete_sha256=TRAIN_COMPLETE_SHA,
        files_sha256={name:sha(out/name) for name in ('manifest.json','loaded_models.json','records.jsonl','summary.json','report.md')},seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','training-run','training-complete-sha256','config','checkpoint','o-checkpoint','repo',
                 'runtime-contract','train-cache','raw-labels','sparse-labels','o-reference','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--anchors',type=int,choices=(2,16),default=16)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('--max-allocated-gib',type=float,required=True)
    a=parser.parse_args()
    require(math.isfinite(a.max_seconds) and a.max_seconds>0 and math.isfinite(a.max_allocated_gib)
            and 0<a.max_allocated_gib<=32,'Positive explicit budget and <=32GiB required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();stopping=[]
    signal.signal(signal.SIGTERM,lambda s,f:stopping.append(s));signal.signal(signal.SIGINT,lambda s,f:stopping.append(s))
    try:run(a,out,started,stopping)
    except BaseException as exc:
        write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),
              optimizer_updates=0,seconds=time.monotonic()-started,source_sha256=sha(__file__)))
        raise


if __name__=='__main__':main()
