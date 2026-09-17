"""Cpl/Fix native four-update engineering preflight; no saved weights/dev scores.

Current predicted objects define a fixed material ownership map. Both arms learn
the same rigid residual and values. Only the Gaussian spatial-address edge uses
learned poses in Cpl and fixed CV poses in Fix. Injection is at the three native
decoder entrances, not into autoregressive memory. Physical motion has no path
through the native future head. No D model is constructed or loaded.

The protocol_template is importable without Torch. The caller freezes sources,
assets and resources before dispatch. This entry never authorizes a formal run.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

SCHEMA = 'shared-rigid-native-preflight-v1'
ARMS = ('Cpl', 'Fix')
COMMON_SHA = '57cf304bafb050fac05b06abe68ce4600a37ca020d9d10d0673ec0ab1edb3bc1'
UTILITY_SHA = '6160b6741f525abcebfccc2b3549b21abffc75b2263bfa5c31c3c8a5231eff9b'
# The historical connected run authenticates the original full O state only.
CONNECTED_PROTOCOL_SHA = 'e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
CONNECTED_COMPLETE_SHA = 'f0bf075fa36622b38c0c5136b187a184d16d3e07ed00857dfe21a0545f1c5f56'
RUNTIME_SHA = '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c'
NUMERICAL = dict(dtype='float32', matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False)
ADDITIONAL_SOURCES = (
    'object_state_forecast_preflight_v2.py', 'object_state_prediction_inputs_v1.py',
    'object_state_conditioner_v2.py', 'rigid_object_motion_v1.py',
    'joint_rigid_object_features_v1.py', 'shared_rigid_native_bridge_v1.py',
    'shared_rigid_prediction_geometry_v1.py', 'crn_full_grid_geometry_v1.py',
    'motion_geometry.py', 'shared_rigid_native_preflight_v1.py')
RECIPE = dict(seed=11, updates=4, examples=16, accumulate=4, train_samples=512,
    planned_formal_updates=512, formal_authorized=False, new_parameters=67366,
    future_head_parameters=13274016, future_head_lr=1e-5, future_head_grad_clip=35.,
    future_head_warmup_updates=50, new_lr=1e-3, new_grad_clip=10., new_warmup_updates=25,
    weight_decay=.01, optimizer='AdamW', betas=[.9,.999], eps=1e-8, amsgrad=False,
    physical_loss_coefficient=1., occupancy_loss_coefficient=1., smooth_l1_beta_m=.5,
    sample_order='first16 of first continuous RandomState11 permutation of 512',
    schedule='original O head 512-update warmup/cosine; train_source_motion_v1.schedule_lr for new module',
    occupancy_loss='original coarse twelve computed; original O CE[1,5]+Lovasz six selected',
    physical_loss='original sparse point_xyz/object/present_group/valid_horizon means',
    arm_difference='Cpl learned future Gaussian addresses; Fix fixed CV addresses; learned values and physical state unchanged',
    scope='whole native future_pred_head plus JointRigidObjectFeatures; non-head O frozen; no D',
    rng='same pre-forward Python/NumPy/CPU/CUDA state restored for paired Cpl/Fix microsteps',
    initialization='same actual O and copied seed11 new module; zero pose/projection give original O/CV',
    initial_loss_comparison='same-logit CE per term rel1e-5 abs1e-7; Lovasz exact; physical scalar descriptive; dense CV bytes exact',
    geometry='current centered predictions and current frame2 G0 only; all GMO boxes; full 640000 material grid before labels',
    physical_field='float64 source_displacement_field; fixed current ownership; no GT filtering or future-head input',
    injection='float32 additive future feature field at all three decoder entrances; no autoregressive memory injection',
    spatial_queries='all BEV y*X+x cells; exact chunks2048; no topk, threshold, ROI or GT mask',
    native_projection_bias=False, late_vjp_microstep_index=12,
    late_vjp_groups=['object_encoder','pose_head'], late_vjp_losses=['occupancy','physical'],
    physical_to_future_head='unused by design; None is expected, not a failure',
    initial_occupancy_to_pose='zero projection initially blocks this path; no nonzero initialization gate',
    address_intervention_after_updates=4, address_intervention_ordinal='fixed selected[12]',
    address_intervention='same weights Cpl versus Fix address only; identical values/physical; t0 exact, future native changes',
    weights_persisted=False, development_evaluation=False)


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def _contract_loader():
    path=Path(__file__).with_name('common_connected_motion_evaluation_v2.py')
    require(sha(path)==COMMON_SHA, 'Source loader changed')
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def expected_sources(connected_protocol):
    """Root can call this on the existing connected protocol to fill the template."""
    contract=_contract_loader()
    expected=dict(contract.check_sources(connected_protocol))
    for name in ADDITIONAL_SOURCES: expected[name]=sha(contract.source_path(name))
    require(expected['object_state_forecast_preflight_v2.py']==UTILITY_SHA, 'Utility source changed')
    return expected


def protocol_template():
    return dict(schema=SCHEMA,status='REVIEW_REQUIRED',recipe=copy.deepcopy(RECIPE),
        numerical_policy=dict(NUMERICAL),sources_sha256={},
        connected_protocol_sha256=CONNECTED_PROTOCOL_SHA,connected_complete_sha256=CONNECTED_COMPLETE_SHA,
        runtime_source_contract_sha256=RUNTIME_SHA,cache_index_sha256=dict(train=None,development=None),
        labels=dict(manifest_sha256=None,complete_sha256=None),
        selection_sha256='5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d',
        predictions=dict(complete_sha256=None,manifest_sha256=None,predictions_sha256=None,inference_source_sha256=None),
        resources=dict(preflight=dict(max_seconds=1200.,max_allocated_gib=32.),formal=None))


def load_contract(a):
    p=read(a.protocol)
    require(p['schema']==SCHEMA and p['status']=='FROZEN' and p['recipe']==RECIPE
            and p['numerical_policy']==NUMERICAL,'Frozen exact shared-rigid preflight required')
    require(p['resources']==dict(preflight=dict(max_seconds=a.max_seconds,
        max_allocated_gib=a.max_allocated_gib),formal=None),'Only frozen preflight resources allowed')
    require(math.isfinite(a.max_seconds) and 0<a.max_seconds<=1200 and
        math.isfinite(a.max_allocated_gib) and 0<a.max_allocated_gib<=32,'Finite bounded resources required')
    require(p['connected_protocol_sha256']==sha(a.connected_protocol)==CONNECTED_PROTOCOL_SHA and
        p['connected_complete_sha256']==sha(Path(a.training_run)/'complete.json')==CONNECTED_COMPLETE_SHA,
        'Historical O provenance changed')
    old=read(Path(a.training_run)/'complete.json')
    require(sha(Path(a.training_run)/'manifest.json')==old['files_sha256']['manifest.json'],
        'Historical full-O state manifest changed')
    contract=_contract_loader();connected=read(a.connected_protocol)
    expected=expected_sources(connected)
    require(p['sources_sha256']==expected,'Exact inherited and shared-rigid source set required')
    require(p['cache_index_sha256']==connected['cache_index_sha256'] and p['labels']==connected['labels'],
        'Native input or physical supervision contract changed')
    require(p['runtime_source_contract_sha256']==sha(a.runtime_contract)==RUNTIME_SHA,'Runtime contract changed')
    for path,digest in read(a.runtime_contract)['runtime_source_sha256'].items():
        require(sha(path)==digest,'Native runtime source changed: '+path)
    names=('train_source_motion_v1','native_state_cache','oracle_transport_probe','common_change_evaluation_v2',
           'train_supported_fusion_v2','objective_supervision_adapters','joint_native_evaluation',
           'object_state_forecast_preflight_v2','object_state_prediction_inputs_v1',
           'object_state_conditioner_v2','shared_rigid_prediction_geometry_v1')
    mods={name:contract.import_bound(name,expected) for name in names}
    oracle=mods['oracle_transport_probe']
    require(sha(a.config)==oracle.CONFIG_SHA and sha(a.checkpoint)==oracle.M0_SHA and
        sha(a.o_checkpoint)==oracle.O_SHA,'Native O assets changed')
    return p,contract,mods


def exact_bytes(left,right):
    """Dtype/shape-aware, including signed zero, for both float32 and float64."""
    return (left.dtype==right.dtype and tuple(left.shape)==tuple(right.shape) and
            left.detach().cpu().contiguous().numpy().tobytes()==right.detach().cpu().contiguous().numpy().tobytes())


def tensor_sha(value):
    value=value.detach().cpu().contiguous()
    h=hashlib.sha256(str(value.dtype).encode()+json.dumps(list(value.shape)).encode())
    h.update(value.numpy().tobytes());return h.hexdigest()


def current_geometry(inputs,ordinal,geometry,oracle):
    """Authenticate the carrier, then discard everything but the current G0."""
    record=inputs.records[ordinal];descriptor=inputs.raw[record['sample_token']]
    identity=('sample_token','scene_token','split','official_index')
    require(descriptor['ordinal']==ordinal and all(descriptor['identity'][k]==record[k] for k in identity),
            'Current pose identity differs')
    path=inputs.raw_root/descriptor['file'];require(sha(path)==descriptor['sha256'],'Current pose carrier changed')
    content=gzip.decompress(path.read_bytes())
    require(hashlib.sha256(content).hexdigest()==descriptor['uncompressed_json_sha256'],'Raw JSON changed')
    carrier=json.loads(content)
    require(carrier['identity']==descriptor['identity'] and carrier['ordinal']==ordinal and
            carrier['selection_sha256']==inputs.receipt['selection_sha256'],'Carrier identity changed')
    frame=carrier['frames'][2]
    require(frame['relative_frame_index']==0 and frame['sample_token']==record['sample_token'],'Wrong current frame')
    G0=frame['lidar_to_global_column_matrix'];del frame,carrier,content
    result=geometry.geometry_from_centered_boxes(record['boxes'],G0,
        grid_shape_xyz=oracle.SHAPE,extent_xyz=oracle.EXTENT)
    result['receipt'].update(ordinal=ordinal,sample_token=record['sample_token'],
        raw_pose_carrier_sha256=descriptor['sha256'])
    return result


def device_geometry(torch,core,geometry,device):
    arrays=geometry['states_numpy']
    def f(key): return torch.as_tensor(arrays[key],dtype=torch.float32,device=device).unsqueeze(0)
    states,valid=core.pack_object_states(f('center'),f('size'),f('rotation'),
        torch.as_tensor(arrays['classes'],dtype=torch.int64,device=device).unsqueeze(0),f('score'),f('velocity'))
    require(bool(valid.all()) and not states.requires_grad,'Retained GMO inputs were padded or trainable')
    result={key:torch.as_tensor(geometry[key],dtype=torch.float64,device=device)
        for key in ('points_R','cv_velocity_R','c0_R','R0_R','wlh','velocity_R','score')}
    result['owner']=torch.as_tensor(geometry['owner'],dtype=torch.int64,device=device)
    result['packed_states']=states[0]
    receipt=dict(geometry['receipt'],packed_states_sha256=tensor_sha(states[0]))
    return result,receipt


def object_forward(module,data,current_bev,oracle,coupled):
    return module(data['packed_states'],current_bev,data['c0_R'],data['R0_R'],data['wlh'],
        data['velocity_R'],data['score'],grid_shape_yx=(oracle.SHAPE[1],oracle.SHAPE[0]),
        extent_xy=(oracle.EXTENT[0],oracle.EXTENT[1],oracle.EXTENT[3],oracle.EXTENT[4]),coupled=coupled)


def material_field(rigid,bridge,data,output,oracle):
    field=rigid.source_displacement_field(data['points_R'],data['owner'],data['cv_velocity_R'],
        data['c0_R'],output['delta_center_R'],output['delta_rotvec_R'])
    return bridge.source_field_to_native_layout(field,oracle.SHAPE)


def cv_reference(torch,bridge,data,oracle):
    # This is the original full-grid CV velocity, not reconstructed from tokens.
    value=torch.stack([data['cv_velocity_R']*h for h in (.5,1.,1.5,2.)])
    value[:,data['owner']<0]=0.
    return bridge.source_field_to_native_layout(value,oracle.SHAPE)


def finite_tensors(torch,values,description):
    require(all(bool(torch.isfinite(v).all()) for v in values),description)


def optimizer_receipt(torch,optimizer,named):
    rows={}
    for name,param in named:
        state=optimizer.state.get(param)
        if not state:
            rows[name]=None
            continue
        finite_tensors(torch,[state[k] for k in ('step','exp_avg','exp_avg_sq')],'Nonfinite optimizer state')
        rows[name]=dict(step=int(state['step'].item()),exp_avg_norm=float(state['exp_avg'].double().norm()),
                       exp_avg_sq_norm=float(state['exp_avg_sq'].double().norm()))
    require(any(v is not None for v in rows.values()) and all(v is None or v['step']==4 for v in rows.values()),
            'AdamW actual endpoint is not four updates')
    return dict(named_parameter_states=rows,parameters_with_state=sum(v is not None for v in rows.values()),
                parameters_without_state=sum(v is None for v in rows.values()),all_present_steps4=True)


def address_probe(torch,u,models,objects,data,sample,oracle,native,rigid,bridge,tree,budget):
    """Same updated weights, toggle only addresses; no labels or metrics."""
    saved=u.rng_state(torch);results={};boundary=tree.tree_digest(sample['inputs'])
    try:
        with torch.no_grad():
            for arm in ARMS:
                budget();models[arm].eval();objects[arm].eval()
                variants=[]
                for coupled in (True,False):
                    u.restore_rng(torch,saved)
                    output=object_forward(objects[arm],data,sample['inputs']['prev_bev_input'][:,-1],oracle,coupled)
                    flow=material_field(rigid,bridge,data,output,oracle)
                    with bridge.inject_future_object_features(models[arm].future_pred_head,output['native_delta']) as hooks:
                        prediction=native.replay(models[arm],sample,training=False)[0]
                    finite_tensors(torch,[prediction,flow]+list(output.values()),'Nonfinite address intervention')
                    variants.append((output,flow,prediction,list(hooks)))
                c,f=variants
                same_keys=('delta_center_R','delta_rotvec_R','values','future_centers_R','future_rotations_R',
                           'current_context','current_sampling_points_R')
                require(all(exact_bytes(c[0][k],f[0][k]) for k in same_keys) and exact_bytes(c[1],f[1]),
                        'Address-only switch changed learned values or physical state')
                require(c[2].shape[0]==5 and exact_bytes(c[2][:1],f[2][:1]),'Address switch changed t0 logits')
                difference=(c[2][1:]-f[2][1:]).detach()
                changed=int(torch.count_nonzero(difference))
                require(changed>0,'Updated spatial address switch has no future native effect')
                results[arm]=dict(same_updated_weights=True,learned_values_and_physical_bytes_exact=True,
                    checked_equal_output_keys=list(same_keys),t0_all_decoder_logits_exact=True,
                    future_changed_values=changed,future_total_values=difference.numel(),
                    future_max_abs_difference=float(difference.abs().max()),
                    future_mean_abs_difference=float(difference.abs().double().mean()),
                    Cpl_hooks=c[3],Fix_hooks=f[3],physical_sha256=tensor_sha(c[1]),
                    Cpl_logits_sha256=tensor_sha(c[2]),Fix_logits_sha256=tensor_sha(f[2]),
                    Cpl_native_delta_sha256=tensor_sha(c[0]['native_delta']),
                    Fix_native_delta_sha256=tensor_sha(f[0]['native_delta']))
                del variants,c,f,difference,output,flow,prediction
        require(tree.tree_digest(sample['inputs'])==boundary,'Address probe mutated native inputs')
    finally:
        u.restore_rng(torch,saved)
    return results


def run(a,out,started,stop):
    p,contract,mods=load_contract(a)
    u=mods['object_state_forecast_preflight_v2'];helper=mods['train_source_motion_v1']
    native=mods['native_state_cache'];oracle=mods['oracle_transport_probe']
    common=mods['common_change_evaluation_v2'];losses=mods['train_supported_fusion_v2']
    adapter=mods['objective_supervision_adapters'];tree=mods['joint_native_evaluation']
    loader=mods['object_state_prediction_inputs_v1'];core=mods['object_state_conditioner_v2']
    geometry_module=mods['shared_rigid_prediction_geometry_v1']
    phases={};cuda_ready=False;updates=examples=0
    def budget():
        require(not stop,'Signal received; no retry or saved weights')
        require(time.monotonic()-started<a.max_seconds,'Whole preflight wall-clock ceiling')
        if cuda_ready:
            require(torch.cuda.max_memory_allocated(a.device)<=a.max_allocated_gib*2**30,'CUDA allocation ceiling')
    def progress(phase,**extra):
        budget()
        value=dict(schema=SCHEMA,phase=phase,updates=updates,examples=examples,seconds=time.monotonic()-started,**extra)
        u.write(out/'progress.json',value);print(json.dumps(value),flush=True)
    progress('authenticated_inputs')
    train_root,train,cache_receipt=helper.cache_index(a.train_cache,'train',p)
    label_root,label_desc=helper.labels_manifest(a.sparse_labels,p)
    orders=helper.planned_orders();selected=orders[0][:16]
    require(p['selection_sha256']==loader.SELECTION_SHA,'Selection contract differs')
    inputs=loader.PredictionInputs(a.predictions,a.raw_metadata,a.selection,p['predictions'],train)
    for ordinal in selected: helper.check_identity(train[ordinal],label_desc[train[ordinal]['sample_token']])
    tick=time.monotonic();geometries={}
    for number,ordinal in enumerate(selected):
        progress('current_full_grid_geometry',geometry_samples_completed=number,ordinal=ordinal)
        geometries[ordinal]=current_geometry(inputs,ordinal,geometry_module,oracle)
        require(geometries[ordinal]['points_R'].shape==(640000,3),'Complete material grid required')
        budget()
    phases['full_current_geometry_seconds']=time.monotonic()-tick
    u.write(out/'geometry.json',dict(schema=SCHEMA,selected_ordinals=selected,
        receipts=[geometries[i]['receipt'] for i in selected],labels_consumed=False,
        full_grid_geometry_seconds=phases['full_current_geometry_seconds']))
    import torch
    require(str(a.device).startswith('cuda') and torch.cuda.is_available(),'Native preflight requires CUDA')
    torch.cuda.set_device(a.device);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(min(1.,a.max_allocated_gib*2**30/
        torch.cuda.get_device_properties(a.device).total_memory),a.device)
    torch.cuda.reset_peak_memory_stats(a.device);cuda_ready=True
    # Import order also binds ordinary intra-module imports to authenticated files.
    rigid=contract.import_bound('rigid_object_motion_v1',p['sources_sha256'])
    object_core=contract.import_bound('joint_rigid_object_features_v1',p['sources_sha256'])
    bridge=contract.import_bound('shared_rigid_native_bridge_v1',p['sources_sha256'])
    progress('native_initialization');tick=time.monotonic()
    model=native.build_native_model(a.config,a.checkpoint,device=a.device,repo=a.repo)
    payload=torch.load(a.o_checkpoint,map_location='cpu',weights_only=False)
    model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True);del payload
    for value in model.parameters():value.requires_grad_(False)
    model.eval()
    require(common.historical_O_head_digest(model.future_pred_head)==common.O_HEAD_SHA,'Historical O head differs')
    O_sha=helper.state_digest(model);initial_head_sha=helper.state_digest(model.future_pred_head)
    require(O_sha==read(Path(a.training_run)/'manifest.json')['frozen_O_state_sha256'],'Original full O differs')
    frozen_sha=u.non_head_digest(model)
    require(sha(inspect.getsourcefile(model.compute_occ_loss))==adapter.DETECTOR_SOURCE_SHA and
        sha(inspect.getsourcefile(model.future_pred_head.loss_occ))==adapter.HEAD_SOURCE_SHA and
        sha(inspect.getsourcefile(model.future_pred_head.loss_voxel))==adapter.HEAD_SOURCE_SHA and
        model.future_pred_head.class_weights.tolist()==[1.,5.],'Native loss formula/weights changed')
    models={'Cpl':model,'Fix':copy.deepcopy(model)}
    u.seed_all(torch);objects={'Cpl':object_core.JointRigidObjectFeatures().to(a.device)}
    objects['Fix']=copy.deepcopy(objects['Cpl']);initial_new_sha=helper.state_digest(objects['Cpl'])
    head_params={};new_params={}
    for arm in ARMS:
        for value in models[arm].future_pred_head.parameters():value.requires_grad_(True)
        head_params[arm]=list(models[arm].future_pred_head.named_parameters())
        new_params[arm]=list(objects[arm].named_parameters())
        require(sum(v.numel() for _,v in head_params[arm])==RECIPE['future_head_parameters'] and
            sum(v.numel() for _,v in new_params[arm])==RECIPE['new_parameters'],'Trainable capacity differs')
        require(all(not v.requires_grad for n,v in models[arm].named_parameters() if not n.startswith('future_pred_head.')),
                'Native observer unfrozen')
        require(helper.state_digest(models[arm])==O_sha and helper.state_digest(objects[arm])==initial_new_sha,
                'Arm initial states differ')
    for modules in (models,objects):
        require(not ({id(v) for v in modules['Cpl'].parameters()} & {id(v) for v in modules['Fix'].parameters()}),
                'Cpl/Fix parameters share storage objects')
    torch.cuda.synchronize(a.device);phases['native_initialization_seconds']=time.monotonic()-tick
    sources=dict(protocol_sha256=sha(a.protocol),sources_sha256=p['sources_sha256'],
        connected_protocol_sha256=CONNECTED_PROTOCOL_SHA,connected_complete_sha256=CONNECTED_COMPLETE_SHA,
        historical_manifest_sha256=sha(Path(a.training_run)/'manifest.json'),
        O_checkpoint_sha256=oracle.O_SHA,M0_checkpoint_sha256=oracle.M0_SHA,config_sha256=oracle.CONFIG_SHA,
        runtime_source_contract_sha256=RUNTIME_SHA,train_cache=cache_receipt,sparse_labels=p['labels'],predictions=inputs.receipt)
    manifest=dict(schema=SCHEMA,mode='preflight',sources=sources,recipe=RECIPE,numerical_policy=NUMERICAL,
        arms=list(ARMS),selected_ordinals=selected,sample_orders_sha256=helper.json_hash(orders),
        selected_identities=[dict(ordinal=i,**common.identity(train[i])) for i in selected],
        initial_O_state_sha256=O_sha,initial_future_head_sha256=initial_head_sha,initial_new_module_sha256=initial_new_sha,
        frozen_non_head_state_sha256=frozen_sha,future_head_parameter_names=[n for n,_ in head_params['Cpl']],
        new_parameter_names=[n for n,_ in new_params['Cpl']],D_model_loaded=False,
        labels_only_after_corresponding_complete_prediction=True,GT_not_passed_to_forward=True,
        physical_support_defined_before_sparse_labels=True,postupdate_t0_may_change=True,
        weights_persisted=False,development_evaluation=False)
    u.write(out/'manifest.json',manifest)
    u.write(out/'loaded_models.json',dict(O_full_state_sha256=O_sha,O_historical_head_sha256=common.O_HEAD_SHA,
        O_future_head_state_sha256=initial_head_sha,new_module_sha256=initial_new_sha,D_model_loaded=False,optimizer_restored=False))
    folders={arm:out/'runs'/arm for arm in ARMS}
    for arm,folder in folders.items():
        folder.mkdir(parents=True)
        u.write(folder/'manifest.json',dict(schema=SCHEMA,arm=arm,common_manifest_sha256=sha(out/'manifest.json'),
            coupled_spatial_address=arm=='Cpl',learned_values_include_pose_residual=True,
            physical_state_learned=True,physical_to_native_future_head=False,
            initial_future_head_sha256=initial_head_sha,initial_new_module_sha256=initial_new_sha))
    progress('initial_O_CV_parity');tick=time.monotonic();parity=[]
    with torch.no_grad():
        for ordinal in selected[:2]:
            budget();record=train[ordinal];sample=common.input_only(native,train_root,record,a.device)
            boundary=tree.tree_digest(sample['inputs']);data,state_receipt=device_geometry(torch,core,geometries[ordinal],a.device)
            base=cv_reference(torch,bridge,data,oracle)
            original=native.replay(models['Cpl'],sample,training=False)[0]
            require(original.shape[0]==5 and original.shape[1]==3,'Original all-5h/3-layer tensor changed')
            predictions={}
            for arm in ARMS:
                objects[arm].eval()
                output=object_forward(objects[arm],data,sample['inputs']['prev_bev_input'][:,-1],oracle,arm=='Cpl')
                flow=material_field(rigid,bridge,data,output,oracle)
                with bridge.inject_future_object_features(models[arm].future_pred_head,output['native_delta']) as hooks:
                    prediction=native.replay(models[arm],sample,training=False)[0]
                finite_tensors(torch,[prediction,flow]+list(output.values()),'Nonfinite initialization')
                require(exact_bytes(prediction,original) and exact_bytes(flow,base),'Initial full O/CV byte parity failed')
                require(int(torch.count_nonzero(output['native_delta']))==0,'Initial native projection is not zero')
                predictions[arm]=prediction
                parity.append(dict(arm=arm,ordinal=ordinal,sample_token=record['sample_token'],
                    all_5h_3layers_logits_O_exact=True,complete_CV_displacement_exact=True,
                    native_logits_sha256=tensor_sha(prediction),physical_field_sha256=tensor_sha(flow),
                    native_projection_zero=True,hook_counts=list(hooks),state_source=state_receipt))
                del output,flow,prediction
            # All initialization predictions and complete material fields precede GT.
            common.load_targets_after_predictions(native,train_root,record,sample,a.device)
            original_hist=common.native_hist(models['Cpl'],original,sample,record)
            for arm in ARMS:
                require(common.native_hist(models[arm],predictions[arm],sample,record)==original_hist,'Native initial histogram differs')
                parity[-2+ARMS.index(arm)]['initial_native_hist_exact']=True
            require(tree.tree_digest(sample['inputs'])==boundary,'Initial parity mutated inputs')
            del data,sample,base,original,predictions
    torch.cuda.synchronize(a.device);phases['initial_O_CV_parity_seconds']=time.monotonic()-tick
    require(all(helper.state_digest(m.future_pred_head)==initial_head_sha for m in models.values()) and
            all(helper.state_digest(m)==initial_new_sha for m in objects.values()),'Parity changed initial trainable state')
    optimizers={arm:{'head':torch.optim.AdamW([v for _,v in head_params[arm]],lr=1e-5,betas=(.9,.999),
            eps=1e-8,weight_decay=.01,amsgrad=False),
        'new':torch.optim.AdamW([v for _,v in new_params[arm]],lr=1e-3,betas=(.9,.999),
            eps=1e-8,weight_decay=.01,amsgrad=False)} for arm in ARMS}
    u.seed_all(torch)
    for arm in ARMS:
        models[arm].train();objects[arm].train()
        for optimizer in optimizers[arm].values():optimizer.zero_grad(set_to_none=True)
    progress('training');tick=time.monotonic();micro_times=[];update_times=[];connectivity={};rng_rows=[]
    initial_comparisons=[]
    for index,ordinal in enumerate(selected):
        budget();torch.cuda.synchronize(a.device);micro_tick=time.monotonic();micro=index%4
        if micro==0:
            update_tick=micro_tick;batches={arm:[] for arm in ARMS}
            for arm in ARMS:
                for group in optimizers[arm]['head'].param_groups:group['lr']=u.head_lr(updates)
                for group in optimizers[arm]['new'].param_groups:group['lr']=helper.schedule_lr(updates)
        record=train[ordinal];sample=common.input_only(native,train_root,record,a.device)
        # A separate forward carrier never acquires the targets added to sample.
        forward_sample=dict(inputs=sample['inputs'],record=sample['record'])
        boundary=tree.tree_digest(sample['inputs']);data,state_receipt=device_geometry(torch,core,geometries[ordinal],a.device)
        data_sha=tree.tree_digest(data);base=cv_reference(torch,bridge,data,oracle) if updates==0 else None
        paired=u.rng_state(torch);before_sha=u.rng_digest(paired);ends={};first=None;comparison=None
        for arm in ARMS:
            u.restore_rng(torch,paired);budget();torch.cuda.synchronize(a.device);arm_tick=time.monotonic()
            output=object_forward(objects[arm],data,forward_sample['inputs']['prev_bev_input'][:,-1],oracle,arm=='Cpl')
            flow=material_field(rigid,bridge,data,output,oracle)
            with bridge.inject_future_object_features(models[arm].future_pred_head,output['native_delta']) as hooks:
                prediction=native.replay(models[arm],forward_sample,training=True)[0]
            finite_tensors(torch,[prediction,flow]+list(output.values()),'Nonfinite native/object/physical forward')
            native_digest=tensor_sha(prediction);flow_digest=tensor_sha(flow)
            require('targets' not in forward_sample,'Forward carrier contains GT')
            # Labels may now enter only the loss. The full physical field already exists.
            if 'targets' not in sample:common.load_targets_after_predictions(native,train_root,record,sample,a.device)
            descriptor=label_desc[record['sample_token']];label=helper.load_sparse(label_root,descriptor,record)
            occupancy,occ_audit=losses.loss_terms(models[arm],prediction,sample['targets'],adapter)
            occ_audit.update(inherited_constant_fixed_t0_field_names_only=True,intermediate_losses_trainable=True,t0_readout_trainable=True)
            sparse=helper.gather_sparse(flow,label);physical,physical_audit=helper.object_group_loss(sparse,label)
            total=occupancy+physical
            finite_tensors(torch,[occupancy,physical,total],'Nonfinite actual train objective')
            if updates==0:
                require(exact_bytes(flow,base),'First four microsteps must preserve the full CV field')
                if arm=='Cpl':first=(native_digest,flow_digest,occ_audit,float(physical.detach()))
                else:
                    require(native_digest==first[0] and flow_digest==first[1],'Initial paired logits/field differ')
                    comparison=u.compare_initial_occupancy(first[2],occ_audit)
                    for row in comparison.values():
                        row['Cpl']=row.pop('V');row['Fix']=row.pop('G');row['Fix_minus_Cpl']=row.pop('G_minus_V')
                    comparison['physical_scalar_descriptive']=dict(Cpl=first[3],Fix=float(physical.detach()),
                        Fix_minus_Cpl=float(physical.detach())-first[3],finite=True,dense_input_bytes_exact=True,
                        no_scalar_equality_gate_due_to_CUDA_index_add=True)
            if index==12:
                require(not physical_audit['empty_target'],'Late fixed anchor lacks physical supervision')
                groups={'future_head':head_params[arm],
                    'object_encoder':list(objects[arm].object_encoder.named_parameters()),
                    'pose_head':list(objects[arm].pose_head.named_parameters()),
                    'value_encoder':list(objects[arm].value_encoder.named_parameters()),
                    'native_projection':list(objects[arm].native_projection.named_parameters())}
                diagnostics={}
                for loss_name,scalar in (('occupancy',occupancy),('physical',physical)):
                    diagnostics[loss_name]={}
                    for group,named in groups.items():
                        grads=torch.autograd.grad(scalar,[v for _,v in named],allow_unused=True,retain_graph=True)
                        stats=u.gradient_stats(named,grads);del grads
                        if group in RECIPE['late_vjp_groups']:
                            require(stats['norm']>0,'Late '+loss_name+' gradient missing at '+group+' '+arm)
                        diagnostics[loss_name][group]=stats
                connectivity[arm]=dict(after_completed_updates=3,microstep_index=index,ordinal=ordinal,
                    sample_token=record['sample_token'],loss_VJPs=diagnostics,
                    total_backward_not_yet_called_for_this_micro=True,
                    physical_future_head_unused_by_design=True,individual_pose_component_nonzero_not_required=True,
                    Fix_value_branch_remains_connected=True)
            (total/4).backward();ends[arm]=u.rng_state(torch)
            require(all(v.grad is None for n,v in models[arm].named_parameters() if not n.startswith('future_pred_head.')),
                    'Task gradient reached frozen observer')
            require(tree.tree_digest(sample['inputs'])==boundary and tree.tree_digest(data)==data_sha,'Training mutated inputs/geometry')
            torch.cuda.synchronize(a.device)
            batches[arm].append(dict(ordinal=ordinal,**common.identity(record),
                inputs_sha256=record['files']['inputs']['sha256'],targets_sha256=record['files']['targets']['sha256'],
                sparse_label_sha256=descriptor['sha256'],input_tree_sha256=boundary,geometry_tree_sha256=data_sha,
                state_source=state_receipt,hook_counts=list(hooks),occupancy_loss=occ_audit,
                physical_loss=float(physical.detach()),physical_audit=physical_audit,total_loss=float(total.detach()),
                native_logits_sha256=native_digest,complete_material_field_sha256=flow_digest,
                paired_rng_before_sha256=before_sha,paired_rng_after_sha256=u.rng_digest(ends[arm]),
                arm_microstep_seconds=time.monotonic()-arm_tick,GT_not_passed_to_forward=True))
            del output,flow,prediction,sparse,label,occupancy,physical,total
        require(u.rng_digest(ends['Cpl'])==u.rng_digest(ends['Fix']),'Paired RNG consumption differs')
        if comparison is not None:
            initial_comparisons.append(dict(microstep_index=index,ordinal=ordinal,terms=comparison,
                all_logits_and_complete_fields_exact=True))
        u.restore_rng(torch,ends['Cpl'])
        rng_rows.append(dict(ordinal=ordinal,before_sha256=before_sha,after_sha256=u.rng_digest(ends['Cpl']),paired_exact=True))
        del sample,forward_sample,data,base,ends,paired
        examples+=1;torch.cuda.synchronize(a.device);micro_times.append(time.monotonic()-micro_tick)
        if micro!=3:continue
        gradients={}
        for arm in ARMS:
            head_stats=u.gradient_stats(head_params[arm],[v.grad for _,v in head_params[arm]])
            new_stats=u.gradient_stats(new_params[arm],[v.grad for _,v in new_params[arm]])
            require(head_stats['norm']>0 and new_stats['norm']>0,'Missing aggregate task gradients')
            hnorm=float(torch.nn.utils.clip_grad_norm_([v for _,v in head_params[arm]],35.,error_if_nonfinite=True))
            nnorm=float(torch.nn.utils.clip_grad_norm_([v for _,v in new_params[arm]],10.,error_if_nonfinite=True))
            gradients[arm]=dict(head=head_stats,new=new_stats,head_preclip_norm=hnorm,new_preclip_norm=nnorm,
                head_clip_factor=min(1.,35./(hnorm+1e-6)),new_clip_factor=min(1.,10./(nnorm+1e-6)))
        budget()
        for arm in ARMS:
            for optimizer in optimizers[arm].values():
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                finite_tensors(torch,[v[k] for v in optimizer.state.values() for k in ('step','exp_avg','exp_avg_sq')],
                    'Nonfinite actual AdamW state')
            finite_tensors(torch,[v for _,v in head_params[arm]+new_params[arm]],'Nonfinite updated weights')
        updates+=1;torch.cuda.synchronize(a.device);update_times.append(time.monotonic()-update_tick)
        for arm in ARMS:
            u.append(folders[arm]/'training.jsonl',dict(update=updates,examples=examples,
                future_head_lr=u.head_lr(updates-1),new_lr=helper.schedule_lr(updates-1),samples=batches[arm],
                gradients=gradients[arm],matched_accum4_update_seconds=update_times[-1],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device)))
        progress('training')
    phases['four_paired_updates_seconds']=time.monotonic()-tick
    require(updates==4 and examples==16 and len(initial_comparisons)==4 and set(connectivity)==set(ARMS),
            'Incomplete actual engineering endpoint')
    progress('same_weight_address_intervention');tick=time.monotonic();ordinal=selected[12]
    sample=common.input_only(native,train_root,train[ordinal],a.device)
    data,_=device_geometry(torch,core,geometries[ordinal],a.device)
    intervention=address_probe(torch,u,models,objects,data,sample,oracle,native,rigid,bridge,tree,budget)
    del data,sample
    torch.cuda.synchronize(a.device);phases['address_intervention_seconds']=time.monotonic()-tick
    require(all(u.non_head_digest(m)==frozen_sha for m in models.values()),'Frozen native observer state changed')
    finalheads={arm:helper.state_digest(m.future_pred_head) for arm,m in models.items()}
    finalobjects={arm:helper.state_digest(m) for arm,m in objects.items()}
    require(all(s!=initial_head_sha for s in finalheads.values()) and all(s!=initial_new_sha for s in finalobjects.values()),
            'Trainable state did not change')
    optimizer_steps={arm:{name:optimizer_receipt(torch,opt,head_params[arm] if name=='head' else new_params[arm])
        for name,opt in optimizers[arm].items()} for arm in ARMS}
    for name,digest in p['sources_sha256'].items():require(sha(contract.source_path(name))==digest,'Source changed during run')
    require(sha(a.protocol)==sources['protocol_sha256'],'Protocol changed during run')
    for field,name in (('complete_sha256','complete.json'),('manifest_sha256','manifest.json'),('predictions_sha256','predictions.json')):
        require(sha(Path(a.predictions)/name)==p['predictions'][field],'Prediction source changed during run')
    budget()
    summary=dict(schema=SCHEMA,status='PASS_SHARED_RIGID_NATIVE_PREFLIGHT',updates=4,examples=16,
        evaluated_samples=0,scores=None,weights_persisted=False,optimizer_restored=False,D_model_loaded=False,
        initial_O_CV_parity=parity,initial_O_CV_parity_pass=True,initial_training_comparisons=initial_comparisons,
        paired_rng=rng_rows,late_dual_loss_connectivity=connectivity,all_train_losses_gradients_parameters_moments_finite=True,
        same_weights_address_interventions=dict(ordinal=ordinal,results=intervention),
        initial_future_head_sha256=initial_head_sha,initial_new_module_sha256=initial_new_sha,
        final_future_head_sha256=finalheads,final_new_module_sha256=finalobjects,
        frozen_non_head_O_unchanged=True,actual_optimizer_steps=optimizer_steps,
        native_logits_not_composed=True,decoder_only_injection=True,postupdate_t0_not_required_to_equal_initial_O=True,
        resources=dict(phases=phases,paired_microstep_seconds=micro_times,matched_accum4_update_seconds=update_times,
            connectivity_VJP_microstep_index=12,peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device),elapsed_seconds=time.monotonic()-started,
            timing_scope='whole wall time includes contract/asset checks, all16 full-grid CPU geometries, native init/parity, four paired accum4 updates, VJPs and address probe'))
    u.write(out/'summary.json',summary)
    for arm in ARMS:
        u.write(folders[arm]/'complete.json',dict(schema=SCHEMA,status=summary['status'],arm=arm,updates=4,examples=16,
            weights_persisted=False,files_sha256={f:sha(folders[arm]/f) for f in ('manifest.json','training.jsonl')}))
    budget();require(not list(out.rglob('*.pt')) and not list(out.rglob('*.pth')),'Weight persistence prohibited')
    u.write(out/'complete.json',dict(schema=SCHEMA,status=summary['status'],mode='preflight',updates=4,examples=16,
        evaluated_samples=0,weights_persisted=False,final_checkpoints={},
        files_sha256={f:sha(out/f) for f in ('manifest.json','loaded_models.json','geometry.json','summary.json')},
        arm_complete_sha256={arm:sha(folders[arm]/'complete.json') for arm in ARMS},elapsed_seconds=time.monotonic()-started))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','connected-protocol','training-run','config','checkpoint','o-checkpoint',
                 'repo','runtime-contract','train-cache','sparse-labels','predictions','raw-metadata','selection','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('--max-allocated-gib',type=float,required=True)
    a=parser.parse_args(argv);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stop=[]
    signal.signal(signal.SIGTERM,lambda s,f:stop.append(s));signal.signal(signal.SIGINT,lambda s,f:stop.append(s))
    utility=_contract_loader().import_bound('object_state_forecast_preflight_v2',{'object_state_forecast_preflight_v2.py':UTILITY_SHA})
    try:run(a,out,started,stop)
    except BaseException as exc:
        utility.write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),
            traceback=traceback.format_exc(),last_progress=read(out/'progress.json') if (out/'progress.json').is_file() else None,
            weights_persisted=False,seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
