"""Native V/G four-update engineering preflight; no saved weights or dev scores.

Both arms train the whole O future head and identical new conditioner/readout.
Only current predicted velocity inputs differ. Frozen actual D supplies the
initial physical field; occupancy is the native head output, not a refiner.
The train512 prediction asset must be complete before this entry can run.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import struct
import sys
import time
import traceback

import numpy as np

SCHEMA = 'object-state-forecast-preflight-v2'
ARMS = ('V', 'G')
CONDITIONER_SHA = 'dad88c05105d85d9b2a6094c687630f1d661cd7a4e35be42a15a5d01b94a8f36'
PREDICTION_INPUT_SHA = '55e39715fdf19e5ae1f227a943a843b194eabed08e788ccb2694a040d6f18256'
CORE_SHA = '96e796dd5bb6d33949a03a6c451919ef2e19c75b81eaea4b10e2c98f8ecc176e'
COMMON_SHA = '57cf304bafb050fac05b06abe68ce4600a37ca020d9d10d0673ec0ab1edb3bc1'
CONNECTED_PROTOCOL_SHA = 'e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
CONNECTED_COMPLETE_SHA = 'f0bf075fa36622b38c0c5136b187a184d16d3e07ed00857dfe21a0545f1c5f56'
D_CHECKPOINT_SHA = '7e2750d56ade7fb36e334b6431e83f51aafa997357744780a507924a696951e3'
RUNTIME_SHA = '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c'
NUMERICAL = dict(dtype='float32', matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False)
RECIPE = dict(seed=11, updates=4, examples=16, accumulate=4, train_samples=512,
    planned_formal_updates=512, formal_authorized=False, conditioner_parameters=95872,
    future_head_parameters=13274016, motion_readout_parameters=745392,
    future_head_lr=1e-5, future_head_grad_clip=35., future_head_warmup_updates=50,
    readout_lr=1e-3, readout_grad_clip=10., readout_warmup_updates=25, new_optimizer_scope='conditioner plus motion readout jointly',
    weight_decay=.01, optimizer='AdamW', betas=[.9,.999], eps=1e-8, amsgrad=False,
    physical_loss_coefficient=1., occupancy_loss_coefficient=1., smooth_l1_beta_m=.5,
    sample_order='first16 of first continuous RandomState11 permutation of 512',
    schedule='original O head 512-update warmup/cosine; original source-motion readout schedule',
    occupancy_loss='original coarse twelve computed; original O CE[1,5]+Lovasz six selected',
    physical_loss='original sparse point_xyz/object/present_group/valid_horizon means',
    arm_difference='V keeps predicted velocity; G zeros only the final three velocity input slots',
    scope='whole native future_pred_head plus new conditioner/readout; D and non-future-head O frozen',
    rng='same pre-forward Python/NumPy/CPU/CUDA state restored for paired V/G microsteps',
    initialization='same actual O and copied seed11 conditioner/readout; zero projections give original O/D',
    initial_loss_comparison='same-logit CE per term rel1e-5 abs1e-7; Lovasz exact; physical index_add scalar difference descriptive, input bytes exact',
    state_tokens='original 27 current fields plus internal p0+h*v hypothesis in R/51.2; G zeros velocity before derivation; no GT or spatial mask',
    outputs='native conditioned future head; no logit composition; physical input keeps shared feature graph',
    late_vjp_microstep_index=12, velocity_intervention_after_updates=4,
    velocity_intervention_ordinal='fixed selected[12]; same weights normal versus zero velocity',
    empty_state_check='updated own native path; never an initial O performance comparison',
    weights_persisted=False, development_evaluation=False)


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())


def head_lr(update):
    require(type(update) is int and 0 <= update < 512, 'Original O schedule index')
    scale = (.1+.9*update/50 if update < 50 else
             .1+.9*.5*(1+math.cos(math.pi*(update-50)/(512-50))))
    return 1e-5*scale


def protocol_template():
    return dict(schema=SCHEMA, status='REVIEW_REQUIRED', recipe=copy.deepcopy(RECIPE),
        numerical_policy=dict(NUMERICAL), sources_sha256={},
        connected_protocol_sha256=CONNECTED_PROTOCOL_SHA,
        connected_complete_sha256=CONNECTED_COMPLETE_SHA, D_checkpoint_sha256=D_CHECKPOINT_SHA,
        runtime_source_contract_sha256=RUNTIME_SHA,
        cache_index_sha256=dict(train=None, development=None),
        labels=dict(manifest_sha256=None, complete_sha256=None),
        selection_sha256='5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d',
        predictions=dict(complete_sha256=None,manifest_sha256=None,predictions_sha256=None,inference_source_sha256=None),
        resources=dict(preflight=dict(max_seconds=600., max_allocated_gib=32.), formal=None))


def load_contract(a):
    p = read(a.protocol)
    require(p['schema']==SCHEMA and p['status']=='FROZEN' and p['recipe']==RECIPE
            and p['numerical_policy']==NUMERICAL, 'Frozen exact V/G preflight recipe required')
    require(p['resources']==dict(preflight=dict(max_seconds=a.max_seconds,
            max_allocated_gib=a.max_allocated_gib), formal=None), 'Only frozen preflight resources allowed')
    require(math.isfinite(a.max_seconds) and 0<a.max_seconds<=600 and
            math.isfinite(a.max_allocated_gib) and 0<a.max_allocated_gib<=32, 'Finite preflight resource cap')
    require(p['connected_protocol_sha256']==sha(a.connected_protocol)==CONNECTED_PROTOCOL_SHA and
            p['connected_complete_sha256']==sha(Path(a.training_run)/'complete.json')==CONNECTED_COMPLETE_SHA and
            p['D_checkpoint_sha256']==D_CHECKPOINT_SHA, 'Actual D provenance changed')
    path=Path(__file__).with_name('common_connected_motion_evaluation_v2.py')
    require(sha(path)==COMMON_SHA, 'Source loader changed')
    spec=importlib.util.spec_from_file_location('common_connected_motion_evaluation_v2', path)
    contract=importlib.util.module_from_spec(spec); sys.modules[spec.name]=contract; spec.loader.exec_module(contract)
    connected=read(a.connected_protocol); inherited=contract.check_sources(connected)
    expected=dict(inherited); expected['future_state_motion_v1.py']=CORE_SHA
    expected['object_state_conditioner_v2.py']=CONDITIONER_SHA
    expected['object_state_prediction_inputs_v1.py']=PREDICTION_INPUT_SHA
    expected[Path(__file__).name]=sha(__file__)
    require(p['sources_sha256']==expected, 'Exact inherited/core/preflight source set required')
    for name, digest in expected.items(): require(sha(contract.source_path(name))==digest, 'Source changed: '+name)
    require(p['cache_index_sha256']==connected['cache_index_sha256'] and p['labels']==connected['labels'],
            'Native inputs or sparse supervision changed')
    require(p['runtime_source_contract_sha256']==sha(a.runtime_contract)==RUNTIME_SHA, 'Runtime contract changed')
    for path, digest in read(a.runtime_contract)['runtime_source_sha256'].items():
        require(sha(path)==digest, 'Native runtime source changed: '+path)
    names=('train_source_motion_v1','native_state_cache','oracle_transport_probe','common_change_evaluation_v2',
           'train_supported_fusion_v2','objective_supervision_adapters','train_connected_motion_v2','joint_native_evaluation')
    mods={name:contract.import_bound(name,expected) for name in names}
    oracle=mods['oracle_transport_probe']
    require(sha(a.config)==oracle.CONFIG_SHA and sha(a.checkpoint)==oracle.M0_SHA and
            sha(a.o_checkpoint)==oracle.O_SHA, 'Native O assets changed')
    return p, contract, mods


def seed_all(torch):
    random.seed(11); np.random.seed(11); torch.manual_seed(11); torch.cuda.manual_seed_all(11)


def rng_state(torch):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()])


def restore_rng(torch, state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def rng_digest(state):
    h=hashlib.sha256(repr(state[:2]).encode())
    for tensor in [state[2]]+state[3]: h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def non_head_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('future_pred_head.'): continue
        value=value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def gradient_stats(named_parameters, gradients):
    values={}; total=0.; none=0
    for (name,_), grad in zip(named_parameters, gradients):
        if grad is None: values[name]=None; none+=1; continue
        require(bool(grad.isfinite().all()), 'Nonfinite gradient: '+name)
        norm=float(grad.detach().double().norm()); values[name]=norm; total+=norm*norm
    return dict(norm=math.sqrt(total), none_parameters=none,
                nonzero_parameters=sum(v is not None and v>0 for v in values.values()),
                parameter_norms=values)


def compare_initial_occupancy(left, right):
    """Existing supported-fusion CE audit tolerance, for all three decoders.

    Paired RNG cannot force CUDA NLL atomic reductions to be byte-identical.
    This changes neither original loss tensors nor the optimizer objective.
    """
    keys=left['optimized_keys']
    expected={'loss_voxel_'+family+'_inter_'+str(i) for family in ('ce','lovasz') for i in range(3)}
    require(keys==right['optimized_keys'] and set(keys)==expected, 'Initial six native loss keys differ')
    def ordered32(value):
        bits=struct.unpack('>I',struct.pack('>f',value))[0]
        return 0x80000000-(bits & 0x7fffffff) if bits & 0x80000000 else 0x80000000+bits
    result={}
    for key in keys:
        a,b=left['original_twelve'][key],right['original_twelve'][key]
        ce=key.startswith('loss_voxel_ce_')
        passed=math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-7) if ce else a==b
        result[key]=dict(V=a,G=b,G_minus_V=b-a,float32_ULPs=abs(ordered32(a)-ordered32(b)),
            comparison='math.isclose' if ce else 'exact',rel_tol=1e-5 if ce else 0.,
            abs_tol=1e-7 if ce else 0.,passed=passed)
    require(all(v['passed'] for v in result.values()), 'Initial per-term CE/Lovasz audit failed')
    return result


def intervention_probe(torch, core, condition_core, models, readouts, conditioners,
                       D, inputs, ordinal, train, train_root, native, common, losses,
                       tree, device, budget):
    """Same learned weights/RNG, only velocity changes; no targets or metrics."""
    saved_rng=rng_state(torch); record=train[ordinal]
    result={}
    try:
        with torch.no_grad():
            sample=common.input_only(native,train_root,record,device)
            boundary=tree.tree_digest(sample['inputs'])
            states,valid,receipt=inputs.get(ordinal,condition_core,device)
            require(bool(valid.any()) and bool((states[...,24:]!=0).any()),
                    'Fixed intervention anchor has no nonzero predicted velocity')
            zero=states.clone();zero[...,24:]=0
            require(losses.exact32(states[...,:24],zero[...,:24]),'Intervention changed geometry')
            base=D(sample['inputs']['prev_bev_input'][:,-1])
            for arm in ARMS:
                budget();models[arm].eval();readouts[arm].eval();conditioners[arm].eval()
                outputs=[];hooks=[]
                for supplied in (states,zero):
                    restore_rng(torch,saved_rng)
                    with condition_core.condition_future_head(models[arm].future_pred_head,
                            conditioners[arm],supplied,valid,use_velocity=arm=='V') as hook:
                        prediction,features=core.capture_native_terminal(models[arm],sample,native,training=False)
                    flow=readouts[arm](features,base,detach_features=False)
                    outputs.append((prediction,features,flow));hooks.append(hook)
                left,right=outputs
                require(losses.exact32(left[0][0],right[0][0]) and losses.exact32(left[1][0],right[1][0]),
                        'Velocity intervention changed unconditioned t0')
                measures={}
                for name,x,y in zip(('occupancy','future_state','physical'),left,right):
                    if name!='physical': x,y=x[1:],y[1:]
                    measures[name]=dict(byte_equal=losses.exact32(x,y),
                        max_abs_difference=float((x-y).abs().max()))
                    require(torch.isfinite(x).all() and torch.isfinite(y).all(),'Nonfinite intervention output')
                if arm=='V': require(all(v['max_abs_difference']>0 for v in measures.values()),
                                     'After four updates velocity does not influence both tasks and future state')
                else: require(all(v['byte_equal'] for v in measures.values()),'G depends on supplied velocity')
                del outputs,left,right,prediction,features,flow,x,y
                # Empty tokens must give this arm's native, unconditioned path,
                # not force the already-updated head back to initial O.
                restore_rng(torch,saved_rng)
                native_prediction,native_features=core.capture_native_terminal(models[arm],sample,native,training=False)
                native_flow=readouts[arm](native_features,base,detach_features=False)
                restore_rng(torch,saved_rng)
                with condition_core.condition_future_head(models[arm].future_pred_head,conditioners[arm],
                        states[:,:0],valid[:,:0],use_velocity=arm=='V') as empty_hook:
                    empty_prediction,empty_features=core.capture_native_terminal(models[arm],sample,native,training=False)
                empty_flow=readouts[arm](empty_features,base,detach_features=False)
                require(losses.exact32(native_prediction,empty_prediction) and
                        losses.exact32(native_features,empty_features) and losses.exact32(native_flow,empty_flow),
                        'Empty object set changed native unconditioned path')
                result[arm]=dict(after_completed_updates=4,ordinal=ordinal,sample_token=record['sample_token'],
                    state_source=receipt,velocity_vs_zero=measures,conditioner_hooks=hooks,
                    t0_self_parity=True,empty_objects_native_path_exact=True,empty_hook=empty_hook,
                    supervision_or_scores_computed=False)
                del native_prediction,native_features,native_flow,empty_prediction,empty_features,empty_flow
            require(tree.tree_digest(sample['inputs'])==boundary and 'targets' not in sample,
                    'Intervention changed inputs or loaded targets')
    finally:
        restore_rng(torch,saved_rng)
    require(rng_digest(rng_state(torch))==rng_digest(saved_rng),'Intervention failed to restore RNG')
    return result


def run(a, out, started, stop):
    p,contract,mods=load_contract(a)
    helper=mods['train_source_motion_v1']; native=mods['native_state_cache']
    oracle=mods['oracle_transport_probe']; common=mods['common_change_evaluation_v2']
    losses=mods['train_supported_fusion_v2']; adapter=mods['objective_supervision_adapters']
    connected=mods['train_connected_motion_v2']; tree=mods['joint_native_evaluation']
    train_root,train,cache_receipt=helper.cache_index(a.train_cache,'train',p)
    label_root,label_desc=helper.labels_manifest(a.sparse_labels,p)
    orders=helper.planned_orders(); selected=orders[0][:16]
    input_loader=contract.import_bound('object_state_prediction_inputs_v1',p['sources_sha256'])
    require(p['selection_sha256']==input_loader.SELECTION_SHA,'Selection contract differs')
    predicted_inputs=input_loader.PredictionInputs(a.predictions,a.raw_metadata,a.selection,p['predictions'],train)
    require(len(set(train[i]['scene_token'] for i in selected[:2]))==2, 'First two original anchors not distinct scenes')
    for i in selected: helper.check_identity(train[i], label_desc[train[i]['sample_token']])
    import torch
    require(str(a.device).startswith('cuda') and torch.cuda.is_available(), 'Native preflight requires CUDA')
    torch.cuda.set_device(a.device); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(min(1., a.max_allocated_gib*2**30/
        torch.cuda.get_device_properties(a.device).total_memory), a.device)
    torch.cuda.reset_peak_memory_stats(a.device)
    def budget():
        require(not stop, 'Signal received; no retry or weight persistence')
        require(time.monotonic()-started<a.max_seconds, 'Preflight wall-clock ceiling')
        require(torch.cuda.max_memory_allocated(a.device)<=a.max_allocated_gib*2**30, 'CUDA memory ceiling')
    budget(); write(out/'progress.json',dict(schema=SCHEMA,phase='initialization',updates=0,examples=0))
    model=native.build_native_model(a.config,a.checkpoint,device=a.device,repo=a.repo)
    payload=torch.load(a.o_checkpoint,map_location='cpu',weights_only=False)
    model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True); del payload
    for param in model.parameters(): param.requires_grad_(False)
    model.eval()
    require(common.historical_O_head_digest(model.future_pred_head)==common.O_HEAD_SHA, 'Historical O head differs')
    O_sha=helper.state_digest(model); initial_head_sha=helper.state_digest(model.future_pred_head)
    require(O_sha==read(Path(a.training_run)/'manifest.json')['frozen_O_state_sha256'], 'Actual full O differs')
    frozen_sha=non_head_digest(model)
    D,unused_gate,D_receipt=connected.load_completed_arm(Path(a.training_run)/'runs'/'D',Path(a.connected_protocol),helper,a.device)
    del unused_gate
    require(D_receipt['checkpoint_sha256']==D_CHECKPOINT_SHA and
            D_receipt['top_complete_sha256']==CONNECTED_COMPLETE_SHA, 'Wrong actual D')
    D_sha=helper.state_digest(D)
    require(sha(inspect.getsourcefile(model.compute_occ_loss))==adapter.DETECTOR_SOURCE_SHA and
            sha(inspect.getsourcefile(model.future_pred_head.loss_occ))==adapter.HEAD_SOURCE_SHA and
            sha(inspect.getsourcefile(model.future_pred_head.loss_voxel))==adapter.HEAD_SOURCE_SHA and
            model.future_pred_head.class_weights.tolist()==[1.,5.], 'Native loss formula/weights changed')
    core=contract.import_bound('future_state_motion_v1',p['sources_sha256'])
    condition_core=contract.import_bound('object_state_conditioner_v2',p['sources_sha256'])
    models={'V':model,'G':copy.deepcopy(model)}
    seed_all(torch); readouts={'V':core.FutureStateMotionReadout().to(a.device)}
    readouts['G']=copy.deepcopy(readouts['V']); initial_readout_sha=helper.state_digest(readouts['V'])
    conditioners={'V':condition_core.ObjectStateConditioner().to(a.device)}
    conditioners['G']=copy.deepcopy(conditioners['V']); initial_conditioner_sha=helper.state_digest(conditioners['V'])
    head_params={}; readout_params={}
    for arm in ARMS:
        for param in models[arm].future_pred_head.parameters(): param.requires_grad_(True)
        head_params[arm]=list(models[arm].future_pred_head.named_parameters())
        readout_params[arm]=[('readout.'+n,v) for n,v in readouts[arm].named_parameters()]+[('conditioner.'+n,v) for n,v in conditioners[arm].named_parameters()]
        require(sum(v.numel() for _,v in head_params[arm])==13274016 and
                sum(v.numel() for _,v in readout_params[arm])==745392+95872, 'Trainable parameter count changed')
        require(all(not v.requires_grad for n,v in models[arm].named_parameters() if not n.startswith('future_pred_head.')),
                'Non-future-head parameter unfrozen')
        require(helper.state_digest(models[arm])==O_sha and helper.state_digest(readouts[arm])==initial_readout_sha and helper.state_digest(conditioners[arm])==initial_conditioner_sha,
                'Initial V/G parameters differ')
    require(not ({id(v) for v in models['V'].parameters()} & {id(v) for v in models['G'].parameters()}), 'Shared native parameters')
    require(not ({id(v) for v in readouts['V'].parameters()} & {id(v) for v in readouts['G'].parameters()}), 'Shared readout parameters')
    require(not ({id(v) for v in conditioners['V'].parameters()} & {id(v) for v in conditioners['G'].parameters()}), 'Shared conditioner parameters')
    sources=dict(protocol_sha256=sha(a.protocol),sources_sha256=p['sources_sha256'],
        connected_protocol_sha256=CONNECTED_PROTOCOL_SHA,connected_complete_sha256=CONNECTED_COMPLETE_SHA,
        D_checkpoint_sha256=D_CHECKPOINT_SHA,O_checkpoint_sha256=oracle.O_SHA,M0_checkpoint_sha256=oracle.M0_SHA,
        config_sha256=oracle.CONFIG_SHA,runtime_source_contract_sha256=RUNTIME_SHA,
        train_cache=cache_receipt,sparse_labels=dict(p['labels']),predictions=predicted_inputs.receipt)
    manifest=dict(schema=SCHEMA,mode='preflight',sources=sources,recipe=RECIPE,numerical_policy=NUMERICAL,
        arms=list(ARMS),initial_O_state_sha256=O_sha,initial_future_head_sha256=initial_head_sha,
        initial_readout_sha256=initial_readout_sha,initial_conditioner_sha256=initial_conditioner_sha,frozen_non_head_state_sha256=frozen_sha,
        frozen_D_motion_state_sha256=D_sha,selected_ordinals=selected,sample_orders_sha256=helper.json_hash(orders),
        selected_identities=[dict(ordinal=i,**common.identity(train[i])) for i in selected],
        future_head_parameter_names=[n for n,_ in head_params['V']],readout_parameter_names=[n for n,_ in readout_params['V']],
        labels_only_in_loss=True,no_source_position_feature_selection=True,initial_O_D_exact_required=True,
        occupancy_logits_not_composed=True,postupdate_t0_may_change=True,weights_persisted=False)
    write(out/'manifest.json',manifest)
    write(out/'loaded_models.json',dict(O_full_state_sha256=O_sha,O_historical_head_sha256=common.O_HEAD_SHA,
        O_future_head_state_sha256=initial_head_sha,D=D_receipt,optimizer_restored=False))
    folders={arm:out/'runs'/arm for arm in ARMS}
    for arm,folder in folders.items():
        folder.mkdir(parents=True)
        write(folder/'manifest.json',dict(schema=SCHEMA,arm=arm,common_manifest_sha256=sha(out/'manifest.json'),
            detach_physical_features=False,use_predicted_velocity=(arm=='V'),initial_future_head_sha256=initial_head_sha,
            initial_readout_sha256=initial_readout_sha))
    parity=[]; parity_start=time.monotonic()
    # Same fixed anchors as training; this is initialization engineering only.
    with torch.no_grad():
        for ordinal in selected[:2]:
            budget(); record=train[ordinal]; sample=common.input_only(native,train_root,record,a.device)
            boundary=tree.tree_digest(sample['inputs']); models['V'].eval()
            original=native.replay(models['V'],sample,training=False)[0]
            base=D(sample['inputs']['prev_bev_input'][:,-1])
            common.load_targets_after_predictions(native,train_root,record,sample,a.device)
            original_hist=common.native_hist(models['V'],original,sample,record)
            for arm in ARMS:
                models[arm].eval(); readouts[arm].eval(); conditioners[arm].eval()
                states,valid,state_receipt=predicted_inputs.get(ordinal,condition_core,a.device)
                with condition_core.condition_future_head(models[arm].future_pred_head,conditioners[arm],states,valid,use_velocity=arm=='V') as hook_audit:
                    prediction,features=core.capture_native_terminal(models[arm],sample,native,training=False)
                native_before=tree.tree_digest(prediction)
                flow=readouts[arm](features,base,detach_features=False)
                require(losses.exact32(prediction,original) and losses.exact32(flow,base), 'Initial O/D byte parity failed')
                require(tree.tree_digest(prediction)==native_before, 'Physical readout modified native logits')
                require(common.native_hist(models[arm],prediction,sample,record)==original_hist, 'Initial native full-GT hist differs')
                parity.append(dict(arm=arm,ordinal=ordinal,sample_token=record['sample_token'],scene_token=record['scene_token'],
                    initial_all_5h_3layers_O_logits_exact=True,initial_D_displacement_exact=True,
                    initial_native_hist_exact=True,physical_readout_did_not_modify_native_logits=True,
                    state_source=state_receipt,conditioner_hook=hook_audit))
                del prediction,features,flow,states,valid
            require(tree.tree_digest(sample['inputs'])==boundary, 'Initialization mutated cache inputs')
            del sample,original,base
    parity_seconds=time.monotonic()-parity_start
    require(all(helper.state_digest(m.future_pred_head)==initial_head_sha for m in models.values()) and
            all(helper.state_digest(r)==initial_readout_sha for r in readouts.values()) and
            all(helper.state_digest(c)==initial_conditioner_sha for c in conditioners.values()),
            'Initialization checks changed weights/buffers')
    optimizers={arm:dict(head=torch.optim.AdamW([v for _,v in head_params[arm]],lr=1e-5,
        betas=(.9,.999),eps=1e-8,weight_decay=.01,amsgrad=False),
        readout=torch.optim.AdamW([v for _,v in readout_params[arm]],lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=.01,amsgrad=False)) for arm in ARMS}
    seed_all(torch)
    for arm in ARMS:
        models[arm].train(); readouts[arm].train(); conditioners[arm].train()
        for opt in optimizers[arm].values(): opt.zero_grad(set_to_none=True)
    initialized_seconds=time.monotonic()-started
    had={arm:{n:False for n,_ in readout_params[arm]} for arm in ARMS}
    micro_times=[]; update_times=[]; connectivity={}; rng_rows=[]; examples=updates=0
    for index,ordinal in enumerate(selected):
        budget(); torch.cuda.synchronize(a.device); micro_tick=time.monotonic()
        micro=index%4
        if micro==0:
            update_tick=micro_tick; batches={arm:[] for arm in ARMS}
            for arm in ARMS:
                for group in optimizers[arm]['head'].param_groups: group['lr']=head_lr(updates)
                for group in optimizers[arm]['readout'].param_groups: group['lr']=helper.schedule_lr(updates)
        record=train[ordinal]; sample=common.input_only(native,train_root,record,a.device)
        boundary=tree.tree_digest(sample['inputs'])
        states,valid,state_receipt=predicted_inputs.get(ordinal,condition_core,a.device)
        state_digest=tree.tree_digest(states)
        with torch.no_grad(): base=D(sample['inputs']['prev_bev_input'][:,-1])
        require(not base.requires_grad and torch.isfinite(base).all(), 'Frozen D field invalid')
        paired_rng=rng_state(torch); before_rng_sha=rng_digest(paired_rng); ends={}; first_values=None; initial_loss_comparison=None
        for arm in ARMS:
            restore_rng(torch,paired_rng); budget(); torch.cuda.synchronize(a.device); arm_tick=time.monotonic()
            with condition_core.condition_future_head(models[arm].future_pred_head,conditioners[arm],states,valid,use_velocity=arm=='V') as hook_audit:
                prediction,features=core.capture_native_terminal(models[arm],sample,native,training=True)
            native_digest=tree.tree_digest(prediction)
            flow=readouts[arm](features,base,detach_features=False)
            require(features.requires_grad and torch.isfinite(flow).all() and torch.isfinite(prediction).all(), 'Native feature/field invalid')
            require(tree.tree_digest(prediction)==native_digest, 'Physical readout modified native occupancy tensor')
            # Targets and sparse contents are consumed only after each prediction.
            if 'targets' not in sample: common.load_targets_after_predictions(native,train_root,record,sample,a.device)
            descriptor=label_desc[record['sample_token']]
            label=helper.load_sparse(label_root,descriptor,record)
            occupancy,occ_audit=losses.loss_terms(models[arm],prediction,sample['targets'],adapter)
            occ_audit['inherited_constant_fixed_t0_field_names_only']=True
            occ_audit['intermediate_losses_trainable']=True
            occ_audit['t0_readout_trainable']=True
            sparse=helper.gather_sparse(flow,label)
            physical,physical_audit=helper.object_group_loss(sparse,label)
            total=occupancy+physical
            require(torch.isfinite(total) and torch.isfinite(physical), 'Nonfinite native/physical objective')
            if updates==0:
                require(losses.exact32(flow,base), 'Before first optimizer update motion must equal D')
                if arm=='V': first_values=(native_digest,occ_audit,float(physical.detach()))
                else:
                    require(native_digest==first_values[0], 'Initial paired native logits differ')
                    initial_loss_comparison=compare_initial_occupancy(first_values[1],occ_audit)
                    initial_loss_comparison['physical_scalar_descriptive']=dict(V=first_values[2],
                        G=float(physical.detach()),G_minus_V=float(physical.detach())-first_values[2],
                        acceptance='finite; both dense fields byte-equal the same D; identical sparse NPZ source/support',
                        no_scalar_equality_gate_due_to_CUDA_index_add=True)
            if index==12:
                require(not physical_audit['empty_target'] and bool(valid.any()),'Late fixed anchor lacks required supervision/state')
                groups={'future_head':head_params[arm],
                        'conditioner':list(conditioners[arm].named_parameters())}
                diagnostics={}
                for loss_name,scalar in (('occupancy',occupancy),('physical',physical)):
                    diagnostics[loss_name]={}
                    for group,named in groups.items():
                        vjp=torch.autograd.grad(scalar,[v for _,v in named],allow_unused=True,retain_graph=True)
                        stats=gradient_stats(named,vjp);del vjp
                        require(stats['norm']>0,'Late '+loss_name+' gradient missing at '+group+' '+arm)
                        if group=='future_head':
                            require(any(v is not None and v>0 for n,v in stats['parameter_norms'].items()
                                if n.startswith('transformer.')),'Late loss did not reach transition')
                        else:
                            require(any(v is not None and v>0 for n,v in stats['parameter_norms'].items()
                                if n.startswith('state_mlp.')),'Late loss did not reach object state encoder')
                        diagnostics[loss_name][group]=stats
                connectivity[arm]=dict(after_completed_updates=3,ordinal=ordinal,sample_token=record['sample_token'],
                    supported=True,loss_VJPs=diagnostics,total_backward_not_yet_called_for_this_micro=True)
            (total/4).backward()
            ends[arm]=rng_state(torch)
            require(all(v.grad is None for n,v in models[arm].named_parameters() if not n.startswith('future_pred_head.')) and
                    all(v.grad is None for v in D.parameters()), 'Gradient reached frozen O observer or D')
            require(not D.training and not any(v.requires_grad for v in D.parameters()), 'D mode/scope changed')
            require(tree.tree_digest(sample['inputs'])==boundary and tree.tree_digest(states)==state_digest, 'Training mutated inputs/state tokens')
            torch.cuda.synchronize(a.device)
            batches[arm].append(dict(ordinal=ordinal,**common.identity(record),
                inputs_sha256=record['files']['inputs']['sha256'],targets_sha256=record['files']['targets']['sha256'],
                sparse_label_sha256=descriptor['sha256'],input_tree_sha256=boundary,state_source=state_receipt,conditioner_hook=hook_audit,
                occupancy_loss=occ_audit,physical_loss=float(physical.detach()),physical_audit=physical_audit,
                total_loss=float(total.detach()),native_logits_sha256=native_digest,native_logits_unmodified_by_readout=True,
                paired_rng_before_sha256=before_rng_sha,paired_rng_after_sha256=rng_digest(ends[arm]),
                arm_microstep_seconds=time.monotonic()-arm_tick))
            del prediction,features,flow,sparse,label,physical,occupancy,total
        require(rng_digest(ends['V'])==rng_digest(ends['G']), 'Paired native forward/backward RNG consumption differs')
        if initial_loss_comparison is not None:
            for arm in ARMS: batches[arm][-1]['initial_occupancy_comparison']=initial_loss_comparison
        restore_rng(torch,ends['V'])
        rng_rows.append(dict(ordinal=ordinal,before_sha256=before_rng_sha,after_sha256=rng_digest(ends['V']),paired_exact=True))
        del sample,base,ends,paired_rng,states,valid
        examples+=1; torch.cuda.synchronize(a.device); micro_times.append(time.monotonic()-micro_tick)
        if micro!=3: continue
        norms={}
        for arm in ARMS:
            hg=gradient_stats(head_params[arm],[v.grad for _,v in head_params[arm]])
            rg=gradient_stats(readout_params[arm],[v.grad for _,v in readout_params[arm]])
            require(hg['norm']>0 and rg['norm']>0 and rg['none_parameters']==0, 'Missing aggregate task gradients')
            for n,v in rg['parameter_norms'].items(): had[arm][n]|=v>0
            hnorm=float(torch.nn.utils.clip_grad_norm_([v for _,v in head_params[arm]],35.,error_if_nonfinite=True))
            rnorm=float(torch.nn.utils.clip_grad_norm_([v for _,v in readout_params[arm]],10.,error_if_nonfinite=True))
            norms[arm]=dict(head=hg,readout=rg,head_preclip_norm=hnorm,readout_preclip_norm=rnorm,
                head_clip_factor=min(1.,35./(hnorm+1e-6)),readout_clip_factor=min(1.,10./(rnorm+1e-6)))
        budget()
        for arm in ARMS:
            for opt in optimizers[arm].values():
                opt.step(); opt.zero_grad(set_to_none=True)
                require(all(torch.isfinite(v[k]).all() for v in opt.state.values() for k in ('exp_avg','exp_avg_sq')), 'Nonfinite AdamW moment')
            require(all(torch.isfinite(v).all() for _,v in head_params[arm]+readout_params[arm]), 'Nonfinite updated parameter')
        updates+=1; torch.cuda.synchronize(a.device); update_times.append(time.monotonic()-update_tick)
        for arm in ARMS:
            append(folders[arm]/'training.jsonl',dict(update=updates,examples=examples,
                future_head_lr=head_lr(updates-1),readout_lr=helper.schedule_lr(updates-1),
                samples=batches[arm],gradients=norms[arm],matched_accum4_update_seconds=update_times[-1],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device)))
        progress=dict(schema=SCHEMA,phase='training',updates=updates,examples=examples,seconds=time.monotonic()-started)
        write(out/'progress.json',progress); print(json.dumps(progress),flush=True); budget()
    require(updates==4 and examples==16 and set(connectivity)==set(ARMS), 'Incomplete engineering endpoint')
    require(all(all(v.values()) for v in had.values()), 'Some new module parameters never received a nonzero task gradient')
    intervention_started=time.monotonic()
    interventions=intervention_probe(torch,core,condition_core,models,readouts,conditioners,D,
        predicted_inputs,selected[12],train,train_root,native,common,losses,tree,a.device,budget)
    torch.cuda.synchronize(a.device)
    intervention_seconds=time.monotonic()-intervention_started
    require(helper.state_digest(D)==D_sha and all(non_head_digest(m)==frozen_sha for m in models.values()), 'Frozen state changed')
    finalheads={arm:helper.state_digest(m.future_pred_head) for arm,m in models.items()}
    finalreadouts={arm:helper.state_digest(m) for arm,m in readouts.items()}
    finalconditioners={arm:helper.state_digest(m) for arm,m in conditioners.items()}
    require(all(s!=initial_head_sha for s in finalheads.values()) and all(s!=initial_readout_sha for s in finalreadouts.values()), 'Trainable state did not change')
    require(all(s!=initial_conditioner_sha for s in finalconditioners.values()),'Conditioner state did not change')
    require(all(int(torch.count_nonzero(m.readout.weight))+int(torch.count_nonzero(m.readout.bias))>0 for m in readouts.values()), 'Zero final readout failed to learn')
    optimizer_steps={}
    for arm in ARMS:
        optimizer_steps[arm]={}
        for name,opt in optimizers[arm].items():
            steps=[int(v['step'].item()) for v in opt.state.values()]
            require(steps and all(step==4 for step in steps), 'Actual AdamW endpoint differs')
            optimizer_steps[arm][name]=dict(parameters_with_state=len(steps),all_steps4=True)
    for name,digest in p['sources_sha256'].items(): require(sha(contract.source_path(name))==digest, 'Source changed during run')
    require(sha(a.protocol)==sources['protocol_sha256'], 'Protocol changed during run')
    for field,name in (('complete_sha256','complete.json'),('manifest_sha256','manifest.json'),('predictions_sha256','predictions.json')):
        require(sha(Path(a.predictions)/name)==p['predictions'][field],'Prediction asset changed during preflight')
    summary=dict(schema=SCHEMA,status='PASS_OBJECT_STATE_FORECAST_PREFLIGHT',updates=4,examples=16,
        evaluated_samples=0,scores=None,weights_persisted=False,optimizer_restored=False,
        initial_O_D_parity=parity,initial_O_D_parity_pass=True,paired_rng=rng_rows,
        late_dual_loss_connectivity=connectivity,all_gradients_finite=True,
        same_weights_velocity_interventions=interventions,
        readout_parameter_received_task_gradient=had,zero_readout_changed=True,
        initial_future_head_sha256=initial_head_sha,initial_readout_sha256=initial_readout_sha,
        initial_conditioner_sha256=initial_conditioner_sha,
        final_future_head_sha256=finalheads,final_readout_sha256=finalreadouts,final_conditioner_sha256=finalconditioners,
        frozen_D_and_non_head_O_unchanged=True,actual_optimizer_steps=optimizer_steps,
        native_logits_never_composed=True,postupdate_t0_is_not_required_to_equal_O=True,
        resources=dict(initialization_including_parity_seconds=initialized_seconds,initial_parity_seconds=parity_seconds,
            postupdate_intervention_seconds=intervention_seconds,
            paired_microstep_seconds=micro_times,first_train_microstep_seconds=micro_times[0],
            warm_paired_microstep_mean_seconds=float(np.mean([v for i,v in enumerate(micro_times) if i not in (0,12)])),
            connectivity_VJP_microstep_index=12,matched_accum4_update_seconds=update_times,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device),
            elapsed_seconds=time.monotonic()-started,
            timing_scope='real train-mode forward/loss/accum4 backward/two AdamW groups per arm plus audit overhead; index12 includes retained-graph dual-loss VJPs'))
    write(out/'summary.json',summary)
    for arm in ARMS:
        write(folders[arm]/'complete.json',dict(schema=SCHEMA,status=summary['status'],arm=arm,updates=4,examples=16,
            weights_persisted=False,files_sha256={f:sha(folders[arm]/f) for f in ('manifest.json','training.jsonl')}))
    budget()
    require(not list(out.rglob('*.pth')) and not list(out.rglob('*.pt')), 'Preflight weight persistence prohibited')
    write(out/'complete.json',dict(schema=SCHEMA,status=summary['status'],mode='preflight',updates=4,examples=16,
        evaluated_samples=0,weights_persisted=False,final_checkpoints={},
        files_sha256={f:sha(out/f) for f in ('manifest.json','loaded_models.json','summary.json')},
        arm_complete_sha256={arm:sha(folders[arm]/'complete.json') for arm in ARMS},elapsed_seconds=time.monotonic()-started))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','connected-protocol','training-run','config','checkpoint','o-checkpoint',
                 'repo','runtime-contract','train-cache','sparse-labels','predictions','raw-metadata','selection','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('--max-allocated-gib',type=float,required=True)
    a=parser.parse_args(argv); out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); stop=[]
    signal.signal(signal.SIGTERM,lambda s,f:stop.append(s)); signal.signal(signal.SIGINT,lambda s,f:stop.append(s))
    try: run(a,out,started,stop)
    except BaseException as exc:
        write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),
            traceback=traceback.format_exc(),last_progress=read(out/'progress.json') if (out/'progress.json').is_file() else None,
            weights_persisted=False,seconds=time.monotonic()-started))
        raise


if __name__=='__main__': main()
