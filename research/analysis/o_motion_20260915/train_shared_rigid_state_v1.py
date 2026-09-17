"""Fixed-final Cpl/Fix training only; development evaluation is a separate stage.

Start from original O and a fresh seed11 JointRigidObjectFeatures, never from
preflight weights. Four train512 passes, 2048 paired examples and 512 accum4
AdamW updates. Only learned versus CV Gaussian spatial addresses differ.
No resume, intermediate checkpoint, best selection, dev inference or D load.
TRAINING_COMPLETE_PENDING_FIXED_DEV200 is not complete experimental evidence.
"""
import argparse
import copy
import importlib.util
import inspect
import math
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

PREFLIGHT_SOURCE_SHA='99a043d557b4001259fc0067257177239262e5cac7af84d866f68852a55696d2'
PREFLIGHT_PROTOCOL_SHA='471d587d1254cd6ce6333a523ad18762bef0aa7ee6b07c068a8e8dd4b535cc56'
PREFLIGHT_COMPLETE_SHA='198384ba7e768901c3c996504c897f5e7154bfdb4ddf61d3eb4da089a76d8212'
CACHE_MODULE='shared_rigid_geometry_cache_v1'
SCHEMA='shared-rigid-state-training-v1'
STATUS='TRAINING_COMPLETE_PENDING_FIXED_DEV200'
ARMS=('Cpl','Fix')


def _preflight_module():
    import hashlib
    path=Path(__file__).with_name('shared_rigid_native_preflight_v1.py')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PREFLIGHT_SOURCE_SHA:
        raise ValueError('Source-matched actual preflight is required')
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


pre=_preflight_module()
require,sha,read=pre.require,pre.sha,pre.read
u=pre._contract_loader().import_bound('object_state_forecast_preflight_v2',
    {'object_state_forecast_preflight_v2.py':pre.UTILITY_SHA})
write,append=u.write,u.append
TRAINING=copy.deepcopy(pre.RECIPE)
for key in list(TRAINING):
    if key.startswith(('late_vjp_','address_intervention_')):del TRAINING[key]
TRAINING.update(updates=512,examples=2048,passes=4,formal_authorized=True,
    sample_order='all four continuous RandomState11 permutations of 512',
    weights_persisted=True,development_evaluation=False,
    checkpoint='fixed final512 only; no resume, intermediate save, best selection or preflight state reuse',
    initialization='rebuild original O and fresh seed11 module; authenticate initial state against discarded preflight',
    geometry='authenticated complete712 geometry cache; train512 only; no CPU geometry recomputation',
    final_status=STATUS,optimizer_none_grad='original AdamW skips None; record per-parameter actual step counts',
    engineering_checks='successful actual preflight bound; no repeated VJP/address intervention during formal training')


def protocol_template():
    template=pre.protocol_template()
    template.pop('recipe');template.update(schema=SCHEMA,status='REVIEW_REQUIRED',training=copy.deepcopy(TRAINING),
        preflight_protocol_sha256=PREFLIGHT_PROTOCOL_SHA,engineering_preflight_complete_sha256=PREFLIGHT_COMPLETE_SHA,
        geometry_cache_complete_sha256=None,resources=None)
    return template


def expected_sources(preflight_protocol):
    contract=pre._contract_loader();expected=dict(preflight_protocol['sources_sha256'])
    expected[CACHE_MODULE+'.py']=sha(contract.source_path(CACHE_MODULE+'.py'))
    cache_module=contract.import_bound(CACHE_MODULE,expected)
    for name,digest in cache_module.SOURCES.items():
        require(name not in expected or expected[name]==digest,'Cache and native source bindings disagree: '+name)
        require(sha(contract.source_path(name))==digest,'Cache dependency changed: '+name)
        expected[name]=digest
    expected[Path(__file__).name]=sha(__file__)
    return expected


def validate_engineering(root,p,pp):
    """Authenticate only actual small engineering receipts; no checkpoint load."""
    root=Path(root);require(not (root/'failed.json').exists(),'Engineering preflight failed')
    require(sha(root/'complete.json')==p['engineering_preflight_complete_sha256']==PREFLIGHT_COMPLETE_SHA,
            'Wrong actual discarded preflight')
    done=read(root/'complete.json')
    require(done['schema']==pre.SCHEMA and done['status']=='PASS_SHARED_RIGID_NATIVE_PREFLIGHT' and
        done['updates']==4 and done['examples']==16 and done['evaluated_samples']==0 and
        done['weights_persisted'] is False and done['final_checkpoints']=={},'Preflight endpoint is not a discarded four-update pass')
    require(set(done['files_sha256'])=={'manifest.json','loaded_models.json','geometry.json','summary.json'} and
        set(done['arm_complete_sha256'])==set(ARMS),'Actual preflight ledger differs')
    for name,digest in done['files_sha256'].items():require(sha(root/name)==digest,'Preflight artifact changed: '+name)
    manifest=read(root/'manifest.json');summary=read(root/'summary.json')
    require(manifest['recipe']==pre.RECIPE and manifest['numerical_policy']==pre.NUMERICAL and
        manifest['sources']['protocol_sha256']==p['preflight_protocol_sha256'] and
        manifest['sources']['sources_sha256']==pp['sources_sha256'],'Actual preflight source/recipe differs')
    require(summary['initial_O_CV_parity_pass'] and summary['all_train_losses_gradients_parameters_moments_finite'] and
        summary['frozen_non_head_O_unchanged'] and summary['weights_persisted'] is False,'Missing engineering pass')
    for arm in ARMS:
        folder=root/'runs'/arm
        require(sha(folder/'complete.json')==done['arm_complete_sha256'][arm],'Preflight arm completion changed')
        ac=read(folder/'complete.json')
        require(ac['updates']==4 and ac['examples']==16 and ac['weights_persisted'] is False,'Preflight arm differs')
        for name,digest in ac['files_sha256'].items():require(sha(folder/name)==digest,'Preflight arm artifact changed')
    return manifest,dict(complete_sha256=sha(root/'complete.json'),manifest_sha256=sha(root/'manifest.json'),
        summary_sha256=sha(root/'summary.json'),actual_preflight_passed=True,weights_loaded=False)


def load_contract(a):
    p=read(a.protocol)
    require(p['schema']==SCHEMA and p['status']=='FROZEN' and p['training']==TRAINING and
        p['numerical_policy']==pre.NUMERICAL,'Exact frozen formal training recipe required')
    require(math.isfinite(a.max_seconds) and a.max_seconds>0 and math.isfinite(a.max_allocated_gib) and
        a.max_allocated_gib>0 and p['resources']==dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib),
        'Finite exact formally frozen resource limits required')
    require(p['preflight_protocol_sha256']==sha(a.preflight_protocol)==PREFLIGHT_PROTOCOL_SHA,
            'Passed preflight protocol changed')
    pp=read(a.preflight_protocol);args=SimpleNamespace(**vars(a));args.protocol=a.preflight_protocol
    args.max_seconds=pp['resources']['preflight']['max_seconds'];args.max_allocated_gib=pp['resources']['preflight']['max_allocated_gib']
    pp,contract,mods=pre.load_contract(args)
    expected=expected_sources(pp);require(p['sources_sha256']==expected,'Exact formal source closure required')
    for name,digest in expected.items():require(sha(contract.source_path(name))==digest,'Formal source changed: '+name)
    for key in ('connected_protocol_sha256','connected_complete_sha256','runtime_source_contract_sha256',
                'cache_index_sha256','labels','predictions','selection_sha256'):
        require(p[key]==pp[key],'Inherited data/source changed: '+key)
    digest=p['geometry_cache_complete_sha256']
    require(isinstance(digest,str) and len(digest)==64 and set(digest)<=set('0123456789abcdef'),
            'An actual completed geometry cache binding is required')
    engineering,receipt=validate_engineering(a.engineering_preflight,p,pp)
    cache_module=contract.import_bound(CACHE_MODULE,p['sources_sha256'])
    return p,contract,mods,cache_module,engineering,receipt


def final_optimizer_receipt(torch,optimizer,named,counts):
    rows={}
    for name,param in named:
        state=optimizer.state.get(param);count=counts[name]
        if not state:
            require(count==0,'Missing actual AdamW state after a gradient update');rows[name]=None;continue
        pre.finite_tensors(torch,[state[k] for k in ('step','exp_avg','exp_avg_sq')],'Nonfinite final optimizer state')
        step=int(state['step'].item());require(step==count and 0<step<=512,'AdamW step disagrees with actual non-None gradients')
        rows[name]=dict(step=step,exp_avg_norm=float(state['exp_avg'].double().norm()),
            exp_avg_sq_norm=float(state['exp_avg_sq'].double().norm()))
    return dict(named_parameter_states=rows,actual_non_none_gradient_updates=dict(counts),
        parameters_with_state=sum(v is not None for v in rows.values()),
        parameters_without_state=sum(v is None for v in rows.values()))


def run(a,out,started,stop):
    p,contract,mods,cache_module,engineering,engineering_receipt=load_contract(a)
    helper=mods['train_source_motion_v1'];native=mods['native_state_cache'];oracle=mods['oracle_transport_probe']
    common=mods['common_change_evaluation_v2'];losses=mods['train_supported_fusion_v2']
    adapter=mods['objective_supervision_adapters'];tree=mods['joint_native_evaluation']
    core=mods['object_state_conditioner_v2'];cuda_ready=False;updates=examples=0
    def budget():
        require(not stop,'Signal received; no retry/resume')
        require(time.monotonic()-started<a.max_seconds,'Formal whole-process time ceiling')
        if cuda_ready:require(torch.cuda.max_memory_allocated(a.device)<=a.max_allocated_gib*2**30,'Formal CUDA allocation ceiling')
    def progress(phase):
        budget();row=dict(schema=SCHEMA,phase=phase,updates=updates,examples=examples,seconds=time.monotonic()-started)
        write(out/'progress.json',row);print(u.json.dumps(row),flush=True)
    progress('authenticated_inputs')
    train_root,train,cache_receipt=helper.cache_index(a.train_cache,'train',p)
    label_root,labels=helper.labels_manifest(a.sparse_labels,p)
    for record in train:helper.check_identity(record,labels[record['sample_token']])
    loader=mods['object_state_prediction_inputs_v1']
    predicted=loader.PredictionInputs(a.predictions,a.raw_metadata,a.selection,p['predictions'],train)
    geometry_cache=cache_module.GeometryCache(a.geometry_cache,p['geometry_cache_complete_sha256'],a.selection)
    cache_source=geometry_cache.receipt
    require(cache_source['complete_sha256']==p['geometry_cache_complete_sha256'] and
        cache_source['source_sha256']==p['sources_sha256'][CACHE_MODULE+'.py'] and
        cache_source['dependencies_sha256']==cache_module.SOURCES and cache_source['selection_sha256']==p['selection_sha256'],
        'Geometry cache implementation/selection differs')
    require(all(cache_source['train_predictions'][key]==value for key,value in p['predictions'].items()) and
        cache_source['raw_manifest_sha256']==predicted.receipt['raw_manifest_sha256'] and
        cache_source['raw_complete_sha256']==predicted.receipt['raw_complete_sha256'],
        'Geometry cache predictions/current pose differ from actual preflight')
    orders=helper.planned_orders();require(len(orders)==4 and all(len(o)==512 for o in orders),'Fixed train512 orders changed')
    import torch
    require(str(a.device).startswith('cuda') and torch.cuda.is_available(),'Native formal training requires CUDA')
    torch.cuda.set_device(a.device);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(min(1.,a.max_allocated_gib*2**30/
        torch.cuda.get_device_properties(a.device).total_memory),a.device)
    torch.cuda.reset_peak_memory_stats(a.device);cuda_ready=True
    rigid=contract.import_bound('rigid_object_motion_v1',p['sources_sha256'])
    object_core=contract.import_bound('joint_rigid_object_features_v1',p['sources_sha256'])
    bridge=contract.import_bound('shared_rigid_native_bridge_v1',p['sources_sha256'])
    progress('fresh_native_initialization')
    model=native.build_native_model(a.config,a.checkpoint,device=a.device,repo=a.repo)
    payload=torch.load(a.o_checkpoint,map_location='cpu',weights_only=False)
    model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True);del payload
    for value in model.parameters():value.requires_grad_(False)
    model.eval();require(common.historical_O_head_digest(model.future_pred_head)==common.O_HEAD_SHA,'Original O head differs')
    O_sha=helper.state_digest(model);head_initial=helper.state_digest(model.future_pred_head);frozen=u.non_head_digest(model)
    require(O_sha==read(Path(a.training_run)/'manifest.json')['frozen_O_state_sha256']==engineering['initial_O_state_sha256'],
            'Fresh original full O differs')
    require(sha(inspect.getsourcefile(model.compute_occ_loss))==adapter.DETECTOR_SOURCE_SHA and
        sha(inspect.getsourcefile(model.future_pred_head.loss_occ))==adapter.HEAD_SOURCE_SHA and
        sha(inspect.getsourcefile(model.future_pred_head.loss_voxel))==adapter.HEAD_SOURCE_SHA and
        model.future_pred_head.class_weights.tolist()==[1.,5.],'Original objective changed')
    models={'Cpl':model,'Fix':copy.deepcopy(model)}
    u.seed_all(torch);objects={'Cpl':object_core.JointRigidObjectFeatures().to(a.device)}
    objects['Fix']=copy.deepcopy(objects['Cpl']);new_initial=helper.state_digest(objects['Cpl'])
    require(head_initial==engineering['initial_future_head_sha256'] and new_initial==engineering['initial_new_module_sha256'],
            'Fresh seed11/O states do not reproduce preflight initialization')
    head_params={};new_params={};opts={};counts={}
    for arm in ARMS:
        for value in models[arm].future_pred_head.parameters():value.requires_grad_(True)
        head_params[arm]=list(models[arm].future_pred_head.named_parameters());new_params[arm]=list(objects[arm].named_parameters())
        require(sum(v.numel() for _,v in head_params[arm])==13274016 and sum(v.numel() for _,v in new_params[arm])==67366,
                'Formal trainable capacity differs')
        require(helper.state_digest(models[arm])==O_sha and helper.state_digest(objects[arm])==new_initial,'Fresh paired states differ')
        require(all(not v.requires_grad for n,v in models[arm].named_parameters() if not n.startswith('future_pred_head.')),
                'Native observer unfrozen')
        opts[arm]={key:torch.optim.AdamW([v for _,v in named],lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01,amsgrad=False)
                   for key,named,lr in (('head',head_params[arm],1e-5),('new',new_params[arm],1e-3))}
        counts[arm]={key:{name:0 for name,_ in named} for key,named in (('head',head_params[arm]),('new',new_params[arm]))}
    for modules in (models,objects):
        require(not ({id(v) for v in modules['Cpl'].parameters()} & {id(v) for v in modules['Fix'].parameters()}),'Shared arm parameters')
    schedules=dict(head=[u.head_lr(i) for i in range(512)],new=[helper.schedule_lr(i) for i in range(512)])
    sources=dict(protocol_sha256=sha(a.protocol),sources_sha256=p['sources_sha256'],
        preflight=engineering_receipt,preflight_protocol_sha256=PREFLIGHT_PROTOCOL_SHA,
        connected_protocol_sha256=pre.CONNECTED_PROTOCOL_SHA,connected_complete_sha256=pre.CONNECTED_COMPLETE_SHA,
        historical_manifest_sha256=sha(Path(a.training_run)/'manifest.json'),
        O_checkpoint_sha256=oracle.O_SHA,M0_checkpoint_sha256=oracle.M0_SHA,config_sha256=oracle.CONFIG_SHA,
        runtime_source_contract_sha256=pre.RUNTIME_SHA,train_cache=cache_receipt,sparse_labels=p['labels'],
        predictions=predicted.receipt,geometry_cache_complete_sha256=p['geometry_cache_complete_sha256'],
        geometry_cache_manifest_sha256=sha(Path(a.geometry_cache)/'manifest.json'),geometry_cache=cache_source)
    manifest=dict(schema=SCHEMA,mode='training_only',training=TRAINING,numerical_policy=pre.NUMERICAL,sources=sources,
        arms=list(ARMS),initial_O_state_sha256=O_sha,initial_future_head_sha256=head_initial,initial_new_module_sha256=new_initial,
        frozen_non_head_state_sha256=frozen,sample_orders=orders,sample_orders_sha256=helper.json_hash(orders),
        schedules=schedules,future_head_parameter_names=[n for n,_ in head_params['Cpl']],
        new_parameter_names=[n for n,_ in new_params['Cpl']],D_model_loaded=False,preflight_weights_loaded=False,
        development_evaluation=False,GT_not_passed_to_forward=True,physical_support_defined_before_sparse_labels=True)
    write(out/'manifest.json',manifest);folders={arm:out/'runs'/arm for arm in ARMS}
    for arm,folder in folders.items():
        folder.mkdir(parents=True);write(folder/'manifest.json',dict(schema=SCHEMA,arm=arm,
            common_manifest_sha256=sha(out/'manifest.json'),coupled_spatial_address=arm=='Cpl',
            learned_values_include_pose_residual=True,physical_state_learned=True,
            initial_future_head_sha256=head_initial,initial_new_module_sha256=new_initial))
    u.seed_all(torch);initial_rng_sha=u.rng_digest(u.rng_state(torch))
    for arm in ARMS:
        models[arm].train();objects[arm].train()
        for optimizer in opts[arm].values():optimizer.zero_grad(set_to_none=True)
    initialization_seconds=time.monotonic()-started;update_times=[];progress('training')
    for pass_index,order in enumerate(orders):
        for ordinal in order:
            budget();torch.cuda.synchronize(a.device);micro=examples%4
            if micro==0:
                tick=time.monotonic();batches={arm:[] for arm in ARMS}
                for arm in ARMS:
                    for key,optimizer in opts[arm].items():
                        for group in optimizer.param_groups:group['lr']=schedules[key][updates]
            record=train[ordinal];sample=common.input_only(native,train_root,record,a.device)
            forward_sample=dict(inputs=sample['inputs'],record=sample['record']);boundary=tree.tree_digest(sample['inputs'])
            geometry=geometry_cache.load('train',ordinal,common.identity(record))
            data,state_receipt=pre.device_geometry(torch,core,geometry,a.device);del geometry
            geometry_sha=tree.tree_digest(data)
            base=pre.cv_reference(torch,bridge,data,oracle) if updates==0 else None
            paired=u.rng_state(torch);before=u.rng_digest(paired);ends={};initial=None;comparison=None
            for arm in ARMS:
                u.restore_rng(torch,paired);budget()
                output=pre.object_forward(objects[arm],data,forward_sample['inputs']['prev_bev_input'][:,-1],oracle,arm=='Cpl')
                flow=pre.material_field(rigid,bridge,data,output,oracle)
                with bridge.inject_future_object_features(models[arm].future_pred_head,output['native_delta']) as hooks:
                    prediction=native.replay(models[arm],forward_sample,training=True)[0]
                pre.finite_tensors(torch,[prediction,flow]+list(output.values()),'Nonfinite complete prediction')
                native_digest=pre.tensor_sha(prediction);flow_digest=pre.tensor_sha(flow)
                require('targets' not in forward_sample,'GT crossed forward boundary')
                if 'targets' not in sample:common.load_targets_after_predictions(native,train_root,record,sample,a.device)
                descriptor=labels[record['sample_token']];label=helper.load_sparse(label_root,descriptor,record)
                occupancy,occ_audit=losses.loss_terms(models[arm],prediction,sample['targets'],adapter)
                occ_audit.update(inherited_constant_fixed_t0_field_names_only=True,intermediate_losses_trainable=True,t0_readout_trainable=True)
                sparse=helper.gather_sparse(flow,label);physical,physical_audit=helper.object_group_loss(sparse,label);total=occupancy+physical
                pre.finite_tensors(torch,[occupancy,physical,total],'Nonfinite formal objective')
                if updates==0:
                    require(pre.exact_bytes(flow,base),'Fresh module must preserve complete CV before first update')
                    if arm=='Cpl':initial=(native_digest,flow_digest,occ_audit)
                    else:
                        require((native_digest,flow_digest)==initial[:2],'Fresh paired logits/material fields differ')
                        comparison=u.compare_initial_occupancy(initial[2],occ_audit)
                        for row in comparison.values():
                            row['Cpl']=row.pop('V');row['Fix']=row.pop('G');row['Fix_minus_Cpl']=row.pop('G_minus_V')
                (total/4).backward();ends[arm]=u.rng_state(torch)
                require(all(v.grad is None for n,v in models[arm].named_parameters() if not n.startswith('future_pred_head.')),
                        'Gradient reached frozen native observer')
                require(tree.tree_digest(sample['inputs'])==boundary and tree.tree_digest(data)==geometry_sha,'Training mutated current inputs')
                batches[arm].append(dict(ordinal=ordinal,**common.identity(record),
                    inputs_sha256=record['files']['inputs']['sha256'],targets_sha256=record['files']['targets']['sha256'],
                    sparse_label_sha256=descriptor['sha256'],input_tree_sha256=boundary,geometry_tree_sha256=geometry_sha,
                    state_source=state_receipt,hook_counts=list(hooks),occupancy_loss=occ_audit,
                    physical_loss=float(physical.detach()),physical_audit=physical_audit,total_loss=float(total.detach()),
                    native_logits_sha256=native_digest,complete_material_field_sha256=flow_digest,
                    paired_rng_before_sha256=before,paired_rng_after_sha256=u.rng_digest(ends[arm])))
                del output,flow,prediction,sparse,label,occupancy,physical,total
            require(u.rng_digest(ends['Cpl'])==u.rng_digest(ends['Fix']),'Paired RNG consumption differs')
            u.restore_rng(torch,ends['Cpl'])
            if comparison is not None:
                for arm in ARMS:batches[arm][-1]['initial_occupancy_comparison']=comparison
            del sample,forward_sample,data,base,paired,ends
            examples+=1
            if micro!=3:continue
            gradients={}
            for arm in ARMS:
                gradients[arm]={}
                for key,named,limit in (('head',head_params[arm],35.),('new',new_params[arm],10.)):
                    stats=u.gradient_stats(named,[v.grad for _,v in named])
                    norm=float(torch.nn.utils.clip_grad_norm_([v for _,v in named],limit,error_if_nonfinite=True))
                    gradients[arm][key]=dict(stats=stats,preclip_norm=norm,clip_factor=min(1.,limit/(norm+1e-6)))
                    for name,value in named:counts[arm][key][name]+=int(value.grad is not None)
            budget()
            for arm in ARMS:
                for optimizer in opts[arm].values():
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                    pre.finite_tensors(torch,[v[k] for v in optimizer.state.values() for k in ('step','exp_avg','exp_avg_sq')],
                        'Nonfinite actual AdamW state')
                pre.finite_tensors(torch,[v for _,v in head_params[arm]+new_params[arm]],'Nonfinite updated parameters')
            updates+=1;torch.cuda.synchronize(a.device);update_times.append(time.monotonic()-tick)
            for arm in ARMS:
                append(folders[arm]/'training.jsonl',dict(update=updates,examples=examples,pass_index=pass_index,
                    future_head_lr=schedules['head'][updates-1],new_lr=schedules['new'][updates-1],samples=batches[arm],
                    gradients=gradients[arm],matched_accum4_update_seconds=update_times[-1],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device)))
            progress('training')
        require(all(u.non_head_digest(m)==frozen for m in models.values()),'Frozen observer changed during train pass')
    require(updates==512 and examples==2048,'Incomplete fixed training endpoint')
    progress('fixed_final_checkpoint')
    finalhead={arm:helper.state_digest(m.future_pred_head) for arm,m in models.items()}
    finalnew={arm:helper.state_digest(m) for arm,m in objects.items()}
    final_rng=u.rng_state(torch);final_rng_sha=u.rng_digest(final_rng);optimizer_steps={};finals={}
    for arm in ARMS:
        optimizer_steps[arm]={key:final_optimizer_receipt(torch,opt,head_params[arm] if key=='head' else new_params[arm],counts[arm][key])
            for key,opt in opts[arm].items()}
        payload=dict(schema=SCHEMA,status='FIXED_FINAL_512_PENDING_FIXED_DEV200',arm=arm,
            future_pred_head=models[arm].future_pred_head.state_dict(),joint_rigid_object_features=objects[arm].state_dict(),
            optimizers={key:opt.state_dict() for key,opt in opts[arm].items()},update=512,examples=2048,
            sample_orders=orders,sample_orders_sha256=helper.json_hash(orders),schedules=schedules,
            schedule_updates_applied=512,rng_state=final_rng,rng_state_sha256=final_rng_sha,initial_train_rng_sha256=initial_rng_sha,
            initial_future_head_sha256=head_initial,initial_new_module_sha256=new_initial,
            final_future_head_sha256=finalhead[arm],final_new_module_sha256=finalnew[arm],
            frozen_non_head_state_sha256=frozen,actual_optimizer_steps=optimizer_steps[arm],
            optimizer_parameter_names={'head':[n for n,_ in head_params[arm]],'new':[n for n,_ in new_params[arm]]},
            manifest_sha256=sha(folders[arm]/'manifest.json'),common_manifest_sha256=sha(out/'manifest.json'),
            sources=sources,preflight_weights_loaded=False,resume_supported=False,development_evaluation_performed=False)
        path=folders[arm]/'final.pth';temporary=path.with_suffix('.tmp')
        torch.save(payload,str(temporary));temporary.replace(path);del payload
        finals[arm]=dict(file=str(path.relative_to(out)),sha256=sha(path),future_head_state_sha256=finalhead[arm],
            new_module_state_sha256=finalnew[arm],rng_state_sha256=final_rng_sha)
    for name,digest in p['sources_sha256'].items():require(sha(contract.source_path(name))==digest,'Source changed during formal training')
    require(sha(a.protocol)==sources['protocol_sha256'],'Formal protocol changed')
    require(sha(Path(a.geometry_cache)/'complete.json')==p['geometry_cache_complete_sha256'] and
            sha(Path(a.geometry_cache)/'manifest.json')==sources['geometry_cache_manifest_sha256'],'Geometry cache contract changed')
    for key,name in (('complete_sha256','complete.json'),('manifest_sha256','manifest.json'),('predictions_sha256','predictions.json')):
        require(sha(Path(a.predictions)/name)==p['predictions'][key],'Prediction asset changed')
    summary=dict(schema=SCHEMA,status=STATUS,updates=512,examples=2048,passes=4,evaluated_samples=0,scores=None,
        development_evaluation_performed=False,experiment_complete=False,final_checkpoints=finals,
        initial_future_head_sha256=head_initial,initial_new_module_sha256=new_initial,
        final_future_head_sha256=finalhead,final_new_module_sha256=finalnew,
        actual_optimizer_steps=optimizer_steps,frozen_non_head_O_unchanged=True,
        paired_rng_all_examples_exact=True,initial_train_rng_sha256=initial_rng_sha,final_rng_sha256=final_rng_sha,
        all_logged_losses_gradients_parameters_moments_finite=True,preflight_weights_loaded=False,D_model_loaded=False,
        resources=dict(initialization_seconds=initialization_seconds,matched_accum4_update_seconds=update_times,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device),
            elapsed_seconds=time.monotonic()-started))
    write(out/'summary.json',summary)
    for arm in ARMS:
        write(folders[arm]/'complete.json',dict(schema=SCHEMA,status=STATUS,arm=arm,updates=512,examples=2048,
            evaluated_samples=0,final_checkpoint=finals[arm],
            files_sha256={name:sha(folders[arm]/name) for name in ('manifest.json','training.jsonl','final.pth')}))
    budget()
    write(out/'complete.json',dict(schema=SCHEMA,status=STATUS,mode='training_only',updates=512,examples=2048,
        evaluated_samples=0,experiment_complete=False,development_evaluation_performed=False,final_checkpoints=finals,
        files_sha256={name:sha(out/name) for name in ('manifest.json','summary.json')},
        arm_complete_sha256={arm:sha(folders[arm]/'complete.json') for arm in ARMS},elapsed_seconds=time.monotonic()-started))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','preflight-protocol','engineering-preflight','connected-protocol','training-run',
                 'config','checkpoint','o-checkpoint','repo','runtime-contract','train-cache','sparse-labels',
                 'predictions','raw-metadata','selection','geometry-cache','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('--max-allocated-gib',type=float,required=True)
    a=parser.parse_args(argv);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();stop=[]
    signal.signal(signal.SIGTERM,lambda s,f:stop.append(s));signal.signal(signal.SIGINT,lambda s,f:stop.append(s))
    try:run(a,out,started,stop)
    except BaseException as exc:
        write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),
            last_progress=read(out/'progress.json') if (out/'progress.json').is_file() else None,
            partial_final_checkpoint_is_not_training_completion=True,seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
