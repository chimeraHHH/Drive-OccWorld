"""Strict, read-only final512 K/B loader. No optimizer is restored.

``authenticated_training`` reads completed small artifacts only. Actual state
and AdamW tensor checks occur only in ``load_completed_arm``. The latter takes
an already authenticated native O model, copies it, and leaves that reference
untouched. D remains external and uses the existing connected-motion loader.

Deploy the frozen runtime/analysis sources, protocol, selection_v1.json and
o_development_records.jsonl with this file, or supply explicit identity paths.
No intermediate checkpoint or incomplete training directory is accepted.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

SCHEMA='future-state-motion-training-v1'
STATUS='COMPLETE_FUTURE_STATE_MOTION_TRAINING'
PROTOCOL_SHA='4129a99bf53a956df7e4467cd582b180d5046226dd3a9fe25f005cd42be652f7'
TRAINER_SHA='2cb16d2d1e15a17b808b23cee8637066c9f5fb85f934950cdf94d680e10bc393'
CORE_SHA='96e796dd5bb6d33949a03a6c451919ef2e19c75b81eaea4b10e2c98f8ecc176e'
ARMS=('K','B')
READOUT_NAMES=('trunk.0.weight','trunk.0.bias','trunk.1.weight','trunk.1.bias',
               'trunk.3.weight','trunk.3.bias','trunk.4.weight','trunk.4.bias','readout.weight','readout.bias')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def source_path(name):
    require(Path(name).name==name and name.endswith('.py'),'Python source basename required')
    for path in (Path(__file__).parent/name,Path(__file__).parent.parent/'m0_improvement_20260915'/name):
        if path.is_file():return path
    raise FileNotFoundError(name)


def import_source(name,digest):
    path=source_path(name+'.py');require(sha(path)==digest,'Source changed: '+name)
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve()==path.resolve(),'Different source alias: '+name)
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def finite_tree(value):
    if isinstance(value,float):require(math.isfinite(value),'Nonfinite recorded number')
    elif isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,list):
        for v in value:finite_tree(v)


def valid_sha(value):return isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef')


def file_chain(root,receipt,expected,skip=()):
    require(set(receipt['files_sha256'])==set(expected),'Completed file set differs')
    for name,digest in receipt['files_sha256'].items():
        require(Path(name).name==name and valid_sha(digest),'Invalid artifact path/hash')
        if name not in skip:require(sha(root/name)==digest,'Completed artifact changed: '+name)


def identity_file(explicit,name,digest):
    if explicit is not None:candidates=[Path(explicit)]
    elif name=='selection_v1.json':
        candidates=[Path(__file__).parent/name,Path(__file__).parent.parent/'m0_improvement_20260915'/name]
    else:
        parent=Path(__file__).parent.parent/'m0_improvement_20260915'
        candidates=[Path(__file__).parent/name,
            parent/'server_results/campaign_objective_v1/runs/O/development_records.jsonl',
            parent/'campaign_objective_v1/runs/O/development_records.jsonl']
    for path in candidates:
        if path.is_file():
            require(sha(path)==digest,'Wrong frozen identity/reference file: '+str(path));return path
    raise FileNotFoundError('Provide authenticated '+name)


def metadata(training_run,protocol_path,helper=None,*,verify_checkpoint_bytes=False,
             selection_path=None,o_reference_path=None):
    # Completion is checked before opening logs, tensors, or any intermediate.
    root=Path(training_run).resolve();done=read(root/'complete.json');complete_sha=sha(root/'complete.json')
    require(not (root/'failed.json').exists() and done['schema']==SCHEMA and done['status']==STATUS
        and done['mode']=='train' and done['updates']==512 and done['examples']==2048
        and done['evaluated_samples']==200,'Only complete final512/dev200 K/B is loadable')
    protocol_path=Path(protocol_path);require(sha(protocol_path)==PROTOCOL_SHA,'Wrong frozen formal protocol')
    p=read(protocol_path);require(p['schema']==SCHEMA and p['status']=='FROZEN','Wrong protocol schema/status')
    bindings=p['sources_sha256'];require(len(bindings)==18 and bindings['future_state_motion_train_v1.py']==TRAINER_SHA
        and bindings['future_state_motion_v1.py']==CORE_SHA,'Wrong native-state training/core sources')
    for name,digest in {**bindings,**p['analysis_sources_sha256']}.items():
        require(valid_sha(digest) and sha(source_path(name))==digest,'Bound source changed: '+name)
    trainer=import_source('future_state_motion_train_v1',TRAINER_SHA)
    require(p['training']==trainer.TRAINING and p['numerical_policy']==trainer.pre.NUMERICAL,'Formal recipe changed')
    if helper is None:helper=import_source('train_source_motion_v1',bindings['train_source_motion_v1.py'])
    require(sha(helper.__file__)==bindings['train_source_motion_v1.py'],'Wrong motion helper')
    selection=read(identity_file(selection_path,'selection_v1.json',p['selection_sha256']))
    require(selection['schema']=='native-state-selection-v1','Wrong selection schema')
    train=[r for r in selection['records'] if r['split']=='train']
    dev=[r for r in selection['records'] if r['split']=='development']
    require(len(train)==512 and len(dev)==200 and len({r['scene_token'] for r in dev})==100
        and not ({r['scene_token'] for r in train}&{r['scene_token'] for r in dev}),'Selection counts/isolation differ')
    original=[json.loads(s) for s in identity_file(o_reference_path,'o_development_records.jsonl',
        p['o_development_records_sha256']).read_text().splitlines()]
    require(len(original)==200,'Wrong original O reference count')
    expected_top=('manifest.json','loaded_models.json','summary.json','physical_gradient_diagnostics.jsonl',
                  'development_records.jsonl','development_objects.jsonl')
    file_chain(root,done,expected_top)
    require(set(done['arm_complete_sha256'])==set(done['final_checkpoints'])==set(ARMS),'Both matched completed arms required')
    manifest=read(root/'manifest.json');summary=read(root/'summary.json');loaded=read(root/'loaded_models.json')
    finite_tree(summary)
    require(manifest['schema']==SCHEMA and manifest['mode']=='train' and manifest['training']==p['training']
        and manifest['numerical_policy']==p['numerical_policy'] and manifest['fresh_initialization']
        and not manifest['optimizer_restored'] and manifest['native_logits_never_composed']
        and manifest['t0_prediction_may_change'] and manifest['labels_only_for_loss_or_evaluation']
        and manifest['no_source_position_feature_selection'] and manifest['no_initial_development_evaluation'],
        'Formal model/data boundary differs')
    sources=manifest['sources']
    require(sources['protocol_sha256']==PROTOCOL_SHA and sources['sources_sha256']==bindings,'Runtime source ledger differs')
    for key in ('preflight_protocol_sha256','engineering_preflight_complete_sha256','connected_protocol_sha256',
                'connected_complete_sha256','D_checkpoint_sha256','runtime_source_contract_sha256',
                'raw_labels','o_development_records_sha256','analysis_sources_sha256','selection_sha256'):
        require(sources[key]==p[key],'Source/analysis binding differs: '+key)
    require(sources['sparse_labels']==p['labels'] and all(sources[role+'_cache']['index_sha256']==p['cache_index_sha256'][role]
        for role in ('train','development')),'Cached input or physical-label binding differs')
    oracle=import_source('oracle_transport_probe',bindings['oracle_transport_probe.py'])
    require(sources['config_sha256']==oracle.CONFIG_SHA and sources['M0_checkpoint_sha256']==oracle.M0_SHA
        and sources['O_checkpoint_sha256']==oracle.O_SHA,'Original native O assets differ')
    engineering=dict(complete_sha256=p['engineering_preflight_complete_sha256'],
        manifest_sha256='568a0862f0c3e87a5a1b3a2cca9460eca8bbcafa2d884442829c12d3f9acb5e1',
        summary_sha256='e5d2f4772c5e76bfef73aa2024e7a411541f7e9e6aef52861ca444084df505b1',
        weights_loaded=False,actual_preflight_passed=True)
    require(manifest['engineering_preflight']==summary['initial_engineering']==engineering,'Fresh preflight provenance differs')
    require(len(manifest['future_head_parameter_names'])==130 and len(set(manifest['future_head_parameter_names']))==130
        and manifest['readout_parameter_names']==list(READOUT_NAMES),'Declared parameter scope differs')
    for key in ('initial_future_head_sha256','initial_readout_sha256'):
        require(valid_sha(manifest[key]) and loaded[key]==summary[key]==manifest[key],'Initial state declaration differs')
    d=loaded['D']
    require(loaded['O_full_state_sha256']==manifest['initial_O_state_sha256'] and not loaded['optimizer_restored']
        and d['actual_motion_state_sha256']==manifest['frozen_D_motion_state_sha256']
        and d['checkpoint_sha256']==p['D_checkpoint_sha256'] and d['top_complete_sha256']==p['connected_complete_sha256']
        and d['protocol_sha256']==p['connected_protocol_sha256'] and d['fixed_final_update']==512
        and d['fixed_examples']==2048 and d['fixed_development_samples']==200 and d['actual_optimizer_parameter_steps_all512']
        and not d['optimizer_restored'] and d['optimizer_updates_in_this_evaluation']==0,'Original O/frozen D load evidence differs')
    require(summary['schema']==SCHEMA and summary['status']==STATUS and summary['updates']==512 and summary['examples']==2048
        and summary['evaluated_samples']==200 and summary['fresh_initialization'] and not summary['optimizer_restored']
        and summary['frozen_D_and_non_head_O_unchanged'] and summary['t0_predictions_may_differ'],'Incomplete final summary')
    orders=helper.planned_orders();order_sha=helper.json_hash(orders)
    require(manifest['sample_orders_sha256']==order_sha,'Wrong seed11 four-pass order')
    logs={};arm_manifests={};arm_completes={};seen_inputs={}
    identity_keys=('sample_token','scene_token','split','official_index')
    for arm in ARMS:
        folder=root/'runs'/arm;require(not (folder/'failed.json').exists(),'Failed arm')
        ac=read(folder/'complete.json');am=read(folder/'manifest.json')
        require(sha(folder/'complete.json')==done['arm_complete_sha256'][arm] and ac['schema']==SCHEMA and ac['status']==STATUS
            and ac['arm']==arm and ac['updates']==512 and ac['examples']==2048 and ac['evaluated_samples']==200,'Arm completion differs')
        file_chain(folder,ac,('manifest.json','training.jsonl','final.pth','development_records.jsonl'),
            skip=() if verify_checkpoint_bytes else ('final.pth',))
        expected=dict(schema=SCHEMA,arm=arm,common_manifest_sha256=sha(root/'manifest.json'),sources=sources,
            detach_physical_features=arm=='B',initial_future_head_sha256=manifest['initial_future_head_sha256'],
            initial_readout_sha256=manifest['initial_readout_sha256'],future_head_parameters=13274016,readout_parameters=745392)
        require(am==expected,'Per-arm model/initialization differs')
        final=done['final_checkpoints'][arm]
        require(final==dict(file='runs/'+arm+'/final.pth',sha256=ac['files_sha256']['final.pth'],
            future_head_state_sha256=ac['final_future_head_sha256'],readout_state_sha256=ac['final_readout_sha256'])
            and summary['final_future_head_sha256'][arm]==final['future_head_state_sha256']
            and summary['final_readout_sha256'][arm]==final['readout_state_sha256'],'Fixed final pointer differs')
        require(summary['actual_optimizer_steps'][arm]==dict(head=dict(parameters_with_state=130,all_steps512=True),
            readout=dict(parameters_with_state=10,all_steps512=True)),'Declared actual optimizer endpoint differs')
        records=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
        require(len(records)==512,'Incomplete actual update log');finite_tree(records)
        for u,row in enumerate(records):
            indices=orders[u//128][4*(u%128):4*(u%128+1)]
            require(row['update']==u+1 and row['examples']==4*(u+1) and row['pass_index']==u//128
                and [r['ordinal'] for r in row['samples']]==indices and row['future_head_lr']>0 and row['readout_lr']>0,
                'Actual budget/order differs')
            if verify_checkpoint_bytes:
                require(row['future_head_lr']==trainer.pre.head_lr(u) and row['readout_lr']==helper.schedule_lr(u),'Actual runtime LR schedule differs')
            grad=row['gradients']
            for component,names,limit in (('head',manifest['future_head_parameter_names'],35.),('readout',list(READOUT_NAMES),10.)):
                require(set(grad[component]['parameter_norms'])==set(names) and grad[component]['norm']>=0
                    and grad[component+'_preclip_norm']>=0 and grad[component+'_clip_factor']==
                    min(1.,limit/(grad[component+'_preclip_norm']+1e-6)),'Gradient/clip scope differs')
                require(all(v is None or v>=0 for v in grad[component]['parameter_norms'].values()),'Invalid gradient norm')
            for item,i in zip(row['samples'],indices):
                require(all(item[k]==train[i][k] for k in identity_keys) and item['native_logits_unmodified_by_readout'],'Logged training identity/route differs')
                fields=('inputs_sha256','targets_sha256','sparse_label_sha256','input_tree_sha256')
                require(all(valid_sha(item[k]) for k in fields),'Invalid training input identity hash')
                identity=tuple(item[k] for k in fields)
                if i in seen_inputs:require(seen_inputs[i]==identity,'Input/GT changed across arms or passes')
                seen_inputs[i]=identity
        logs[arm]=records;arm_manifests[arm]=am;arm_completes[arm]=ac
    require(len(seen_inputs)==512,'Incomplete four-pass source exposure')
    for k,b in zip(logs['K'],logs['B']):
        require(k['future_head_lr']==b['future_head_lr'] and k['readout_lr']==b['readout_lr'],'Matched arm LR differs')
        for left,right in zip(k['samples'],b['samples']):
            require(all(left[key]==right[key] for key in ('ordinal',*identity_keys,'inputs_sha256','targets_sha256',
                'sparse_label_sha256','input_tree_sha256','paired_rng_before_sha256','paired_rng_after_sha256')),'Paired input or RNG differs')
    diagnostics=[json.loads(x) for x in (root/'physical_gradient_diagnostics.jsonl').read_text().splitlines()]
    require(diagnostics==summary['physical_gradient_diagnostics'] and len(diagnostics)==4,'Physical diagnostic receipt differs')
    for row,(i,arm) in zip(diagnostics,((4,'K'),(4,'B'),(2044,'K'),(2044,'B'))):
        require(row['arm']==arm and row['microstep_index']==i and row['after_completed_updates']==i//4
            and row['ordinal']==orders[i//512][i%512] and row['sample_token']==train[row['ordinal']]['sample_token']
            and row['diagnostic_only'] and not row['used_for_training_decisions'],'Diagnostic selection/scope differs')
    rows=[json.loads(x) for x in (root/'development_records.jsonl').read_text().splitlines()]
    require(len(rows)==200,'Incomplete final development records')
    armdev={arm:[json.loads(x) for x in (root/'runs'/arm/'development_records.jsonl').read_text().splitlines()] for arm in ARMS}
    require(all(len(v)==200 for v in armdev.values()),'Incomplete per-arm final development')
    for i,row in enumerate(rows):
        require(row['ordinal']==i and all(row[k]==dev[i][k] for k in identity_keys) and
            set(row['hist_by_arm'])==set(row['metrics_by_arm'])=={'O','K','B'} and
            all(row[k] for k in ('O_reference_hist_exact','native_CPU_hist_exact','GT_common_denominators_exact',
                                'GT_raw_sparse_read_after_predictions')) and row['t0_prediction_policy']=='model_specific_native'
            and row['t0_equality_gate_applied'] is False,'Wrong development identity or t0 policy')
        require(row['sample_token']==original[i]['sample_token'] and row['scene_token']==original[i]['scene_token']
            and row['hist_by_arm']['O']==original[i]['hist_by_horizon'],'Original O reference not reproduced')
        trainer.verify_common_denominators(row['metrics_by_arm'])
        for name in ('O','K','B'):
            require([h['occupancy']['confusion'] for h in row['metrics_by_arm'][name]['horizons']]==row['hist_by_arm'][name],
                'Recorded native and CPU histograms differ')
        for arm in ARMS:
            other=armdev[arm][i]
            require(other['ordinal']==i and all(other[k]==dev[i][k] for k in identity_keys)
                and other['hist_by_horizon']==row['hist_by_arm'][arm] and other['horizon_seconds']==[0,.5,1,1.5,2],
                'Per-arm final evaluation differs')
    require(sha(root/'complete.json')==complete_sha,'Completion changed during validation')
    return dict(root=root,protocol_path=protocol_path,protocol=p,manifest=manifest,summary=summary,complete=done,
        loaded_models=loaded,sources=sources,arm_manifests=arm_manifests,arm_completes=arm_completes,
        orders=orders,order_sha256=order_sha,top_complete_sha256=complete_sha,
        checkpoint_bytes_verified=verify_checkpoint_bytes,actual_checkpoint_tensors_verified=False,
        cached_input_tensor_bytes_reread=False)


def authenticated_training(training_run,protocol_path,*,selection_path=None,o_reference_path=None):
    """No torch import or pth read; completion, hashes, logs and identities only."""
    return metadata(training_run,protocol_path,selection_path=selection_path,o_reference_path=o_reference_path)


def validate_state(module,state,torch):
    expected=module.state_dict();require(set(state)==set(expected),'Final state keys differ')
    for name,reference in expected.items():
        value=state[name]
        require(isinstance(value,torch.Tensor) and value.shape==reference.shape and value.dtype==reference.dtype
            and bool(torch.isfinite(value).all()),'Invalid state tensor: '+name)
    module.load_state_dict(state,strict=True)


def validate_optimizer(optimizer,module,expected_lr,expected_count,torch):
    groups=optimizer['param_groups'];states=optimizer['state']
    require(len(groups)==1,'Unexpected AdamW group count');group=groups[0]
    parameters=list(module.parameters())
    require(len(parameters)==expected_count and group['params']==list(range(expected_count))
        and set(states)==set(group['params']),'Missing/misordered optimizer parameter state')
    require(group['lr']==expected_lr and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
        and group['weight_decay']==.01 and group['amsgrad'] is False and not group.get('maximize',False)
        and not group.get('capturable',False) and not group.get('differentiable',False),'Final AdamW policy differs')
    for i,param in enumerate(parameters):
        state=states[i]
        require(set(state)=={'step','exp_avg','exp_avg_sq'} and isinstance(state['step'],torch.Tensor)
            and state['step'].numel()==1 and bool(torch.isfinite(state['step']).all()) and float(state['step'])==512.,
            'Actual AdamW step differs')
        for key in ('exp_avg','exp_avg_sq'):
            require(isinstance(state[key],torch.Tensor) and state[key].shape==param.shape and state[key].dtype==param.dtype
                and bool(torch.isfinite(state[key]).all()),'Invalid actual AdamW moment')
        require(bool((state['exp_avg_sq']>=0).all()),'Negative AdamW second moment')


def load_completed_arm(training_run,arm,protocol_path,helper,native_O_model,device='cuda:0',*,
                       selection_path=None,o_reference_path=None):
    """Return (copied frozen native candidate, frozen physical readout, receipt).

    native_O_model must already be constructed by the bound native O builder.
    Its original state, mode and gradients are verified and left untouched.
    """
    require(arm in ARMS,'Expected K or B')
    meta=metadata(training_run,protocol_path,helper,verify_checkpoint_bytes=True,
        selection_path=selection_path,o_reference_path=o_reference_path)
    import torch
    p=meta['protocol'];manifest=meta['manifest'];root=meta['root'];folder=root/'runs'/arm
    trainer=import_source('future_state_motion_train_v1',TRAINER_SHA)
    common=import_source('common_change_evaluation_v2',p['sources_sha256']['common_change_evaluation_v2.py'])
    adapter=import_source('objective_supervision_adapters',p['sources_sha256']['objective_supervision_adapters.py'])
    core=import_source('future_state_motion_v1',CORE_SHA)
    require(not any(m.training for m in native_O_model.modules()) and
        all(not v.requires_grad and v.grad is None for v in native_O_model.parameters()),'O reference must be frozen eval without gradients')
    require(sha(inspect.getsourcefile(native_O_model.compute_occ_loss))==adapter.DETECTOR_SOURCE_SHA and
        sha(inspect.getsourcefile(native_O_model.future_pred_head.loss_occ))==adapter.HEAD_SOURCE_SHA,'Native model source differs')
    require(helper.state_digest(native_O_model)==manifest['initial_O_state_sha256'] and
        helper.state_digest(native_O_model.future_pred_head)==manifest['initial_future_head_sha256'] and
        common.historical_O_head_digest(native_O_model.future_pred_head)==common.O_HEAD_SHA and
        trainer.pre.non_head_digest(native_O_model)==manifest['frozen_non_head_state_sha256'],'Caller O template differs from original initialization')
    checkpoint=folder/'final.pth';before=checkpoint.stat();payload=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    require(payload['schema']==SCHEMA and payload['status']=='FIXED_FINAL_512' and payload['arm']==arm
        and payload['update']==512 and payload['examples']==2048 and not payload['resume_supported'],'Wrong actual final payload')
    require(payload['sources']==meta['sources'] and payload['manifest_sha256']==sha(folder/'manifest.json')
        and payload['common_manifest_sha256']==sha(root/'manifest.json') and payload['sample_orders']==meta['orders']
        and payload['sample_orders_sha256']==meta['order_sha256'],'Actual payload provenance/order differs')
    for key in ('initial_future_head_sha256','initial_readout_sha256'):require(payload[key]==manifest[key],'Actual initial state binding differs')
    candidate=copy.deepcopy(native_O_model)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(11);readout=core.FutureStateMotionReadout()
    require(helper.state_digest(readout)==manifest['initial_readout_sha256'],'Fresh seed11 physical readout differs')
    require(sum(v.numel() for v in candidate.future_pred_head.parameters())==13274016
        and [n for n,_ in candidate.future_pred_head.named_parameters()]==manifest['future_head_parameter_names']
        and sum(v.numel() for v in readout.parameters())==745392
        and [n for n,_ in readout.named_parameters()]==list(READOUT_NAMES),'Actual parameter scope differs')
    validate_state(candidate.future_pred_head,payload['future_pred_head'],torch)
    validate_state(readout,payload['motion_readout'],torch)
    head_sha=helper.state_digest(candidate.future_pred_head);readout_sha=helper.state_digest(readout)
    final=meta['complete']['final_checkpoints'][arm]
    require(head_sha==payload['final_future_head_sha256']==final['future_head_state_sha256'] and
        readout_sha==payload['final_readout_sha256']==final['readout_state_sha256'],'Actual final parameter bytes differ')
    require(set(payload['optimizers'])=={'head','readout'} and
        payload['actual_optimizer_steps']==meta['summary']['actual_optimizer_steps'][arm],'Optimizer component/receipt differs')
    validate_optimizer(payload['optimizers']['head'],candidate.future_pred_head,trainer.pre.head_lr(511),130,torch)
    validate_optimizer(payload['optimizers']['readout'],readout,helper.schedule_lr(511),10,torch)
    after=checkpoint.stat()
    require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
        and sha(checkpoint)==final['sha256'] and sha(root/'complete.json')==meta['top_complete_sha256'], 'Final artifact changed during load')
    del payload
    for network in (candidate,readout):
        for param in network.parameters():param.requires_grad_(False)
        network.to(device).eval()
    require(trainer.pre.non_head_digest(candidate)==manifest['frozen_non_head_state_sha256'] and
        helper.state_digest(native_O_model)==manifest['initial_O_state_sha256'],'Candidate observer or caller O changed')
    receipt=dict(schema='future-state-motion-final-load-v1',arm=arm,checkpoint_sha256=final['sha256'],
        actual_future_head_state_sha256=head_sha,actual_readout_state_sha256=readout_sha,
        actual_candidate_full_state_sha256=helper.state_digest(candidate),
        initial_O_state_sha256=manifest['initial_O_state_sha256'],initial_future_head_sha256=manifest['initial_future_head_sha256'],
        initial_readout_sha256=manifest['initial_readout_sha256'],frozen_non_head_state_sha256=manifest['frozen_non_head_state_sha256'],
        frozen_D_motion_state_sha256=manifest['frozen_D_motion_state_sha256'],D_loaded_by_this_function=False,
        source_sha256=sha(__file__),sources_sha256=p['sources_sha256'],analysis_sources_sha256=p['analysis_sources_sha256'],
        selection_sha256=p['selection_sha256'],protocol_sha256=PROTOCOL_SHA,top_complete_sha256=meta['top_complete_sha256'],
        arm_complete_sha256=meta['complete']['arm_complete_sha256'][arm],manifest_sha256=sha(folder/'manifest.json'),
        common_manifest_sha256=sha(root/'manifest.json'),sample_orders_sha256=meta['order_sha256'],
        fixed_final_update=512,fixed_examples=2048,fixed_development_samples=200,
        future_head_parameters=13274016,readout_parameters=745392,
        actual_optimizer_parameter_steps_all512=dict(head=True,readout=True),optimizer_restored=False,
        optimizer_updates_in_this_evaluation=0,caller_O_template_unchanged=True)
    return candidate,readout,receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description='CPU metadata authentication only; never loads checkpoint tensors.')
    parser.add_argument('--training-run',required=True);parser.add_argument('--protocol',required=True)
    parser.add_argument('--selection');parser.add_argument('--o-reference')
    a=parser.parse_args(argv)
    result=authenticated_training(a.training_run,a.protocol,selection_path=a.selection,o_reference_path=a.o_reference)
    print(json.dumps(dict(status='PASS_COMPLETED_METADATA_ONLY',top_complete_sha256=result['top_complete_sha256'],
        actual_checkpoint_tensors_verified=False,checkpoint_bytes_verified=False),allow_nan=False))


if __name__=='__main__':main()
