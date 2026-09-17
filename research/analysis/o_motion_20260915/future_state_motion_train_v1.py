"""Fresh fixed-final K/B training, using the verified future-state preflight.

Only the physical readout's feature detach differs between arms. Both native
future heads train on the original O occupancy objective. No transport/refiner
is applied to occupancy logits. The native t0 output is allowed to change.
There is no resume, best-checkpoint selection, or intermediate development run.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

PREFLIGHT_SOURCE_SHA = '5397cb38b258f1ef2ef27b6e1bf47ef4b4085fa8b43a52e10d5c7b7fc34e91ab'
PREFLIGHT_PROTOCOL_SHA = 'd03a261909cfe3bcdd36da4b0a622580a610647e1f06f0d643514514da822234'
PREFLIGHT_COMPLETE_SHA = '8d2e2dd9094ffb897eccd60febc95d2e06b30f93e7b8aaad74dc4657bde20dbd'
SCHEMA = 'future-state-motion-training-v1'
ARMS = ('K','B')
RAW_MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
RAW_COMPLETE_SHA = 'e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69'
O_REFERENCE_SHA = '3f426891f6811a595542a1e9694783521ae89cf18d6385896012e5f52a8e9b9a'


def _preflight_module():
    path=Path(__file__).with_name('future_state_motion_preflight_v1.py')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PREFLIGHT_SOURCE_SHA:
        raise ValueError('Frozen preflight helper changed')
    name='future_state_motion_preflight_v1'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


pre=_preflight_module()
require,sha,read,write,append=pre.require,pre.sha,pre.read,pre.write,pre.append
TRAINING=copy.deepcopy(pre.RECIPE)
TRAINING.update(updates=512,examples=2048,passes=4,formal_authorized=True,weights_persisted=True,
    development_evaluation=True,development_samples=200,development_scenes=100,
    sample_order='all four continuous RandomState11 permutations of 512',
    checkpoint='fixed final512 only; no resume, best, or intermediate development',
    diagnostic_physical_VJP_microsteps=[4,2044],diagnostic_VJP_controls_training=False)


def protocol_template():
    return dict(schema=SCHEMA,status='REVIEW_REQUIRED',training=copy.deepcopy(TRAINING),
        numerical_policy=dict(pre.NUMERICAL),sources_sha256={},
        preflight_protocol_sha256=PREFLIGHT_PROTOCOL_SHA,
        engineering_preflight_complete_sha256=PREFLIGHT_COMPLETE_SHA,
        connected_protocol_sha256=pre.CONNECTED_PROTOCOL_SHA,
        connected_complete_sha256=pre.CONNECTED_COMPLETE_SHA,D_checkpoint_sha256=pre.D_CHECKPOINT_SHA,
        runtime_source_contract_sha256=pre.RUNTIME_SHA,
        cache_index_sha256=dict(train=None,development=None),labels=dict(manifest_sha256=None,complete_sha256=None),
        raw_labels=dict(manifest_sha256=RAW_MANIFEST_SHA,complete_sha256=RAW_COMPLETE_SHA),
        o_development_records_sha256=O_REFERENCE_SHA,resources=None,
        analysis_sources_sha256={},selection_sha256=None)


def append_many(path, rows):
    """Same ordered JSONL records, one durable flush per anchor/arm."""
    with Path(path).open('a') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())


def validate_engineering(root,protocol,preflight_protocol,initial_head=None,initial_readout=None):
    """Small authenticated engineering artifacts only; never load probe weights."""
    root=Path(root);done=read(root/'complete.json')
    require(not (root/'failed.json').exists() and sha(root/'complete.json')==
        protocol['engineering_preflight_complete_sha256']==PREFLIGHT_COMPLETE_SHA,'Wrong actual preflight endpoint')
    require(done['schema']==pre.SCHEMA and done['status']=='PASS_FUTURE_STATE_MOTION_PREFLIGHT' and
        done['updates']==4 and done['examples']==16 and done['evaluated_samples']==0 and
        done['weights_persisted'] is False and done['final_checkpoints']=={},'Preflight not a successful discarded four-update run')
    require(set(done['files_sha256'])=={'manifest.json','loaded_models.json','summary.json'},'Preflight file set differs')
    for name,digest in done['files_sha256'].items():require(sha(root/name)==digest,'Preflight file changed: '+name)
    manifest=read(root/'manifest.json');summary=read(root/'summary.json')
    require(manifest['recipe']==pre.RECIPE and manifest['numerical_policy']==pre.NUMERICAL and
        manifest['sources']['protocol_sha256']==PREFLIGHT_PROTOCOL_SHA and
        manifest['sources']['sources_sha256']==preflight_protocol['sources_sha256'],'Preflight recipe/source differs')
    require(summary['initial_O_D_parity_pass'] and summary['all_gradients_finite'] and summary['zero_readout_changed']
        and summary['frozen_D_and_non_head_O_unchanged'] and not summary['weights_persisted'], 'Preflight engineering boundary not passed')
    parity=summary['initial_O_D_parity']
    require(len(parity)==4 and all(all(r[k] for k in ('initial_all_5h_3layers_O_logits_exact',
        'initial_D_displacement_exact','initial_native_hist_exact','physical_readout_did_not_modify_native_logits')) for r in parity),
        'Missing initial O/D numerical proof')
    connection=summary['physical_connectivity_after_first_update']
    kg=connection['K']['physical_future_head_gradient'];bg=connection['B']['physical_future_head_gradient']
    require(kg['norm']>0 and any(v is not None and v>0 for n,v in kg['parameter_norms'].items() if n.startswith('transformer.'))
        and bg['none_parameters']==len(bg['parameter_norms']),'Actual preflight did not establish K/B physical connection boundary')
    require(len(summary['paired_rng'])==16 and all(r['paired_exact'] for r in summary['paired_rng']), 'Incomplete paired RNG proof')
    for arm in ARMS:
        folder=root/'runs'/arm;ac=read(folder/'complete.json')
        require(sha(folder/'complete.json')==done['arm_complete_sha256'][arm] and ac['updates']==4 and
            ac['examples']==16 and not ac['weights_persisted'],'Preflight arm endpoint differs')
        require(set(ac['files_sha256'])=={'manifest.json','training.jsonl'}, 'Unexpected preflight artifact')
        for name,digest in ac['files_sha256'].items():require(sha(folder/name)==digest,'Preflight arm file changed')
    if initial_head is not None:require(initial_head==manifest['initial_future_head_sha256'],'Fresh O head differs from preflight')
    if initial_readout is not None:require(initial_readout==manifest['initial_readout_sha256'],'Fresh seed11 readout differs from preflight')
    return dict(complete_sha256=sha(root/'complete.json'),manifest_sha256=sha(root/'manifest.json'),
        summary_sha256=sha(root/'summary.json'),weights_loaded=False,actual_preflight_passed=True)


def load_contract(a):
    p=read(a.protocol)
    require(p['schema']==SCHEMA and p['status']=='FROZEN' and p['training']==TRAINING and
        p['numerical_policy']==pre.NUMERICAL,'Exact frozen formal K/B recipe required')
    require(math.isfinite(a.max_seconds) and 0<a.max_seconds<=42600 and math.isfinite(a.max_allocated_gib)
        and 0<a.max_allocated_gib<=32 and p['resources']==dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib),
        'Finite formally frozen resource budget required')
    require(sha(a.preflight_protocol)==p['preflight_protocol_sha256']==PREFLIGHT_PROTOCOL_SHA,'Frozen preflight protocol differs')
    args=SimpleNamespace(**vars(a));args.protocol=a.preflight_protocol;args.max_seconds=600.;args.max_allocated_gib=32.
    pp,contract,mods=pre.load_contract(args)
    expected=dict(pp['sources_sha256']);expected[Path(__file__).name]=sha(__file__)
    require(p['sources_sha256']==expected,'Exact inherited plus formal trainer source set required')
    for name,digest in expected.items():require(sha(contract.source_path(name))==digest,'Formal source changed: '+name)
    for key in ('connected_protocol_sha256','connected_complete_sha256','D_checkpoint_sha256',
                'runtime_source_contract_sha256','cache_index_sha256','labels'):
        require(p[key]==pp[key],'Inherited data/source changed: '+key)
    require(p['raw_labels']==dict(manifest_sha256=RAW_MANIFEST_SHA,complete_sha256=RAW_COMPLETE_SHA)
        and p['o_development_records_sha256']==O_REFERENCE_SHA,'Development data definition changed')
    def valid_digest(value):return isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef')
    require(set(p['analysis_sources_sha256'])=={'summarize_future_state_common_v1.py','summarize_future_state_physical_v1.py'}
        and all(valid_digest(v) for v in p['analysis_sources_sha256'].values()) and valid_digest(p['selection_sha256']),
        'Predeclared separate analysis/selection bindings required')
    validate_engineering(a.engineering_preflight,p,pp)
    return p,pp,contract,mods


def gt_only(value):
    if isinstance(value,dict):return {k:gt_only(v) for k,v in value.items() if k not in ('TP','FN','recall')}
    if isinstance(value,list):return [gt_only(v) for v in value]
    return value


def verify_common_denominators(metrics):
    """Prediction-specific t0 values are deliberately NOT an equality gate."""
    reference=metrics['O']
    for name in ARMS:
        item=metrics[name]
        for key in ('identity','shape_hxyz','extent_xyz_m'):require(item[key]==reference[key], 'Common geometry/identity differs')
        for key in ('gt_valid_mask_sha256','valid_voxels'):
            require(item['t0_boundary'][key]==reference['t0_boundary'][key], 'Common t0 GT domain differs')
        for left,right in zip(item['horizons'],reference['horizons']):
            require(np.array_equal(np.asarray(left['occupancy']['confusion']).sum(1),
                np.asarray(right['occupancy']['confusion']).sum(1)) and left['ignored_voxels']==right['ignored_voxels'],
                'Common full-GT class counts differ')
            if left['horizon_index']:
                require(gt_only(left['motion_positive_attribution'])==gt_only(right['motion_positive_attribution']),
                    'Common motion-group GT support differs')
        for left,right in zip(item['transitions'],reference['transitions']):
            require(left['domain']==right['domain'] and np.array_equal(np.asarray(left['confusion']).sum(1),
                np.asarray(right['confusion']).sum(1)), 'Common transition GT domain differs')


def run(a,out,started,stop):
    p,pp,contract,mods=load_contract(a)
    helper=mods['train_source_motion_v1'];native=mods['native_state_cache'];oracle=mods['oracle_transport_probe']
    common=mods['common_change_evaluation_v2'];losshelper=mods['train_supported_fusion_v2']
    adapter=mods['objective_supervision_adapters'];connected=mods['train_connected_motion_v2'];tree=mods['joint_native_evaluation']
    train_root,train,train_receipt=helper.cache_index(a.train_cache,'train',p)
    dev_root,dev,dev_receipt=helper.cache_index(a.dev_cache,'development',p)
    require(len({r['scene_token'] for r in dev})==100 and not ({r['scene_token'] for r in train}&{r['scene_token'] for r in dev}),
        'Fixed development scenes or train/dev isolation changed')
    label_root,labels=helper.labels_manifest(a.sparse_labels,p)
    for record in train+dev:helper.check_identity(record,labels[record['sample_token']])
    require(sha(a.o_development_records)==O_REFERENCE_SHA,'Original O development record changed')
    reference=[json.loads(x) for x in Path(a.o_development_records).read_text().splitlines()]
    require(len(reference)==200 and all(all(r[k]==v[k] for k in ('sample_token','scene_token')) for r,v in zip(reference,dev)),
        'Original O evaluation identity/order differs')
    raw_root=Path(a.raw_labels)
    require(sha(raw_root/'manifest.json')==RAW_MANIFEST_SHA and sha(raw_root/'complete.json')==RAW_COMPLETE_SHA,'Raw label manifest changed')
    raw_desc={r['identity']['sample_token']:r for r in read(raw_root/'manifest.json')['records']}
    for record in dev:require(raw_desc[record['sample_token']]['identity']==common.identity(record),'Raw/cache identity differs')
    import torch
    require(str(a.device).startswith('cuda') and torch.cuda.is_available(),'Native training requires CUDA')
    torch.cuda.set_device(a.device);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(min(1.,a.max_allocated_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory),a.device)
    torch.cuda.reset_peak_memory_stats(a.device)
    def budget():
        require(not stop,'Signal received; no automatic resume/retry')
        require(time.monotonic()-started<a.max_seconds,'Formal wall-clock ceiling')
        require(torch.cuda.max_memory_allocated(a.device)<=a.max_allocated_gib*2**30,'CUDA memory ceiling')
    budget();write(out/'progress.json',dict(schema=SCHEMA,phase='initialization',updates=0,examples=0))
    model=native.build_native_model(a.config,a.checkpoint,device=a.device,repo=a.repo)
    payload=torch.load(a.o_checkpoint,map_location='cpu',weights_only=False)
    model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True);del payload
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval()
    require(common.historical_O_head_digest(model.future_pred_head)==common.O_HEAD_SHA,'Initial historical O head differs')
    O_sha=helper.state_digest(model);head_initial=helper.state_digest(model.future_pred_head)
    require(O_sha==read(Path(a.training_run)/'manifest.json')['frozen_O_state_sha256'],'Initial whole O differs')
    initial_head_state={k:v.detach().cpu().clone() for k,v in model.future_pred_head.state_dict().items()}
    frozen_sha=pre.non_head_digest(model)
    D,unused_gate,D_receipt=connected.load_completed_arm(Path(a.training_run)/'runs'/'D',Path(a.connected_protocol),helper,a.device)
    del unused_gate;D_sha=helper.state_digest(D)
    require(D_receipt['checkpoint_sha256']==pre.D_CHECKPOINT_SHA and D_receipt['top_complete_sha256']==pre.CONNECTED_COMPLETE_SHA,'Actual D differs')
    require(sha(inspect.getsourcefile(model.compute_occ_loss))==adapter.DETECTOR_SOURCE_SHA and
        sha(inspect.getsourcefile(model.future_pred_head.loss_occ))==adapter.HEAD_SOURCE_SHA and
        sha(inspect.getsourcefile(model.future_pred_head.loss_voxel))==adapter.HEAD_SOURCE_SHA and
        model.future_pred_head.class_weights.tolist()==[1.,5.],'Native occupancy loss changed')
    core=contract.import_bound('future_state_motion_v1',p['sources_sha256'])
    metric=contract.import_bound('common_occupancy_change_metrics_v1',p['sources_sha256'])
    models={'K':model,'B':copy.deepcopy(model)};pre.seed_all(torch)
    readouts={'K':core.FutureStateMotionReadout().to(a.device)};readouts['B']=copy.deepcopy(readouts['K'])
    readout_initial=helper.state_digest(readouts['K'])
    engineering=validate_engineering(a.engineering_preflight,p,pp,head_initial,readout_initial)
    head_params={};readout_params={}
    for arm in ARMS:
        for param in models[arm].future_pred_head.parameters():param.requires_grad_(True)
        head_params[arm]=list(models[arm].future_pred_head.named_parameters());readout_params[arm]=list(readouts[arm].named_parameters())
        require(sum(v.numel() for _,v in head_params[arm])==13274016 and sum(v.numel() for _,v in readout_params[arm])==745392,'Parameter count differs')
        require(helper.state_digest(models[arm])==O_sha and helper.state_digest(readouts[arm])==readout_initial,'Fresh arm initialization differs')
        require(all(not v.requires_grad for n,v in models[arm].named_parameters() if not n.startswith('future_pred_head.')),'Frozen scope differs')
    require(not ({id(v) for v in models['K'].parameters()}&{id(v) for v in models['B'].parameters()}),'Shared native parameters')
    require(not ({id(v) for v in readouts['K'].parameters()}&{id(v) for v in readouts['B'].parameters()}),'Shared readout parameters')
    opts={arm:dict(head=torch.optim.AdamW([v for _,v in head_params[arm]],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,amsgrad=False),
        readout=torch.optim.AdamW(readouts[arm].parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=.01,amsgrad=False)) for arm in ARMS}
    orders=helper.planned_orders()
    sources=dict(protocol_sha256=sha(a.protocol),sources_sha256=p['sources_sha256'],preflight_protocol_sha256=PREFLIGHT_PROTOCOL_SHA,
        engineering_preflight_complete_sha256=PREFLIGHT_COMPLETE_SHA,connected_protocol_sha256=pre.CONNECTED_PROTOCOL_SHA,
        connected_complete_sha256=pre.CONNECTED_COMPLETE_SHA,D_checkpoint_sha256=pre.D_CHECKPOINT_SHA,
        O_checkpoint_sha256=oracle.O_SHA,M0_checkpoint_sha256=oracle.M0_SHA,config_sha256=oracle.CONFIG_SHA,
        runtime_source_contract_sha256=pre.RUNTIME_SHA,train_cache=train_receipt,development_cache=dev_receipt,
        sparse_labels=p['labels'],raw_labels=p['raw_labels'],o_development_records_sha256=O_REFERENCE_SHA,
        analysis_sources_sha256=p['analysis_sources_sha256'],selection_sha256=p['selection_sha256'])
    manifest=dict(schema=SCHEMA,mode='train',sources=sources,training=TRAINING,numerical_policy=pre.NUMERICAL,
        initial_O_state_sha256=O_sha,initial_future_head_sha256=head_initial,initial_readout_sha256=readout_initial,
        frozen_non_head_state_sha256=frozen_sha,frozen_D_motion_state_sha256=D_sha,sample_orders_sha256=helper.json_hash(orders),
        future_head_parameter_names=[n for n,_ in head_params['K']],readout_parameter_names=[n for n,_ in readout_params['K']],
        engineering_preflight=engineering,fresh_initialization=True,optimizer_restored=False,
        native_logits_never_composed=True,t0_prediction_may_change=True,physical_reference_is_D_not_native_O_flow=True,
        labels_only_for_loss_or_evaluation=True,no_source_position_feature_selection=True,no_initial_development_evaluation=True)
    write(out/'manifest.json',manifest);write(out/'loaded_models.json',dict(O_full_state_sha256=O_sha,D=D_receipt,
        initial_future_head_sha256=head_initial,initial_readout_sha256=readout_initial,optimizer_restored=False))
    folders={arm:out/'runs'/arm for arm in ARMS}
    for arm,folder in folders.items():
        folder.mkdir(parents=True)
        write(folder/'manifest.json',dict(schema=SCHEMA,arm=arm,common_manifest_sha256=sha(out/'manifest.json'),
            sources=sources,detach_physical_features=arm=='B',initial_future_head_sha256=head_initial,
            initial_readout_sha256=readout_initial,future_head_parameters=13274016,readout_parameters=745392))
    pre.seed_all(torch)
    for arm in ARMS:
        models[arm].train();readouts[arm].train()
        for opt in opts[arm].values():opt.zero_grad(set_to_none=True)
    initialization_seconds=time.monotonic()-started;updates=examples=evaluated=0
    had={arm:{n:False for n,_ in readout_params[arm]} for arm in ARMS}
    micro_times=[];update_times=[];connections=[];rows=[];objects=[]
    for pass_index,order in enumerate(orders):
        for ordinal in order:
            budget();torch.cuda.synchronize(a.device);mtick=time.monotonic();micro=examples%4
            if micro==0:
                utick=mtick;batches={arm:[] for arm in ARMS}
                for arm in ARMS:
                    for group in opts[arm]['head'].param_groups:group['lr']=pre.head_lr(updates)
                    for group in opts[arm]['readout'].param_groups:group['lr']=helper.schedule_lr(updates)
            record=train[ordinal];sample=common.input_only(native,train_root,record,a.device);boundary=tree.tree_digest(sample['inputs'])
            with torch.no_grad():base=D(sample['inputs']['prev_bev_input'][:,-1])
            require(not base.requires_grad and torch.isfinite(base).all(),'Frozen D field invalid')
            paired=pre.rng_state(torch);before_rng=pre.rng_digest(paired);ends={};initial_values=None;initial_comparison=None
            for arm in ARMS:
                pre.restore_rng(torch,paired);budget();prediction,features=core.capture_native_terminal(models[arm],sample,native,training=True)
                native_digest=tree.tree_digest(prediction)
                flow=readouts[arm](features,base,detach_features=arm=='B')
                require(features.requires_grad and torch.isfinite(flow).all() and torch.isfinite(prediction).all(),'Nonfinite forward')
                require(tree.tree_digest(prediction)==native_digest,'Readout modified native occupancy tensor')
                if 'targets' not in sample:common.load_targets_after_predictions(native,train_root,record,sample,a.device)
                descriptor=labels[record['sample_token']];label=helper.load_sparse(label_root,descriptor,record)
                occupancy,occ_audit=losshelper.loss_terms(models[arm],prediction,sample['targets'],adapter)
                occ_audit.update(inherited_constant_fixed_t0_field_names_only=True,intermediate_losses_trainable=True,t0_readout_trainable=True)
                sparse=helper.gather_sparse(flow,label);physical,physical_audit=helper.object_group_loss(sparse,label);total=occupancy+physical
                require(torch.isfinite(total) and torch.isfinite(physical),'Nonfinite objective')
                if updates==0:
                    require(losshelper.exact32(flow,base),'Fresh zero readout must equal D')
                    if arm=='K':initial_values=(native_digest,occ_audit,float(physical.detach()))
                    else:
                        require(native_digest==initial_values[0],'Initial paired native prediction differs')
                        initial_comparison=pre.compare_initial_occupancy(initial_values[1],occ_audit)
                        initial_comparison['physical_scalar_descriptive']=dict(K=initial_values[2],B=float(physical.detach()),
                            B_minus_K=float(physical.detach())-initial_values[2],same_D_fields_exact=True,same_sparse_source=True)
                if examples in (4,2044):
                    vjp=torch.autograd.grad(physical,[v for _,v in head_params[arm]],allow_unused=True,retain_graph=True)
                    stats=pre.gradient_stats(head_params[arm],vjp);del vjp
                    event=dict(arm=arm,microstep_index=examples,after_completed_updates=updates,ordinal=ordinal,
                        sample_token=record['sample_token'],empty_target=physical_audit['empty_target'],
                        physical_future_head_gradient=stats,diagnostic_only=True,used_for_training_decisions=False)
                    connections.append(event);append(out/'physical_gradient_diagnostics.jsonl',event)
                (total/4).backward();ends[arm]=pre.rng_state(torch)
                require(all(v.grad is None for n,v in models[arm].named_parameters() if not n.startswith('future_pred_head.')) and
                    all(v.grad is None for v in D.parameters()) and not D.training,'Frozen state received gradient or D changed mode')
                require(tree.tree_digest(sample['inputs'])==boundary,'Cache inputs mutated')
                batches[arm].append(dict(ordinal=ordinal,**common.identity(record),inputs_sha256=record['files']['inputs']['sha256'],
                    targets_sha256=record['files']['targets']['sha256'],sparse_label_sha256=descriptor['sha256'],input_tree_sha256=boundary,
                    occupancy_loss=occ_audit,physical_loss=float(physical.detach()),physical_audit=physical_audit,total_loss=float(total.detach()),
                    native_logits_sha256=native_digest,native_logits_unmodified_by_readout=True,
                    paired_rng_before_sha256=before_rng,paired_rng_after_sha256=pre.rng_digest(ends[arm])))
                del prediction,features,flow,sparse,label,physical,occupancy,total
            require(pre.rng_digest(ends['K'])==pre.rng_digest(ends['B']),'Paired RNG consumption differs')
            pre.restore_rng(torch,ends['K'])
            if initial_comparison is not None:
                for arm in ARMS:batches[arm][-1]['initial_occupancy_comparison']=initial_comparison
            del sample,base,paired,ends
            examples+=1;torch.cuda.synchronize(a.device);micro_times.append(time.monotonic()-mtick)
            if micro!=3:continue
            norms={}
            for arm in ARMS:
                hg=pre.gradient_stats(head_params[arm],[v.grad for _,v in head_params[arm]])
                rg=pre.gradient_stats(readout_params[arm],[v.grad for _,v in readout_params[arm]])
                require(rg['none_parameters']==0,'Missing physical readout parameter graph')
                for name,norm in rg['parameter_norms'].items():had[arm][name]|=norm>0
                hn=float(torch.nn.utils.clip_grad_norm_([v for _,v in head_params[arm]],35.,error_if_nonfinite=True))
                rn=float(torch.nn.utils.clip_grad_norm_(readouts[arm].parameters(),10.,error_if_nonfinite=True))
                norms[arm]=dict(head=hg,readout=rg,head_preclip_norm=hn,readout_preclip_norm=rn,
                    head_clip_factor=min(1.,35./(hn+1e-6)),readout_clip_factor=min(1.,10./(rn+1e-6)))
            budget()
            for arm in ARMS:
                for opt in opts[arm].values():
                    opt.step();opt.zero_grad(set_to_none=True)
                    require(all(torch.isfinite(v[k]).all() for v in opt.state.values() for k in ('exp_avg','exp_avg_sq')),'Nonfinite AdamW moments')
                require(all(torch.isfinite(v).all() for _,v in head_params[arm]+readout_params[arm]),'Nonfinite updated parameter')
            updates+=1;torch.cuda.synchronize(a.device);update_times.append(time.monotonic()-utick)
            for arm in ARMS:
                append(folders[arm]/'training.jsonl',dict(update=updates,examples=examples,pass_index=pass_index,
                    future_head_lr=pre.head_lr(updates-1),readout_lr=helper.schedule_lr(updates-1),samples=batches[arm],
                    gradients=norms[arm],matched_accum4_update_seconds=update_times[-1],peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device)))
            progress=dict(schema=SCHEMA,phase='training',updates=updates,examples=examples,seconds=time.monotonic()-started)
            write(out/'progress.json',progress);print(json.dumps(progress),flush=True);budget()
    require(updates==512 and examples==2048 and len(connections)==4,'Fixed formal training endpoint incomplete')
    require(helper.state_digest(D)==D_sha and all(pre.non_head_digest(m)==frozen_sha for m in models.values()),'Frozen model state changed')
    final_head={arm:helper.state_digest(m.future_pred_head) for arm,m in models.items()}
    final_readout={arm:helper.state_digest(m) for arm,m in readouts.items()}
    optimizer_steps={};finals={}
    for arm in ARMS:
        optimizer_steps[arm]={}
        for name,opt in opts[arm].items():
            steps=[int(v['step'].item()) for v in opt.state.values()]
            require(steps and all(s==512 for s in steps),'Actual AdamW endpoint differs')
            optimizer_steps[arm][name]=dict(parameters_with_state=len(steps),all_steps512=True)
        payload=dict(schema=SCHEMA,status='FIXED_FINAL_512',arm=arm,future_pred_head=models[arm].future_pred_head.state_dict(),
            motion_readout=readouts[arm].state_dict(),optimizers={name:opt.state_dict() for name,opt in opts[arm].items()},
            update=512,examples=2048,sample_orders=orders,sample_orders_sha256=helper.json_hash(orders),
            initial_future_head_sha256=head_initial,initial_readout_sha256=readout_initial,
            final_future_head_sha256=final_head[arm],final_readout_sha256=final_readout[arm],
            manifest_sha256=sha(folders[arm]/'manifest.json'),common_manifest_sha256=sha(out/'manifest.json'),sources=sources,
            actual_optimizer_steps=optimizer_steps[arm],resume_supported=False)
        path=folders[arm]/'final.pth';tmp=path.with_suffix('.tmp');torch.save(payload,str(tmp));tmp.replace(path);del payload
        finals[arm]=dict(file=str(path.relative_to(out)),sha256=sha(path),future_head_state_sha256=final_head[arm],readout_state_sha256=final_readout[arm])
    # Copy the unchanged observer and restore its authenticated original head.
    # It is an O reference, not a new initialization or fine-tuned comparator.
    baseline=copy.deepcopy(models['K']);baseline.future_pred_head.load_state_dict(initial_head_state,strict=True)
    for param in baseline.parameters():param.requires_grad_(False)
    baseline.eval();require(helper.state_digest(baseline)==O_sha,'Restored frozen O reference differs')
    for arm in ARMS:models[arm].eval();readouts[arm].eval()
    train_seconds=time.monotonic()-started;dev_tick=time.monotonic()
    with torch.no_grad():
        for ordinal,record in enumerate(dev):
            budget();tick=time.monotonic();sample=common.input_only(native,dev_root,record,a.device)
            boundary=tree.tree_digest(sample['inputs'])
            predictions={'O':native.replay(baseline,sample,training=False)[0]}
            base=D(sample['inputs']['prev_bev_input'][:,-1]);flows={'D':base}
            for arm in ARMS:
                predictions[arm],features=core.capture_native_terminal(models[arm],sample,native,training=False)
                flows[arm]=readouts[arm](features,base,detach_features=arm=='B');del features
            # All O/K/B native predictions and D/K/B fields exist before labels.
            gt=common.load_targets_after_predictions(native,dev_root,record,sample,a.device)
            ld=labels[record['sample_token']];label=helper.load_sparse(label_root,ld,record)
            rd=raw_desc[record['sample_token']];path=raw_root/rd['file'];require(sha(path)==rd['sha256'],'Raw label bytes changed')
            raw_bytes=gzip.decompress(path.read_bytes());require(hashlib.sha256(raw_bytes).hexdigest()==rd['uncompressed_json_sha256'],'Raw decoded bytes changed')
            raw=json.loads(raw_bytes);hist={};metrics={}
            for arm,prediction in predictions.items():
                hist[arm]=common.native_hist(baseline,prediction,sample,record)
                if arm=='O':require(hist[arm]==reference[ordinal]['hist_by_horizon'],'Original O dev200 histogram differs')
                binary=common.fine_binary(prediction,oracle,budget)
                metrics[arm]=metric.evaluate_common_occupancy_change(binary,gt,oracle.EXTENT,raw)
                require([h['occupancy']['confusion'] for h in metrics[arm]['horizons']]==hist[arm],'Native/full-GT CPU histogram differs')
                del binary
            verify_common_denominators(metrics)
            physical_counts={}
            for arm,flow in flows.items():
                physical_rows=helper.epe_records(helper.gather_sparse(flow,label),label,record)
                physical_counts[arm]=len(physical_rows)
                physical_rows=[dict(item,arm=arm,ordinal=ordinal) for item in physical_rows]
                append_many(out/'development_objects.jsonl',physical_rows);objects.extend(physical_rows)
            require(len(set(physical_counts.values()))==1,'Physical support count differs')
            require(tree.tree_digest(sample['inputs'])==boundary,'Development inputs mutated')
            row=dict(ordinal=ordinal,**common.identity(record),hist_by_arm=hist,metrics_by_arm=metrics,
                physical_object_rows_by_arm=physical_counts,inputs_sha256=record['files']['inputs']['sha256'],
                targets_sha256=record['files']['targets']['sha256'],raw_label_sha256=rd['sha256'],sparse_label_sha256=ld['sha256'],
                O_reference_hist_exact=True,native_CPU_hist_exact=True,GT_common_denominators_exact=True,
                t0_prediction_policy='model_specific_native',
                t0_predictions_may_differ=True,t0_equality_gate_applied=False,GT_raw_sparse_read_after_predictions=True,
                seconds=time.monotonic()-tick)
            append(out/'development_records.jsonl',row);rows.append(row)
            for arm in ARMS:append(folders[arm]/'development_records.jsonl',dict(ordinal=ordinal,**common.identity(record),
                hist_by_horizon=hist[arm],horizon_seconds=[0,.5,1,1.5,2]))
            evaluated+=1;write(out/'progress.json',dict(schema=SCHEMA,phase='development',updates=512,examples=2048,
                samples=evaluated,seconds=time.monotonic()-started))
            del sample,predictions,base,flows,gt,label,raw_bytes,raw,hist,metrics
    require(evaluated==200 and helper.state_digest(D)==D_sha and helper.state_digest(baseline)==O_sha,'Incomplete evaluation/frozen state changed')
    require(all(helper.state_digest(models[a].future_pred_head)==final_head[a] and helper.state_digest(readouts[a])==final_readout[a]
        and pre.non_head_digest(models[a])==frozen_sha for a in ARMS),'Evaluation changed final model state')
    for name,digest in p['sources_sha256'].items():require(sha(contract.source_path(name))==digest,'Source changed during run')
    require(sha(a.protocol)==sources['protocol_sha256'],'Formal protocol changed')
    # Existing descriptive pure scoring has no cross-model t0 equality gate.
    scores=common.summarize(rows,('O','K','B'),metric)
    physical_summary={arm:helper.epe_summary([r for r in objects if r['arm']==arm]) for arm in ('D','K','B')}
    summary=dict(schema=SCHEMA,status='COMPLETE_FUTURE_STATE_MOTION_TRAINING',updates=512,examples=2048,evaluated_samples=200,
        scores=scores,physical=physical_summary,initial_future_head_sha256=head_initial,initial_readout_sha256=readout_initial,
        final_future_head_sha256=final_head,final_readout_sha256=final_readout,actual_optimizer_steps=optimizer_steps,
        readout_task_gradient_received=had,physical_gradient_diagnostics=connections,frozen_D_and_non_head_O_unchanged=True,
        initial_engineering=engineering,fresh_initialization=True,optimizer_restored=False,bootstrap_performed=False,
        t0_predictions_may_differ=True,physical_reference_is_privately_trained_D_not_native_O_flow=True,
        resources=dict(initialization_seconds=initialization_seconds,paired_microstep_seconds=micro_times,
            matched_accum4_update_seconds=update_times,train_endpoint_seconds=train_seconds,
            development_seconds=time.monotonic()-dev_tick,elapsed_seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device)))
    write(out/'summary.json',summary)
    for arm in ARMS:
        files=('manifest.json','training.jsonl','final.pth','development_records.jsonl')
        write(folders[arm]/'complete.json',dict(schema=SCHEMA,status=summary['status'],arm=arm,updates=512,examples=2048,
            evaluated_samples=200,final_future_head_sha256=final_head[arm],final_readout_sha256=final_readout[arm],
            files_sha256={f:sha(folders[arm]/f) for f in files}))
    budget();files=('manifest.json','loaded_models.json','summary.json','physical_gradient_diagnostics.jsonl',
                   'development_records.jsonl','development_objects.jsonl')
    write(out/'complete.json',dict(schema=SCHEMA,status=summary['status'],mode='train',updates=512,examples=2048,
        evaluated_samples=200,final_checkpoints=finals,files_sha256={f:sha(out/f) for f in files},
        arm_complete_sha256={arm:sha(folders[arm]/'complete.json') for arm in ARMS},elapsed_seconds=time.monotonic()-started))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','preflight-protocol','engineering-preflight','connected-protocol','training-run','config',
                 'checkpoint','o-checkpoint','repo','runtime-contract','train-cache','dev-cache','sparse-labels',
                 'raw-labels','o-development-records','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('--max-allocated-gib',type=float,required=True)
    a=parser.parse_args(argv);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();stop=[]
    signal.signal(signal.SIGTERM,lambda s,f:stop.append(s));signal.signal(signal.SIGINT,lambda s,f:stop.append(s))
    try:run(a,out,started,stop)
    except BaseException as exc:
        write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),
            last_progress=read(out/'progress.json') if (out/'progress.json').is_file() else None,
            final_checkpoint_does_not_authorize_evaluation_until_complete=True,seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
