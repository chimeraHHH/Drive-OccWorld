"""Matched feature-residual R/R0 training on frozen O and actual final D motion.

The only arm difference is D displacement versus zeros. Core parameters train;
O/D, their inputs, original GT, all native loss formulas and native evaluator
remain frozen. --preflight performs four actual accum4 AdamW updates and saves
no weights. Formal training requires a separately bound successful preflight.
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
import random
import signal
import sys
import time
import traceback

import numpy as np

SCHEMA = 'future-feature-residual-training-v1'
ARMS = ('R', 'R0')
CONNECTED_PROTOCOL_SHA = 'e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
CONNECTED_COMPLETE_SHA = 'f0bf075fa36622b38c0c5136b187a184d16d3e07ed00857dfe21a0545f1c5f56'
D_CHECKPOINT_SHA = '7e2750d56ade7fb36e334b6431e83f51aafa997357744780a507924a696951e3'
COMMON_SHA = '57cf304bafb050fac05b06abe68ce4600a37ca020d9d10d0673ec0ab1edb3bc1'
RUNTIME_SHA = '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c'
O_REFERENCE_SHA = '3f426891f6811a595542a1e9694783521ae89cf18d6385896012e5f52a8e9b9a'
RAW_MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
RAW_COMPLETE_SHA = 'e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69'
NUMERICAL = dict(dtype='float32', matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False)
TRAINING = dict(seed=11, train_samples=512, development_samples=200, passes=4,
    accumulate=4, updates=512, parameters_per_arm=41137, lr=1e-3,
    grad_clip=35., weight_decay=.01, optimizer='AdamW', betas=[.9,.999], eps=1e-8,
    amsgrad=False, warmup_updates=25, final_lr_ratio=.1,
    order='continuous_numpy_RandomState11_permutation_each_pass',
    lr_schedule='train_source_motion_v1.schedule_lr(u); u=0..511',
    loss='unchanged native coarse twelve computed; original O CE[1,5]+Lovasz six selected',
    scope='only independent new 41137-parameter residual modules; O and actual final D motion frozen',
    architecture='shared Linear256->(Z*8) projection; source 3D trilinear feature transport; Conv3d19->16 k3 ReLU -> zero Conv3d16->1 k1; no GN',
    transport='R actual final D displacement; R0 zero displacement; same original O current hard mask',
    feature_input='O terminal readout prehook clone [5,B,YX,256]; no raw boxes or motion labels',
    target='signed residual on final decoder future foreground logit only; t0/intermediate/background unchanged',
    checkpoint='fixed final512 only; no resume, best selection or initial-development scoring')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()


def seed_all(torch):
    random.seed(11);np.random.seed(11);torch.manual_seed(11);torch.cuda.manual_seed_all(11)


def protocol_template():
    return dict(schema=SCHEMA,status='REVIEW_REQUIRED',training=copy.deepcopy(TRAINING),
        numerical_policy=dict(NUMERICAL),sources_sha256={},
        connected_protocol_sha256=CONNECTED_PROTOCOL_SHA,connected_complete_sha256=CONNECTED_COMPLETE_SHA,
        D_checkpoint_sha256=D_CHECKPOINT_SHA,runtime_source_contract_sha256=RUNTIME_SHA,
        cache_index_sha256=dict(train=None,development=None),o_development_records_sha256=O_REFERENCE_SHA,
        raw_labels=dict(manifest_sha256=RAW_MANIFEST_SHA,complete_sha256=RAW_COMPLETE_SHA),
        engineering=dict(preflight_complete_sha256=None),
        resources=dict(preflight=dict(max_seconds=600.,max_allocated_gib=32.),train=None))


def load_contract(a):
    p=read(a.protocol)
    require(p['schema']==SCHEMA and p['status']=='FROZEN' and p['training']==TRAINING and
            p['numerical_policy']==NUMERICAL,'Frozen exact feature-residual recipe required')
    require(p['connected_protocol_sha256']==sha(a.connected_protocol)==CONNECTED_PROTOCOL_SHA and
            p['connected_complete_sha256']==sha(Path(a.training_run)/'complete.json')==CONNECTED_COMPLETE_SHA and
            p['D_checkpoint_sha256']==D_CHECKPOINT_SHA and p['runtime_source_contract_sha256']==RUNTIME_SHA,
            'Actual frozen D provenance changed')
    require(math.isfinite(a.max_seconds) and 0<a.max_seconds and math.isfinite(a.max_allocated_gib)
            and 0<a.max_allocated_gib<=32,'Finite positive budget and <=32GiB required')
    mode='preflight' if a.preflight else 'train'
    require(p['resources'][mode]==dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib),
            'Resources must be explicitly frozen for this execution mode')
    cp=Path(__file__).with_name('common_connected_motion_evaluation_v2.py')
    require(sha(cp)==COMMON_SHA,'Common source loader changed')
    spec=importlib.util.spec_from_file_location('common_connected_motion_evaluation_v2',cp)
    contract=importlib.util.module_from_spec(spec);sys.modules[spec.name]=contract;spec.loader.exec_module(contract)
    connected=read(a.connected_protocol)
    bindings=contract.check_sources(connected)
    for name,digest in bindings.items():require(p['sources_sha256'].get(name)==digest,'Inherited source changed: '+name)
    for name in (Path(__file__).name,'future_feature_residual_v1.py'):
        require(name in p['sources_sha256'],'Missing feature source '+name)
    for name,digest in p['sources_sha256'].items():require(sha(contract.source_path(name))==digest,'Source changed: '+name)
    require(p['cache_index_sha256']==connected['cache_index_sha256'] and p['o_development_records_sha256']==O_REFERENCE_SHA,
            'Frozen native inputs/reference changed')
    runtime=read(a.runtime_contract);require(sha(a.runtime_contract)==RUNTIME_SHA,'Runtime source contract changed')
    for path,digest in runtime['runtime_source_sha256'].items():require(sha(path)==digest,'Native runtime changed: '+path)
    modules={name:contract.import_bound(name,p['sources_sha256']) for name in
        ('train_source_motion_v1','native_state_cache','oracle_transport_probe','common_change_evaluation_v2',
         'train_supported_fusion_v2','objective_supervision_adapters','train_connected_motion_v2','joint_native_evaluation')}
    require(sha(a.config)==modules['oracle_transport_probe'].CONFIG_SHA and
            sha(a.checkpoint)==modules['oracle_transport_probe'].M0_SHA and
            sha(a.o_checkpoint)==modules['oracle_transport_probe'].O_SHA,'O/M0/config changed')
    return p,connected,contract,modules


def validate_preflight(path, protocol, initial_sha):
    root=Path(path);done=read(root/'complete.json')
    require(not (root/'failed.json').exists() and sha(root/'complete.json')==protocol['engineering']['preflight_complete_sha256'],
            'Actual successful preflight receipt required')
    require(done['schema']==SCHEMA and done['status']=='PASS_FUTURE_FEATURE_RESIDUAL_PREFLIGHT' and
            done['mode']=='preflight' and done['updates']==4 and done['examples']==16 and
            done['evaluated_samples']==0 and done['final_checkpoints']=={},'Wrong preflight endpoint')
    for f,h in done['files_sha256'].items():require(sha(root/f)==h,'Preflight artifact changed')
    manifest=read(root/'manifest.json');summary=read(root/'summary.json')
    require(manifest['sources']['sources_sha256']==protocol['sources_sha256'] and
            manifest['training']==TRAINING and manifest['numerical_policy']==NUMERICAL and
            manifest['initial_module_sha256']==initial_sha,'Preflight source/recipe/initial state mismatch')
    require(summary['initial_O_parity_pass'] and summary['all_gradients_finite'] and
            summary['upstream_task_gradient_reached'] and summary['zero_readout_changed'] and
            not summary['weights_persisted'] and summary['frozen_O_D_unchanged'],'Preflight did not validate learning boundary')
    for arm,h in done['arm_complete_sha256'].items():
        require(arm in ARMS and sha(root/'runs'/arm/'complete.json')==h,'Preflight arm receipt changed')
        ac=read(root/'runs'/arm/'complete.json')
        for f,digest in ac['files_sha256'].items():require(sha(root/'runs'/arm/f)==digest,'Preflight arm file changed')
    return dict(complete_sha256=sha(root/'complete.json'),initial_module_sha256=initial_sha)


def output_stats(result):
    """Small descriptive field statistics; no threshold or local support gate."""
    import torch
    residual=result['residual'];support=result['support_weight']
    require(torch.isfinite(residual).all() and torch.isfinite(support).all(),'Nonfinite residual/transport')
    rows=[]
    for h in range(4):
        value=residual[0,h,0];mask=support[0,h,0]>0
        rows.append(dict(horizon_index=h+1,nominal_seconds=(h+1)*.5,
            supported_voxels=int(mask.sum()),residual_nonzero_voxels=int(torch.count_nonzero(value)),
            residual_nonzero_outside_transport_support=int(torch.count_nonzero(value[~mask])),
            residual_rms=float(value.detach().square().mean().sqrt()),residual_max_abs=float(value.detach().abs().max())))
    return rows


def run(a,out,started,stop):
    p,connected,contract,mods=load_contract(a)
    helper=mods['train_source_motion_v1'];native=mods['native_state_cache'];oracle=mods['oracle_transport_probe']
    common=mods['common_change_evaluation_v2'];losshelper=mods['train_supported_fusion_v2']
    adapter=mods['objective_supervision_adapters'];trainer=mods['train_connected_motion_v2'];tree=mods['joint_native_evaluation']
    train_root,train,train_receipt=helper.cache_index(a.train_cache,'train',p)
    dev_root,dev,dev_receipt,reference,rawroot,rawdesc=None,[],None,[],None,{}
    if not a.preflight:
        require(a.dev_cache and a.o_development_records and a.raw_labels and a.engineering_preflight,
                'Formal run requires dev cache/reference/raw labels and successful preflight')
        dev_root,dev,dev_receipt=helper.cache_index(a.dev_cache,'development',p)
        require(sha(a.o_development_records)==O_REFERENCE_SHA,'Original O development reference changed')
        reference=[json.loads(x) for x in Path(a.o_development_records).read_text().splitlines()]
        require(len(reference)==200 and all(all(old[k]==r[k] for k in ('sample_token','scene_token')) for old,r in zip(reference,dev)),
                'Development order differs from O reference')
        require(not ({r['scene_token'] for r in train}&{r['scene_token'] for r in dev}),'Train/development scenes overlap')
        require(p['raw_labels']==dict(manifest_sha256=RAW_MANIFEST_SHA,complete_sha256=RAW_COMPLETE_SHA),'Raw metadata contract changed')
        rawroot=Path(a.raw_labels)
        require(sha(rawroot/'manifest.json')==RAW_MANIFEST_SHA and sha(rawroot/'complete.json')==RAW_COMPLETE_SHA,'Raw labels changed')
        rawdesc={r['identity']['sample_token']:r for r in read(rawroot/'manifest.json')['records']}
        for r in dev:require(rawdesc[r['sample_token']]['identity']==common.identity(r),'Raw development identity differs')
    write(out/'progress.json',dict(schema=SCHEMA,phase='model_initialization',updates=0,examples=0))
    import torch
    require(str(a.device).startswith('cuda') and torch.cuda.is_available(),'Native frozen O requires CUDA')
    torch.cuda.set_device(a.device);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(min(1.,a.max_allocated_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory),a.device)
    torch.cuda.reset_peak_memory_stats(a.device)
    def budget():
        require(not stop,'Signal received; no retry')
        require(time.monotonic()-started<a.max_seconds,'Wall-clock ceiling')
        require(torch.cuda.max_memory_allocated(a.device)<=a.max_allocated_gib*2**30,'CUDA memory ceiling')
    budget()
    model=native.build_native_model(a.config,a.checkpoint,device=a.device,repo=a.repo)
    payload=torch.load(a.o_checkpoint,map_location='cpu',weights_only=False)
    model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True);del payload
    for param in model.parameters():param.requires_grad_(False)
    model.eval()
    require(common.historical_O_head_digest(model.future_pred_head)==common.O_HEAD_SHA,'Historical O state changed')
    O_sha=helper.state_digest(model)
    require(O_sha==read(Path(a.training_run)/'manifest.json')['frozen_O_state_sha256'],'Actual O differs from connected training')
    motion,unused_gate,D_receipt=trainer.load_completed_arm(Path(a.training_run)/'runs'/'D',Path(a.connected_protocol),helper,a.device)
    del unused_gate
    D_sha=helper.state_digest(motion)
    require(D_receipt['top_complete_sha256']==CONNECTED_COMPLETE_SHA and D_receipt['checkpoint_sha256']==D_CHECKPOINT_SHA,
            'Wrong actual frozen D motion')
    require(sha(inspect.getsourcefile(model.compute_occ_loss))==adapter.DETECTOR_SOURCE_SHA and
            sha(inspect.getsourcefile(model.future_pred_head.loss_occ))==adapter.HEAD_SOURCE_SHA and
            sha(inspect.getsourcefile(model.future_pred_head.loss_voxel))==adapter.HEAD_SOURCE_SHA and
            model.future_pred_head.class_weights.tolist()==[1.,5.],'Native loss source/weights changed')
    contract.import_bound('transport_ops',p['sources_sha256'])
    core=contract.import_bound('future_feature_residual_v1',p['sources_sha256'])
    seed_all(torch)
    heads={'R':core.FutureFeatureResidual().to(a.device).train()};heads['R0']=copy.deepcopy(heads['R'])
    initial={k:v.detach().cpu().clone() for k,v in heads['R'].state_dict().items()}
    initial_sha=helper.state_digest(heads['R'])
    require(helper.state_digest(heads['R0'])==initial_sha,'Initial residual modules differ')
    names=[name for name,_ in heads['R'].named_parameters()]
    require(all(sum(v.numel() for v in head.parameters())==41137 for head in heads.values()),'Feature residual architecture changed')
    require(names==['projection.weight','projection.bias','decoder.0.weight','decoder.0.bias','decoder.2.weight','decoder.2.bias'],
            'Residual parameter scope changed')
    require(all(torch.count_nonzero(head.decoder[-1].weight)==0 and torch.count_nonzero(head.decoder[-1].bias)==0 for head in heads.values()),
            'Final residual readout is not zero initialized')
    require(not ({id(v) for v in heads['R'].parameters()}&{id(v) for v in heads['R0'].parameters()}),'Arms share parameters')
    engineering_receipt=None if a.preflight else validate_preflight(a.engineering_preflight,p,initial_sha)
    optimizers={arm:torch.optim.AdamW(head.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=.01,amsgrad=False) for arm,head in heads.items()}
    orders=helper.planned_orders();limit=4 if a.preflight else 512;mode='preflight' if a.preflight else 'train'
    sources=dict(protocol_sha256=sha(a.protocol),sources_sha256=p['sources_sha256'],connected_protocol_sha256=CONNECTED_PROTOCOL_SHA,
        connected_complete_sha256=CONNECTED_COMPLETE_SHA,D_checkpoint_sha256=D_CHECKPOINT_SHA,
        train_cache=train_receipt,development_cache=dev_receipt,O_checkpoint_sha256=oracle.O_SHA,
        M0_checkpoint_sha256=oracle.M0_SHA,config_sha256=oracle.CONFIG_SHA,runtime_source_contract_sha256=RUNTIME_SHA)
    manifest=dict(schema=SCHEMA,mode=mode,sources=sources,training=TRAINING,numerical_policy=NUMERICAL,
        arms=list(ARMS),initial_module_sha256=initial_sha,frozen_O_state_sha256=O_sha,frozen_D_motion_state_sha256=D_sha,
        sample_orders_sha256=helper.json_hash(orders),trainable_names=names,same_initial_parameters=True,independent_optimizers=True,
        planned_updates=limit,engineering_preflight=engineering_receipt,seed=11,optimizer_scope='new feature residual only',
        current_mask='original frozen O current hard argmax; no GT mask',raw_or_sparse_labels_used_for_training=False,
        GT_used_only_by_original_occupancy_loss=True,terminal_features='no-modification prehook clone',
        final_decoder_future_foreground_only=True,t0_intermediate_and_background_exact=True,
        residual_allowed_outside_transport_support=True,no_local_W_influence_domain_claim=True,
        no_new_flow_EPE_claim=True,development_exposure='historical development200; not full validation')
    write(out/'manifest.json',manifest);write(out/'loaded_models.json',dict(O_full_state_sha256=O_sha,D=D_receipt,optimizer_restored=False))
    folders={arm:out/'runs'/arm for arm in ARMS}
    for arm,folder in folders.items():
        folder.mkdir(parents=True)
        write(folder/'manifest.json',dict(manifest,arm=arm,transport='actual_D' if arm=='R' else 'zero',common_manifest_sha256=sha(out/'manifest.json')))
    updates=examples=micro=evaluated=0;actual={arm:0 for arm in ARMS}
    had={arm:{name:False for name in names} for arm in ARMS};parity=[];devrows=[];microtimes=[];updatetimes=[]
    init_seconds=time.monotonic()-started
    def checkpoint(arm,filename,status):
        path=folders[arm]/filename
        payload=dict(schema=SCHEMA,mode=mode,arm=arm,status=status,residual_module=heads[arm].state_dict(),
            optimizer=optimizers[arm].state_dict(),update=actual[arm],examples=examples,matched_updates_completed=updates,
            incomplete_accumulation_examples=micro,sample_orders=orders,sample_orders_sha256=helper.json_hash(orders),
            initial_module_sha256=initial_sha,final_module_sha256=helper.state_digest(heads[arm]),
            manifest_sha256=sha(folders[arm]/'manifest.json'),sources=sources,resume_supported=False)
        temporary=path.with_suffix('.tmp');torch.save(payload,str(temporary));temporary.replace(path)
        return dict(file=str(path.relative_to(out)),sha256=sha(path),module_state_sha256=payload['final_module_sha256'])
    def shared(record,root):
        with torch.no_grad():
            sample=common.input_only(native,root,record,a.device)
            boundary=tree.tree_digest(sample['inputs'])
            original,features=core.capture_O_terminal(model,sample,native)
            base=oracle.predictions_to_xyz(original)
            flow=motion(sample['inputs']['prev_bev_input'][:,-1])
            foreground=base[0:1,1:2]>base[0:1,0:1]
            future=(base[1:,1]-base[1:,0])[None,:,None]
            require(not features.requires_grad and not flow.requires_grad and torch.isfinite(features).all() and torch.isfinite(flow).all(),'Frozen features/flow invalid')
        return sample,original,features,base,flow,foreground,future,boundary
    def prediction(arm,original,features,base,flow,foreground,future):
        result=heads[arm](features,foreground,flow if arm=='R' else torch.zeros_like(flow),future)
        changed=core.compose_prediction(original,base,result['residual'],oracle)
        require(losshelper.exact32(changed[0],original[0]) and losshelper.exact32(changed[:,:-1],original[:,:-1]) and
                losshelper.exact32(changed[:,-1,...,0],original[:,-1,...,0]),'Protected native logits changed')
        return changed,result
    def unchanged(sample,boundary):
        require(tree.tree_digest(sample['inputs'])==boundary,'Input mutated')
        require(all(v.grad is None for v in model.parameters()) and all(v.grad is None for v in motion.parameters()),'Gradient reached frozen O/D')
        require(not any(x.training for x in model.modules()) and not motion.training,'Frozen model mode changed')
    try:
        seed_all(torch)
        for opt in optimizers.values():opt.zero_grad(set_to_none=True)
        for pass_index,order in enumerate(orders):
            for ordinal in order:
                budget();torch.cuda.synchronize(a.device);mtick=time.monotonic()
                if micro==0:
                    utick=mtick;batches={arm:[] for arm in ARMS}
                    for opt in optimizers.values():
                        for group in opt.param_groups:group['lr']=helper.schedule_lr(updates)
                record=train[ordinal]
                sample,original,features,base,flow,foreground,future,boundary=shared(record,train_root)
                common.load_targets_after_predictions(native,train_root,record,sample,a.device)
                audits={};original_hist=common.native_hist(model,original,sample,record) if updates==0 and micro<2 else None
                for arm in ARMS:
                    changed,result=prediction(arm,original,features,base,flow,foreground,future)
                    if original_hist is not None:
                        require(torch.count_nonzero(result['residual'])==0 and losshelper.exact32(changed,original),'Initial residual is not bitwise O')
                        require(common.native_hist(model,changed,sample,record)==original_hist,'Initial native histogram differs')
                        parity.append(dict(arm=arm,ordinal=ordinal,sample_token=record['sample_token'],scene_token=record['scene_token'],all_five_horizons_three_layers_logits_exact=True,native_hist_exact=True))
                    loss,audit=losshelper.loss_terms(model,changed,sample['targets'],adapter)
                    stats=output_stats(result)
                    (loss/4).backward();audits[arm]=audit
                    batches[arm].append(dict(ordinal=ordinal,sample_token=record['sample_token'],scene_token=record['scene_token'],
                        inputs_sha256=record['files']['inputs']['sha256'],targets_sha256=record['files']['targets']['sha256'],
                        input_tree_sha256=boundary,loss=audit,fields=stats))
                    del loss,changed,result
                comparison=losshelper.compare_constant_losses(audits['R'],audits['R0'])
                require(all(x['passed'] for x in comparison.values()),'Original constant CE tolerance/Lovasz exact gate failed')
                for arm in ARMS:batches[arm][-1]['constant_loss_comparison']=comparison
                unchanged(sample,boundary)
                del sample,original,features,base,flow,foreground,future,audits
                examples+=1;micro+=1;torch.cuda.synchronize(a.device)
                microtimes.append(time.monotonic()-mtick)
                if micro<4:continue
                norms={};namednorms={}
                for arm,head in heads.items():
                    require(all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters()),'Missing/nonfinite residual gradient')
                    namednorms[arm]={name:float(v.grad.norm()) for name,v in head.named_parameters()}
                    for name,norm in namednorms[arm].items():had[arm][name]|=norm>0
                    norms[arm]=float(torch.nn.utils.clip_grad_norm_(head.parameters(),35.,error_if_nonfinite=True))
                budget()
                for arm,opt in optimizers.items():
                    opt.step();actual[arm]+=1;opt.zero_grad(set_to_none=True)
                    require(all(torch.isfinite(v).all() for v in heads[arm].parameters()),'Nonfinite residual parameter')
                    require(all(torch.isfinite(v[k]).all() for v in opt.state.values() for k in ('exp_avg','exp_avg_sq')),'Nonfinite AdamW moment')
                updates+=1;micro=0;torch.cuda.synchronize(a.device);updatetimes.append(time.monotonic()-utick)
                for arm in ARMS:
                    append(folders[arm]/'training.jsonl',dict(update=updates,pass_index=pass_index,examples=examples,
                        lr=helper.schedule_lr(updates-1),loss=float(np.mean([x['loss']['objective_sum'] for x in batches[arm]])),
                        preclip_grad_norm=norms[arm],parameter_gradient_norms=namednorms[arm],clip_factor=min(1.,35./(norms[arm]+1e-6)),
                        samples=batches[arm],sample_group_sha256=helper.json_hash([x['sample_token'] for x in batches[arm]]),
                        shared_update_wall_seconds=updatetimes[-1],peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device)))
                progress=dict(schema=SCHEMA,phase='training',updates=updates,examples=examples,actual_updates=actual,seconds=time.monotonic()-started)
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                budget()
                if updates==limit:break
            if updates==limit:break
        require(updates==limit and examples==4*limit and micro==0 and all(x==limit for x in actual.values()),'Fixed update budget incomplete')
        require(len(parity)==4 and all(all(x.values()) for x in had.values()),'Initial parity or task gradient propagation failed')
        changes={arm:{name:float((value.detach().cpu()-initial[name]).abs().max()) for name,value in head.state_dict().items()} for arm,head in heads.items()}
        # Every parameter tensor must have received a task gradient at least once;
        # the first update can correctly have zero projection/trunk gradients.
        require(all(max(v.values())>0 for v in changes.values()),'Residual modules did not change')
        require(all(torch.count_nonzero(head.decoder[-1].weight)+torch.count_nonzero(head.decoder[-1].bias)>0 for head in heads.values()),
                'Zero residual readout unchanged; weight decay alone cannot satisfy this check')
        finalshas={arm:helper.state_digest(head) for arm,head in heads.items()}
        finals={} if a.preflight else {arm:checkpoint(arm,'final.pth','FIXED_FINAL_512') for arm in ARMS}
        if not a.preflight:
            metric=contract.import_bound('common_occupancy_change_metrics_v1',p['sources_sha256'])
            for head in heads.values():head.eval()
            with torch.no_grad():
                for i,record in enumerate(dev):
                    budget();tick=time.monotonic()
                    sample,original,features,base,flow,foreground,future,boundary=shared(record,dev_root)
                    predictions={'O':original};diags={}
                    for arm in ARMS:
                        predictions[arm],result=prediction(arm,original,features,base,flow,foreground,future)
                        diags[arm]=output_stats(result);del result
                    # Full candidate predictions are finished before evaluation
                    # GT and raw box contents enter the common metric only.
                    gt=common.load_targets_after_predictions(native,dev_root,record,sample,a.device)
                    desc=rawdesc[record['sample_token']];path=rawroot/desc['file']
                    require(sha(path)==desc['sha256'],'Raw label bytes changed')
                    rawbytes=gzip.decompress(path.read_bytes());require(hashlib.sha256(rawbytes).hexdigest()==desc['uncompressed_json_sha256'],'Decoded raw labels changed')
                    raw=json.loads(rawbytes);hist={};metrics={}
                    for arm,pred in predictions.items():
                        hist[arm]=common.native_hist(model,pred,sample,record)
                        if arm=='O':require(hist[arm]==reference[i]['hist_by_horizon'],'Original O dev200 changed')
                        binary=common.fine_binary(pred,oracle,budget)
                        metrics[arm]=metric.evaluate_common_occupancy_change(binary,gt,oracle.EXTENT,raw)
                        require([x['occupancy']['confusion'] for x in metrics[arm]['horizons']]==hist[arm],'CPU/native full GT histogram differs')
                        if arm!='O':require(metrics[arm]['t0_boundary']==metrics['O']['t0_boundary'],'Common t0 boundary differs')
                        del binary
                    unchanged(sample,boundary);evaluated+=1
                    row=dict(ordinal=i,**common.identity(record),hist_by_arm=hist,metrics_by_arm=metrics,diagnostics=diags,
                        inputs_sha256=record['files']['inputs']['sha256'],targets_sha256=record['files']['targets']['sha256'],raw_label_sha256=desc['sha256'],
                        O_reference_hist_exact=True,native_CPU_hist_exact=True,t0_all_arms_exact=True,
                        GT_raw_read_after_predictions=True,seconds=time.monotonic()-tick)
                    append(out/'development_records.jsonl',row);devrows.append(row)
                    for arm in ARMS:append(folders[arm]/'development_records.jsonl',dict(ordinal=i,**common.identity(record),hist_by_horizon=hist[arm],horizon_seconds=[0,.5,1,1.5,2]))
                    write(out/'progress.json',dict(schema=SCHEMA,phase='development',updates=updates,samples=evaluated,seconds=time.monotonic()-started))
                    del sample,original,features,base,flow,foreground,future,predictions,gt,rawbytes,raw,metrics,hist
            require(evaluated==200,'Incomplete final development evaluation')
        require(helper.state_digest(model)==O_sha and helper.state_digest(motion)==D_sha,'Frozen O/D state changed')
        require(all(helper.state_digest(heads[arm])==finalshas[arm] for arm in ARMS),'Evaluation changed residual weights')
        for name,digest in p['sources_sha256'].items():require(sha(contract.source_path(name))==digest,'Source changed during run')
        require(sha(a.protocol)==sources['protocol_sha256'],'Execution protocol changed')
        summary=dict(schema=SCHEMA,mode=mode,updates=updates,examples=examples,evaluated_samples=evaluated,
            scores=None if a.preflight else common.summarize(devrows,('O',*ARMS),metric),
            initial_module_sha256=initial_sha,final_module_sha256=finalshas,parameter_max_abs_change=changes,
            initial_O_parity_pass=True,initial_O_parity=parity,all_gradients_finite=True,
            task_gradient_reached_by_parameter=had,upstream_task_gradient_reached=True,zero_readout_changed=True,
            frozen_O_D_unchanged=True,weights_persisted=not a.preflight,physical_motion_unchanged_no_new_EPE=True,
            seed=11,no_threshold_selection=True,bootstrap_performed=False,
            resources=dict(initialization_seconds=init_seconds,microstep_seconds=microtimes,
                first_microstep_seconds=microtimes[0],warm_microstep_mean_seconds=float(np.mean(microtimes[2:])),
                matched_accum4_update_seconds=updatetimes,elapsed_seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device),
                timing_scope='both arms shared frozen O/D; actual forward/loss/backward/AdamW including checks; first two micros additionally include initial parity hist'))
        write(out/'summary.json',summary)
        for arm in ARMS:
            files=['manifest.json','training.jsonl']+([] if a.preflight else ['final.pth','development_records.jsonl'])
            write(folders[arm]/'complete.json',dict(schema=SCHEMA,mode=mode,arm=arm,status='PASS_FUTURE_FEATURE_RESIDUAL_PREFLIGHT' if a.preflight else 'COMPLETE_FUTURE_FEATURE_RESIDUAL_TRAINING',
                updates=updates,examples=examples,evaluated_samples=evaluated,files_sha256={f:sha(folders[arm]/f) for f in files},final_module_sha256=finalshas[arm]))
        files=['manifest.json','loaded_models.json','summary.json']+([] if a.preflight else ['development_records.jsonl'])
        budget();write(out/'complete.json',dict(schema=SCHEMA,mode=mode,status='PASS_FUTURE_FEATURE_RESIDUAL_PREFLIGHT' if a.preflight else 'COMPLETE_FUTURE_FEATURE_RESIDUAL_TRAINING',
            updates=updates,examples=examples,evaluated_samples=evaluated,files_sha256={f:sha(out/f) for f in files},
            arm_complete_sha256={arm:sha(folders[arm]/'complete.json') for arm in ARMS},final_checkpoints=finals,
            elapsed_seconds=time.monotonic()-started))
    except BaseException as exc:
        partial={}
        if not a.preflight:
            for arm in ARMS:
                try:partial[arm]=checkpoint(arm,'interrupted.pth','INCOMPLETE_NOT_RESUMABLE')
                except BaseException as error:partial[arm]=dict(error=repr(error))
        write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),
            updates=updates,examples=examples,incomplete_accumulation_examples=micro,actual_updates=actual,
            evaluated_samples=evaluated,partial_checkpoints=partial,seconds=time.monotonic()-started))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','connected-protocol','training-run','config','checkpoint','o-checkpoint',
                 'repo','runtime-contract','train-cache','out'):
        parser.add_argument('--'+name,required=True)
    for name in ('dev-cache','o-development-records','raw-labels','engineering-preflight'):parser.add_argument('--'+name)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=float,required=True);parser.add_argument('--max-allocated-gib',type=float,required=True)
    a=parser.parse_args(argv)
    require(not (a.preflight and any((a.dev_cache,a.o_development_records,a.raw_labels,a.engineering_preflight))),
            'Preflight must not read development/raw labels or preflight weights')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();stop=[]
    signal.signal(signal.SIGTERM,lambda s,f:stop.append(s));signal.signal(signal.SIGINT,lambda s,f:stop.append(s))
    try:run(a,out,started,stop)
    except BaseException as exc:
        if not (out/'failed.json').exists():write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),phase='initialization',seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
