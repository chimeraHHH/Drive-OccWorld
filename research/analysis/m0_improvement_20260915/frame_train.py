"""Fixed-budget A/reference training with the completed native1 as control.

Only the reference arm may train. This script refuses an unfrozen frame protocol,
missing engineering evidence, changed sources/cache identities, or an unmatched
native1 control. Its optimizer/training/final-evaluation block is compiled from
the exact parent memory_experiment.train AST, rather than reimplementing its LR,
sample order, task loss, accumulation or checkpoint schedule. The only inserted
statement audits AdamW defaults against the completed control before any update.

An initial full development200 evaluation measures initialization impact, then
seed11 is reset before the original optimizer block. No checkpoint/threshold is
chosen from development scores. Probe weights are never read. The original M0,
old runs, all caches, and eight parent sources are read-only. No retry, resume,
automatic larger budget, second seed or full-validation dispatch is provided.

CLI: python frame_train.py --frame-protocol frame_protocol_v1.json
 --protocol protocol_v2.json --repo REPO --config CONFIG --checkpoint M0
 --train-cache TRAIN --dev-cache DEV --control-run RUNS/native1
 --out NEW_REFERENCE_RUN --max-seconds FROZEN_CAP
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import signal
import textwrap
import time

PARENT_PROTOCOL_SHA256 = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
INITIAL_HEAD_SHA256 = '6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'
CONTROL_FILES = ('manifest.json','training_complete.json','complete.json','latest.pth',
                 'training.jsonl','development_records.jsonl','frozen_native_fp32_records.jsonl')
IDENTITY_KEYS = ('sample_token','scene_token','official_index','split')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path,expected=None):
    path=Path(path);data=path.read_bytes()
    if expected is not None:require(hashlib.sha256(data).hexdigest()==expected,'Hash mismatch: '+str(path))
    return json.loads(data)


def relative_to_protocol(path,protocol_path):
    path=Path(path)
    return path if path.is_absolute() else Path(protocol_path).resolve().parent/path


def planned_orders(hp):
    import numpy as np
    rng=np.random.RandomState(hp['seed'])
    return [rng.permutation(hp['train_samples']).tolist() for _ in range(hp['passes'])]


def schedule_lr(update,hp):
    """Audit only: execution uses the unchanged expression in the native AST."""
    total=hp['passes']*hp['train_samples']//hp['accumulate']
    scale=(0.1+0.9*update/max(1,hp['warmup_updates'])) if update<hp['warmup_updates'] else (
        0.1+0.9*0.5*(1+math.cos(math.pi*(update-hp['warmup_updates'])/max(1,total-hp['warmup_updates']))))
    return hp['lr']*scale


def validate_engineering(frame,frame_path,parent):
    evidence=frame['engineering_evidence']
    require(set(evidence)=={'frame_preflight','real_geometry'},'Both engineering gates are required')
    result={}
    for name,binding in evidence.items():
        directory=relative_to_protocol(binding['directory'],frame_path)
        require(not (directory/'failed.json').exists(),'Failed engineering gate: '+name)
        complete=read(directory/'complete.json',binding['complete_sha256'])
        require(binding['source'] in frame['source_sha256'],'Engineering source is not frozen')
        require(complete['status']==binding['status'],'Engineering gate status mismatch')
        if name=='frame_preflight':
            require(complete['status']=='PASS_FRAME_ENGINEERING_PREFLIGHT' and
                    complete['performance_claim'] is False and complete['candidate_checkpoint_saved'] is False,
                    'Require successful engineering-only frame preflight')
            manifest=read(directory/'manifest.json',binding['manifest_sha256'])
            require(complete['manifest_sha256']==binding['manifest_sha256'] and
                    manifest['protocol_sha256']==PARENT_PROTOCOL_SHA256 and
                    manifest['m0_sha256']==parent['m0_sha256'] and manifest['seed']==11 and
                    manifest['precision']==parent['numerical_policy'],'Preflight source/precision differs')
            for filename,digest in manifest['source_sha256'].items():
                require(frame['source_sha256'].get(filename)==digest,'Preflight source differs: '+filename)
            reports=complete['reports'];require([r['mode'] for r in reports]==['native1','reference'],
                                               'Both coordinate modes must pass preflight')
            for report in reports:
                require(report['initial_head_sha256']==INITIAL_HEAD_SHA256 and report['trainable_parameters']==13274016,
                        'Preflight weights/capacity differ')
                require(report['migration']['adapter_sha256']==frame['source_sha256']['frame_consistent_adapter.py'],
                        'Preflight used a different adapter')
                gradients=report['last_horizon_gradient_norms']+[report['observed_t0_gradient_norm']]
                require(len(gradients)==5 and all(math.isfinite(g) and g>0 for g in gradients),
                        'Missing finite positive complete BPTT preflight')
        else:
            require(complete['status']=='PASS_REAL_LOADER_AND_GEOMETRY_AUDIT' and complete['optimizer_steps']==0,
                    'Require successful real-data coordinate contract audit')
            audit=read(directory/'result.json',complete['result_sha256'])
            require(audit['status']==complete['status'] and audit['config_sha256']==parent['config_sha256'] and
                    audit['seven_frame_GT_dataset_equals_loader_bitwise'] is True and
                    audit['prediction_based_selection'] is False and audit['data_modified'] is False and
                    audit['optimizer_steps']==0,'Real geometry result is not the expected read-only proof')
            source_values={Path(k).name:v for k,v in audit['source_files_sha256'].items()}
            require(source_values.get(binding['source'])==frame['source_sha256'][binding['source']],
                    'Real geometry proof used a different script')
            require(source_values.get('native_state_cache.py')==parent['source_sha256']['native_state_cache.py'],
                    'Real geometry proof used a different native boundary')
        result[name]=dict(directory=str(directory.resolve()),complete_sha256=binding['complete_sha256'],status=complete['status'])
    return result


def validate_inputs(a):
    """Read-only contract validation before model/optimizer construction."""
    pkg=Path(__file__).resolve().parent
    parent=read(a.protocol,PARENT_PROTOCOL_SHA256);frame=read(a.frame_protocol)
    require(frame['schema']=='frame-consistent-training-protocol-v1' and
            frame['status']=='FROZEN_BEFORE_REFERENCE_TRAINING' and frame['arm']=='reference',
            'Only frozen A/reference training is accepted')
    require(frame['parent_protocol_sha256']==PARENT_PROTOCOL_SHA256,'Wrong parent training protocol')
    for key in ('training','numerical_policy','config_sha256','m0_sha256','selection_sha256'):
        require(frame[key]==parent[key],'Reference/C contract differs: '+key)
    hp=parent['training']
    require(hp['seed']==11 and hp['train_samples']==512 and hp['development_samples']==200 and
            hp['passes']==4 and hp['accumulate']==4 and hp['lr']==1e-5 and hp['weight_decay']==.01 and
            hp['grad_clip']==35 and hp['warmup_updates']==50 and hp['scope']=='whole_future_pred_head',
            'Unexpected registered training schedule')
    require(type(a.max_seconds) is int and 0<a.max_seconds<=frame['resource_policy']['max_seconds'],
            'Deadline exceeds frozen resource cap')
    require(sha(a.config)==parent['config_sha256'] and sha(a.checkpoint)==parent['m0_sha256'],
            'Initial config/checkpoint differs from C')
    for filename,digest in parent['source_sha256'].items():
        require(Path(filename).name==filename and sha(pkg/filename)==digest,'Parent frozen source changed: '+filename)
    required={'frame_train.py','frame_consistent_adapter.py','frame_preflight.py','joint_native_evaluation.py',
              frame['engineering_evidence']['real_geometry']['source']}
    require(required<=set(frame['source_sha256']),'Frame sources are not completely bound')
    for filename,digest in frame['source_sha256'].items():
        require(Path(filename).name==filename and sha(pkg/filename)==digest,'Frame source changed: '+filename)
    require(parent['runtime_source_sha256'].items()<=frame['runtime_source_sha256'].items(),
            'Frame runtime source contract must include all parent bindings')
    for filename,digest in frame['runtime_source_sha256'].items():
        require(Path(filename).is_absolute() and sha(filename)==digest,'Native runtime source changed: '+filename)
    from frame_consistent_adapter import GRID_SOURCE_SHA256
    require(any(Path(k).name=='e2e_predictor_utils.py' and v==GRID_SOURCE_SHA256
                for k,v in frame['runtime_source_sha256'].items()),'Grid geometry helper must be explicitly frozen')
    selection_path=Path(a.protocol).with_name('selection_v1.json')
    selected=read(selection_path,parent['selection_sha256'])['records']
    from memory_experiment import cached_records
    records={};indices={}
    for role,cache in [('train',a.train_cache),('development',a.dev_cache)]:
        directory=Path(cache);index=read(directory/'index.json',frame['cache_index_sha256'][role])
        rows=cached_records(directory);expected=[r for r in selected if r['split']==role]
        require([[r[k] for k in IDENTITY_KEYS] for r in rows]==[[r[k] for k in IDENTITY_KEYS] for r in expected],
                'Cache exact identities/order/role differ: '+role)
        require(index['split']==role and index['config_sha256']==parent['config_sha256'] and
                index['selection_sha256']==parent['selection_sha256'] and
                index['extractor_sha256']==parent['source_sha256']['native_state_cache.py'],
                'Cache provenance differs: '+role)
        require(index['future_occupancy_is_input'] is False and index['inputs_targets_separate'] is True and
                index['native_model']['radar_contract']=='native_loader_B_without_common_source_override',
                'Cache input/target contract changed')
        require(all(r['parity']['bitwise_all_five_horizons_three_layers'] and r['parity']['exact_native_confusion'] for r in rows),
                'Incomplete original native cache parity')
        records[role]=rows;indices[role]=index
    require(len(records['train'])==512 and len(records['development'])==200,'Wrong data counts')
    require(not {r['scene_token'] for r in records['train']}&{r['scene_token'] for r in records['development']},
            'Training/development scenes overlap')
    evidence=validate_engineering(frame,a.frame_protocol,parent)
    return parent,frame,records,evidence


def validate_control(a,parent,frame,records):
    """Strict reuse of completed C, including actual final payload and log LR."""
    import torch
    directory=Path(a.control_run)
    require(not (directory/'failed.json').exists(),'Cannot reuse failed native1 control')
    require(set(frame['control_files_sha256'])==set(CONTROL_FILES),'Incomplete control file binding')
    for filename,digest in frame['control_files_sha256'].items():
        require(sha(directory/filename)==digest,'Fixed C artifact changed: '+filename)
    manifest=read(directory/'manifest.json');trained=read(directory/'training_complete.json');done=read(directory/'complete.json')
    hp=parent['training'];planned=planned_orders(hp)
    require(manifest['arm']=='native1' and manifest['seed']==11 and manifest['protocol_sha256']==PARENT_PROTOCOL_SHA256 and
            manifest['initial_head_sha256']==INITIAL_HEAD_SHA256 and manifest['m0_sha256']==parent['m0_sha256'],
            'Control is not the registered native1 initialization')
    require(manifest['numerical_policy']==parent['numerical_policy'] and manifest['trainable_parameters']==13274016 and
            manifest['passes']==4 and manifest['examples_per_pass']==512 and
            manifest['migration']['mode']=='native1','Control precision/capacity/exposure differs')
    for filename,digest in manifest['source_sha256'].items():
        require(parent['source_sha256'][filename]==digest,'Control trained source differs')
    require(manifest['train_index_sha256']==frame['cache_index_sha256']['train'] and
            manifest['dev_index_sha256']==frame['cache_index_sha256']['development'],'Control used different cached states')
    require(trained['status']=='TRAINED_FIXED_FINAL' and done['status']=='TRAINED_AND_DEVELOPMENT_EVALUATED',
            'Only completed fixed final C can be reused')
    for receipt in [trained,done]:
        require(receipt['updates']==512 and receipt['examples']==2048 and
                receipt['checkpoint_sha256']==frame['control_files_sha256']['latest.pth'],'Control endpoint differs')
    require(done['evaluation']['samples']==200 and done['evaluation']['sha256']==frame['control_files_sha256']['development_records.jsonl'],
            'Control final development is incomplete')
    reference=manifest['frozen_precision_reference']
    require(reference['samples']==200 and reference['optimizer_updates']==0 and
            reference['initial_head_sha256']==INITIAL_HEAD_SHA256 and reference['source_m0_sha256']==parent['m0_sha256'] and
            reference['matmul_tf32'] is False and reference['cudnn_tf32'] is True and
            reference['sha256']==frame['control_files_sha256']['frozen_native_fp32_records.jsonl'],
            'Control frozen M0_fp32 reference is not the same source')
    log=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines() if line.strip()]
    require(len(log)==512,'Incomplete control optimizer log')
    for index,row in enumerate(log):
        require(row['update']==index+1 and row['examples']==4*(index+1) and row['pass_index']==index//128+1,
                'Control update/exposure sequence differs')
        require(math.isclose(row['lr'],schedule_lr(index,hp),rel_tol=1e-14,abs_tol=0.),
                'Control warmup/cosine LR differs at update '+str(index+1))
        require(all(math.isfinite(row[key]) for key in ['loss_mean','loss_min','loss_max','grad_norm']),
                'Control training log is nonfinite')
    # No candidate weights are imported. This CPU payload is inspected solely
    # for control shape/order/optimizer metadata; A is freshly built from M0.
    payload=torch.load(str(directory/'latest.pth'),map_location='cpu')
    require(payload['arm']=='native1' and payload['metadata']==manifest and payload['pass_index']==4 and payload['update']==512 and
            payload['sample_orders']==planned,'Control final tensor payload/order differs')
    for filename in ('development_records.jsonl','frozen_native_fp32_records.jsonl'):
        rows=[json.loads(line) for line in (directory/filename).read_text().splitlines() if line.strip()]
        require([(r['sample_token'],r['scene_token']) for r in rows]==[(r['sample_token'],r['scene_token']) for r in records['development']],
                'Control evaluation identities/order differ')
    return manifest,payload,dict(directory=str(directory.resolve()),files_sha256=frame['control_files_sha256'],
        sample_orders_sha256=hashlib.sha256(json.dumps(planned,separators=(',',':')).encode()).hexdigest(),
        reused=True,new_control_training=False,exact_updates=512,exact_examples=2048,training_log_all_LRs_verified=True)


def verify_optimizer(optimizer,control_groups,hp):
    """Verify current AdamW defaults against the actual completed C optimizer."""
    current=optimizer.state_dict()['param_groups']
    require(len(current)==len(control_groups)==1,'Expected one native AdamW parameter group')
    left,right=current[0],control_groups[0]
    require(set(left)==set(right),'AdamW defaults/schema differ from reused C runtime')
    for key in left:
        if key=='lr':
            require(left[key]==hp['lr'] and math.isclose(right[key],schedule_lr(511,hp),rel_tol=1e-14),
                    'Control/default optimizer LR differs')
        else:require(left[key]==right[key],'AdamW/default parameter order differs: '+key)
    require(not optimizer.state,'Reference optimizer must start with zero state')


def native_training_block(parent):
    """Compile the exact native optimizer-through-final-evaluation block."""
    import memory_experiment as original
    from frame_consistent_adapter import ast_sha
    require(sha(original.__file__)==parent['source_sha256']['memory_experiment.py'],'Native trainer source changed')
    node=ast.parse(textwrap.dedent(inspect.getsource(original.train))).body[0]
    starts=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and len(n.targets)==1 and
            isinstance(n.targets[0],ast.Name) and n.targets[0].id=='optimizer']
    require(len(starts)==1,'Ambiguous native optimizer block')
    original_body=copy.deepcopy(node.body[starts[0]:]);body=copy.deepcopy(original_body)
    body.insert(1,ast.parse('_ft_verify_optimizer(optimizer, control_optimizer_groups, hp)').body[0])
    tree=ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]))
    receipt=dict(parent_train_ast_sha256=ast_sha(node),original_block_ast_sha256=ast_sha(ast.Module(body=original_body,type_ignores=[])),
        executed_block_ast_sha256=ast_sha(tree),unchanged_native_training_statements=len(original_body),
        added_statements=['verify fresh AdamW defaults/parameter order against fixed completed native1'],
        modified_native_statements=0,full_task_losses_unchanged=True,additional_loss_terms=0)
    return compile(tree,'<reference:original-native-training-block>','exec'),receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['frame-protocol','protocol','repo','config','checkpoint','train-cache','dev-cache','control-run','out']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-seconds',type=int,required=True)
    a=parser.parse_args(argv);a.arm='reference';out=Path(a.out).resolve()
    require(not out.exists(),'Reference output must be a new directory')
    require(a.max_seconds>0,'Deadline must be positive')
    started=time.monotonic();context={}
    def stop(*_):raise TimeoutError('Fixed frame-training deadline or termination signal; no retry/resume')
    previous_handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.alarm(a.max_seconds)
    from memory_experiment import write,event,seed_all,state_digest,evaluate
    try:
        parent,frame,records,evidence=validate_inputs(a)
        control_manifest,control_payload,control_receipt=validate_control(a,parent,frame,records)
        block,block_receipt=native_training_block(parent)
        import numpy as np
        import torch
        from native_state_cache import build_native_model,load_sample,replay
        from frame_consistent_adapter import install_frame_consistent_adapter
        seed_all(11)
        model=build_native_model(a.config,a.checkpoint,device='cuda',repo=a.repo)
        migration=install_frame_consistent_adapter(model,mode='reference')
        for parameter in model.parameters():parameter.requires_grad_(False)
        for parameter in model.future_pred_head.parameters():parameter.requires_grad_(True)
        params={name:p for name,p in model.named_parameters() if p.requires_grad}
        require(params and all(n.startswith('future_pred_head.') for n in params),'Wrong trainable scope')
        require(list(params)==control_manifest['trainable_parameter_names'] and
                sum(p.numel() for p in params.values())==control_manifest['trainable_parameters'],
                'Reference parameter scope/order/capacity differs from C')
        initial=state_digest(model.future_pred_head)
        require(initial==INITIAL_HEAD_SHA256==control_manifest['initial_head_sha256'],'Reference initial tensor values differ from C')
        native_state=model.future_pred_head.state_dict();control_state=control_payload['future_pred_head']
        require(set(native_state)==set(control_state),'Reference/C head state keys differ')
        for name,value in native_state.items():
            require(tuple(value.shape)==tuple(control_state[name].shape) and value.dtype==control_state[name].dtype,
                    'Reference/C head shape or dtype differs: '+name)
            require(torch.isfinite(control_state[name]).all().item(),'Nonfinite completed control head')
        control_optimizer_groups=copy.deepcopy(control_payload['optimizer']['param_groups'])
        del control_payload,control_state,native_state
        seed_all(11) # Exactly the C reset after initialization/migration.
        out.mkdir(parents=True,exist_ok=False)
        write(out/'control_reuse.json',control_receipt)
        hp=parent['training'];train_rows=records['train'];dev_rows=records['development']
        metadata=dict(arm='reference',seed=11,initial_head_sha256=initial,protocol_sha256=sha(a.frame_protocol),
            parent_protocol_sha256=PARENT_PROTOCOL_SHA256,m0_sha256=parent['m0_sha256'],migration=migration,
            train_index_sha256=frame['cache_index_sha256']['train'],dev_index_sha256=frame['cache_index_sha256']['development'],
            source_sha256=frame['source_sha256'],parent_source_sha256=parent['source_sha256'],
            runtime_source_sha256=frame['runtime_source_sha256'],trainable_parameters=sum(p.numel() for p in params.values()),
            trainable_parameter_names=list(params),numerical_policy=parent['numerical_policy'],
            examples_per_pass=512,passes=4,augmentation='none; identical deterministic native t0 cache in all arms',
            engineering_evidence=evidence,control_reuse=control_receipt,training_block=block_receipt,
            treatment='single R-coordinate state geometry only; original physical ego/action unchanged',
            checkpoint_selection='fixed final 512 updates; no best/dev/threshold selection',
            full_validation_automatically_requested=False)
        initial_eval=evaluate(model,a.dev_cache,dev_rows,out/'initial_development_records.jsonl')
        require(state_digest(model.future_pred_head)==initial,'Initial evaluation mutated model state')
        initial_eval.update(optimizer_updates=0,head_state_sha256=initial,role='initialization impact diagnostic, not checkpoint selection')
        metadata['initial_development']=initial_eval;write(out/'manifest.json',metadata)
        seed_all(11) # Same as C: reset AFTER full frozen initial development.
        import memory_experiment as native
        context=dict(native.__dict__)
        context.update(a=a,hp=hp,out=out,model=model,migration=migration,params=params,metadata=metadata,
            train_rows=train_rows,dev_rows=dev_rows,frozen_sha=parent['m0_sha256'],np=np,torch=torch,
            load_sample=load_sample,replay=replay,control_optimizer_groups=control_optimizer_groups,
            _ft_verify_optimizer=verify_optimizer)
        exec(block,context)
        complete=read(out/'complete.json')
        require(complete['updates']==512 and complete['examples']==2048,'Incomplete fixed reference endpoint')
        complete.update(frame_protocol_sha256=sha(a.frame_protocol),parent_protocol_sha256=PARENT_PROTOCOL_SHA256,
            manifest_sha256=sha(out/'manifest.json'),control_reuse_sha256=sha(out/'control_reuse.json'),
            initial_development_sha256=sha(out/'initial_development_records.jsonl'),
            training_log_sha256=sha(out/'training.jsonl'),final_head_state_sha256=state_digest(model.future_pred_head),
            total_process_seconds=time.monotonic()-started,new_training_seeds=1,training_seed=11,
            automatic_followup=False)
        write(out/'complete.json',complete)
        event('FRAME_TRAIN_COMPLETE',out=str(out),updates=512,examples=2048)
    except BaseException as exc:
        # Diagnostic interruption checkpoint only, never eligible for evaluation
        # or automatic resume. A prior completed-pass latest.pth remains intact.
        signal.alarm(0)
        for sig in previous_handlers:signal.signal(sig,signal.SIG_IGN)
        if out.is_dir():
            emergency=None
            if 'model' in context and 'optimizer' in context:
                try:
                    import torch
                    torch.save(dict(arm='reference',status='INTERRUPTED_NOT_FINAL_NOT_RESUMABLE',
                        metadata=context.get('metadata'),updates=context.get('update',0),examples=context.get('example_count',0),
                        future_pred_head={k:v.detach().cpu() for k,v in context['model'].future_pred_head.state_dict().items()},
                        optimizer=context['optimizer'].state_dict(),sample_orders=context.get('all_order',[])),out/'interrupted.pth')
                    emergency=sha(out/'interrupted.pth')
                except BaseException as save_error:emergency='CHECKPOINT_SAVE_FAILED: '+repr(save_error)
            write(out/'failed.json',dict(status='FAILED_NO_RETRY',error=repr(exc),seconds=time.monotonic()-started,
                completed_updates=context.get('update',0),examples=context.get('example_count',0),
                interrupted_checkpoint=emergency,automatic_retry=False,full_validation_requested=False))
        raise
    finally:
        signal.alarm(0)
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)


if __name__=='__main__':main()
