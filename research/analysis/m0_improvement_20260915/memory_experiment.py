"""Whole forecasting-head training on frozen native observation states.

The unchanged M0 loss/evaluator are called directly. All arms share sample order,
seed, optimizer, loss and examples. This file never modifies the M0 checkpoint.
"""
import argparse, copy, datetime, gc, hashlib, json, math, random, time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(path, data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def event(name, **kw):
    print(json.dumps(dict(event=name, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kw), allow_nan=False), flush=True)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True


def state_digest(module):
    h=hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare_model(config, checkpoint, mode):
    from native_state_cache import build_native_model
    from observation_memory import install_observation_memory
    model=build_native_model(config, checkpoint, device='cuda')
    migration=install_observation_memory(model, mode=mode)
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.future_pred_head.parameters(): p.requires_grad_(True)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert params and all(n.startswith('future_pred_head.') for n in params)
    return model,migration,params


def cached_records(directory):
    directory=Path(directory)
    assert (directory/'complete.json').is_file(), 'Incomplete cache may not train'
    complete=json.loads((directory/'complete.json').read_text())
    assert complete['index_sha256']==sha(directory/'index.json'), 'Cache final index hash mismatch'
    d=json.loads((directory/'index.json').read_text())
    assert complete['status']=='COMPLETE_NATIVE_STATE_CACHE'
    assert d['schema']=='m0-native-state-cache-v1' and d['status']=='COMPLETE'
    assert complete['samples']==len(d['records']) and d['records']
    assert d['m0_sha256']=='0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
    assert d['source_model_sha256']=='67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5'
    return d['records']


def evaluate(model, cache, records, path):
    import torch
    from native_state_cache import load_sample,replay
    model.eval(); all_rows=[]; start=time.monotonic()
    with torch.no_grad(), Path(path).open('x') as f:
        for ordinal, record in enumerate(records):
            sample=load_sample(cache, record, device='cuda')
            outputs=replay(model,sample,training=False)
            rows=model.evaluate_occ_records(outputs[0], sample['targets'], sample['inputs']['img_metas'])
            assert len(rows)==1
            row=rows[0]
            assert row['sample_token']==record['sample_token'] and row['scene_token']==record['scene_token']
            row['hist_by_horizon']=row['hist_by_horizon'].tolist()
            f.write(json.dumps(row)+'\n'); f.flush(); all_rows.append(row)
            if (ordinal+1)%10==0: event('EVAL',completed=ordinal+1,planned=len(records),seconds=time.monotonic()-start)
            del outputs,sample
    return dict(samples=len(all_rows), seconds=time.monotonic()-start, sha256=sha(path))


def preflight(a):
    import torch
    from native_state_cache import load_sample,replay
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    records=cached_records(a.train_cache)
    assert len(records)>=2 and records[0]['scene_token']!=records[1]['scene_token']
    baseline=None; two_slots=None; two_slot_sha=None; reports=[]
    for mode in ['native1','persistent2','rolling2']:
        seed_all(11); model,migration,params=prepare_model(a.config,a.checkpoint,mode)
        initial_sha=state_digest(model.future_pred_head)
        if mode=='persistent2':two_slot_sha=initial_sha
        if mode=='rolling2':assert initial_sha==two_slot_sha, 'Two-slot initial state differs'
        sample=load_sample(a.train_cache,records[0],device='cuda')
        torch.cuda.reset_peak_memory_stats(); begin=time.monotonic()
        model.eval()
        with torch.no_grad(): pred=replay(model,sample,training=False)[0]
        torch.cuda.synchronize(); forward_s=time.monotonic()-begin
        if baseline is None:
            baseline=pred.detach().cpu()
        else:
            # h=0 and h=0.5 are equivalent to duplicated native source.
            torch.testing.assert_close(pred[:2].cpu(),baseline[:2],rtol=1e-4,atol=1e-4)
        if mode=='persistent2':two_slots=pred.detach().cpu()
        if mode=='rolling2':
            # Their queues differ only after h=1.0, so outputs through h=1.0 match.
            torch.testing.assert_close(pred[:3].cpu(),two_slots[:3],rtol=1e-5,atol=1e-5)
        max_init_difference=float((pred[:2].cpu()-baseline[:2]).abs().max())
        del pred
        captured=[]
        def hook(module,inputs,outputs):
            feat=outputs[0]
            assert feat.requires_grad
            feat.retain_grad();captured.append(feat)
        handle=model.future_pred_head.register_forward_hook(hook)
        seed_all(11);begin=time.monotonic()
        pred=replay(model,sample,training=True)[0]
        assert len(captured)==4
        # Isolate the last time step: direct earlier losses must not mask detach.
        scalar=pred[-1,-1].square().mean()
        scalar.backward();torch.cuda.synchronize()
        bptt_grad=[float(x.grad.norm()) if x.grad is not None else None for x in captured]
        assert all(x is not None and math.isfinite(x) and x>0 for x in bptt_grad), 'Last-horizon signal does not reach all earlier generated features'
        handle.remove();del pred,scalar,captured,sample
        model.zero_grad(set_to_none=True)
        optimizer=torch.optim.AdamW(list(params.values()),lr=1e-5,weight_decay=.01)
        # Two warmup updates allocate AdamW state and warm the CUDA/loss kernels.
        # Three later complete updates provide the resource profile. Discard all
        # probe weights; none of these optimized models enters candidate results.
        update_receipts=[];probe_counter=0
        monitored_name=next(n for n in params if '.transformer.' in n and params[n].numel()>256)
        before= params[monitored_name].detach().clone()
        for probe_update in range(5):
            begin=time.monotonic();optimizer.zero_grad(set_to_none=True);probe_losses=[]
            for micro in range(4):
                record=records[probe_counter % min(2,len(records))];probe_counter+=1
                sample=load_sample(a.train_cache,record,device='cuda')
                pred=replay(model,sample,training=True)[0]
                losses=model.compute_occ_loss(pred,sample['targets'])
                assert losses and all(torch.isfinite(v).all() for v in losses.values())
                total=sum(losses.values());(total/4).backward()
                probe_losses.append(float(total.detach()))
                del sample,pred,losses,total
            grad=torch.nn.utils.clip_grad_norm_(list(params.values()),35,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            seconds=time.monotonic()-begin
            row=dict(update=probe_update+1,warmup=probe_update<2,examples=4,seconds=seconds,
                     mean_loss=sum(probe_losses)/4,grad_norm=float(grad),max_memory_bytes=torch.cuda.max_memory_allocated())
            update_receipts.append(row);event('PROFILE_UPDATE',mode=mode,**row)
        assert torch.any(before!=params[monitored_name]).item(), 'Optimizer did not update transition weights'
        measured=[r['seconds'] for r in update_receipts if not r['warmup']]
        row=dict(mode=mode,migration=migration,initial_head_sha256=initial_sha,trainable_parameters=sum(p.numel() for p in params.values()),
                 forward_seconds=forward_s,update_seconds_mean=sum(measured)/len(measured),update_seconds_max=max(measured),
                 max_memory_bytes=torch.cuda.max_memory_allocated(),native_first_step_max_abs=max_init_difference,
                 last_horizon_only_gradient_norms=bptt_grad,optimizer_updates=update_receipts,
                 optimizer_parameter_checked=monitored_name,
                 optimizer_parameter_max_abs_change=float((params[monitored_name]-before).abs().max()))
        reports.append(row);write(out/'progress.json',reports);event('PREFLIGHT_ARM',**row)
        del before,optimizer,model,params;gc.collect();torch.cuda.empty_cache()
    assert reports[1]['trainable_parameters']==reports[2]['trainable_parameters']
    write(out/'complete.json',dict(status='PASS_NATIVE_TASK_GRADIENT_AND_MEMORY_PARITY',reports=reports,
          source_sha256={p:sha(Path(__file__).with_name(p)) for p in ['memory_experiment.py','native_state_cache.py','observation_memory.py']},
          numerical_policy=dict(future_matmul_tf32=False,future_cudnn_tf32=True,observer_cache_matmul_tf32=True),
          evidence='Engineering preflight only; no candidate trained or performance superiority established'))


def train(a):
    import numpy as np
    import torch
    from native_state_cache import load_sample,replay
    protocol=json.loads(Path(a.protocol).read_text()); hp=protocol['training']
    assert hp['seed']==11 and hp['scope']=='whole_future_pred_head'
    assert protocol['numerical_policy']['future_matmul_tf32'] is False and protocol['numerical_policy']['future_cudnn_tf32'] is True
    assert a.arm in ['native1','persistent2','rolling2']
    train_rows=cached_records(a.train_cache);dev_rows=cached_records(a.dev_cache)
    assert len(train_rows)==hp['train_samples'] and len(dev_rows)==hp['development_samples']
    assert sha(a.config)==protocol['config_sha256']
    selection_path=Path(a.protocol).with_name('selection_v1.json')
    assert sha(selection_path)==protocol['selection_sha256']
    planned=json.loads(selection_path.read_text())['records']
    keys=['sample_token','scene_token','official_index','split']
    for actual,role in [(train_rows,'train'),(dev_rows,'development')]:
        expected=[r for r in planned if r['split']==role]
        assert [[r[k] for k in keys] for r in actual]==[[r[k] for k in keys] for r in expected], 'Cache identities/order/role differ from frozen selection'
    for cache,role in [(a.train_cache,'train'),(a.dev_cache,'development')]:
        index=json.loads((Path(cache)/'index.json').read_text())
        assert index['split']==role
        assert index['config_sha256']==protocol['config_sha256']
        assert index['selection_sha256']==protocol['selection_sha256']
        assert index['extractor_sha256']==protocol['source_sha256']['native_state_cache.py']
        assert index['future_occupancy_is_input'] is False and index['inputs_targets_separate'] is True
        assert index['native_model']['radar_contract']=='native_loader_B_without_common_source_override'
        assert all(r['parity']['bitwise_all_five_horizons_three_layers'] and r['parity']['exact_native_confusion'] for r in index['records'])
    for file_name,expected_sha in protocol['source_sha256'].items():
        assert sha(Path(__file__).with_name(file_name))==expected_sha, 'Frozen source changed: '+file_name
    assert not set(r['scene_token'] for r in train_rows)&set(r['scene_token'] for r in dev_rows)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    seed_all(11); model,migration,params=prepare_model(a.config,a.checkpoint,a.arm)
    seed_all(11)  # Same training RNG start after architecture-specific initialization.
    frozen_sha=sha(a.checkpoint); model_sha=sha(Path(__file__).with_name('memory_experiment.py'))
    assert frozen_sha=='0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
    metadata=dict(arm=a.arm,seed=11,initial_head_sha256=state_digest(model.future_pred_head),protocol_sha256=sha(a.protocol),m0_sha256=frozen_sha,migration=migration,
                  train_index_sha256=sha(Path(a.train_cache)/'index.json'),dev_index_sha256=sha(Path(a.dev_cache)/'index.json'),
                  source_sha256={p:sha(Path(__file__).with_name(p)) for p in ['memory_experiment.py','native_state_cache.py','observation_memory.py']},
                  trainable_parameters=sum(p.numel() for p in params.values()),trainable_parameter_names=list(params),
                  numerical_policy=protocol['numerical_policy'],examples_per_pass=len(train_rows),passes=hp['passes'],augmentation='none; identical deterministic native t0 cache in all arms')
    if a.arm=='native1':
        reference=evaluate(model,a.dev_cache,dev_rows,out/'frozen_native_fp32_records.jsonl')
        assert state_digest(model.future_pred_head)==metadata['initial_head_sha256']
        reference.update(initial_head_sha256=metadata['initial_head_sha256'],source_m0_sha256=frozen_sha,
                         optimizer_updates=0,matmul_tf32=False,cudnn_tf32=True)
        metadata['frozen_precision_reference']=reference
    write(out/'manifest.json',metadata)
    seed_all(11)  # Reset after the native-only frozen reference evaluation.
    optimizer=torch.optim.AdamW(list(params.values()),lr=hp['lr'],weight_decay=hp['weight_decay'])
    accumulated=hp['accumulate']; assert len(train_rows)%accumulated==0
    total_updates=hp['passes']*len(train_rows)//accumulated
    order_rng=np.random.RandomState(11); update=0;example_count=0;start=time.monotonic();all_order=[]
    log=(out/'training.jsonl').open('x')
    try:
        for epoch in range(hp['passes']):
            order=order_rng.permutation(len(train_rows)).tolist();all_order.append(order)
            window=[];optimizer.zero_grad(set_to_none=True)
            for local_step,index in enumerate(order):
                lr_scale=(0.1+0.9*update/max(1,hp['warmup_updates'])) if update<hp['warmup_updates'] else (0.1+0.9*0.5*(1+math.cos(math.pi*(update-hp['warmup_updates'])/max(1,total_updates-hp['warmup_updates']))))
                for group in optimizer.param_groups:group['lr']=hp['lr']*lr_scale
                begin=time.monotonic(); sample=load_sample(a.train_cache,train_rows[index],device='cuda')
                outputs=replay(model,sample,training=True)
                losses=model.compute_occ_loss(outputs[0],sample['targets'])
                total=sum(losses.values()); assert torch.isfinite(total).item(), 'Nonfinite task loss; stop'
                (total/accumulated).backward(); example_count+=1
                window.append(float(total.detach()))
                del outputs,sample,losses,total
                if (local_step+1)%accumulated==0:
                    grad=torch.nn.utils.clip_grad_norm_(list(params.values()),hp['grad_clip'],error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True);update+=1
                    row=dict(pass_index=epoch+1,update=update,examples=example_count,loss_mean=float(np.mean(window)),
                             loss_min=min(window),loss_max=max(window),grad_norm=float(grad),lr=optimizer.param_groups[0]['lr'],
                             seconds=time.monotonic()-start,example_last_seconds=time.monotonic()-begin)
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();window=[]
                    write(out/'progress.json',dict(status='TRAINING',planned_updates=total_updates,**row))
                    if update%10==0:event('TRAIN',**row)
            # Fixed final checkpoint selection; intermediate snapshots are for recovery only.
            state=dict(arm=a.arm,metadata=metadata,pass_index=epoch+1,update=update,
                       future_pred_head={k:v.detach().cpu() for k,v in model.future_pred_head.state_dict().items()},
                       optimizer=optimizer.state_dict(),sample_orders=all_order)
            target=out/'latest.pth';temp=out/'latest.tmp.pth';torch.save(state,temp);temp.replace(target);del state
        assert update==total_updates
        write(out/'training_complete.json',dict(status='TRAINED_FIXED_FINAL',updates=update,examples=example_count,
                seconds=time.monotonic()-start,checkpoint_sha256=sha(out/'latest.pth')))
        eval_receipt=evaluate(model,a.dev_cache,dev_rows,out/'development_records.jsonl')
        assert sha(a.checkpoint)==frozen_sha
        write(out/'complete.json',dict(status='TRAINED_AND_DEVELOPMENT_EVALUATED',updates=update,examples=example_count,
                seconds=time.monotonic()-start,checkpoint_sha256=sha(out/'latest.pth'),evaluation=eval_receipt,
                performance_claim='No conclusion before paired aggregation and controls'))
    finally:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','train'],required=True)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--train-cache',required=True)
    p.add_argument('--dev-cache');p.add_argument('--protocol');p.add_argument('--arm');p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.mode=='preflight':preflight(a)
    else:train(a)
