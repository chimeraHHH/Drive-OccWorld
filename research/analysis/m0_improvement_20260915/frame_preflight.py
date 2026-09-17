"""Engineering checks for a reference-frame state, on two fixed train scenes.

Probe weights are discarded. This is not a candidate training/effect result.
The original cache, targets, checkpoint and eight campaign sources are read-only.
"""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import signal
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','config','checkpoint','train-cache','protocol','out']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--max-seconds',type=int,default=1200)
    a=p.parse_args()
    def timeout(*_):raise TimeoutError('Engineering preflight deadline reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(a.max_seconds)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    from memory_experiment import sha,write,seed_all,state_digest,cached_records,event
    try:
        protocol=json.loads(Path(a.protocol).read_text())
        assert sha(a.protocol)=='071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
        assert sha(a.config)==protocol['config_sha256'] and sha(a.checkpoint)==protocol['m0_sha256']
        for name,digest in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
        for name,digest in protocol['runtime_source_sha256'].items():assert sha(name)==digest
        assert sha(Path(a.train_cache)/'index.json')=='1dc14643df86f85c5ae87b17b83c60cada99f189a025675d150062acb046e7b1'
        import numpy as np
        import torch
        from native_state_cache import build_native_model,load_sample,replay,_tree_map
        from frame_consistent_adapter import install_frame_consistent_adapter
        from joint_native_evaluation import tree_digest
        records=[];scenes=set()
        for row in cached_records(a.train_cache):
            assert row['split']=='train'
            if row['scene_token'] not in scenes:records.append(row);scenes.add(row['scene_token'])
            if len(records)==2:break
        assert len(records)==2 and len(scenes)==2
        manifest=dict(status='ENGINEERING_PREFLIGHT',selection_rule='first anchor of first two distinct frozen training-cache scenes; no score selection',
            identities=[{k:r[k] for k in ['sample_token','scene_token','official_index','split']} for r in records],
            source_sha256={name:sha(Path(__file__).with_name(name)) for name in ['frame_preflight.py','frame_consistent_adapter.py','joint_native_evaluation.py']},
            protocol_sha256=sha(a.protocol),m0_sha256=sha(a.checkpoint),seed=11,
            precision=protocol['numerical_policy'],probe_updates_per_mode=4,accumulation=4,probe_weights_saved=False)
        write(out/'manifest.json',manifest)
        native={};reports=[]
        for mode in ['native1','reference']:
            seed_all(11);model=build_native_model(a.config,a.checkpoint,repo=a.repo)
            head_sha=state_digest(model.future_pred_head)
            assert head_sha=='6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'
            original_keys={k:(tuple(v.shape),str(v.dtype)) for k,v in model.state_dict().items()}
            migration=install_frame_consistent_adapter(model,mode=mode)
            assert state_digest(model.future_pred_head)==head_sha
            assert original_keys=={k:(tuple(v.shape),str(v.dtype)) for k,v in model.state_dict().items()}
            for v in model.parameters():v.requires_grad_(False)
            for v in model.future_pred_head.parameters():v.requires_grad_(True)
            params={n:v for n,v in model.named_parameters() if v.requires_grad}
            assert params and all(n.startswith('future_pred_head.') for n in params)
            traces=[]
            def before_head(module,args,kwargs):
                traces.append(dict(horizon=int(args[2]),action=tree_digest(args[3]),condition=tree_digest(args[4]),
                    targets=kwargs['tgt_points'].detach().cpu().clone(),references=kwargs['ref_points'].detach().cpu().clone()))
            handle=model.future_pred_head.register_forward_pre_hook(before_head,with_kwargs=True)
            forward_checks=[]
            for record in records:
                for identity in [False,True]:
                    sample=load_sample(a.train_cache,record,device='cuda')
                    if identity:
                        # Isolate the geometry contract using the same real observed state.
                        # This synthetic pose intervention is never a benchmark sample.
                        for meta in sample['inputs']['img_metas']:
                            for key in ['future2ref_lidar_transform','ref2future_lidar_transform']:
                                meta[key]=[np.eye(4,dtype=np.asarray(v).dtype) for v in meta[key]]
                        for metas in sample['inputs']['prev_img_metas']:
                            entries=metas.values() if isinstance(metas,dict) else metas
                            for meta in entries:meta['ref_lidar_to_cur_lidar']=np.eye(4,dtype=np.asarray(meta['ref_lidar_to_cur_lidar']).dtype)
                    inputs_digest=tree_digest(sample['inputs']);target_digest=tree_digest(sample['targets'])
                    traces.clear();seed_all(11);prediction=replay(model,sample,training=False)[0]
                    assert torch.isfinite(prediction).all() and len(traces)==4
                    assert tree_digest(sample['inputs'])==inputs_digest and tree_digest(sample['targets'])==target_digest
                    key=(record['sample_token'],identity);pred_cpu=prediction.detach().cpu()
                    row=dict(sample_token=key[0],identity_pose_intervention=identity,input_sha256=inputs_digest,target_sha256=target_digest)
                    if mode=='native1':native[key]=(pred_cpu,copy.deepcopy(traces))
                    else:
                        baseline,native_trace=native[key]
                        torch.testing.assert_close(pred_cpu[0],baseline[0],rtol=0,atol=0)
                        for t,c in zip(traces,native_trace):
                            assert t['horizon']==c['horizon'] and t['action']==c['action'] and t['condition']==c['condition']
                            torch.testing.assert_close(t['targets'],c['targets'],rtol=0,atol=0)
                            expected=t['targets'].unsqueeze(2)
                            torch.testing.assert_close(t['references'],expected,rtol=0,atol=1e-6)
                        if identity:torch.testing.assert_close(pred_cpu,baseline,rtol=1e-4,atol=1e-4)
                        row.update(max_logit_delta_vs_native=float((pred_cpu-baseline).abs().max()),
                            physical_action_condition_identical=True,reference_frame_grid_verified=True,
                            geometry_max_delta_vs_native=max(float((t['references']-c['references']).abs().max()) for t,c in zip(traces,native_trace)))
                    forward_checks.append(row);del sample,prediction,pred_cpu
            if mode=='reference':
                assert any(r['geometry_max_delta_vs_native']>1e-5 for r in forward_checks if not r['identity_pose_intervention']), 'No real nonidentity geometry intervention observed'
            handle.remove();traces.clear()
            # Last-horizon-only autograd test cannot be rescued by earlier losses.
            sample=load_sample(a.train_cache,records[0],device='cuda')
            sample['inputs']['prev_bev_input'].requires_grad_(True);features=[]
            def retain(module,args,outputs):
                f=outputs[0];f.retain_grad();features.append(f)
            handle=model.future_pred_head.register_forward_hook(retain)
            seed_all(11);pred=replay(model,sample,training=True)[0]
            loss=pred[-1,-1].square().mean();loss.backward();torch.cuda.synchronize()
            gradients=[float(v.grad.norm()) if v.grad is not None else None for v in features]
            observed_gradient=float(sample['inputs']['prev_bev_input'].grad.norm())
            assert len(gradients)==4 and all(v is not None and math.isfinite(v) and v>0 for v in gradients+[observed_gradient])
            handle.remove();del pred,loss,sample,features;model.zero_grad(set_to_none=True)
            # Same fixed small-data optimization probe, never a saved candidate.
            seed_all(11);optimizer=torch.optim.AdamW(list(params.values()),lr=1e-5,weight_decay=.01)
            monitored=next(n for n in params if '.transformer.' in n and params[n].numel()>256)
            before=params[monitored].detach().clone();updates=[];torch.cuda.reset_peak_memory_stats()
            for update in range(4):
                begin=time.monotonic();optimizer.zero_grad(set_to_none=True);values=[]
                for micro in range(4):
                    sample=load_sample(a.train_cache,records[micro%2],device='cuda')
                    pred=replay(model,sample,training=True)[0];losses=model.compute_occ_loss(pred,sample['targets'])
                    assert losses and all(torch.isfinite(v).all() for v in losses.values())
                    total=sum(losses.values());(total/4).backward();values.append(float(total.detach()))
                    del sample,pred,losses,total
                norm=torch.nn.utils.clip_grad_norm_(list(params.values()),35,error_if_nonfinite=True)
                monitored_grad=params[monitored].grad
                assert monitored_grad is not None and torch.isfinite(monitored_grad).all()
                monitored_grad_norm=float(monitored_grad.norm());assert monitored_grad_norm>0
                optimizer.step();torch.cuda.synchronize()
                row=dict(update=update+1,mean_loss=sum(values)/4,grad_norm=float(norm),monitored_parameter_grad_norm=monitored_grad_norm,seconds=time.monotonic()-begin)
                updates.append(row);event('FRAME_PROBE_UPDATE',mode=mode,**row)
            change=float((params[monitored]-before).abs().max());assert math.isfinite(change) and change>0
            report=dict(mode=mode,migration=migration,initial_head_sha256=head_sha,trainable_parameters=sum(v.numel() for v in params.values()),
                forward_checks=forward_checks,last_horizon_gradient_norms=gradients,observed_t0_gradient_norm=observed_gradient,
                optimizer_probe=updates,update_seconds_mean_warm=sum(r['seconds'] for r in updates[1:])/3,
                maximum_cuda_bytes=torch.cuda.max_memory_allocated(),monitored_parameter=monitored,parameter_max_abs_change=change)
            reports.append(report);write(out/'progress.json',reports)
            del before,optimizer,model,params;gc.collect();torch.cuda.empty_cache()
        assert reports[0]['trainable_parameters']==reports[1]['trainable_parameters']
        assert sha(a.checkpoint)==manifest['m0_sha256']
        write(out/'complete.json',dict(status='PASS_FRAME_ENGINEERING_PREFLIGHT',reports=reports,
            manifest_sha256=sha(out/'manifest.json'),seconds=time.monotonic()-started,performance_claim=False,
            candidate_checkpoint_saved=False,meaning='Geometry, information-preservation, gradient and resource checks only'))
    except BaseException as exc:
        write(out/'failed.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-started));raise


if __name__=='__main__':main()
