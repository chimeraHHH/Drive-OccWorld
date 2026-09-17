"""Train-only motion learnability / task-competition experiment.

All arms start from the authenticated shared v2 codec. Encoder and decoder
stay frozen, so the intervention is the objective/readout on evolving state,
not extra shape fitting. No development samples, best selection, or resume.
P optimizes 0.1 physical; J and D optimize occupancy + 0.1 physical.
P/J are rendered with transport at evaluation; D uses identity readout.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path); tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def append(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data, allow_nan=False)+'\n'); f.flush()


def run(a):
    start = time.monotonic(); p = json.loads(Path(a.protocol).read_text())
    assert sha(a.protocol) == a.protocol_sha256 and p['status'] == 'FROZEN'
    assert p['schema'] == 'dense-state-learning-curve-v1'
    for name, digest in p['sources_sha256'].items():
        assert Path(name).name == name and sha(Path(__file__).parent/name) == digest, name
    for path, digest in p['runtime_source_sha256'].items():
        assert sha(path) == digest, path
    prior = Path(a.prior_fit)
    assert sha(prior/'complete.json') == p['prior_complete_sha256']
    done = json.loads((prior/'complete.json').read_text())
    assert done['status'] == 'COMPLETE_TRAIN_ONLY_ENGINEERING_FIT'
    for name in ('manifest.json','protocol.json','shared_codec.pth'):
        expected = done['protocol_sha256'] if name == 'protocol.json' else done['files_sha256'][name]
        assert sha(prior/name) == expected, name
    assert sha(prior/'shared_codec.pth') == p['shared_codec_sha256']
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', p)
    import torch
    import torch.nn.functional as F
    import native_state_cache as native
    import train_source_motion_v1 as helper
    from dense_task_state_v2 import DenseTaskState
    native._prepare_repo(a.repo)
    lossmod = importlib.import_module('projects.mmdet3d_plugin.bevformer.dense_heads.world_head_v1')
    torch.set_num_threads(2); torch.manual_seed(11); np.random.seed(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    free, total = torch.cuda.mem_get_info(); assert free >= 32*2**30
    torch.cuda.set_per_process_memory_fraction(p['resources']['max_allocated_gib']*2**30/total)
    torch.cuda.reset_peak_memory_stats()
    stopped = []
    signal.signal(signal.SIGTERM, lambda s, f: stopped.append(s))
    signal.signal(signal.SIGINT, lambda s, f: stopped.append(s))

    def check():
        assert not stopped, 'Stop requested'
        assert time.monotonic()-start < p['resources']['max_seconds'], 'Time budget exceeded'
        assert torch.cuda.max_memory_allocated() <= p['resources']['max_allocated_gib']*2**30

    trainroot, records, receipt = helper.cache_index(a.train_cache, 'train', p)
    labelroot, labels = helper.labels_manifest(a.sparse_labels, p)
    selected = json.loads((prior/'manifest.json').read_text())['selected_ordinals']
    assert selected == p['selected_ordinals'] == [396,445,60,115]
    payload = torch.load(str(prior/'shared_codec.pth'), map_location='cpu')
    assert payload['protocol_sha256'] == done['protocol_sha256']
    shared = payload['state_dict']; del payload

    def make_model():
        torch.manual_seed(11)
        m = DenseTaskState().cuda(); m.load_state_dict(shared, strict=True)
        for module in (m.encoder, m.decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        return m

    init = make_model(); dataset = []
    for i in selected:
        check(); r = records[i]
        tokens = helper.load_tokens(trainroot, r, native, 'cuda')
        with torch.no_grad():
            first = init(tokens, 'direct')
            assert first['displacement'].count_nonzero() == 0
        path = native._sample_directory(trainroot, r)/r['files']['targets']['file']
        native._verify_file(path, r['files']['targets'])
        target = np.load(path, allow_pickle=False)[0,2:].copy()
        assert target.shape == (5,512,512,40) and np.isin(target,[0,1,255]).all()
        coarse = lossmod._downsample_occ_target(torch.from_numpy(target).cuda().long()).cpu()
        label = helper.load_sparse(labelroot, labels[r['sample_token']], r)
        dataset.append(dict(tokens=tokens.cpu(), target=target, coarse=coarse, label=label,
            sample_token=r['sample_token'], scene_token=r['scene_token']))
        del first, tokens
    del init; torch.cuda.empty_cache()
    weights = torch.tensor([1.,5.], device='cuda')

    def occ_loss(logits, target):
        logits = F.interpolate(logits, size=(256,256,20), mode='trilinear', align_corners=False)
        return (lossmod.CE_ssc_loss(logits, target, weights, ignore_index=255) +
                lossmod.lovasz_softmax(torch.softmax(logits,1), target, ignore=255))

    def grad_pair(model, d, arm):
        """Diagnostic only: autograd.grad never touches optimizer/.grad state."""
        check(); mode = 'direct' if arm == 'D' else 'transport'
        forecast = model(d['tokens'].cuda(), mode)
        occ = occ_loss(forecast['logits'][0], d['coarse'].cuda())
        phys, _ = helper.object_group_loss(helper.gather_sparse(forecast['displacement'],d['label']),d['label'])
        named = [(n,v) for n,v in model.named_parameters() if v.requires_grad]
        go = torch.autograd.grad(occ,[v for n,v in named],retain_graph=True,allow_unused=True)
        gp = torch.autograd.grad(p['physical_weight']*phys,[v for n,v in named],allow_unused=True)
        rows = {}
        for group, accept in [('all_dynamics',lambda n:True),('velocity_only',lambda n:n.startswith('velocity.'))]:
            aa=bb=ab=0.; on=pn=0
            for (name,value),left,right in zip(named,go,gp):
                if not accept(name):
                    continue
                if left is not None:
                    assert torch.isfinite(left).all(); aa += float(left.double().square().sum()); on += 1
                if right is not None:
                    assert torch.isfinite(right).all(); bb += float(right.double().square().sum()); pn += 1
                if left is not None and right is not None:
                    ab += float((left.double()*right.double()).sum())
            rows[group] = dict(occupancy_norm=aa**.5 if on else None, weighted_physical_norm=bb**.5 if pn else None,
                dot=ab, cosine=ab/(aa*bb)**.5 if aa>0 and bb>0 else None,
                occupancy_used_parameters=on, physical_used_parameters=pn)
        if arm == 'D':
            assert rows['velocity_only']['occupancy_used_parameters'] == 0
        return rows

    def evaluate(model, arm, step):
        check(); mode='direct' if arm=='D' else 'transport'
        hists={c:np.zeros((5,2,2),dtype=np.int64) for c in ['normal','zero','reverse']}
        objects=[]; losses=[]; sample_hists=[]; dense_motion=[]
        with torch.no_grad():
            for d in dataset:
                check(); tokens=d['tokens'].cuda(); f=model(tokens,mode)
                assert torch.isfinite(f['displacement']).all()
                lengths=f['displacement'][0].norm(dim=1).flatten(1)
                dense_motion.append(dict(sample_token=d['sample_token'],
                    mean_m=lengths.mean(1).cpu().tolist(),p95_m=torch.quantile(lengths,.95,dim=1).cpu().tolist(),
                    max_m=lengths.max(1).values.cpu().tolist()))
                phys,_=helper.object_group_loss(helper.gather_sparse(f['displacement'],d['label']),d['label'])
                losses.append(dict(occupancy=float(occ_loss(f['logits'][0],d['coarse'].cuda())),physical=float(phys)))
                pred=helper.gather_sparse(f['displacement'],d['label']).cpu().numpy(); lab=d['label']
                error=np.linalg.norm(pred-lab['target_displacement_m'],axis=-1)
                magnitude=np.linalg.norm(pred,axis=-1); zero=np.linalg.norm(lab['target_displacement_m'],axis=-1)
                for h in range(4):
                    for obj,g in enumerate(lab['object_speed_group'][h]):
                        if g<0:continue
                        mask=lab['valid'][h]&(lab['object_index']==obj); assert mask.any()
                        objects.append(dict(sample_token=d['sample_token'],instance_token=str(lab['instance_tokens'][obj]),
                            h=h,group=int(g),points=int(mask.sum()),epe=float(error[h,mask].mean(dtype=np.float64)),
                            magnitude=float(magnitude[h,mask].mean(dtype=np.float64)),zero_epe=float(zero[h,mask].mean(dtype=np.float64))))
                sh={}
                for condition in hists:
                    output=f if condition=='normal' else model(tokens,mode,condition)
                    if condition!='normal':assert torch.equal(output['displacement'],f['displacement'])
                    counts=[]
                    for h in range(5):
                        pred_fine=F.interpolate(output['logits'][:,h],size=(512,512,40),mode='trilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)
                        valid=d['target'][h]!=255
                        hist=np.bincount((2*d['target'][h][valid]+pred_fine[valid]).astype(np.int64),minlength=4).reshape(2,2)
                        hists[condition][h]+=hist;counts.append(hist.tolist())
                    sh[condition]=counts
                sample_hists.append(dict(sample_token=d['sample_token'],hist=sh))
        occ={}
        for condition,hist in hists.items():
            tp,fp,fn=hist[:,1,1],hist[:,0,1],hist[:,1,0]
            iou=tp/(tp+fp+fn);recall=tp/(tp+fn)
            occ[condition]=dict(hist=hist.tolist(),iou_by_horizon=iou.tolist(),recall_by_horizon=recall.tolist(),future_mean_iou=float(iou[1:].mean()))
        phys=[]
        for h in range(4):
            row=dict(horizon_seconds=(h+1)*.5)
            for g,name in enumerate(['stationary','ambiguous','moving']):
                values=[r for r in objects if r['h']==h and r['group']==g]
                row[name]=dict(objects=len(values),points=sum(r['points'] for r in values),
                    **{k:float(np.mean([r[k] for r in values])) if values else None for k in ['epe','magnitude','zero_epe']})
            phys.append(row)
        gradients=[grad_pair(model,d,arm) for d in dataset]
        append(out/'objects.jsonl',dict(arm=arm,step=step,objects=objects))
        result=dict(arm=arm,step=step,occupancy=occ,physical=phys,losses=losses,
                    gradients=gradients,sample_hists=sample_hists,dense_motion=dense_motion,seconds=time.monotonic()-start)
        append(out/'milestones.jsonl',result)
        print(json.dumps(dict(event='milestone',arm=arm,step=step,future_iou=occ['normal']['future_mean_iou'],
            moving2s=phys[-1]['moving'],stationary2s=phys[-1]['stationary'])),flush=True)
        return result

    manifest=dict(protocol_sha256=a.protocol_sha256,prior_complete_sha256=p['prior_complete_sha256'],
        cache=receipt,selected_ordinals=selected,scene_tokens=[d['scene_token'] for d in dataset],
        seed=11,development_read=0,external_models_trained=False,
        sample_dt_seconds=[d['label']['dt_future_seconds'].tolist() for d in dataset],
        sample_dt_offset_seconds=[(d['label']['dt_future_seconds']-np.array([.5,1.,1.5,2.])).tolist() for d in dataset],arms={})
    write(out/'manifest.json',manifest);summary=dict(scope='train4 learnability, not generalization or O comparison',arms={})
    for arm in ['P','J','D']:
        model=make_model(); params=[v for v in model.parameters() if v.requires_grad]
        initial=native._parameter_digest(model)
        frozen={prefix:native._parameter_digest(getattr(model,prefix)) for prefix in ['encoder','decoder']}
        manifest['arms'][arm]=dict(initial_parameters_sha256=initial,frozen_sha256=frozen,
            trainable_parameters=sum(v.numel() for v in params),all_parameters=sum(v.numel() for v in model.parameters()))
        write(out/'manifest.json',manifest)
        optimizer=torch.optim.AdamW(params,lr=p['lr'],weight_decay=.01)
        curve=[evaluate(model,arm,0)]
        for step in range(1,p['updates_per_arm']+1):
            check(); d=dataset[(step-1)%len(dataset)]
            # P's skipped render does not change displacement computation.
            mode='transport' if arm=='J' else 'direct'
            forecast=model(d['tokens'].cuda(),mode)
            physical,_=helper.object_group_loss(helper.gather_sparse(forecast['displacement'],d['label']),d['label'])
            occ=occ_loss(forecast['logits'][0],d['coarse'].cuda()) if arm!='P' else None
            loss=p['physical_weight']*physical+(occ if occ is not None else 0.)
            assert torch.isfinite(loss);loss.backward()
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in params)
            gradient_stats={}
            before_step={n:v.detach().clone() for n,v in model.named_parameters() if v.requires_grad}
            for prefix in ['dynamics.','content_increment.','velocity.']:
                gs=[v.grad for n,v in model.named_parameters() if n.startswith(prefix) and v.grad is not None]
                gradient_stats[prefix]=dict(used_parameters=len(gs),norm=float(sum(g.double().square().sum() for g in gs).sqrt()) if gs else None)
            norm=torch.nn.utils.clip_grad_norm_(params,10.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            update_norm={prefix:float(sum((v.detach()-before_step[n]).double().square().sum()
                for n,v in model.named_parameters() if n.startswith(prefix)).sqrt())
                for prefix in ['dynamics.','content_increment.','velocity.']}
            append(out/'training.jsonl',dict(arm=arm,step=step,sample_index=(step-1)%len(dataset),
                total=float(loss.detach()),physical=float(physical.detach()),occupancy=float(occ.detach()) if occ is not None else None,
                grad_norm=float(norm),gradient_stats=gradient_stats,update_norm=update_norm,seconds=time.monotonic()-start))
            del forecast,physical,occ,loss
            if step in p['milestones']:
                assert all(native._parameter_digest(getattr(model,k))==v for k,v in frozen.items())
                curve.append(evaluate(model,arm,step))
                torch.save(dict(state_dict=model.state_dict(),protocol_sha256=a.protocol_sha256,arm=arm,step=step),out/(arm+'_step'+str(step)+'.pth'))
                summary['arms'][arm]=curve;write(out/'summary.json',summary)
        del model,optimizer;torch.cuda.empty_cache()
    assert len({v['initial_parameters_sha256'] for v in manifest['arms'].values()})==1
    assert len({v['trainable_parameters'] for v in manifest['arms'].values()})==1
    reference=np.asarray(summary['arms']['P'][0]['occupancy']['normal']['hist']).sum(2)
    t0=np.asarray(summary['arms']['P'][0]['occupancy']['normal']['hist'])[0]
    for curve in summary['arms'].values():
        for point in curve:
            for value in point['occupancy'].values():
                assert np.array_equal(np.asarray(value['hist']).sum(2),reference)
                assert np.array_equal(np.asarray(value['hist'])[0],t0)
    summary.update(seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   same_gt_denominators=True,frozen_codec_and_t0_verified=True)
    write(out/'summary.json',summary)
    files=['protocol.json','manifest.json','summary.json','training.jsonl','milestones.jsonl','objects.jsonl']
    files += [path.name for path in out.glob('*.pth')]
    write(out/'complete.json',dict(status='COMPLETE_TRAIN4_LEARNING_CURVES',protocol_sha256=a.protocol_sha256,
        updates=3*p['updates_per_arm'],development_read=0,files_sha256={name:sha(out/name) for name in files}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['protocol','protocol-sha256','prior-fit','repo','train-cache','sparse-labels','out']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    try:run(args)
    except BaseException as error:
        if Path(args.out).is_dir():write(Path(args.out)/'failed.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
