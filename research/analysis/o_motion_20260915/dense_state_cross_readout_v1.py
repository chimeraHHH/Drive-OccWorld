"""Fixed-final train4 content/flow swap, without optimization or GT input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    start=time.monotonic();root=Path(a.run);package=Path(a.package)
    assert sha(root/'complete.json')==a.expected_complete_sha256
    done=json.loads((root/'complete.json').read_text());p=json.loads((root/'protocol.json').read_text())
    assert done['status']=='COMPLETE_TRAIN4_LEARNING_CURVES' and done['updates']==1536
    for name in ['protocol.json','manifest.json','P_step512.pth','J_step512.pth','D_step512.pth']:
        assert sha(root/name)==done['files_sha256'][name],name
    for name,digest in p['sources_sha256'].items():assert sha(package/name)==digest,name
    for path,digest in p['runtime_source_sha256'].items():assert sha(path)==digest,path
    sys.path.insert(0,str(package))
    import torch
    import torch.nn.functional as F
    import native_state_cache as native
    import train_source_motion_v1 as helper
    from dense_task_state_v2 import DenseTaskState
    from transport_ops import forward_splat_3d
    native._prepare_repo(a.repo)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    free,total=torch.cuda.mem_get_info();assert free>20*2**30
    torch.cuda.set_per_process_memory_fraction(16*2**30/total);torch.cuda.reset_peak_memory_stats()
    cache,records,_=helper.cache_index(a.train_cache,'train',p)
    models={};model_hashes={}
    for arm in ['P','J','D']:
        model=DenseTaskState().cuda().eval();payload=torch.load(str(root/(arm+'_step512.pth')),map_location='cpu')
        assert payload['arm']==arm and payload['step']==512 and payload['protocol_sha256']==done['protocol_sha256']
        model.load_state_dict(payload['state_dict'],strict=True);model.requires_grad_(False);models[arm]=model
        model_hashes[arm]=native._parameter_digest(model)
    for part in ['encoder','decoder']:
        assert len({native._parameter_digest(getattr(m,part)) for m in models.values()})==1
    combinations=[(c,f) for c in ['current','P','J','D'] for f in ['zero','P','J','D']]
    totals={c+'__'+f:np.zeros((5,2,2),dtype=np.int64) for c,f in combinations}
    samples=[];decoder=models['P'];max_error=0.
    for index in p['selected_ordinals']:
        assert time.monotonic()-start<360
        r=records[index];tokens=helper.load_tokens(cache,r,native,'cuda')
        contents={};flows={};native_predictions={};zero_predictions={};digests={}
        with torch.no_grad():
            for arm,model in models.items():
                captures={'deltas':[]}
                def origin_hook(module,args,value):captures['origin']=value.detach().clone()
                def delta_hook(module,args,value):captures['deltas'].append(value.detach().clone())
                handles=[model.encoder.register_forward_hook(origin_hook),model.content_increment.register_forward_hook(delta_hook)]
                try:forecast=model(tokens,'direct' if arm=='D' else 'transport')
                finally:
                    for handle in handles:handle.remove()
                assert len(captures['deltas'])==4
                state=captures['origin'];current=model.to_xyz(state,8)
                if 'current' not in contents:contents['current']=[current]*4
                else:assert torch.equal(current,contents['current'][0])
                future=[]
                for delta in captures['deltas']:
                    state=state+delta;future.append(model.to_xyz(state,8))
                contents[arm]=future;flows[arm]=forecast['displacement'][0]
                native_predictions[arm]=forecast['logits'][0]
                # A separate numerical reference only: never replaces the
                # once-extracted content/flow used in the matrix.
                zero_predictions[arm]=model(tokens,'direct' if arm=='D' else 'transport','zero')['logits'][0]
                digests[arm]=dict(content_sha256=native._tensor_digest(torch.stack(future,dim=1)),
                                  displacement_sha256=native._tensor_digest(flows[arm]))
            flows['zero']=torch.zeros_like(flows['P'])
            digests['current']=dict(content_sha256=native._tensor_digest(torch.stack(contents['current'],dim=1)))
            digests['zero']=dict(displacement_sha256=native._tensor_digest(flows['zero']))
            # Every combination uses these once-extracted arrays, never another
            # forward pass or labels to select a source/flow/mask.
            path=native._sample_directory(cache,r)/r['files']['targets']['file'];native._verify_file(path,r['files']['targets'])
            target=np.load(path,allow_pickle=False)[0,2:]
            t0=decoder.decode(contents['current'][0],torch.ones_like(contents['current'][0][:,:1]))
            sample_hist={};native_parity={}
            for c,f in combinations:
                key=c+'__'+f;logits=[t0[0]]
                for h in range(4):
                    source=contents[c][h]
                    if f=='zero':feature=source;coverage=torch.ones_like(source[:,:1])
                    else:
                        splat=forward_splat_3d(source,flows[f][h:h+1],extent=decoder.extent)
                        feature,coverage=splat['normalized'],splat['coverage']
                    logits.append(decoder.decode(feature,coverage)[0])
                logits=torch.stack(logits)
                if (c==f and c in ['P','J']) or (c=='D' and f=='zero'):
                    reference=native_predictions[c];error=float((logits-reference).abs().max())
                    assert torch.allclose(logits,reference,atol=1e-5,rtol=1e-5)
                    max_error=max(max_error,error);native_parity[key]=error
                if c in ['P','J','D'] and f=='zero':
                    error=float((logits-zero_predictions[c]).abs().max())
                    assert torch.allclose(logits,zero_predictions[c],atol=1e-5,rtol=1e-5)
                    max_error=max(max_error,error);native_parity[key+'_zero_reference']=error
                counts=[]
                for h in range(5):
                    fine=F.interpolate(logits[h:h+1],size=(512,512,40),mode='trilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)
                    valid=target[h]!=255
                    hist=np.bincount((2*target[h][valid]+fine[valid]).astype(np.int64),minlength=4).reshape(2,2)
                    totals[key][h]+=hist;counts.append(hist.tolist())
                sample_hist[key]=counts
        samples.append(dict(sample_token=r['sample_token'],scene_token=r['scene_token'],source_digests=digests,
                            native_parity_max_abs=native_parity,hist=sample_hist))
        print(json.dumps(dict(event='sample_complete',sample_token=r['sample_token'],seconds=time.monotonic()-start)),flush=True)
    metrics={};reference=None
    for key,hist in totals.items():
        if reference is None:reference=hist.sum(2)
        assert np.array_equal(hist.sum(2),reference)
        tp,fp,fn=hist[:,1,1],hist[:,0,1],hist[:,1,0];iou=tp/(tp+fp+fn)
        metrics[key]=dict(hist=hist.tolist(),iou_by_horizon=iou.tolist(),future_mean_iou=float(iou[1:].mean()))
    assert all(native._parameter_digest(models[arm])==digest for arm,digest in model_hashes.items())
    result=dict(status='COMPLETE_FIXED_TRAIN4_CROSS_READOUT',scope='fixed final512 train4 representation/readout attribution, not held-out or independent causal effects',
        source_sha256=sha(__file__),learning_curve_complete_sha256=sha(root/'complete.json'),
        combinations=metrics,samples=samples,checkpoint_parameter_sha256=model_hashes,parameters_unchanged=True,
        input='observed BEV only; content/flow once extracted; no GT masks or new optimization',
        native_parity_max_abs=max_error,parity_atol=1e-5,parity_rtol=1e-5,
        same_gt_denominators=True,seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    with Path(a.out).open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['run','package','repo','train-cache','out','expected-complete-sha256']:parser.add_argument('--'+name,required=True)
    main(parser.parse_args())
