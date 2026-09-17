"""Independently recompute prespecified no-radar future AUCs from saved arrays.

No production analysis/scoring code is imported. Weighted AUC is calculated by
searching cumulative negative mass at each positive score, with half credit for
ties. This audit excludes privileged-past and FB summary recomputation.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def load(root, row):
    p = root/row['file']
    assert sha(p) == row['sha256'], p
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def auc(score, benefit, w):
    positive, negative = benefit > 0, benefit < 0
    pw, nw = w[positive].sum(), w[negative].sum()
    if not pw or not nw:
        return None
    order = np.argsort(score[negative], kind='stable')
    ns = score[negative][order]
    prefix = np.concatenate(([0.], np.cumsum(w[negative][order])))
    ps = score[positive]
    left = np.searchsorted(ns, ps, side='left')
    right = np.searchsorted(ns, ps, side='right')
    return float(np.dot(w[positive], .5*(prefix[left]+prefix[right]))/(pw*nw))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--analysis', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    started = time.monotonic()
    analysis = read(a.analysis)
    roots = {k: a.root/v for k, v in dict(flow='history_frozen_flow_evidence_train_shared_v2',
        motion='motion_evidence_train_v1', radar='radar_evidence_train_v1', camera='history_camera_evidence_train_v1').items()}
    assert sha(roots['flow']/'complete.json') == '27374a36bb06a55d0d03429f8f28a22c5b9da8b7fb66c0afbe5a0b59d6941917'
    indexes = {}
    for name, root in roots.items():
        endpoint = read(root/'complete.json')
        for file, digest in endpoint['files_sha256'].items():
            assert sha(root/file) == digest
        indexes[name] = read(root/'index.json')['records']
        assert len(indexes[name]) == 512
    names = ['flow_score_px','broken_flow_score_px','score_true','score_broken','lsq_speed_xy_mps','hypothesis_flow_separation_px']
    h = np.asarray([.5,1.,1.5,2.])
    parts = [dict(scores=[],benefit=[],weight=[],group=[],reason=[]) for _ in h]
    source_points = 0
    for i in range(512):
        rows = {k: v[i] for k,v in indexes.items()}
        assert all(r['identity'] == rows['flow']['identity'] for r in rows.values())
        v = {k: load(roots[k], row) for k,row in rows.items()}
        flow,motion,radar,camera = [v[k] for k in ('flow','motion','radar','camera')]
        assert np.array_equal(flow['source_flat_indices'],motion['source_flat_indices'])
        assert np.array_equal(flow['reason_code'],camera['reason_code'])
        idx = motion['source_flat_indices'];source_points += len(idx)
        xx,yy,_ = np.unravel_index(idx,(200,200,16))
        absent = radar['radar_bev'][0,0,yy,xx] == 0
        obj = motion['object_index'];sizes = np.bincount(obj,minlength=len(motion['instance_tokens']))
        weight = 1./sizes[obj]
        d = motion['D_displacement_m'].astype(np.float64)
        vel = np.sum(h[:,None,None]*d,axis=0)/np.sum(h*h)
        speed = np.sqrt(np.sum(vel[:,:2]**2,axis=1))
        scores = np.full((len(idx),6),np.nan)
        reason = flow['reason_code'];eligible = np.isin(reason,[0,4])
        cur,past = camera['uv'][eligible,:,0],camera['uv'][eligible,:,1]
        for col, field in [(0,'sampled_flow'),(1,'sampled_broken_flow')]:
            diff = cur + flow[field][eligible]-past
            residual = np.sqrt(np.sum(diff*diff,axis=-1))
            scores[eligible,col] = residual[:,0]-residual[:,1]
        scores[:,2] = camera['score'];scores[:,3] = camera['broken_score'];scores[:,4]=speed
        delta = past-cur
        scores[eligible,5]=np.sqrt(np.sum((delta[:,1]-delta[:,0])**2,axis=-1))
        assert np.array_equal(scores[eligible,0],flow['score'][eligible])
        assert np.array_equal(scores[eligible,1],flow['broken_score'][eligible])
        for hi in range(4):
            take = motion['valid'][hi] & (motion['owner']<0) & absent
            y = motion['target_displacement_m'][hi].astype(np.float64)
            benefit = np.sqrt(np.sum(y[:,:2]**2,axis=1))-np.sqrt(np.sum((d[hi,:,:2]-y[:,:2])**2,axis=1))
            values = dict(scores=scores[take],benefit=benefit[take],weight=weight[take],
                          group=motion['object_speed_group'][hi,obj[take]],reason=reason[take])
            for key,val in values.items():parts[hi][key].append(val)
    assert source_points == 1740053
    checks=[];maximum=0.
    for hi,item in enumerate(parts):
        e={k:np.concatenate(v) for k,v in item.items()}
        frozen=analysis['horizons'][hi]['strata']['no_radar']
        for group,gg in frozen.items():
            groupmask=np.ones(len(e['reason']),bool) if group=='all' else e['group']==('stationary','ambiguous','moving').index(group)
            for record in gg['bins']:
                mask=groupmask.copy();bi=record['bin_index']
                if bi is not None:
                    edges=[0,.1,.5,1,2,5,10,np.inf]
                    mask &= (e['scores'][:,4]>=edges[bi]) & (e['scores'][:,4]<edges[bi+1])
                for branch,reason,cols in [('primary_photo_valid',0,range(6)),('low_texture_extension',4,[0,1,4,5])]:
                    selected=mask & (e['reason']==reason)
                    ref=record[branch]['primary_original_future_benefit_auc']
                    assert int(selected.sum())==ref['points']
                    b,w=e['benefit'][selected],e['weight'][selected]
                    for ci in cols:
                        score=e['scores'][selected,ci]
                        assert np.isfinite(score).all()
                        val=auc(score,b,w);target=ref['features_same_population'][names[ci]]
                        assert int((b>0).sum())==target['positive_points'] and int((b<0).sum())==target['negative_points']
                        if val is None:assert target['auc'] is None;error=0.
                        else:
                            error=abs(val-target['auc']);assert error<1e-10,(hi,group,bi,branch,names[ci],val,target['auc'])
                        maximum=max(maximum,error)
                        checks.append(dict(horizon=float(h[hi]),group=group,bin=bi,branch=branch,feature=names[ci],auc=val,abs_difference=error))
    result=dict(status='PASS_INDEPENDENT_NO_RADAR_FUTURE_AUC_RECOMPUTATION',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(__file__),analysis_sha256=sha(a.analysis),samples=512,source_points=source_points,
        comparisons=len(checks),max_abs_auc_difference=maximum,checks=checks,seconds=time.monotonic()-started,
        scope='All four horizons, prespecified no-radar groups and speed bins, primary/low-texture separately; score algebra and original object weights recomputed',
        limitations=['Privileged-past AUC and FB summaries not independently recomputed here','No raw-image flow re-inference','Training-internal descriptive evidence, not model gain'],
        model_forward=False,training=False,threshold_selection=False)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__=='__main__':main()
