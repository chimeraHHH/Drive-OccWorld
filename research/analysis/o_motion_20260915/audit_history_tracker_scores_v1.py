"""Independent direct per-object loops for the completed surface diagnostic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

n=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--evaluation',type=Path,required=True);args=parser.parse_args();root=args.evaluation
motion=n/'server_results/diagnostics/motion_evidence_train_v1'
rows=list(csv.DictReader((root/'objects.csv').open()))
lookup={(r['sample_token'],r['instance_token'],r['clock'],float(r['horizon_s']),r['arm']):r for r in rows}
assert len(lookup)==len(rows)
index=json.loads((motion/'index.json').read_text())['records']
audits=json.loads((root/'sample_audit.json').read_text())
count=0;maximum=0.;support_counts={k:0 for k in ['in_roi','quality_in_roi']};distances={k:[] for k in support_counts}
for audit in audits:
    r=index[audit['ordinal']];assert r['identity']==audit['identity']
    with np.load(motion/r['file'],allow_pickle=False) as z:v={k:z[k] for k in z.files}
    with np.load(root/'queried'/Path(r['file']).name,allow_pickle=False) as z:q={k:z[k] for k in z.files}
    assert np.array_equal(q['source_flat_indices'],v['source_flat_indices'])
    for key in support_counts:
        valid=q[key+'_support'];distance=q[key+'_distance'];vel=q[key+'_velocity']
        assert np.array_equal(valid,distance<=1.) and not np.any(vel[~valid])
        support_counts[key]+=int(valid.sum());distances[key].extend(distance[valid].tolist())
    obj=v['object_index'];uncovered=v['owner']<0
    for hi,h in enumerate([.5,1.,1.5,2.]):
        for clock,dt in [('nominal',h),('actual',v['dt_future_seconds'][hi])]:
            cv=v['CV_velocity_mps']*dt;d=v['D_displacement_m'][hi]
            a=q['in_roi_velocity']*dt;f=q['quality_in_roi_velocity']*dt
            arms={'zero':np.zeros_like(cv),'D':d,'CRN_CV':cv,
                  'CV_D_speed05':np.where((uncovered & (v['observable_features'][:,0]>=.5))[:,None],d,cv),
                  'surface_all_zero':a,'surface_quality_zero':f,
                  'CV_surface_all_uncovered':np.where(uncovered[:,None],a,cv),
                  'CV_surface_quality_uncovered':np.where(uncovered[:,None],f,cv)}
            for oi in np.flatnonzero(v['object_future_valid'][hi]):
                take=obj==oi;target=v['target_displacement_m'][hi,take].astype(float)
                for arm,pred in arms.items():
                    row=lookup[(r['identity']['sample_token'],str(v['instance_tokens'][oi]),clock,h,arm)]
                    assert int(row['points'])==int(take.sum())
                    for dim,name in [(2,'xy_epe_m'),(3,'xyz_epe_m')]:
                        error=pred[take,:dim].astype(float)-target[:,:dim]
                        value=np.sqrt((error*error).sum(axis=1)).mean()
                        delta=abs(value-float(row[name]));maximum=max(maximum,delta);count+=1
                        assert delta<1e-10
assert count==2*len(rows)
result={'status':'PASS_INDEPENDENT_OBJECT_LOOP_RECOMPUTATION','scalar_errors_checked':count,
        'maximum_absolute_difference_m':maximum,'all_original_population_retained':True,
        'surface_supported_original_points':support_counts,
        'supported_nearest_distance_m_quantiles':{k:np.quantile(v,[0,.5,.9,1]).tolist() for k,v in distances.items()},
        'notes':'Direct object loops versus bincount scoring; no model re-inference or geometry re-estimation',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (root/'independent_score_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
