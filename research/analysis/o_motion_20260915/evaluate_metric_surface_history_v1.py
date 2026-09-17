"""Score frozen image-derived motion on every original material point.

No fitting or threshold selection. Surface predictions are fixed before labels
are opened; nearest-neighbor interpolation uses geometry only, never GT owner.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
HORIZONS = (.5, 1., 1.5, 2.)
ARMS = ('zero', 'D', 'CRN_CV', 'CV_D_speed05', 'surface_all_zero',
        'surface_fb1_zero', 'CV_surface_all_uncovered', 'CV_surface_fb1_uncovered')
GROUPS = ('all', 'stationary', 'ambiguous', 'moving')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def query(points, surfaces, velocities):
    """Euclidean 3D nearest neighbor <=1m; unsupported queries return zero."""
    out = np.zeros_like(points, dtype=np.float64)
    if not len(surfaces):
        return out, np.zeros(len(points), bool), np.full(len(points), np.inf)
    distance, idx = cKDTree(surfaces).query(points, k=1, workers=1)
    supported = distance <= 1.
    out[supported] = velocities[idx[supported]]
    return out, supported, distance


def object_mean(obj, values, counts):
    return np.bincount(obj, weights=values, minlength=len(counts))/counts


def analytic_checks():
    points=np.array([[0.,0,0],[1.,0,0],[3.,0,0]])
    v,m,d=query(points,np.array([[0.,0,0]]),np.array([[2.,0,0]]))
    assert m.tolist()==[True,True,False] and v[:,0].tolist()==[2,2,0]
    assert np.allclose(object_mean(np.array([0,0,1]),np.array([1.,3.,10.]),np.array([2,1])),[2.,10.])
    return ['1m boundary retained; unsupported zero fallback', 'unequal object sizes remain object-equal']


def main(a):
    start=time.monotonic()
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'queried').mkdir()
    qa=analytic_checks()
    done=read(a.surface/'complete.json')
    assert done['status']=='COMPLETE_FROZEN_SURFACE_HISTORY' and done['optimizer_updates']==0 and not done['GT_read']
    assert sha(a.surface/'protocol.json')==done['protocol_sha256']
    assert sha(a.surface/'records.jsonl')==done['records_sha256']
    protocol=read(a.surface/'protocol.json')
    sr=[json.loads(x) for x in (a.surface/'records.jsonl').read_text().splitlines()]
    assert len(sr)==done['pairs']==6*len(protocol['anchors'])
    # Authenticate original material-point geometry and all baseline records.
    md=read(a.motion/'complete.json')
    for name,digest in md['files_sha256'].items(): assert sha(a.motion/name)==digest
    index=read(a.motion/'index.json')['records']
    by_token={r['identity']['sample_token']:r for r in index}
    points_path=HERE/'shared_rigid_geometry_cache_dev0_roundtrip_v1/points_R.npy'
    assert sha(points_path)=='91bcc80882e41ffd5234cecce4b517ef4081839c89dbb6a048c4226b723ca6fd'
    grid=np.load(points_path,allow_pickle=False)
    rule={'status':'FIXED_RULES_NO_FIT','source_sha256':sha(__file__),'surface_complete_sha256':sha(a.surface/'complete.json'),
          'baseline_complete_sha256':sha(a.motion/'complete.json'),'geometry_sha256':sha(points_path),
          'radius_m':1.,'surface_variants':['in_roi','fb1_in_roi'],'arms':list(ARMS),
          'primary_time':'nominal horizon for CV and surface, matching original D/CV protocol',
          'sensitivity':'actual future elapsed time for both CV and surface; native D unchanged',
          'population':'all original material points and all original future-valid objects; no support filtering',
          'surface_interpolation':'nearest 3D point within 1m; fixed image grid; no GT ownership or label used to choose neighbor',
          'GT_use':'original points sample a fixed input-only surface field; GT motion only scores that field',
          'baseline_is_O':False,'occupancy_evaluated':False,'selection_or_fitting':False,'checks':qa}
    write(a.out/'rules.json',rule)
    object_rows=[]; audits=[]
    for identity in protocol['anchors']:
        row=by_token[identity['sample_token']]
        assert row['identity']==identity
        candidates=[r for r in sr if r['identity']==identity]
        assert len(candidates)==6 and len({r['camera'] for r in candidates})==6
        clouds={key:[[],[]] for key in ('in_roi','fb1_in_roi')}
        # Read completed prediction arrays first; labels cannot influence them.
        for rec in candidates:
            p=a.surface/rec['file']; assert sha(p)==rec['sha256']
            with np.load(p,allow_pickle=False) as z:
                for key,(ps,vs) in clouds.items():
                    m=z[key];ps.append(z['point_t0_R_m'][m]);vs.append(z['velocity_R_mps'][m])
        p=a.motion/row['file']; assert sha(p)==row['sha256']
        with np.load(p,allow_pickle=False) as z: v={k:z[k] for k in z.files}
        assert json.loads(str(v['identity_json']))==identity
        points=grid[v['source_flat_indices']]; obj=v['object_index']; k=len(v['instance_tokens'])
        counts=np.bincount(obj,minlength=k); assert (counts>0).all()
        queried={}
        for key,(ps,vs) in clouds.items():
            ps,vs=np.concatenate(ps),np.concatenate(vs)
            assert np.isfinite(ps).all() and np.isfinite(vs).all()
            queried[key]=query(points,ps,vs)
        audit=dict(ordinal=row['ordinal'],identity=identity,source_points=len(points),objects=k,
                   valid_points=v['valid'].sum(axis=1).tolist(),valid_objects=v['object_future_valid'].sum(axis=1).tolist(),
                   nearest_support={key:int(value[1].sum()) for key,value in queried.items()},
                   predicted_surface_counts={key:sum(len(x) for x in pair[0]) for key,pair in clouds.items()},
                   actual_future_dt_s=v['dt_future_seconds'].tolist())
        save={'source_flat_indices':v['source_flat_indices']}
        for key,(velocity,support,distance) in queried.items():
            save.update({key+'_velocity':velocity,key+'_support':support,key+'_distance':distance})
        qpath=a.out/'queried'/Path(row['file']).name
        np.savez_compressed(qpath,**save); audit['queried_sha256']=sha(qpath);audits.append(audit)
        uncovered=v['owner']<0; select_D=uncovered & (v['observable_features'][:,0]>=.5)
        cover={key:object_mean(obj,values[1].astype(float),counts) for key,values in queried.items()}
        for hi,horizon in enumerate(HORIZONS):
            valid=v['object_future_valid'][hi]
            assert np.array_equal(v['valid'][hi],valid[obj])
            target=v['target_displacement_m'][hi].astype(float)
            for clock,dt in [('nominal',horizon),('actual',float(v['dt_future_seconds'][hi]))]:
                cv=v['CV_velocity_mps']*dt;d=v['D_displacement_m'][hi].astype(float)
                all_v=queried['in_roi'][0]*dt;fb_v=queried['fb1_in_roi'][0]*dt
                predictions=(np.zeros_like(d),d,cv,np.where(select_D[:,None],d,cv),all_v,fb_v,
                             np.where(uncovered[:,None],all_v,cv),np.where(uncovered[:,None],fb_v,cv))
                for arm,pred in zip(ARMS,predictions):
                    e_xy=np.linalg.norm((pred-target)[:,:2],axis=1)
                    e_xyz=np.linalg.norm(pred-target,axis=1)
                    xy=object_mean(obj,e_xy,counts);xyz=object_mean(obj,e_xyz,counts)
                    speed=object_mean(obj,np.linalg.norm(pred[:,:2]/dt,axis=1),counts)
                    for oi in np.flatnonzero(valid):
                        object_rows.append(dict(sample_token=identity['sample_token'],scene_token=identity['scene_token'],
                            instance_token=str(v['instance_tokens'][oi]),horizon_s=horizon,clock=clock,
                            group=GROUPS[int(v['object_speed_group'][hi,oi])+1],arm=arm,points=int(counts[oi]),
                            xy_epe_m=float(xy[oi]),xyz_epe_m=float(xyz[oi]),predicted_speed_mps=float(speed[oi]),
                            all_support_fraction=float(cover['in_roi'][oi]),fb1_support_fraction=float(cover['fb1_in_roi'][oi])))
    rows=[]
    for clock in ('nominal','actual'):
        for h in HORIZONS:
            for group in GROUPS:
                for arm in ARMS:
                    selected=[r for r in object_rows if r['clock']==clock and r['horizon_s']==h and r['arm']==arm and (group=='all' or r['group']==group)]
                    if not selected: continue
                    rows.append(dict(clock=clock,horizon_s=h,group=group,arm=arm,objects=len(selected),
                        points=sum(r['points'] for r in selected),
                        **{name:float(np.mean([r[name] for r in selected])) for name in
                           ('xy_epe_m','xyz_epe_m','predicted_speed_mps','all_support_fraction','fb1_support_fraction')}))
    for name,data in [('objects.csv',object_rows),('summary.csv',rows)]:
        with (a.out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write(a.out/'sample_audit.json',audits)
    result={'status':'COMPLETE_FULL_ORIGINAL_POPULATION_TRAIN16_DIAGNOSTIC','anchors':len(audits),
            'scenes':len({r['identity']['scene_token'] for r in audits}),'source_points':sum(r['source_points'] for r in audits),
            'valid_objects_by_horizon':np.sum([r['valid_objects'] for r in audits],axis=0).tolist(),
            'valid_points_by_horizon':np.sum([r['valid_points'] for r in audits],axis=0).tolist(),
            'seconds':time.monotonic()-start,'optimizer_updates':0,'O_occupancy_evaluated':False,
            'files_sha256':{name:sha(a.out/name) for name in ('rules.json','objects.csv','summary.csv','sample_audit.json')},
            'summary':rows}
    write(a.out/'complete.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='summary'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface',type=Path,required=True)
    p.add_argument('--motion',type=Path,default=HERE/'server_results/diagnostics/motion_evidence_train_v1')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
