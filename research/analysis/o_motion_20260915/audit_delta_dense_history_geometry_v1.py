"""Independently reconstruct dense tracker geometry and LSQ from raw outputs.

Uses explicit K solves and np.linalg.lstsq, not the inference lift/fit helpers.
No target data, model loading, or GPU access.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    done=read(a.dense/'complete.json')
    assert done['status']=='COMPLETE_DENSE_DELTA_ENDPOINT_AND_LSQ_HISTORY'
    assert done['pairs']==96 and done['anchors']==16 and done['optimizer_updates']==0 and not done['GT_read']
    hd=read(a.history/'complete.json')
    for f,h in hd['files_sha256'].items():assert sha(a.history/f)==h
    rows=read(a.history/'records.json')['records']
    methods=('delta_dense_endpoint','delta_dense_lsq')
    records={}
    for name,root in [(x,a.dense/x) for x in methods]+[('twoframe',a.twoframe)]:
        complete=read(root/'complete.json')
        assert complete['status']=='COMPLETE_FROZEN_HISTORY_TRACKER'
        assert sha(root/'protocol.json')==complete['protocol_sha256']
        assert sha(root/'records.jsonl')==complete['records_sha256']
        records[name]=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
        assert len(records[name])==complete['pairs']==96
    maximum={k:0. for k in ('raw_points_m','endpoint_velocity_mps','lsq_velocity_mps','endpoint_t0_m','lsq_t0_m')}
    masks={name:{key:0 for key in ('in_roi','quality_in_roi')} for name in methods}
    for i,rec in enumerate(records['delta_dense_endpoint']):
        row=next(r for r in rows if r['identity']==rec['identity'])
        pair=next(p for p in row['cameras'] if p['channel']==rec['camera'])
        frames=pair['frames'];arrays={}
        for name in (*methods,'twoframe'):
            rr=records[name][i]
            assert (rr['identity'],rr['camera'])==(rec['identity'],rec['camera'])
            root=a.twoframe if name=='twoframe' else a.dense/name
            assert sha(root/rr['file'])==rr['sha256']
            with np.load(root/rr['file'],allow_pickle=False) as z:arrays[name]={k:z[k] for k in z.files}
        ep,ls,old=(arrays[x] for x in (*methods,'twoframe'))
        for key in ('uv','depth_current_m','point_current_R_m','camera_dt_s','exposure_to_t0_s'):
            assert np.array_equal(ep[key],ls[key]) and np.array_equal(ep[key],old[key]),key
        for key in ('raw_track_uv','raw_track_depth_m','raw_track_points_R_m','relative_frame_times_s','visibility_all','confidence_all'):
            assert np.array_equal(ep[key],ls[key],equal_nan=True),key
        uv=ep['raw_track_uv'];depth=ep['raw_track_depth_m'];T,N=depth.shape
        assert T==len(frames) and N==1400
        points=[]
        for t,f in enumerate(frames):
            q=ep['uv'] if t==T-1 else uv[t]
            d=ep['depth_current_m'] if t==T-1 else depth[t]
            rays=np.linalg.solve(np.asarray(f['K']),np.concatenate([q,np.ones((N,1))],1).T).T
            transform=np.linalg.solve(np.asarray(row['lidar_to_global']),np.asarray(f['camera_to_global']))
            xyz=(np.concatenate([rays*d[:,None],np.ones((N,1))],1)@transform.T)[:,:3]
            points.append(xyz)
        points=np.stack(points)
        delta=float(np.nanmax(abs(points-ep['raw_track_points_R_m'])))
        maximum['raw_points_m']=max(maximum['raw_points_m'],delta);assert delta<1e-9
        times=np.asarray([(f['sample_data']['timestamp']-frames[-1]['sample_data']['timestamp'])/1e6 for f in frames])
        assert np.array_equal(times,ep['relative_frame_times_s']) and np.all(np.diff(times)>0)
        assert float(ep['camera_dt_s'])==-times[0]
        velocities={'delta_dense_endpoint':(points[-1]-points[0])/-times[0],
                    'delta_dense_lsq':np.linalg.lstsq(times[:,None],(points-points[-1]).reshape(T,-1),rcond=None)[0].reshape(N,3)}
        offset=(row['t0_lidar_us']-frames[-1]['sample_data']['timestamp'])/1e6
        for name,tag in zip(methods,('endpoint','lsq')):
            z=arrays[name];v=velocities[name];p0=points[-1]+offset*v
            for label,actual,expected in [(tag+'_velocity_mps',z['velocity_R_mps'],v),(tag+'_t0_m',z['point_t0_R_m'],p0)]:
                diff=float(np.nanmax(abs(actual.astype(float)-expected)))
                maximum[label]=max(maximum[label],diff)
                assert np.allclose(actual,expected,rtol=2e-7,atol=2e-6,equal_nan=True),(label,diff)
            valid_frames=np.isfinite(points).all(2)&(depth>0)&(depth<300)&(uv[:,:,0]>=0)&(uv[:,:,0]<=1599)&(uv[:,:,1]>=0)&(uv[:,:,1]<=899)
            selected=[0,T-1] if tag=='endpoint' else list(range(T))
            valid=valid_frames[selected].all(0)&np.isfinite(v).all(1)
            roi=valid&(np.abs(p0[:,:2])<51.2).all(1)&(p0[:,2]>-5)&(p0[:,2]<3)
            quality=roi&z['visibility_all'][selected].all(0)
            for key,value in [('correspondence_valid',valid),('in_roi',roi),('quality_in_roi',quality)]:
                assert np.array_equal(z[key],value),(name,key)
            for key in masks[name]:masks[name][key]+=int(z[key].sum())
    result=dict(status='PASS_INDEPENDENT_DENSE_GEOMETRY_AND_SOURCE_MATCHING',pairs=96,
                exact_twoframe_current_queries_depths_points=True,source_queries=96*1400,
                maximum_absolute_errors=maximum,surface_mask_counts=masks,
                lsq='independent numpy.linalg.lstsq; current anchor fixed; all actual timestamps',
                mask_criteria_differ=True,GT_read=False,source_sha256=sha(Path(__file__)))
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('dense','twoframe','history','out'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
