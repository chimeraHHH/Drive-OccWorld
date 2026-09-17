"""Official frozen DELTA with every authenticated intermediate history frame.

Endpoint and anchored least-squares velocity use the same fixed tracks.
Original query grid, endpoints, depth source and calibration are retained.
Additional input frames are explicit; no GT or future frames are read.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

from extract_metric_surface_history_v1 import (authenticate, load_models, infer_depth,
    lift, sample, sha, read, write, CAMERAS)

DELTA_SHA='7da306765904ec0b02e9cc8a33406250818680a1ad0b63384ae73cd58d4ef6cc'


def field(row,pair,uv,current_depth,past_uv,past_depth,quality):
    g0=np.asarray(row['lidar_to_global'],float)
    cur,past=pair['current'],pair['past']
    current_point=lift(uv,current_depth,np.asarray(cur['K']),np.linalg.inv(g0)@np.asarray(cur['camera_to_global']))
    past_point=lift(past_uv,past_depth,np.asarray(past['K']),np.linalg.inv(g0)@np.asarray(past['camera_to_global']))
    dt=(cur['sample_data']['timestamp']-past['sample_data']['timestamp'])/1e6
    offset=(row['t0_lidar_us']-cur['sample_data']['timestamp'])/1e6
    velocity=(current_point-past_point)/dt
    p0=current_point+offset*velocity
    inside=(past_uv[:,0]>=0)&(past_uv[:,0]<=1599)&(past_uv[:,1]>=0)&(past_uv[:,1]<=899)
    valid=inside&np.isfinite(velocity).all(1)&(current_depth>0)&(current_depth<300)&(past_depth>0)&(past_depth<300)
    roi=valid&(np.abs(p0[:,:2])<51.2).all(1)&(p0[:,2]>-5)&(p0[:,2]<3)
    return dict(uv=uv.astype('float32'),past_uv=past_uv.astype('float32'),
                depth_current_m=current_depth.astype('float32'),depth_past_m=past_depth.astype('float32'),
                point_current_R_m=current_point.astype('float32'),point_past_R_m=past_point.astype('float32'),
                point_t0_R_m=p0.astype('float32'),velocity_R_mps=velocity.astype('float32'),
                correspondence_valid=valid,in_roi=roi,quality_in_roi=roi&quality,
                camera_dt_s=np.asarray(dt),exposure_to_t0_s=np.asarray(offset))


def anchored_velocity(points,times):
    t=np.asarray(times,float)
    assert t[-1]==0 and np.all(np.diff(t)>0)
    return np.sum(t[:,None,None]*(points-points[-1:]),axis=0)/np.sum(t*t)


def main(a):
    from PIL import Image
    start=time.monotonic();a.out.mkdir(parents=True,exist_ok=False)
    dm,rm=authenticate(a)
    manifest=read(a.delta_asset/'source_manifest.json')
    assert manifest['commit']=='3367cda1c74d19e73296165f9826b213211678dd'
    for rec in manifest['files']:assert sha(a.delta_asset/'source'/rec['file'])==rec['sha256']
    for f,h in read(a.delta_asset/'dependency_manifest.json')['files'].items():assert sha(a.delta_asset/'deps'/f)==h
    assert sha(a.delta_asset/'densetrack3d.pth')==DELTA_SHA
    old_done=read(a.twoframe/'complete.json')
    assert old_done['status']=='COMPLETE_FROZEN_HISTORY_TRACKER'
    assert sha(a.twoframe/'records.jsonl')==old_done['records_sha256']
    old_records=[json.loads(s) for s in (a.twoframe/'records.jsonl').read_text().splitlines()]
    dense_done=read(a.dense_history/'complete.json')
    assert dense_done['status']=='COMPLETE_AUTHENTICATED_DENSE_HISTORY_INPUTS'
    for f,h in dense_done['files_sha256'].items():assert sha(a.dense_history/f)==h
    rows=read(a.dense_history/'records.json')['records'][:a.anchors]
    original=read(a.history/'records.json')['records'][:a.anchors]
    assert [r['identity'] for r in rows]==[r['identity'] for r in original]
    root=Path(read(a.history/'manifest.json')['source_data_root'])
    for name,quality in [('delta_dense_endpoint','official first/last visibility >0.9'),('delta_dense_lsq','official all-frame visibility >0.9')]:
        out=a.out/name;(out/'samples').mkdir(parents=True)
        write(out/'protocol.json',dict(status='INPUT_ONLY_FROZEN_INFERENCE',method=name,anchors=[r['identity'] for r in rows],
              source_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('extract_metric_surface_history_v1.py')),
              delta_weight_sha256=DELTA_SHA,depth_weight_sha256=dm['weight']['sha256'],
              current_grid_stride=32,current_grid_phase=4,queries_per_camera=1400,
              input_frames='all raw frames between original past/current endpoints, inclusive; added observations explicitly change input information',official_backward_tracking=True,
              input_deadline='authenticated current sensor packet availability, possibly later than t0 LiDAR; not zero-latency LiDAR-time causality',
              window_len=16,padding='official replicate; actual frame count recorded per camera',quality_mask=quality,
              common_depth='Metric3D outputs; DELTA official nearest resize to384x512 then bilinear query sample',
              GT_read=False,optimizer_updates=0,seed=11,dense_input_complete_sha256=sha(a.dense_history/'complete.json'),
              twoframe_reference_complete_sha256=sha(a.twoframe/'complete.json'),
              fit='endpoint difference or current-anchored unweighted least squares over actual timestamps; no threshold fitting',
              comparison='same original endpoint times/current source query depths and 1400-query grid; same complete original population; extra real observations only; endpoint and LSQ separately reported'))
    torch,depth,unused_raft,unused_padder=load_models(a,dm,rm)
    del unused_raft,unused_padder
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(24*1024**3/torch.cuda.get_device_properties(0).total_memory)
    sys.path[:0]=[str(a.delta_asset/'deps'),str(a.delta_asset/'source')]
    from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
    from densetrack3d.models.predictor.predictor import Predictor3D
    from densetrack3d.models.model_utils import bilinear_sample2d
    torch.manual_seed(11)
    model=DenseTrack3D(stride=4,window_len=16,add_space_attn=True,num_virtual_tracks=64,model_resolution=(384,512),upsample_factor=4)
    state=torch.load(a.delta_asset/'densetrack3d.pth',map_location='cpu',weights_only=True)
    if 'model' in state:state=state['model']
    keys=model.load_state_dict(state,strict=True);del state
    # Official DenseTrack3D.train() omits returning self, so eval() cannot chain.
    model.eval();model.requires_grad_(False);model.cuda()
    predictor=Predictor3D(model);predictor.eval();predictor.cuda()
    write(a.out/'delta_load.json',dict(missing_keys=keys.missing_keys,unexpected_keys=keys.unexpected_keys,parameters=sum(p.numel() for p in model.parameters())))
    saved={name:[] for name in ('delta_dense_endpoint','delta_dense_lsq')};pair_count=0
    with torch.inference_mode():
        for row in rows:
            pairs={p['channel']:p for p in row['cameras']}
            for channel in CAMERAS:
                if a.max_pairs and pair_count>=a.max_pairs:break
                tick=time.monotonic();pair=pairs[channel]
                write(a.out/'progress.json',dict(phase='inference',pairs=pair_count,ordinal=row['ordinal'],camera=channel,seconds=tick-start))
                frames=pair['frames'];T=len(frames)
                assert 2<=T<=16
                rgb=[]
                for frame in frames:
                    p=root/frame['image']['file']
                    assert frame['sample_data']['timestamp']<=row['input_availability_us'] and sha(p)==frame['image']['sha256']
                    rgb.append(np.asarray(Image.open(p).convert('RGB')))
                depths=[infer_depth(torch,depth,x,f['K']) for x,f in zip(rgb,frames)]
                video=torch.from_numpy(np.stack(rgb).transpose(0,3,1,2).copy()).float().cuda()[None]
                videodepth=torch.from_numpy(np.stack(depths)).float().cuda()[None,:,None]
                yy,xx=np.meshgrid(np.arange(4,900,32),np.arange(4,1600,32),indexing='ij')
                uv=np.column_stack([xx.ravel(),yy.ravel()]).astype(float);assert len(uv)==1400
                queries=torch.from_numpy(np.column_stack([np.full(len(uv),T-1),uv])).float().cuda()[None]
                output=predictor(video,videodepth,queries=queries,grid_query_frame=T-1,backward_tracking=True,
                                 predefined_intrs=torch.tensor(pair['current']['K'],device='cuda',dtype=torch.float32))
                track=output['trajs_uv'][0].cpu().numpy().astype(float)
                track_d=output['trajs_depth'][0,...,0].cpu().numpy().astype(float)
                vis=output['vis'][0].cpu().numpy();conf=output['conf'][0].cpu().numpy()
                assert track.shape==(T,1400,2) and np.max(abs(track[-1]-uv))<.001
                low=torch.nn.functional.interpolate(videodepth[0],(384,512),mode='nearest')
                scale=np.array([511/1599,383/899]);luv=queries[:,:,1:]*queries.new_tensor(scale)
                common_depth=bilinear_sample2d(low[-1:],luv[:,:,0],luv[:,:,1]).flatten().cpu().numpy().astype(float)
                assert np.max(abs(track_d[-1]-common_depth))<1e-4
                old=next(r for r in old_records if r['identity']==row['identity'] and r['camera']==channel)
                path=a.twoframe/old['file'];assert sha(path)==old['sha256']
                with np.load(path,allow_pickle=False) as z:
                    assert np.array_equal(z['uv'],uv)
                    repeat_depth_difference=float(np.max(abs(z['depth_current_m']-common_depth)))
                assert repeat_depth_difference==0,'Current query depths differ from matched two-frame run'
                endpoint=field(row,pair,uv,common_depth,track[0],track_d[0],vis[0]&vis[-1])
                G0=np.asarray(row['lidar_to_global'],float)
                points=np.stack([lift(track[i] if i<T-1 else uv,track_d[i] if i<T-1 else common_depth,
                         np.asarray(f['K']),np.linalg.inv(G0)@np.asarray(f['camera_to_global'])) for i,f in enumerate(frames)])
                times=np.array([(f['sample_data']['timestamp']-frames[-1]['sample_data']['timestamp'])/1e6 for f in frames])
                assert times[-1]==0 and np.all(np.diff(times)>0)
                velocity=anchored_velocity(points,times)
                p0=points[-1]+float(endpoint['exposure_to_t0_s'])*velocity
                valid_frames=np.isfinite(points).all(2)&(track_d>0)&(track_d<300)&(track[:,:,0]>=0)&(track[:,:,0]<=1599)&(track[:,:,1]>=0)&(track[:,:,1]<=899)
                valid=valid_frames.all(0)&np.isfinite(velocity).all(1)
                roi=valid&(np.abs(p0[:,:2])<51.2).all(1)&(p0[:,2]>-5)&(p0[:,2]<3)
                lsq=dict(endpoint,point_t0_R_m=p0.astype('float32'),velocity_R_mps=velocity.astype('float32'),
                         correspondence_valid=valid,in_roi=roi,quality_in_roi=roi&vis.all(0))
                results={'delta_dense_endpoint':endpoint,'delta_dense_lsq':lsq}
                for arrays in results.values():
                    arrays.update(raw_track_uv=track,raw_track_depth_m=track_d,raw_track_points_R_m=points,
                                  visibility_all=vis,confidence_all=conf,relative_frame_times_s=times)
                for name,arrays in results.items():
                    out=a.out/name;fname=f"samples/{row['ordinal']:04d}_{channel}.npz"
                    np.savez_compressed(out/fname,**arrays)
                    rec=dict(ordinal=row['ordinal'],identity=row['identity'],camera=channel,file=fname,sha256=sha(out/fname),
                             queries=len(uv),input_frames=T,roi_queries=int(arrays['in_roi'].sum()),quality_roi_queries=int(arrays['quality_in_roi'].sum()),
                             repeated_metric_depth_max_difference_m=repeat_depth_difference,seconds=time.monotonic()-tick)
                    saved[name].append(rec)
                    with (out/'records.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
                pair_count+=1
                print(json.dumps(dict(pairs=pair_count,ordinal=row['ordinal'],camera=channel,seconds=time.monotonic()-tick,
                                     input_frames=T,endpoint_quality=int(endpoint['quality_in_roi'].sum()),lsq_quality=int(lsq['quality_in_roi'].sum()))),flush=True)
                del video,videodepth,output,low,results
    torch.cuda.synchronize()
    for name,records in saved.items():
        out=a.out/name
        write(out/'complete.json',dict(status='COMPLETE_FROZEN_HISTORY_TRACKER' if not a.max_pairs else 'COMPLETE_RESOURCE_PROBE_ONLY',
              anchors=len(rows),pairs=len(records),optimizer_updates=0,GT_read=False,
              protocol_sha256=sha(out/'protocol.json'),records_sha256=sha(out/'records.jsonl')))
    complete=dict(status='COMPLETE_RESOURCE_PROBE_ONLY' if a.max_pairs else 'COMPLETE_DENSE_DELTA_ENDPOINT_AND_LSQ_HISTORY',
                  pairs=pair_count,anchors=len(rows),seconds=time.monotonic()-start,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,optimizer_updates=0,GT_read=False)
    write(a.out/'complete.json',complete);print(json.dumps(complete),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('depth-asset','raft-asset','history','delta-asset','twoframe','dense-history','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--anchors',type=int,default=16);p.add_argument('--max-pairs',type=int,default=0)
    p.add_argument('--max-seconds',type=int,default=1800);a=p.parse_args()
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('bounded inference time exceeded')))
    signal.alarm(a.max_seconds)
    try:main(a)
    except BaseException:
        if a.out.exists():write(a.out/'failed.json',{'error':traceback.format_exc()})
        raise
