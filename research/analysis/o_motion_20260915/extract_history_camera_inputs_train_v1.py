"""Bounded CPU input-only audit/export: train512 current/previous-keyframe images.

No annotations, targets, model/checkpoint, Torch, GPU, or future image access.
Raw sensor timestamps/poses and image hashes are preserved for later scoring.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
from PIL import Image

SCHEMA='history-camera-inputs-train-v1'
CAMERAS=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
SELECTION_SHA='5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
METADATA_SHA={
    'sample':'6035ac58b6e971622be2bb1be15b917e7cb4e05ae984d6b339c27c1699c4ad9d',
    'sample_data':'6dcad49f0b9bd7b1cef04e2a0ff2ac2879b46b938e8d48153f00693cebb4de21',
    'ego_pose':'be12bd501f694b344628ba0680a37d6e1ec83e62b37f310c205c1dbe41cd09ff',
    'calibrated_sensor':'67781a5dd7b2504b046ef89d6dcb267b12d1cc91af21f1ec5758624588e99865',
    'sensor':'4d5c96570e2d8b09b88ce4c605e40ee43c4909f97bea5741185f18f92eb491ae'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def require(ok,msg):
    if not ok:raise ValueError(msg)


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def run(a,out,started):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only input audit required')
    require(sha(a.selection)==SELECTION_SHA,'Selection changed')
    require(sha(a.reader)==a.reader_sha,'Frozen streaming/geometry source changed')
    spec=importlib.util.spec_from_file_location('raw_table_geometry_reader',a.reader)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    root=Path(a.data_root);tables=root/'v1.0-trainval';ledger={}
    def rows(name):
        path=tables/(name+'.json')
        yield from reader.table(path,ledger)
        require(ledger[str(path)]['sha256']==METADATA_SHA[name],'Official metadata bytes changed: '+name)
    samples={r['token']:r for r in rows('sample')}
    sensors={r['token']:r for r in rows('sensor')}
    calibration={r['token']:r for r in rows('calibrated_sensor')}
    selection=json.loads(Path(a.selection).read_text())
    selected=[r for r in selection['records'] if r['split']=='train']
    require(len(selected)==512 and len({r['scene_token'] for r in selected})==256,'Expected train512')
    pairs=[];wanted=set()
    for r in selected:
        current=samples[r['sample_token']];previous=samples[current['prev']]
        require(current['scene_token']==previous['scene_token']==r['scene_token'] and
                previous['next']==current['token'] and previous['timestamp']<current['timestamp'],
                'Wrong previous keyframe or scene')
        pairs.append((current,previous));wanted.update([current['token'],previous['token']])
    data={};current_radars={}
    for sd in rows('sample_data'):
        if sd['sample_token'] not in wanted or not sd['is_key_frame']:continue
        cs=calibration[sd['calibrated_sensor_token']];channel=sensors[cs['sensor_token']]['channel']
        key=(sd['sample_token'],channel)
        if channel in CAMERAS+['LIDAR_TOP']:
            require(key not in data,'Duplicate keyframe sensor');data[key]=sd
        elif channel.startswith('RADAR_'):
            current_radars.setdefault(sd['sample_token'],[]).append(sd['timestamp'])
    require(all((t,c) in data for t in wanted for c in CAMERAS+['LIDAR_TOP']),'Missing camera/LiDAR keyframe')
    needed={d['ego_pose_token'] for d in data.values()}
    poses={r['token']:r for r in rows('ego_pose') if r['token'] in needed}
    require(set(poses)==needed,'Missing actual sensor-time ego pose')
    images={};records=[]
    def sensor_to_global(sd):
        cs=calibration[sd['calibrated_sensor_token']];ep=poses[sd['ego_pose_token']]
        mat=np.asarray(reader.lidar_to_global(ep,cs),dtype=np.float64)
        require(np.allclose(mat[:3,:3].T@mat[:3,:3],np.eye(3),atol=1e-10,rtol=0),'Invalid sensor rotation')
        return mat
    def image_record(sd,channel,G0,t0):
        rel=Path(sd['filename']);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe image path')
        path=root/rel;require(path.is_file() and path.stat().st_size<8*1024**2,'Missing/oversized source image')
        if sd['filename'] not in images:
            before=path.stat()
            with Image.open(path) as im:
                require(im.format=='JPEG' and im.size==(sd['width'],sd['height']),'Source JPEG header differs')
                size=list(im.size);mode=im.mode
            images[sd['filename']]=dict(file=sd['filename'],bytes=before.st_size,sha256=sha(path),size_wh=size,mode=mode)
            after=path.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Image changed during audit')
        GC=sensor_to_global(sd);T=np.linalg.inv(GC)@G0
        cs=calibration[sd['calibrated_sensor_token']];K=np.asarray(cs['camera_intrinsic'],dtype=float)
        require(K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0,'Invalid intrinsics')
        return dict(channel=channel,sample_data=dict(sd),calibrated_sensor=dict(cs),
                    ego_pose=dict(poses[sd['ego_pose_token']]),seconds_from_lidar=(sd['timestamp']-t0)/1e6,
                    R_to_camera=T.tolist(),camera_to_global=GC.tolist(),K=K.tolist(),image=images[sd['filename']])
    for i,(r,(current,previous)) in enumerate(zip(selected,pairs)):
        require(time.monotonic()-started<a.max_seconds,'Input audit time budget exceeded')
        lidar=data[(current['token'],'LIDAR_TOP')];t0=lidar['timestamp'];G0=sensor_to_global(lidar)
        camera=[]
        for channel in CAMERAS:
            c=data[(current['token'],channel)];p=data[(previous['token'],channel)]
            require(p['timestamp']<c['timestamp'],'Past image timestamp is not earlier')
            require(calibration[c['calibrated_sensor_token']]['sensor_token']==calibration[p['calibrated_sensor_token']]['sensor_token'],
                    'Camera channel/sensor changed')
            camera.append(dict(channel=channel,current=image_record(c,channel,G0,t0),past=image_record(p,channel,G0,t0)))
        availability=max([t0]+[v['current']['sample_data']['timestamp'] for v in camera]+current_radars.get(current['token'],[]))
        require(all(v['past']['sample_data']['timestamp']<=v['current']['sample_data']['timestamp']<=availability for v in camera),
                'Image exceeds current sensor availability')
        records.append(dict(ordinal=i,identity=r,previous_sample_token=previous['token'],
             lidar_sample_data=lidar,lidar_calibrated_sensor=calibration[lidar['calibrated_sensor_token']],
             lidar_ego_pose=poses[lidar['ego_pose_token']],lidar_to_global=G0.tolist(),t0_lidar_us=t0,
             input_availability_us=availability,availability_delay_seconds=(availability-t0)/1e6,cameras=camera))
        print(json.dumps(dict(event='sample_complete',completed=i+1,seconds=time.monotonic()-started)),flush=True)
    require('torch' not in sys.modules,'Input audit imported Torch')
    write(out/'records.json',dict(schema=SCHEMA,records=records))
    write(out/'manifest.json',dict(schema=SCHEMA,source_sha256=sha(__file__),reader_sha256=sha(a.reader),
          selection_sha256=SELECTION_SHA,source_data_root=str(root),metadata=ledger,images=list(images.values()),
          scope='raw current and immediately preceding sample keyframes only; no annotations or GT depth',
          camera_order=CAMERAS,coordinates='R=current LIDAR_TOP; column SE3; raw K, no crop/resize/flip',
          availability='maximum of selected current keyframe sensor timestamps; does not imply prediction at exact lidar timestamp',
          python=sys.version,numpy=np.__version__))
    delta=[v[k]['seconds_from_lidar'] for r in records for v in r['cameras'] for k in ['current']]
    summary=dict(schema=SCHEMA,status='COMPLETE_AUTHENTICATED_HISTORY_CAMERA_INPUTS',samples=512,scenes=256,
        image_occurrences=512*12,unique_images=len(images),image_bytes=sum(v['bytes'] for v in images.values()),
        current_camera_lidar_offset_minmax_seconds=[min(delta),max(delta)],
        availability_delay_max_seconds=max(r['availability_delay_seconds'] for r in records),
        source_images_transformed=False,images_decoded_for_scoring=False,header_and_all_bytes_checked=True,
        future_images_read=False,annotations_read=False,model_forward=False,optimizer_updates=0,
        seconds=time.monotonic()-started)
    write(out/'summary.json',summary)
    write(out/'complete.json',dict(schema=SCHEMA,status=summary['status'],samples=512,source_sha256=sha(__file__),
          files_sha256={f:sha(out/f) for f in ('records.json','manifest.json','summary.json')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['selection','reader','reader-sha','data-root','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--max-seconds',type=int,default=300);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def stop(sig,frame):raise TimeoutError('Signal '+str(sig))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(a.max_seconds)
    try:
        require(a.max_seconds==300,'Resource bound changed');run(a,out,started)
    except BaseException as exc:
        signal.alarm(0);write(out/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
