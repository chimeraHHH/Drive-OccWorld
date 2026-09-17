"""Authenticate every raw camera sample between the fixed historical endpoints.

CPU only. No annotation/target tables, model, GPU or future image access.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
from PIL import Image
import build_full_motion_targets_v1 as reader


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)


def main(a):
    start=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    a.out.mkdir(parents=True,exist_ok=False)
    assert sha(reader.__file__)=='dddea6d4e4a9b3a45037484b3165cf6cc4f52584da9749469ca1fd3ff1a03557'
    assert sha(a.history/'complete.json')=='040d62242a541b9685f3f39f1534b3663b0e1142b4cd7d9737cdbfd15d94314d'
    for f,h in read(a.history/'complete.json')['files_sha256'].items():assert sha(a.history/f)==h
    old=read(a.history/'manifest.json');root=Path(old['source_data_root']);rows=read(a.history/'records.json')['records'][:a.anchors]
    ledger={}
    def table(name):
        p=root/'v1.0-trainval'/(name+'.json')
        yield from reader.table(p,ledger)
        assert ledger[str(p)]['sha256']==old['metadata'][str(p)]['sha256'],name
    samples={r['token']:r for r in table('sample')}
    calibration={r['token']:r for r in table('calibrated_sensor')}
    sensors={r['token']:r for r in table('sensor')}
    windows={};sequences={}
    for row in rows:
        for pair in row['cameras']:
            key=(row['ordinal'],pair['channel']);sequences[key]=[]
            prefix=Path(pair['current']['sample_data']['filename']).name.rsplit('__',1)[0]
            windows.setdefault(prefix,[]).append((key,pair['past']['sample_data']['timestamp'],pair['current']['sample_data']['timestamp']))
    for sd in table('sample_data'):
        filename=sd['filename']
        if not filename.endswith('.jpg'):continue
        prefix=Path(filename).name.rsplit('__',1)[0]
        for key,t0,t1 in windows.get(prefix,[]):
            if t0<=sd['timestamp']<=t1:sequences[key].append(sd)
    needed={sd['ego_pose_token'] for seq in sequences.values() for sd in seq}
    poses={p['token']:p for p in table('ego_pose') if p['token'] in needed};assert set(poses)==needed
    records=[];counts=[];intervals=[];images={}
    for row in rows:
        updated=dict(row);updated['cameras']=[]
        for pair in row['cameras']:
            seq=sorted(sequences[(row['ordinal'],pair['channel'])],key=lambda x:x['timestamp'])
            assert len(seq)>=2 and seq[0]==pair['past']['sample_data'] and seq[-1]==pair['current']['sample_data']
            for p,c in zip(seq[:-1],seq[1:]):
                assert p['next']==c['token'] and c['prev']==p['token'] and p['timestamp']<c['timestamp']
                intervals.append((c['timestamp']-p['timestamp'])/1e6)
            frames=[]
            for sd in seq:
                assert samples[sd['sample_token']]['scene_token']==row['identity']['scene_token']
                assert sd['timestamp']<=row['input_availability_us']
                cs=calibration[sd['calibrated_sensor_token']];ep=poses[sd['ego_pose_token']]
                assert sensors[cs['sensor_token']]['channel']==pair['channel']
                rel=Path(sd['filename']);assert not rel.is_absolute() and '..' not in rel.parts
                p=root/rel;assert p.is_file()
                if str(rel) not in images:
                    with Image.open(p) as im:assert im.format=='JPEG' and im.size==(1600,900) and im.mode=='RGB'
                    images[str(rel)]={'file':str(rel),'bytes':p.stat().st_size,'sha256':sha(p),'size_wh':[1600,900],'mode':'RGB'}
                G=np.asarray(reader.lidar_to_global(ep,cs));K=np.asarray(cs['camera_intrinsic'])
                assert K.shape==(3,3) and np.isfinite(K).all() and np.allclose(G[:3,:3].T@G[:3,:3],np.eye(3),atol=1e-10)
                frames.append(dict(sample_data=sd,K=K.tolist(),camera_to_global=G.tolist(),image=images[str(rel)],
                                   seconds_from_lidar=(sd['timestamp']-row['t0_lidar_us'])/1e6))
            for i,key in [(0,'past'),(-1,'current')]:
                assert frames[i]['image']['sha256']==pair[key]['image']['sha256']
                assert np.array_equal(frames[i]['camera_to_global'],pair[key]['camera_to_global'])
                assert np.array_equal(frames[i]['K'],pair[key]['K'])
            updated['cameras'].append(dict(pair,frames=frames));counts.append(len(frames))
        records.append(updated)
        print(json.dumps({'completed_anchors':len(records),'seconds':time.monotonic()-start}),flush=True)
    write(a.out/'records.json',{'records':records})
    write(a.out/'manifest.json',dict(source_sha256=sha(__file__),reader_sha256=sha(reader.__file__),
          endpoint_history_complete_sha256=sha(a.history/'complete.json'),source_data_root=str(root),metadata=ledger,
          selection='first16 original train anchors, unchanged endpoints',temporal_support='all consecutive raw camera sample_data between original past and current endpoints, inclusive',
          no_GT_tables_read=True,future_images_read=False,images=list(images.values())))
    summary=dict(status='COMPLETE_AUTHENTICATED_DENSE_HISTORY_INPUTS',anchors=len(rows),sequences=len(counts),
         frames_per_sequence_minmax=[min(counts),max(counts)],image_occurrences=sum(counts),unique_images=len(images),
         interframe_dt_s_minmax=[min(intervals),max(intervals)],source_bytes=sum(x['bytes'] for x in images.values()),
         seconds=time.monotonic()-start,GT_read=False,model_forward=False,optimizer_updates=0)
    write(a.out/'summary.json',summary)
    write(a.out/'complete.json',dict(summary,files_sha256={f:sha(a.out/f) for f in ['records.json','manifest.json','summary.json']}))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--anchors',type=int,default=16);p.add_argument('--max-seconds',type=int,default=300);a=p.parse_args()
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('bounded input audit')))
    signal.alarm(a.max_seconds)
    try:main(a)
    except BaseException:
        if a.out.exists():write(a.out/'failed.json',{'error':traceback.format_exc()})
        raise
