"""CPU fixed-history correspondence queries on every original train source point.

Only source indices and frozen D displacements are read from motion NPZs;
future GT arrays are never opened. No model, training, target-based association,
threshold fitting, new view search, or full dense-field performance claim.
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

SCHEMA='history-camera-evidence-train-v1'
HISTORY_SHA='040d62242a541b9685f3f39f1534b3663b0e1142b4cd7d9737cdbfd15d94314d'
MOTION_SHA='3bd429effb1fbe50ac0aeb77330bc0100ee144bf8406fbd078097fb336301a73'
GEOMETRY_SHA='a2504a0389cb2756032531961e305e22287ec890c96085e5ea29b2e61e4f1c99'
POINTS_SHA='91bcc80882e41ffd5234cecce4b517ef4081839c89dbb6a048c4226b723ca6fd'
SELECTION_SHA='5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def authenticated_ledger(root,digest):
    require(sha(root/'complete.json')==digest and not (root/'failed.json').exists(),'Wrong/failed completed input')
    c=read(root/'complete.json')
    for name,h in c['files_sha256'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe input ledger')
        require(sha(root/p)==h,'Completed input file changed: '+name)
    return c


def run(a,out,started):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU job must have no visible GPU')
    protocol=read(a.protocol)
    require(protocol['schema']==SCHEMA and protocol['status']=='FROZEN' and
            protocol['resources']==dict(max_seconds=600,cpu_threads=2) and
            protocol['history_inputs_complete_sha256']==HISTORY_SHA and
            protocol['motion_complete_sha256']==MOTION_SHA and
            protocol['geometry_complete_sha256']==GEOMETRY_SHA and
            protocol['selection_sha256']==SELECTION_SHA,'Frozen extraction protocol differs')
    require(sha(a.module)==protocol['module_sha256'],'Core module changed')
    spec=importlib.util.spec_from_file_location('history_camera_evidence_core',a.module)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    constants=dict(camera_order=list(core.CAMERA_ORDER),horizons_seconds=list(core.HORIZONS_SECONDS),
                   patch_radius=core.PATCH_RADIUS,patch_size=core.PATCH_SIZE,zncc_eps=core.ZNCC_EPS,
                   texture_std_min=core.TEXTURE_STD_MIN,reason_codes={str(k):v for k,v in core.REASON_CODES.items()},
                   axes=core.AXES)
    require(constants==protocol['module_constants'],'Module constants differ')
    history=Path(a.history_inputs);motion=Path(a.motion_run);geometry=Path(a.geometry_cache)
    hc=authenticated_ledger(history,HISTORY_SHA);mc=authenticated_ledger(motion,MOTION_SHA)
    require(sha(geometry/'complete.json')==GEOMETRY_SHA and sha(geometry/'points_R.npy')==POINTS_SHA,'Original geometry changed')
    points=np.load(geometry/'points_R.npy',allow_pickle=False)
    require(points.shape==(640000,3) and points.dtype==np.float64,'Grid layout changed')
    hrecords=read(history/'records.json')['records'];mrecords=read(motion/'index.json')['records']
    require(len(hrecords)==len(mrecords)==512,'Complete train support required')
    hmanifest=read(history/'manifest.json');root=Path(hmanifest['source_data_root'])
    manifest=dict(schema=SCHEMA,source_sha256=sha(__file__),extraction_protocol_sha256=sha(a.protocol),
        module_sha256=sha(a.module),module_constants=constants,history_inputs_complete_sha256=HISTORY_SHA,
        history_inputs_files_sha256=hc['files_sha256'],motion_complete_sha256=MOTION_SHA,
        motion_files_sha256=mc['files_sha256'],geometry_complete_sha256=GEOMETRY_SHA,points_R_sha256=POINTS_SHA,
        selection_sha256=SELECTION_SHA,source_data_root=str(root),scope=protocol['scope'],
        runtime=dict(python=sys.version,numpy=np.__version__,PIL_version=Image.__version__))
    write(out/'manifest.json',manifest);(out/'samples').mkdir();records=[]
    total=0;reason_counts={str(k):0 for k in core.REASON_CODES};decoded=0
    for ordinal,(h,m) in enumerate(zip(hrecords,mrecords)):
        require(time.monotonic()-started<a.max_seconds,'CPU correspondence budget exceeded')
        require(h['ordinal']==m['ordinal']==ordinal and h['identity']==m['identity'],'Input sample identity differs')
        path=motion/m['file'];require(sha(path)==m['sha256'] and path.stat().st_size==m['bytes'],'Frozen D source changed')
        with np.load(path,allow_pickle=False) as z:
            idx=z['source_flat_indices'];d=z['D_displacement_m']
        require(idx.dtype==np.int64 and len(idx)==m['source_points'] and
                np.all((idx>=0)&(idx<640000)),'Original point query support changed')
        velocity=core.nominal_lsq_velocity(d);camera_pairs={}
        for pair in h['cameras']:
            frames={}
            for name in ['current','past']:
                f=pair[name];desc=f['image'];file=root/desc['file']
                require(file.stat().st_size==desc['bytes'] and sha(file)==desc['sha256'],'Authenticated image changed')
                with Image.open(file) as im:
                    require(im.size==tuple(desc['size_wh']) and im.format=='JPEG','Image header differs')
                    rgb=np.array(im.convert('RGB'),dtype=np.uint8)
                decoded+=1
                frames[name]=dict(timestamp_us=f['sample_data']['timestamp'],camera_to_global=np.asarray(f['camera_to_global']),
                                  intrinsic=np.asarray(f['K']),rgb=rgb)
            camera_pairs[pair['channel']]=frames
        result=core.evaluate_history_correspondence(points[idx],velocity,np.asarray(h['lidar_to_global']),
                    h['t0_lidar_us'],h['input_availability_us'],camera_pairs,chunk_size=4096)
        require(np.array_equal(result['valid'],result['reason_code']==0),'Common scoring mask inconsistent')
        require(np.isfinite(result['score'][result['valid']]).all() and
                np.isnan(result['score'][~result['valid']]).all() and
                np.isfinite(result['broken_score'][result['valid']]).all() and
                np.isnan(result['broken_score'][~result['valid']]).all(),'Actual/control missing score mismatch')
        arrays=dict(identity_json=np.asarray(json.dumps(h['identity'],sort_keys=True)),source_flat_indices=idx,
                    nominal_lsq_velocity_mps=velocity,**result)
        name='samples/%04d_%s.npz'%(ordinal,h['identity']['sample_token'])
        with (out/name).open('xb') as f:np.savez_compressed(f,**arrays)
        with np.load(out/name,allow_pickle=False) as z:
            require(set(z.files)==set(arrays),'Saved NPZ keys changed')
            for k,v in arrays.items():
                require(np.array_equal(v,z[k],equal_nan=True) if v.dtype.kind in 'fc' else np.array_equal(v,z[k]),
                        'Saved query arrays changed: '+k)
        counts={str(k):int((result['reason_code']==k).sum()) for k in core.REASON_CODES}
        for k,v in counts.items():reason_counts[k]+=v
        records.append(dict(ordinal=ordinal,identity=h['identity'],file=name,bytes=(out/name).stat().st_size,
            sha256=sha(out/name),source_points=len(idx),input_sha256=m['input_sha256'],source_motion_npz_sha256=m['sha256'],
            reason_counts=counts,arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()}))
        total+=len(idx);del arrays,result,camera_pairs,d,velocity
        print(json.dumps(dict(event='sample_complete',completed=ordinal+1,seconds=time.monotonic()-started)),flush=True)
    require(total==1740053 and decoded==6144 and sum(reason_counts.values())==total,'Incomplete original source/image support')
    require('torch' not in sys.modules,'CPU image diagnosis must not import Torch')
    write(out/'index.json',dict(schema=SCHEMA,records=records,source_points=total,reason_counts=reason_counts))
    write(out/'summary.json',dict(schema=SCHEMA,status='COMPLETE_HISTORY_CAMERA_TRAIN_EVIDENCE',samples=512,scenes=256,
        source_points=total,images_decoded=decoded,reason_counts=reason_counts,seconds=time.monotonic()-started,
        model_forward=False,optimizer_updates=0,future_GT_arrays_opened=False,threshold_selection=False,
        fitting=False,all_query_arrays_roundtrip_exact=True,images_authenticated_again_before_decode=True,
        scope='Pointwise input-only field queried at every original source index; not a full640000-grid computation'))
    write(out/'complete.json',dict(schema=SCHEMA,status='COMPLETE_HISTORY_CAMERA_TRAIN_EVIDENCE',samples=512,
        source_sha256=sha(__file__),extraction_protocol_sha256=sha(a.protocol),module_sha256=sha(a.module),
        files_sha256={name:sha(out/name) for name in ['manifest.json','index.json','summary.json']}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['protocol','module','history-inputs','motion-run','geometry-cache','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--max-seconds',type=int,default=600);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def stop(sig,frame):raise TimeoutError('Signal '+str(sig))
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    signal.alarm(a.max_seconds)
    try:
        require(a.max_seconds==600,'Resource bound changed');run(a,out,started)
    except BaseException as exc:
        signal.alarm(0);write(out/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
