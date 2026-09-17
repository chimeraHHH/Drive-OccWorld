"""Frozen D inference on train512; collect observable fields before any GT.

No optimizer, router, threshold selection, development forward or O decoder.
The training scenes were seen by D: this is a mechanism diagnostic, not test
generalization. Future-label groups/instances are stored for scoring only.
"""
import argparse
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

SCHEMA = 'motion-evidence-train-extraction-v1'
BASE_SHA = '1a730762d74dda2fcf367d19c80a159e212a9be8884a4cb33e605e9341006211'
FEATURES = ['lsq_speed_xy_mps', 'min_horizon_speed_xy_mps',
            'max_horizon_speed_xy_mps', 'non_cv_velocity_rms_xy_mps',
            'relative_non_cv_velocity_rms_xy', 'local_3x3_velocity_rms_xy_mps',
            'local_3x3_predicted_coverage_fraction']
RECIPE = dict(split='train', samples=512, scenes=256, source_points=1740053,
    valid_objects_by_horizon=[11254,10967,10636,10332],
    valid_points_by_horizon=[1709467,1675202,1639983,1608197],
    horizons_seconds=[.5,1.,1.5,2.], feature_names=FEATURES,
    lsq_velocity='sum(h * predicted_displacement_h) / sum(h**2)',
    non_cv='sqrt(mean_h(||predicted_displacement_h/h - lsq_velocity||_xy**2))',
    relative_non_cv='non_cv / (lsq_speed_xy_mps + 0.1 m/s)',
    local_roughness='sqrt(mean_3x3_XY(||neighbor_lsq_velocity - center_lsq_velocity||_xy**2)); fixed z; replicated border',
    local_coverage='mean_3x3_XY(current_prediction_owner>=0); fixed z; replicated border',
    forward_order='all 640000 D vectors, CV velocity and seven fields before sparse label read',
    sparse_support='all original train source points and all original valid future labels; no new filtering',
    D_seen_all_training_scenes=True, development_forward=False,
    O_model_loaded=False, optimizer_updates=0, fitting=False, threshold_selection=False)


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
    return d.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def base_module():
    path=Path(__file__).with_name('fixed_D_coverage_complementarity_v1.py')
    require(sha(path)==BASE_SHA,'Reviewed inference helper changed')
    spec=importlib.util.spec_from_file_location('fixed_D_coverage_complementarity_v1',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def observable_fields(displacements,owner):
    """Only predictions enter this pure function; XYZ grid and nominal h fixed."""
    require(displacements.shape==(4,640000,3) and np.isfinite(displacements).all(),'Invalid dense prediction')
    h=np.asarray(RECIPE['horizons_seconds'],dtype=np.float64)
    d=displacements.astype(np.float64)
    velocity=np.sum(h[:,None,None]*d,axis=0)/np.sum(h*h)
    v_by_h=d/h[:,None,None]
    speeds=np.linalg.norm(v_by_h[:,:,:2],axis=2)
    speed=np.linalg.norm(velocity[:,:2],axis=1)
    non_cv=np.sqrt(np.mean(np.sum((v_by_h[:,:,:2]-velocity[None,:,:2])**2,axis=2),axis=0))
    xy=velocity[:,:2].reshape(200,200,16,2)
    padded=np.pad(xy,((1,1),(1,1),(0,0),(0,0)),mode='edge')
    rough=np.zeros((200,200,16),dtype=np.float64)
    for i in range(3):
        for j in range(3):rough+=np.sum((padded[i:i+200,j:j+200]-xy)**2,axis=-1)
    rough=np.sqrt(rough/9).reshape(-1)
    require(owner.shape==(640000,) and owner.dtype==np.int64,'Full predicted owner array required')
    mask=(owner>=0).reshape(200,200,16).astype(np.float64)
    border=np.pad(mask,((1,1),(1,1),(0,0)),mode='edge');coverage=np.zeros_like(mask)
    for i in range(3):
        for j in range(3):coverage+=border[i:i+200,j:j+200]/9
    fields=np.stack([speed,speeds.min(axis=0),speeds.max(axis=0),non_cv,non_cv/(speed+.1),rough,coverage.reshape(-1)],axis=1)
    require(fields.shape==(640000,7) and fields.dtype==np.float64 and np.isfinite(fields).all(),'Feature field invalid')
    return fields


def expected_sources(connected_protocol,source_root):
    b=base_module();r=b.expected_sources(connected_protocol,source_root)
    r[Path(__file__).name]=sha(__file__);return r


def protocol_template(connected_protocol,source_root):
    b=base_module()
    return dict(schema=SCHEMA,status='REVIEW_REQUIRED',recipe=RECIPE,
        sources_sha256=expected_sources(connected_protocol,source_root),
        connected_protocol_sha256=b.CONNECTED_SHA,d_complete_sha256=b.D_COMPLETE_SHA,
        geometry_complete_sha256=b.GEOMETRY_SHA,runtime_contract_sha256=b.RUNTIME_SHA,
        selection_sha256=b.SELECTION_SHA,resources=dict(max_seconds=1800,max_allocated_gib=8,cpu_threads=2),
        numerical_policy=b.RECIPE['numerical_policy'])


def run(a,out,started):
    b=base_module();p=read(a.protocol);expected=protocol_template(a.connected_protocol,a.source_root)
    expected['status']='FROZEN';require(p==expected,'Exact frozen extraction protocol differs')
    require(p['resources']==dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib,cpu_threads=2),'Resource cap changed')
    require(sha(a.selection)==b.SELECTION_SHA and sha(a.runtime_contract)==b.RUNTIME_SHA,'Source contract changed')
    for path,digest in read(a.runtime_contract)['runtime_source_sha256'].items():require(sha(path)==digest,'Runtime source changed')
    require(sha(Path(a.d_run)/'complete.json')==b.D_COMPLETE_SHA,'Wrong D endpoint')
    sys.path.insert(0,str(Path(a.source_root).resolve()))
    sources=p['sources_sha256']
    helper=b.bind('train_source_motion_v1',sources,a.source_root)
    trainer=b.bind('train_connected_motion_v2',sources,a.source_root)
    native=b.bind('native_state_cache',sources,a.source_root)
    cache=b.bind('shared_rigid_geometry_cache_v1',sources,a.source_root)
    original=read(a.connected_protocol)
    root,selected,cache_receipt=helper.cache_index(a.train_cache,'train',original)
    labels,descriptors=helper.labels_manifest(a.sparse_labels,original)
    geometry=cache.GeometryCache(a.geometry_cache,b.GEOMETRY_SHA,a.selection)
    require([b.identity(r) for r in selected]==geometry.chosen['train'],'Train identity/geometry differs')
    require(len({r['scene_token'] for r in selected})==256,'Wrong training scene count')
    dev_scenes={r['scene_token'] for r in geometry.chosen['development']}
    require(not dev_scenes.intersection(r['scene_token'] for r in selected),'Train/dev scenes overlap')
    for key in ['valid_objects_by_horizon','valid_points_by_horizon']:
        actual=[sum(descriptors[r['sample_token']]['counts'][key][h] for r in selected) for h in range(4)]
        require(actual==RECIPE[key],'Training support counts differ')
    if a.check:
        require('torch' not in sys.modules,'Metadata check imported Torch')
        write(out/'check.json',dict(schema=SCHEMA,status='AUTHENTICATED_METADATA_ONLY',
            protocol_sha256=sha(a.protocol),sources_sha256=sources,train_cache=cache_receipt,
            geometry=geometry.receipt,samples=512,scenes=256,torch_imported=False,GPU_used=False,
            D_payload_loaded=False,actual_input_bytes_checked=False,optimizer_updates=0))
        return
    import torch
    require(torch.cuda.is_available() and str(a.device).startswith('cuda'),'Original CUDA environment required')
    torch.set_num_threads(2);torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(min(1.,8*2**30/torch.cuda.get_device_properties(a.device).total_memory),a.device)
    torch.cuda.reset_peak_memory_stats(a.device);native._prepare_repo(a.repo)
    D,gate,receipt=trainer.load_completed_arm(Path(a.d_run)/'runs/D',a.connected_protocol,helper,a.device)
    del gate
    digest=helper.state_digest(D);require(not D.training and all(not x.requires_grad for x in D.parameters()),'D not frozen')
    write(out/'manifest.json',dict(schema=SCHEMA,protocol_sha256=sha(a.protocol),sources_sha256=sources,
        recipe=RECIPE,D_loader=receipt,train_cache=cache_receipt,geometry_cache=geometry.receipt,
        sparse_manifest_sha256=original['labels']['manifest_sha256'],selection_sha256=b.SELECTION_SHA,
        environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))))
    (out/'samples').mkdir();records=[];total_source=0;total_valid=np.zeros(4,dtype=np.int64);total_objects=np.zeros(4,dtype=np.int64)
    def budget():
        require(time.monotonic()-started<a.max_seconds,'Wallclock budget exceeded')
        require(torch.cuda.max_memory_allocated(a.device)<=a.max_allocated_gib*2**30,'Memory budget exceeded')
    for i,r in enumerate(selected):
        budget();g=geometry.load('train',i,b.identity(r));tokens=helper.load_tokens(root,r,native,a.device)
        with torch.inference_mode():prediction=D(tokens)
        require(prediction.shape==(1,4,3,200,200,16) and prediction.dtype==torch.float32 and bool(torch.isfinite(prediction).all()),'Invalid D prediction')
        dense=prediction[0].reshape(4,3,-1).permute(0,2,1).detach().cpu().numpy()
        features=observable_fields(dense,g['owner'])
        # All observable fields have been defined; only now open future labels.
        desc=descriptors[r['sample_token']];label=helper.load_sparse(labels,desc,r);idx=label['source_flat_indices']
        sparse=helper.gather_sparse(prediction,label).detach().cpu().numpy()
        require(np.array_equal(sparse,dense[:,idx]),'XYZ native gather mismatch')
        arrays=dict(identity_json=np.asarray(json.dumps(b.identity(r),sort_keys=True)),
            source_flat_indices=idx,owner=g['owner'][idx],owner_original_indices=g['owner_original_indices'][idx],
            observable_features=features[idx],D_displacement_m=sparse,
            CV_velocity_mps=g['cv_velocity_R'][idx],target_displacement_m=label['target_displacement_m'],
            valid=label['valid'],object_index=label['object_index'],instance_tokens=label['instance_tokens'],
            object_speed_group=label['object_speed_group'],object_future_valid=label['object_future_valid'],
            dt_future_seconds=label['dt_future_seconds'])
        filename='samples/%04d_%s.npz'%(i,r['sample_token']);file=out/filename
        with file.open('xb') as f:np.savez_compressed(f,**arrays)
        with np.load(file,allow_pickle=False) as z:
            require(set(z.files)==set(arrays) and all(np.array_equal(z[k],v) for k,v in arrays.items()),'Saved array roundtrip differs')
        count=len(idx);total_source+=count;total_valid+=label['valid'].sum(axis=1);total_objects+=label['object_future_valid'].sum(axis=1)
        records.append(dict(ordinal=i,identity=b.identity(r),file=filename,sha256=sha(file),bytes=file.stat().st_size,
            source_points=count,covered_points=int((arrays['owner']>=0).sum()),input_sha256=r['files']['inputs']['sha256'],
            sparse_sha256=desc['sha256'],geometry=g['receipt'],
            arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()}))
        del tokens,prediction,dense,features,sparse,arrays,label,g
        budget();print(json.dumps(dict(event='sample_complete',completed=i+1,seconds=time.monotonic()-started)),flush=True)
    require(len(records)==512 and total_source==RECIPE['source_points'] and total_valid.tolist()==RECIPE['valid_points_by_horizon'] and total_objects.tolist()==RECIPE['valid_objects_by_horizon'],'Incomplete original training support')
    require(helper.state_digest(D)==digest and all(q.grad is None for q in D.parameters()),'D changed')
    require(expected_sources(a.connected_protocol,a.source_root)==sources,'Sources changed')
    budget();write(out/'index.json',dict(schema=SCHEMA,records=records,feature_names=FEATURES,
        source_points=total_source,valid_points_by_horizon=total_valid.tolist(),valid_objects_by_horizon=total_objects.tolist()))
    write(out/'summary.json',dict(schema=SCHEMA,status='COMPLETE_FIXED_D_TRAIN_FEATURES',samples=512,scenes=256,
        source_points=total_source,object_horizon_rows=int(total_objects.sum()),optimizer_updates=0,threshold_selection=False,
        model_fitted=False,development_forward=False,O_model_loaded=False,D_state_unchanged=True,
        all_saved_arrays_roundtrip_exact=True,seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated(a.device)/2**30,
        interpretation='D-training-internal observability diagnostic; not unseen-scene performance'))
    budget();write(out/'complete.json',dict(schema=SCHEMA,status='COMPLETE_FIXED_D_TRAIN_FEATURES',samples=512,scenes=256,
        protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),files_sha256={k:sha(out/k) for k in ['manifest.json','index.json','summary.json']},optimizer_updates=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','connected-protocol','d-run','train-cache','sparse-labels','geometry-cache','selection','repo','runtime-contract','source-root','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--check',action='store_true')
    p.add_argument('--max-seconds',type=int,required=True);p.add_argument('--max-allocated-gib',type=float,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def terminate(signum,frame):raise TimeoutError('Signal '+str(signum))
    for s in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(s,terminate)
    try:
        require(a.max_seconds>0,'Positive time limit required');signal.alarm(a.max_seconds);run(a,out,started)
    except BaseException as e:
        signal.alarm(0);write(out/'failed.json',dict(schema=SCHEMA,error=repr(e),traceback=traceback.format_exc(),optimizer_updates=0,seconds=time.monotonic()-started));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
