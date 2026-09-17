"""Read-only real-data coordinate audit; CPU, no model/checkpoint/optimizer.

Use the two frozen joint-preflight anchors in their fixed order, selecting the
first with nonzero future ego motion. Observe the actual test-pipeline loader
without replacing its outputs. Separate continuous rigid-transform agreement,
quantization, clipping, collisions and the exact full-GT majority-vote result.
Execute the original geometric _align_bev_coordnates AST on CPU with a dummy
configuration object; only get_bev_grids' device is explicitly set to CPU.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
import types
import numpy as np

SELECTION_SHA='e616af491e77ad977dfddfd931c1110f9580921bc38ed6ad84281dfe23a9754d'
CONFIG_SHA='c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303'
LOADER_SHA='ac9190193630eca9f3118136e2d0b46bf528217f50244e37e6fc1cbff4841d1e'
DETECTOR_SHA='67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    with temp.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def event(name,**kwargs):print(json.dumps(dict(event=name,**kwargs)),flush=True)


def source_function(path,name):
    tree=ast.parse(Path(path).read_text())
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    require(len(matches)==1,'Ambiguous source method')
    node=copy.deepcopy(matches[0]);require(not node.decorator_list,'Unexpected decorated geometry function')
    return compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec')


def vote_and_remap(voxels,labels,shape):
    """Independent exact sparse voting: native uint16 counter, low-class tie.

    Uses the observed native prequantized coordinates, so float operation-order
    differences in the independent transform cannot create false GT failures.
    Counter overflow is explicitly rejected, not silently approximated.
    """
    clipped=np.clip(voxels,np.zeros(3),np.asarray(shape)-1).astype(np.int64)
    labels=np.asarray(labels).astype(np.int64)
    require(np.isfinite(voxels).all() and ((labels>=0)&(labels<=255)).all(),'Invalid sparse occupancy')
    linear=np.ravel_multi_index(clipped.T,tuple(shape))
    packed,count=np.unique(linear*256+labels,return_counts=True)
    require(count.max(initial=0)<65536,'Native uint16 voting overflow present; audit refuses approximate reconstruction')
    cell,cls=packed//256,packed%256
    order=np.lexsort((cls,-count,cell));sorted_cell=cell[order]
    first=np.r_[True,sorted_cell[1:]!=sorted_cell[:-1]]
    winners=order[first]
    dense=np.zeros(int(np.prod(shape)),dtype=np.uint8);dense[cell[winners]]=cls[winners]
    mapping=np.arange(256,dtype=np.uint8)
    mapping[[0,1,8,11,12,13,14,15,16]]=0;mapping[[2,3,4,5,6,7,9,10]]=1
    dense=mapping[dense].reshape(tuple(shape))
    _,cell_count=np.unique(linear,return_counts=True)
    return dense,clipped,dict(unique_destination_voxels=len(cell_count),
        multi_source_destination_voxels=int((cell_count>1).sum()),
        max_source_rows_per_destination=int(cell_count.max(initial=0)),
        maximum_per_class_vote=int(count.max(initial=0)),uint16_vote_overflow=False)


def roi_counts(voxels,clipped,shape):
    shape=np.asarray(shape)
    physical=((voxels<0)|(voxels>=shape)).any(1)
    numeric=((voxels<0)|(voxels>shape-1)).any(1)
    changed=(np.floor(voxels).astype(np.int64)!=clipped).any(1)
    return dict(source_rows=len(voxels),physical_outside_anchor_ROI_rows=int(physical.sum()),
        native_clip_changed_numeric_coordinates_rows=int(numeric.sum()),
        native_clip_changed_integer_voxel_rows=int(changed.sum()),
        note='Uppermost in-ROI voxel centers can be numerically clipped without changing their integer voxel. Physical ROI is half-open [0,shape).')


def run(a,out):
    require(sha(a.selection)==a.selection_sha256==SELECTION_SHA,'Frozen two-anchor selection hash mismatch')
    require(sha(a.config)==CONFIG_SHA,'Native S0 config hash mismatch')
    selection=json.loads(Path(a.selection).read_text())
    require(selection['scope']=='engineering_only_two_cached_validation_anchors' and len(selection['records'])==2,'Only fixed two-anchor engineering scope')
    # Keep potential NumPy/Numba import caches out of frozen repository paths.
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    import torch
    from mmcv import Config
    from mmcv.parallel import DataContainer
    from mmdet3d.datasets import build_dataset
    from native_state_cache import _prepare_repo,_selected_rows,_identity_from_metas
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    torch.set_num_threads(2)
    try:
        import cv2
        cv2.setNumThreads(2)
    except ImportError:pass
    random.seed(11);np.random.seed(11);torch.manual_seed(11)
    _prepare_repo(a.repo)
    cfg=Config.fromfile(a.config);dc=copy.deepcopy(cfg.data.test);dc.test_mode=True;dc.pop('samples_per_gpu',None)
    require(dc.radar_cfg.cache_readonly is True and dc.radar_cfg.nsweeps==5 and dc.future_metadata_only is True,'Readonly native test pipeline required')
    require(dc.get('radar_observation_cfg') is None and not dc.get('allow_dual_radar_inputs'),'Alternate radar pipeline forbidden')
    require(sha(dc.ann_file)==selection['ann_sha256'],'Annotation hash mismatch')
    event('DATASET_BUILD_NO_MODEL')
    dataset=build_dataset(dc)
    require(dataset.queue_length==2 and dataset.future_length==4 and list(dataset.rand_frame_interval)==[1],'Native seven-frame temporal identity required')
    rows=_selected_rows(selection,'validation',dataset)
    require([r['official_index'] for r in rows]==sorted(r['official_index'] for r in rows),'Selection order changed')
    transforms=[x for x in dataset.pipeline.transforms if type(x).__name__=='LoadOccupancy']
    require(len(transforms)==1,'Expected exactly one active LoadOccupancy')
    loader=transforms[0];loader_source=Path(inspect.getsourcefile(type(loader))).resolve()
    require(sha(loader_source)==LOADER_SHA,'Actual active loader differs from audited source')
    require(loader.time_history_field==2 and loader.time_future_field==4 and loader.use_fine_occ and
            not loader.use_separate_classes and not loader.use_background_classes,'Unexpected occupancy label contract')
    shape=np.asarray(loader.grid_size);require(shape.tolist()==[512,512,40],'Native GT shape required')
    def pose(info,kind):
        return transform_matrix(info[kind+'_translation'],Quaternion(info[kind+'_rotation']),inverse=False)
    candidates=[];chosen=None
    for row in rows:
        index=row['data_info_index'];origin=dataset.data_infos[index];E0=pose(origin,'ego2global')
        conditions=[]
        for h in range(1,5):
            info=dataset.data_infos[index+h]
            require(str(info['scene_token'])==row['scene_token'],'Cross-scene future')
            relative=np.linalg.inv(E0)@pose(info,'ego2global')
            translation=float(np.linalg.norm(relative[:3,3]))
            angle=float(np.arccos(np.clip((np.trace(relative[:3,:3])-1)/2,-1,1)))
            conditions.append(dict(horizon=h,translation_m=translation,rotation_radians=angle,
                                   nonzero=translation>1e-4 or angle>1e-6))
        candidates.append(dict(identity=row,future_ego=conditions))
        if any(x['nonzero'] for x in conditions):chosen=row;break
    write(out/'selection_receipt.json',dict(rule='First frozen anchor with any future ego translation>1e-4m or rotation>1e-6rad; no prediction or label-content selection.',candidates_checked=candidates,chosen=chosen))
    require(chosen is not None,'Both preselected anchors have zero future ego motion; no replacement')
    capture=dict(world=[],voxels=[]);native_get=loader.get_seq_occ;native_world=loader.world2voxel
    def observe_get(self,results):
        require('metadata' not in capture,'Multiple label-sequence calls')
        capture['metadata']={k:copy.deepcopy(results[k]) for k in ['scene_token_list','lidar_token_list','egopose_list','ego2lidar_list']}
        result=native_get(results);capture['gt']=result.detach().cpu().clone();return result
    def observe_world(self,world):
        result=native_world(world);capture['world'].append(np.array(world,copy=True));capture['voxels'].append(np.array(result,copy=True));return result
    loader.get_seq_occ=types.MethodType(observe_get,loader);loader.world2voxel=types.MethodType(observe_world,loader)
    event('ACTUAL_TEST_PIPELINE',sample_token=chosen['sample_token'],official_index=chosen['official_index'])
    try:example=dataset._prepare_data_info(chosen['data_info_index'],rand_interval=None)
    finally:loader.get_seq_occ=native_get;loader.world2voxel=native_world
    require(example is not None,'Fixed selected example failed; no replacement')
    metas=example['img_metas'].data;identity=_identity_from_metas(metas)
    require(all(identity[k]==chosen[k] for k in ['sample_token','scene_token']),'Actual pipeline anchor identity mismatch')
    require(len(capture['world'])==len(capture['voxels'])==7,'Seven actual loader frames required')
    gt=example['segmentation'];gt=gt.data if isinstance(gt,DataContainer) else gt
    require(torch.is_tensor(gt) and gt.shape==(7,512,512,40) and torch.equal(gt.cpu(),capture['gt']),'Dataset altered loaded GT sequence')
    gt=gt.cpu().numpy();meta=metas[max(metas)];index=chosen['data_info_index']
    frame_infos=[dataset.data_infos[index-2+k] for k in range(7)]
    G0=pose(frame_infos[2],'ego2global')@pose(frame_infos[2],'lidar2ego')
    require(capture['metadata']['lidar_token_list'][2]==str(frame_infos[2]['lidar_token']),'Loader present pose is not anchor')
    frames=[];source_files={str(Path(a.config).resolve()):sha(a.config),str(Path(a.selection).resolve()):sha(a.selection),
        str(Path(dc.ann_file).resolve()):sha(dc.ann_file),str(loader_source):sha(loader_source)}
    for k,info in enumerate(frame_infos):
        require(capture['metadata']['scene_token_list'][k]==str(info['scene_token']) and
                capture['metadata']['lidar_token_list'][k]==str(info['lidar_token']),'Pose/source sequence identity mismatch')
        rawpath=Path(loader.occ_path)/('scene_'+str(info['scene_token']))/'occupancy'/(str(info['lidar_token'])+'.npy')
        raw=np.load(rawpath,allow_pickle=False);source_files[str(rawpath.resolve())]=sha(rawpath)
        local=loader.voxel2world(raw[:,[2,1,0]]+.5)
        G=pose(info,'ego2global')@pose(info,'lidar2ego');to_ref=np.linalg.inv(G0)@G
        direct=(np.c_[local,np.ones(len(local))]@to_ref.T)[:,:3]
        observed=capture['world'][k];voxels=capture['voxels'][k]
        transform_error=float(np.max(np.abs(direct-observed),initial=0.))
        require(transform_error<1e-7,'Loader differs from independent current-reference rigid conversion')
        rebuilt,clipped,vote=vote_and_remap(voxels,raw[:,-1],shape)
        differences=int(np.count_nonzero(rebuilt!=gt[k]));require(differences==0,'Exact native-coordinate majority-vote/GMO GT reconstruction failed')
        direct_voxels=loader.world2voxel(direct)
        independent_indices=np.clip(direct_voxels,0,shape-1).astype(np.int64)
        quantization_difference=int(np.any(independent_indices!=clipped,axis=1).sum())
        # Stable source-row examples only, never selected by model predictions.
        foreground=np.isin(raw[:,-1],[2,3,4,5,6,7,9,10]);outside=((voxels<0)|(voxels>=shape)).any(1)
        witness_ids=list(dict.fromkeys(np.flatnonzero(foreground&~outside)[:3].tolist()+np.flatnonzero(outside)[:2].tolist()+[0]))[:6]
        witnesses=[dict(source_row=int(j),source_voxel_zyx=raw[j,:3].tolist(),source_class=int(raw[j,-1]),
            source_lidar_xyz=local[j].tolist(),anchor_xyz=observed[j].tolist(),anchor_voxel_before_clip=voxels[j].tolist(),
            actual_destination_voxel_xyz=clipped[j].tolist(),actual_binary_GT=int(gt[k][tuple(clipped[j])])) for j in witness_ids]
        frames.append(dict(sequence_index=k,horizon_seconds=(k-2)*.5,source_lidar_token=str(info['lidar_token']),
            source_file=str(rawpath.resolve()),source_sha256=source_files[str(rawpath.resolve())],
            source_shape=list(raw.shape),source_to_anchor_column_matrix=to_ref.tolist(),
            continuous_transform_max_abs_error_m=transform_error,
            independent_float_order_quantized_index_difference_rows=quantization_difference,
            exact_native_coordinate_GT_reconstruction_mismatches=differences,
            reconstructed_GT_sha256=array_sha(rebuilt),actual_GT_sha256=array_sha(gt[k]),
            GT_counts_0_1_255=[int((gt[k]==c).sum()) for c in [0,1,255]],
            clip=roi_counts(voxels,clipped,shape),votes=vote,witnesses=witnesses))
    # Execute the actual native geometry only; construct neither nn.Module nor checkpoint.
    detector_module=importlib.import_module('projects.mmdet3d_plugin.bevformer.detectors.drive_occworld')
    detector_source=Path(detector_module.__file__).resolve();require(sha(detector_source)==DETECTOR_SHA,'Native geometry source changed')
    utils=detector_module.e2e_predictor_utils;grid_source=Path(utils.__file__).resolve()
    grid_cpu=types.SimpleNamespace(get_bev_grids=lambda *args,**kw:utils.get_bev_grids(*args,**dict(kw,device='cpu')),
        bev_grids_to_coordinates=utils.bev_grids_to_coordinates,bev_coords_to_grids=utils.bev_coords_to_grids)
    namespace=dict(np=np,torch=torch,e2e_predictor_utils=grid_cpu)
    exec(source_function(detector_source,'_align_bev_coordnates'),namespace)
    dummy=types.SimpleNamespace(bev_h=200,bev_w=200,point_cloud_range=list(loader.pc_range),
                               future_pred_head=types.SimpleNamespace(use_plan_traj=False))
    geometries=[];pc=loader.pc_range;T_ref_history=torch.eye(4,dtype=torch.float32).reshape(1,1,4,4)
    for h in range(1,5):
        native=namespace['_align_bev_coordnates'](dummy,h,T_ref_history,[meta],torch.zeros(1,h,2))
        target,aligned,ref2future,_,future_to_ref_grid=native
        expected=frames[h+2]['source_to_anchor_column_matrix'];actual=np.asarray(meta['future2ref_lidar_transform'][h])
        require(np.allclose(actual,np.asarray(expected).T,rtol=0,atol=1e-9),'Dataset future2ref differs from actual loader pose chain')
        query=utils.bev_grids_to_coordinates(target,dummy.point_cloud_range).numpy()[0]
        ref_native=utils.bev_grids_to_coordinates(aligned[:,:,0],dummy.point_cloud_range).numpy()[0]
        ref_direct=(np.c_[query,np.ones((len(query),2))]@actual)[:,:2]
        error=float(np.abs(ref_direct-ref_native).max());require(error<1e-4,'Native CPU query mapping differs from matrix')
        inside=lambda x:np.logical_and(x>=pc[:2],x<pc[3:5]).all(1)
        future_from_ref=(np.c_[query,np.ones((len(query),2))]@np.asarray(meta['ref2future_lidar_transform'][h]))[:,:2]
        grids=aligned.numpy()[0,:,0];strict=(np.logical_and(grids>np.array([.5/200,.5/200]),grids<np.array([199.5/200,199.5/200]))).all(1)
        selected_indices=[0,101*200+73,20000,39999]
        geometries.append(dict(horizon_seconds=h*.5,query_grid_shape=list(target.shape),query_plane_z=1.0,
            native_cpu_query_max_abs_vs_independent_m=error,total_query_centers=len(query),
            future_grid_queries_sampling_inside_anchor_xy_ROI=int(inside(ref_native).sum()),
            future_grid_queries_sampling_outside_anchor_xy_ROI=int((~inside(ref_native)).sum()),
            future_grid_queries_inside_strict_interpolation_center_bounds=int(strict.sum()),
            anchor_grid_centers_covered_by_future_xy_ROI=int(inside(future_from_ref).sum()),
            anchor_grid_centers_outside_future_xy_ROI=int((~inside(future_from_ref)).sum()),
            future2ref_row_matrix=actual.tolist(),reference_to_future_row_matrix=ref2future.numpy()[0].tolist(),
            witnesses=[dict(token_index=i,query_future_xy=query[i].tolist(),sampled_anchor_xy=ref_native[i].tolist()) for i in selected_indices],
            limitation='Counts are native 200x200 BEV query centers on the z=1 plane, not exact 3D volume coverage or visibility. Deformable offsets are not used.'))
    for module in [detector_module,utils,importlib.import_module(type(dataset).__module__),
        importlib.import_module('projects.mmdet3d_plugin.datasets.nuscenes_world_dataset_template')]:
        path=Path(module.__file__).resolve();source_files[str(path)]=sha(path)
    for filename in ['native_state_cache.py','real_coordinate_audit.py']:
        path=Path(__file__).with_name(filename).resolve();source_files[str(path)]=sha(path)
    return dict(schema='m0-real-coordinate-audit-v1',status='PASS_REAL_LOADER_AND_GEOMETRY_AUDIT',
        chosen=chosen,selection_rule=candidates,source_files_sha256=source_files,
        config_sha256=CONFIG_SHA,selection_sha256=SELECTION_SHA,
        seven_frame_GT_dataset_equals_loader_bitwise=True,frames=frames,future_query_geometry=geometries,
        conclusion='For this fixed real anchor, actual pipeline GT is reconstructed exactly in the observation-anchor LiDAR frame; native geometry queries use future-to-reference sampling. This verifies the explicit geometric-frame contract, not the frame semantics of learned checkpoint logits.',
        limitations=['No model/checkpoint loaded, no learned-offset or accuracy diagnosis.',
            'Exact GT reconstruction uses captured native floating coordinates before clipping/voting; independent rigid transforms are compared in meters and any quantized boundary differences reported separately.',
            'Source occupancy arrays are interpreted by the unchanged native loader as per-time LiDAR-local grids; no independent external survey of dataset label generation is claimed.',
            'Clipped out-of-ROI source points are retained by the loader at the boundary; no new mask or label policy was applied.'],
        optimizer_steps=0,prediction_based_selection=False,data_modified=False)


def self_test():
    shape=[4,5,2];coords=np.array([[1.2,2.2,.5],[1.2,2.2,.5],[1.2,2.2,.5],[7.,2.,.1],[-2.,0.,.1]])
    labels=np.array([4,1,1,4,255]);dense,indices,counts=vote_and_remap(coords,labels,shape)
    assert dense[1,2,0]==0 and dense[3,2,0]==1 and dense[0,0,0]==255
    assert roi_counts(coords,indices,shape)['physical_outside_anchor_ROI_rows']==2
    # Tie chooses original lower class BEFORE binary grouping.
    d,_,_=vote_and_remap(np.array([[1.1,1.1,.1]]*2),np.array([4,1]),shape);assert d[1,1,0]==0
    # Upper in-ROI fractional coordinate clipping changes no integer index.
    v=np.array([[3.5,2.,.1]]);_,c,_=vote_and_remap(v,np.array([4]),shape);r=roi_counts(v,c,shape)
    assert r['native_clip_changed_numeric_coordinates_rows']==1 and r['physical_outside_anchor_ROI_rows']==r['native_clip_changed_integer_voxel_rows']==0
    print(json.dumps(dict(status='PASS_NUMPY_VOTE_CLIP_TESTS')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','selection','selection-sha256','out','repo']:p.add_argument('--'+name)
    p.add_argument('--max-seconds',type=int,default=600);p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return 0
    require(all(getattr(a,k) for k in ['config','selection','selection_sha256','out']),'Missing fixed audit inputs')
    require(0<a.max_seconds<=1200,'Bounded CPU audit required')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def stop(number,_frame):raise InterruptedError('CPU coordinate audit signal '+str(number))
    for s in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(s,stop)
    signal.alarm(a.max_seconds)
    try:
        result=run(a,out);result['seconds']=time.monotonic()-start;result['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(out/'result.json',result);write(out/'complete.json',dict(status=result['status'],result_sha256=sha(out/'result.json'),seconds=result['seconds'],optimizer_steps=0))
        event('COMPLETE',out=str(out));return 0
    except BaseException as exc:
        write(out/'failed.json',dict(status='FAILED_OR_REJECTED',error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start,optimizer_steps=0))
        traceback.print_exc();return 1
    finally:signal.alarm(0)


if __name__=='__main__':raise SystemExit(main())
