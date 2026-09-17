"""Authenticated current CRN train512 states; labels never define state tokens.

The prediction file already contains geometric centres. No second origin
correction is applied. The raw metadata carrier is opened only to extract the
current calibrated LiDAR pose; annotations and future poses are not returned.
"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
RAW_MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
RAW_COMPLETE_SHA = 'e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69'
ORIGIN_SHA = 'aa05e4cd8d2c9ffbf41e751ebc0bda7114040753e404c0560975ee134e23a101'
CHECKPOINT_SHA = 'f725aafc7f033f484f13704134178f2377b3e12bb7e85207b447f8a3ebfa0cb3'
GEOMETRY_SHA = 'e9d232c3f7aaacd073cfda645868e357afad9e5a2684d07b2f3f2394bd796a31'
GMO = ('car','bus','truck','trailer','construction_vehicle','motorcycle','bicycle','pedestrian')
IDENTITY = ('sample_token','scene_token','official_index','split')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def geometry():
    path=Path(__file__).with_name('motion_geometry.py')
    require(sha(path)==GEOMETRY_SHA,'Frozen geometry differs')
    spec=importlib.util.spec_from_file_location('_object_state_geometry_v1',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def current_states_numpy(boxes, G0):
    """Global geometric boxes -> current LiDAR R, with no GT argument."""
    geo=geometry(); pose=geo._rigid_pose(G0,'current LiDAR to global')
    inverse=np.linalg.inv(pose); rows=[]; retained=[]
    for index,box in enumerate(boxes):
        if box['detection_name'] not in GMO: continue
        center=geo._finite_array(box['translation'],(3,),'predicted center')
        size=geo._finite_array(box['size'],(3,),'predicted wlh')
        rotation=geo.quaternion_wxyz_to_matrix(box['rotation'])
        velocity=geo._finite_array(box['velocity'],(2,),'predicted global xy velocity')
        score=float(box['detection_score'])
        require(np.isfinite(score) and np.all(size>0),'Invalid predicted size/score')
        rows.append((inverse[:3,:3]@center+inverse[:3,3],size,
                     inverse[:3,:3]@rotation,GMO.index(box['detection_name']),score,
                     inverse[:3,:3]@np.array([velocity[0],velocity[1],0.],dtype=np.float64)))
        retained.append(index)
    n=len(rows)
    result=dict(center=np.empty((n,3)),size=np.empty((n,3)),rotation=np.empty((n,3,3)),
                classes=np.empty(n,dtype=np.int64),score=np.empty(n),velocity=np.empty((n,3)))
    for i,row in enumerate(rows):
        for key,value in zip(result,row): result[key][i]=value
    return result,retained


class PredictionInputs:
    def __init__(self, prediction_root, raw_root, selection_path, contract, train):
        root=Path(prediction_root); self.raw_root=Path(raw_root)
        require(set(contract)=={'complete_sha256','manifest_sha256','predictions_sha256',
                'inference_source_sha256'},'Prediction contract fields differ')
        for key,name in (('complete_sha256','complete.json'),('manifest_sha256','manifest.json'),
                         ('predictions_sha256','predictions.json')):
            require(sha(root/name)==contract[key],'Prediction asset SHA differs: '+name)
        complete=read(root/'complete.json');manifest=read(root/'manifest.json');data=read(root/'predictions.json')
        require(complete['schema']=='crn-train512-inference-complete-v1' and
                complete['status']=='COMPLETE_CRN_TRAIN512_INFERENCE' and complete['mode']=='full'
                and complete['samples']==512 and complete['optimizer_updates']==0,'Not completed official train512 inference')
        require(set(complete['files_sha256'])=={'manifest.json','predictions.json','raw_export/results_nusc.json'},
                'Prediction ledger differs')
        for name,digest in complete['files_sha256'].items(): require(sha(root/name)==digest,'Prediction file differs: '+name)
        require(manifest['schema']=='crn-train512-inference-v1' and manifest['mode']=='full'
                and manifest['selected_samples']==512 and manifest['selection_ordinals']==list(range(512))
                and manifest['full_train_infos_samples']==28130,'Prediction sample selection differs')
        require(manifest['official_commit']=='5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d'
                and manifest['checkpoint_sha256']==complete['checkpoint_sha256']==CHECKPOINT_SHA
                and manifest['source_sha256']==complete['source_sha256']==contract['inference_source_sha256']
                and manifest['origin_adapter_sha256']==complete['origin_adapter_sha256']==ORIGIN_SHA,
                'Official checkpoint/source/origin chain differs')
        for key in ('assets_complete_sha256','source_contract_sha256','selection_sha256'):
            require(manifest[key]==complete[key],'Prediction provenance differs: '+key)
        require(manifest['selection_sha256']==sha(selection_path)==SELECTION_SHA and
                manifest['model_training'] is False and manifest['optimizer_updates']==0 and
                manifest['GT_batch_passed_to_model'] is False and manifest['initial_model_state_equals_final'] is True,
                'Prediction input or training scope differs')
        require(manifest['raw_export_sha256']==complete['files_sha256']['raw_export/results_nusc.json'],
                'Raw export chain differs')
        require(data['schema']=='crn-state-train512-predictions-v1' and data['box_origin']=='global_geometric_center'
                and data['raw_export_origin']=='global_transformed_ego_bottom_center'
                and data['center_conversion']=='translation + R(export_quaternion) @ [0,0,height/2]',
                'Origin contract differs; do not apply origin correction twice')
        chosen=[r for r in read(selection_path)['records'] if r['split']=='train']
        records=data['records']
        require(len(records)==len(train)==len(chosen)==512 and len({r['scene_token'] for r in records})==256,
                'Train512/256 scene coverage differs')
        for i,(r,t,s) in enumerate(zip(records,train,chosen)):
            require(r['ordinal']==i and all(r[k]==t[k]==s[k] for k in IDENTITY) and r['split']=='train',
                    'Prediction/train/selection identity mismatch')
            require(all(b['sample_token']==r['sample_token'] for b in r['boxes']),'Box token differs')
        require(sum(len(r['boxes']) for r in records)==manifest['selected_boxes'],'Box count differs')
        self.records=records
        require(sha(self.raw_root/'manifest.json')==RAW_MANIFEST_SHA and
                sha(self.raw_root/'complete.json')==RAW_COMPLETE_SHA,'Raw pose metadata chain differs')
        raw_manifest=read(self.raw_root/'manifest.json');raw_complete=read(self.raw_root/'complete.json')
        require(raw_complete['manifest_sha256']==RAW_MANIFEST_SHA and raw_manifest['selection']['sha256']==SELECTION_SHA,
                'Raw metadata selection differs')
        self.raw={r['identity']['sample_token']:r for r in raw_manifest['records'] if r['identity']['split']=='train'}
        self.receipt=dict(contract,selection_sha256=SELECTION_SHA,checkpoint_sha256=CHECKPOINT_SHA,
            origin_adapter_sha256=ORIGIN_SHA,raw_manifest_sha256=RAW_MANIFEST_SHA,raw_complete_sha256=RAW_COMPLETE_SHA,
            samples=512,scenes=256,boxes=manifest['selected_boxes'],origin_adaptation_in_consumer=False,
            current_pose_only=True,annotation_or_future_pose_input=False,new_score_threshold=False)

    def get(self, ordinal, core, device):
        import torch
        record=self.records[ordinal];descriptor=self.raw[record['sample_token']]
        require(descriptor['ordinal']==ordinal and all(descriptor['identity'][k]==record[k] for k in IDENTITY),
                'Current pose identity mismatch')
        path=self.raw_root/descriptor['file'];require(sha(path)==descriptor['sha256'],'Raw pose carrier differs')
        content=gzip.decompress(path.read_bytes())
        require(hashlib.sha256(content).hexdigest()==descriptor['uncompressed_json_sha256'],'Raw JSON differs')
        carrier=json.loads(content)
        require(carrier['identity']==descriptor['identity'] and carrier['ordinal']==ordinal and
                carrier['selection_sha256']==SELECTION_SHA,'Raw carrier identity differs')
        # Only this calibrated current pose enters any token. Discard annotations.
        frame=carrier['frames'][2]
        require(frame['relative_frame_index']==0 and frame['sample_token']==record['sample_token'],'Wrong current pose')
        G0=frame['lidar_to_global_column_matrix'];del frame,carrier,content
        arrays,retained=current_states_numpy(record['boxes'],G0)
        def f(key): return torch.as_tensor(arrays[key],dtype=torch.float32,device=device).unsqueeze(0)
        states,valid=core.pack_object_states(f('center'),f('size'),f('rotation'),
            torch.as_tensor(arrays['classes'],dtype=torch.int64,device=device).unsqueeze(0),f('score'),f('velocity'))
        require(not states.requires_grad and not valid.requires_grad,'State input cannot be trainable')
        receipt=dict(ordinal=ordinal,sample_token=record['sample_token'],raw_pose_carrier_sha256=descriptor['sha256'],
            current_G0_sha256=hashlib.sha256(np.asarray(G0,dtype='<f8').tobytes()).hexdigest(),
            retained_original_box_indices=retained,total_boxes=len(record['boxes']),GMO_boxes=len(retained),
            packed_states_sha256=hashlib.sha256(states.detach().cpu().numpy().tobytes()).hexdigest(),
            geometry_sha256=hashlib.sha256(states[...,:24].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            velocity_slots='global [vx,vy,0] rotated to R, /20; G zeros only these slots',
            pose_source='current calibration/ego pose only; no annotations/future frames input')
        return states,valid,receipt
