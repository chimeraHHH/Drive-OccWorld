"""Formal-only current CRN dev200 states from a separately centered asset.

The old raw extraction and one-time origin adapter remain fully identified by
manifest hashes. This consumer never applies the correction a second time.
No metric, GT matching, detector forward or threshold selection occurs.
"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

INPUT_LOADER_SHA='55e39715fdf19e5ae1f227a943a843b194eabed08e788ccb2694a040d6f18256'
COMPLETE_SHA='ddeff445a45608955c7b6b5b164d9b70786d95e042c05a1bfce87c39fd0ea226'
MANIFEST_SHA='152c86fb03ee3695358fc7ad4d5731f6b21a77db560af6815329535d7c44c275'
PREDICTIONS_SHA='b673ebbce36d7fa101a0c085b7e0115fec11a5e4bd2874bb4d28d4448a2d647e'
EXPORT_SHA='fec6541461bf6eb0ad5f3e113e4fcb13c937896731203ef5de003a6ac0055653'
ORIGIN_CONTRACT_SHA='c4b99aadab4bdd20485c3a544ba9cd735441101f59ec228230a6d79f5d46a51b'
ORIGIN_PROOF_SHA='2d9b711857e82ff1c3e88be16064a05d12b03d63f23f1d2545941c5b6fca0149'


def bound(name,digest):
    p=Path(__file__).with_name(name+'.py')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source differs: '+name)
    s=importlib.util.spec_from_file_location('_formal_prediction_'+name,p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


inputs=bound('object_state_prediction_inputs_v1',INPUT_LOADER_SHA)
require,sha,read=inputs.require,inputs.sha,inputs.read


class DevelopmentPredictionInputs:
    def __init__(self,root,raw_root,selection_path,dev,contract):
        root=Path(root);self.raw_root=Path(raw_root)
        require(set(contract)=={'complete_sha256','manifest_sha256','predictions_sha256'},'Centered dev contract fields differ')
        for name,key in (('complete.json','complete_sha256'),('manifest.json','manifest_sha256'),('predictions.json','predictions_sha256')):
            require(sha(root/name)==contract[key],'Centered development asset differs: '+name)
        done=read(root/'complete.json');manifest=read(root/'manifest.json');data=read(root/'predictions.json')
        require(done['schema']=='crn-state-dev200-centering-complete-v1' and
                done['status']=='COMPLETE_CRN_STATE_DEV200_CENTERING' and
                done['files_sha256']=={'predictions.json':contract['predictions_sha256'],'manifest.json':contract['manifest_sha256']},
                'Centered development extraction incomplete')
        require(manifest['schema']=='crn-state-dev200-centering-manifest-v1' and
                manifest['selection_sha256']==sha(selection_path)==inputs.SELECTION_SHA and
                manifest['raw_extraction']==dict(complete_sha256=COMPLETE_SHA,manifest_sha256=MANIFEST_SHA,predictions_sha256=PREDICTIONS_SHA)
                and manifest['checkpoint_sha256']==inputs.CHECKPOINT_SHA and
                manifest['official_commit']=='5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d','Dev official provenance changed')
        require(manifest['origin_adapter_sha256']==inputs.ORIGIN_SHA and
                manifest['origin_contract_sha256']==ORIGIN_CONTRACT_SHA and manifest['origin_proof_sha256']==ORIGIN_PROOF_SHA,
                'Missing actual CRN bottom-centre evidence')
        require(data['schema']=='crn-state-dev200-centered-predictions-v1' and data['box_origin']=='global_geometric_center',
                'Expected separate centered dev asset; do not apply origin adaptation again')
        chosen=[r for r in read(selection_path)['records'] if r['split']=='development']
        records=data['records'];require(len(records)==len(dev)==len(chosen)==200 and
                len({r['scene_token'] for r in records})==100,'Development coverage changed')
        for i,(r,d,s) in enumerate(zip(records,dev,chosen)):
            require(r['ordinal']==i and all(r[k]==d[k]==s[k] for k in inputs.IDENTITY), 'Dev identity/order differs')
            require(all(b['sample_token']==r['sample_token'] for b in r['boxes']),'Dev box token differs')
        require(sum(len(r['boxes']) for r in records)==94892==manifest['selected_boxes'],'Dev box coverage differs')
        require(manifest['selected_samples']==200 and manifest['selected_scenes']==100,'Manifest coverage differs')
        self.records=records
        require(sha(self.raw_root/'manifest.json')==inputs.RAW_MANIFEST_SHA and
                sha(self.raw_root/'complete.json')==inputs.RAW_COMPLETE_SHA,'Current pose metadata chain differs')
        self.raw={r['identity']['sample_token']:r for r in read(self.raw_root/'manifest.json')['records']
                  if r['identity']['split']=='development'}
        self.receipt=dict(contract,raw_extraction=dict(complete_sha256=COMPLETE_SHA,manifest_sha256=MANIFEST_SHA,predictions_sha256=PREDICTIONS_SHA),
            export_sha256=EXPORT_SHA,checkpoint_sha256=inputs.CHECKPOINT_SHA,origin_adapter_sha256=inputs.ORIGIN_SHA,
            origin_contract_sha256=ORIGIN_CONTRACT_SHA,origin_proof_sha256=ORIGIN_PROOF_SHA,
            samples=200,scenes=100,boxes=94892,origin_adapted_once_by_asset_producer=True,origin_adaptation_in_consumer=False,detector_rerun=False,
            raw_manifest_sha256=inputs.RAW_MANIFEST_SHA,raw_complete_sha256=inputs.RAW_COMPLETE_SHA)

    def get(self,ordinal,core,device):
        import torch
        record=self.records[ordinal];desc=self.raw[record['sample_token']]
        require(desc['ordinal']==512+ordinal and all(desc['identity'][k]==record[k] for k in inputs.IDENTITY),
                'Development current-pose identity differs')
        path=self.raw_root/desc['file'];require(sha(path)==desc['sha256'],'Current pose carrier differs')
        content=gzip.decompress(path.read_bytes())
        require(hashlib.sha256(content).hexdigest()==desc['uncompressed_json_sha256'],'Raw JSON differs')
        carrier=json.loads(content);frame=carrier['frames'][2]
        require(carrier['identity']==desc['identity'] and carrier['ordinal']==512+ordinal and
                carrier['selection_sha256']==inputs.SELECTION_SHA and frame['relative_frame_index']==0 and
                frame['sample_token']==record['sample_token'],'Current metadata identity differs')
        G0=frame['lidar_to_global_column_matrix'];del carrier,frame,content
        arrays,retained=inputs.current_states_numpy(record['boxes'],G0)
        def f(key):return torch.as_tensor(arrays[key],dtype=torch.float32,device=device).unsqueeze(0)
        states,valid=core.pack_object_states(f('center'),f('size'),f('rotation'),
            torch.as_tensor(arrays['classes'],dtype=torch.int64,device=device).unsqueeze(0),f('score'),f('velocity'))
        receipt=dict(ordinal=ordinal,raw_global_ordinal=512+ordinal,sample_token=record['sample_token'],
            raw_pose_carrier_sha256=desc['sha256'],retained_original_box_indices=retained,
            current_G0_sha256=hashlib.sha256(np.asarray(G0,dtype='<f8').tobytes()).hexdigest(),
            packed_states_sha256=hashlib.sha256(states.detach().cpu().numpy().tobytes()).hexdigest(),
            geometry_sha256=hashlib.sha256(states[...,:24].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            total_boxes=len(record['boxes']),GMO_boxes=len(retained),origin_adaptation_in_consumer=False)
        return states,valid,receipt
