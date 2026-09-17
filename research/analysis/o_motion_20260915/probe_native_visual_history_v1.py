"""Real-image H3/native parity and legal H1 inference, without training.

Uses the original H3 dataset/GT and fixed O head. Selection currently occurs
after image preparation, so backbone counts are not an end-to-end speed test.
Radar retains native five sweeps; this is not a Doppler attribution experiment.
"""
import argparse,contextlib,copy,functools,hashlib,inspect,json,os,signal,time,types
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def equivalent(a,b,torch):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and torch.equal(a,b)
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict) or isinstance(b,dict):
        return isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b) and all(equivalent(a[k],b[k],torch) for k in a)
    if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(equivalent(x,y,torch) for x,y in zip(a,b))
    return a==b


@contextlib.contextmanager
def capture(model,native,torch,trace):
    future=model.future_pred;evaluate=model.evaluate_occ_records;ref=model.obtain_ref_bev
    signature=inspect.signature(future)
    def observed_ref(self,img,*args,**kwargs):
        trace['current_img_tensor_sha256']=native._tensor_digest(img)
        trace['current_img_shape']=list(img.shape)
        return ref(img,*args,**kwargs)
    def observed_future(self,*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();inputs=dict(bound.arguments)
        native.validate_inputs(inputs)
        assert 'inputs' not in trace
        trace['inputs']=native._tree_map(inputs,lambda x:x.detach().cpu().clone())
        previous=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        try:result=future(*args,**kwargs)
        finally:torch.backends.cuda.matmul.allow_tf32=previous
        assert result[0].dtype==torch.float32 and bool(torch.isfinite(result[0]).all())
        trace['preds']=result[0].detach().cpu().clone().numpy()
        return result
    def observed_eval(self,preds,targets,metas):
        output=evaluate(preds,targets,metas)
        trace['targets']=targets.detach().cpu().to(torch.uint8).numpy()
        trace['hist']=np.asarray(output[0]['hist_by_horizon'],dtype=np.int64)
        return output
    def image_hook(module,args):trace['backbone_image_batches'].append(int(args[0].shape[0]))
    trace['backbone_image_batches']=[]
    handle=model.img_backbone.register_forward_pre_hook(image_hook)
    model.future_pred=types.MethodType(observed_future,model)
    model.evaluate_occ_records=types.MethodType(observed_eval,model)
    model.obtain_ref_bev=types.MethodType(observed_ref,model)
    try:yield
    finally:
        handle.remove();model.future_pred=future;model.evaluate_occ_records=evaluate;model.obtain_ref_bev=ref


def run(a):
    started=time.monotonic();p=read(a.protocol)
    assert sha(a.protocol)==a.protocol_sha256 and p['status']=='FROZEN'
    assert p['schema']=='native-visual-history-input-probe-v1'
    for name,digest in p['sources_sha256'].items():assert sha(Path(__file__).parent/name)==digest,name
    for name,digest in p['runtime_source_sha256'].items():assert sha(name)==digest,name
    assert sha(a.config)==p['config_sha256'] and sha(a.o_checkpoint)==p['o_checkpoint_sha256']
    import torch
    import native_state_cache as native
    import common_change_evaluation_v2 as common
    from native_visual_history_window_v1 import observation_window
    native._prepare_repo(a.repo);os.chdir(a.repo)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel,collate
    from mmdet3d.datasets import build_dataset
    torch.set_num_threads(2);torch.manual_seed(11);np.random.seed(11)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    free,total=torch.cuda.mem_get_info();assert free>=40*2**30
    torch.cuda.set_per_process_memory_fraction(p['resources']['max_allocated_gib']*2**30/total)
    stop=[];signal.signal(signal.SIGTERM,lambda s,f:stop.append(s))
    def check():
        assert not stop and time.monotonic()-started<p['resources']['max_seconds']
        assert torch.cuda.max_memory_allocated()<=p['resources']['max_allocated_gib']*2**30
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);save(out/'protocol.json',p)
    index=read(Path(a.train_cache)/'index.json');assert sha(Path(a.train_cache)/'index.json')==p['train_cache_index_sha256']
    assert index['split']=='train' and len(index['records'])==512
    selected=[index['records'][i] for i in p['original_train_ordinals']]
    cfg=Config.fromfile(a.config);dc=copy.deepcopy(cfg.data.test);dc.test_mode=True;dc.pop('samples_per_gpu',None)
    dc.ann_file=cfg.data.train.ann_file
    assert dc.queue_length==2 and dc.future_metadata_only
    assert dc.radar_cfg.nsweeps==5 and dc.radar_cfg.cache_readonly
    dataset=build_dataset(dc)
    selected=native._selected_rows({'records':selected},'train',dataset)
    assert len(selected)==2 and len({r['scene_token'] for r in selected})==2
    model=native.build_native_model(a.config,a.checkpoint,'cuda',repo=a.repo)
    payload=torch.load(a.o_checkpoint,map_location='cpu');assert payload['arm']=='O' and payload['update']==512
    model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True);del payload
    assert common.historical_O_head_digest(model.future_pred_head)==p['o_head_state_sha256']
    model.requires_grad_(False);model.eval();initial=native._parameter_digest(model)
    assert model.future_pred_head.history_queue_length==2
    wrapped=MMDataParallel(model,device_ids=[0]);records=[]
    for ordinal,row in enumerate(selected):
        check();example=dataset._prepare_data_info(row['data_info_index'],rand_interval=None)
        assert example is not None
        identity=native._identity_from_metas(example['img_metas'].data)
        assert identity['sample_token']==row['sample_token'] and identity['scene_token']==row['scene_token']
        assert tuple(example['img'].data.shape[:2])==(3,6)
        contexts={};windows={}
        for mode,h in [('native_H3',None),('window_H3',3),('window_H1',1)]:
            check();data=collate([copy.deepcopy(example)],samples_per_gpu=1);data.pop('instance',None)
            trace={};wt={};manager=observation_window(model,h,wt) if h else contextlib.nullcontext()
            with manager:
                with capture(model,native,torch,trace),torch.no_grad():
                    wrapped(return_loss=False,rescale=True,**data)
            torch.cuda.synchronize();contexts[mode]=trace;windows[mode]=wt;del data
        base=contexts['native_H3'];same=contexts['window_H3'];short=contexts['window_H1']
        assert equivalent(base['inputs'],same['inputs'],torch),'H3 wrapper altered native future inputs'
        assert np.array_equal(base['preds'],same['preds']),'H3 wrapper changed full native logits'
        assert np.array_equal(base['targets'],same['targets']) and np.array_equal(base['hist'],same['hist'])
        assert sum(base['backbone_image_batches'])==sum(same['backbone_image_batches'])==18
        assert sum(short['backbone_image_batches'])==6
        assert base['current_img_tensor_sha256']==same['current_img_tensor_sha256']==short['current_img_tensor_sha256']
        assert np.array_equal(base['targets'],short['targets'])
        assert tuple(short['targets'].shape)==(1,7,512,512,40)
        for key in ['action_condition_dict','cond_norm_dict','plan_dict','radar_bev','valid_frames','occ_flow','motion_state']:
            assert equivalent(base['inputs'][key],short['inputs'][key],torch),key
        for left,right in zip(base['inputs']['img_metas'],short['inputs']['img_metas']):
            assert set(left)==set(right)
            for key in left:
                if key not in ['can_bus','prev_bev_exists']:assert equivalent(left[key],right[key],torch),key
            assert not right['prev_bev_exists'] and np.all(right['can_bus'][:3]==0) and right['can_bus'][-1]==0
        assert short['inputs']['num_frames']==1 and base['inputs']['num_frames']==3
        assert tuple(short['inputs']['prev_bev_input'].shape)==(1,1,40000,256)
        # The native decoder's original history/GT layout remains untouched.
        assert model.future_pred_head.history_queue_length==2 and model.memory_queue_len==1
        item=dict(ordinal=ordinal,sample_token=row['sample_token'],scene_token=row['scene_token'],
            original_train_ordinal=p['original_train_ordinals'][ordinal],native_H3_wrapper_all_logits_exact=True,
            native_H3_wrapper_future_inputs_exact=True,GT_full_tensor_exact=True,GT_shape=list(base['targets'].shape),
            GT_sha256=hashlib.sha256(base['targets'].tobytes()).hexdigest(),
            future_ego_action_and_query_metadata_exact=True,current_img_tensor_sha256=base['current_img_tensor_sha256'],
            radar5_tensor_sha256=native._tensor_digest(base['inputs']['radar_bev']),
            camera_images_encoded={k:sum(v['backbone_image_batches']) for k,v in contexts.items()},
            backbone_batches={k:v['backbone_image_batches'] for k,v in contexts.items()},
            H1_current_state_equal_H3=torch.equal(short['inputs']['prev_bev_input'],base['inputs']['prev_bev_input']),
            H1_state_abs_difference_mean=float((short['inputs']['prev_bev_input']-base['inputs']['prev_bev_input']).abs().mean()),
            history_windows=windows,optimizer_updates=0)
        records.append(item);save(out/'records.json',records)
        save(out/'progress.json',dict(completed=len(records),total=2,elapsed_seconds=time.monotonic()-started))
        del contexts,base,same,short,example
    assert native._parameter_digest(model)==initial
    done=dict(status='COMPLETE_NATIVE_H1_INPUT_PREFLIGHT',samples=2,scenes=2,
        protocol_sha256=a.protocol_sha256,optimizer_updates=0,model_parameters_unchanged=True,
        parameters_sha256=initial,elapsed_seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        claims=['H3 wrapper reproduces native input and all logits exactly','H1 encodes six current images with explicit empty history','Original GT and future conditions preserved'],
        limitations=['H1 is inference-time intervention, not a trained baseline','Native five-sweep radar retained','Original H3 preprocessing still occurs; no end-to-end speed or efficiency claim','Two train anchors only; no performance conclusion'],
        files_sha256={name:sha(out/name) for name in ['protocol.json','records.json']})
    save(out/'complete.json',done);print(json.dumps(done),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ['protocol','protocol-sha256','repo','config','checkpoint','o-checkpoint','train-cache','out']:ap.add_argument('--'+name,required=True)
    run(ap.parse_args())
