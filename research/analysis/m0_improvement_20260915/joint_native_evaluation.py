"""Stream five fixed occupancy models over identical native validation inputs.

Engineering pilot is the default and requires the existing development cache.
Full validation requires explicit --mode full and the frozen 5119-row selection.
Each sample performs one original M0 image/radar forward (TF32 enabled), captures
its original future_pred boundary/GT via native_state_cache._capture_native,
then replays four fixed future heads on cloned identical inputs, with matmul
TF32 disabled. Original future ego/action, SE(3), label slicing and full-GT
evaluator are unchanged. Only small JSON records/hashes are persisted.

Two shards use selection ordinal modulo shard_count (full official index equals
ordinal); each shard directory and global token directory are exclusive-create.
There is no resume, sample replacement, fitting, threshold selection, automatic
retry or dispatch. Dataset construction below is the exact frozen extractor
recipe; its identity/capture/replay helpers are imported, not reimplemented.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback


PROTOCOL_SHA='071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
SELECTION_SHA={
    'pilot':'e616af491e77ad977dfddfd931c1110f9580921bc38ed6ad84281dfe23a9754d',
    'full':'60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391'}
MODELS=('M0','M0_fp32','native1','persistent2','rolling2')
ARMS=MODELS[2:]
HORIZONS=[0.,.5,1.,1.5,2.]
# Observed in native_boundary_special_types.json, never inferred from a name.
# Hash the class identity AND its implementation source, not repr/address or an
# omitted metadata field. Other class-valued metadata remains unsupported.
BOUNDARY_METADATA_CLASS_SOURCES={
    ('mmdet3d.core.bbox.structures.lidar_box3d','LiDARInstance3DBoxes'):
        '8badeb5b932fd162768e25d3a94467cc0dd0aef9c11caac043cae7eca8cc00c3'}


def require(ok,message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path,value,exclusive=False):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    try:
        with tmp.open('x') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        if exclusive:os.link(str(tmp),str(path));tmp.unlink()
        else:os.replace(str(tmp),str(path))
        fd=os.open(str(path.parent),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if tmp.exists():tmp.unlink()


def event(name,**kwargs):print(json.dumps(dict(event=name,utc=utc(),**kwargs)),flush=True)


def array_digest(array):
    import numpy as np
    a=np.ascontiguousarray(array);h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def metadata_class_tag(value):
    """Stable tag for the one observed native class, with strict source binding."""
    import inspect
    import sys
    key=(value.__module__,value.__qualname__)
    if key not in BOUNDARY_METADATA_CLASS_SOURCES:
        raise TypeError('Unsupported boundary metadata class: '+'.'.join(key))
    module=sys.modules.get(key[0])
    require(module is not None and getattr(module,key[1],None) is value,
            'Boundary metadata class is not its canonical module export')
    source=inspect.getsourcefile(value)
    require(source is not None and Path(source).is_file(),
            'Boundary metadata class source is unavailable')
    source_sha=sha(source)
    require(source_sha==BOUNDARY_METADATA_CLASS_SOURCES[key],
            'Boundary metadata class source differs from observed native source')
    identity=dict(module=key[0],qualname=key[1],source_sha256=source_sha)
    return b'python_type:'+json.dumps(identity,sort_keys=True,separators=(',',':')).encode()+b';'


def tree_digest(tree):
    """Typed, deterministic full boundary hash; labels are a separate digest."""
    import numpy as np
    import torch
    h=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            h.update(b'torch:'+str(value.dtype).encode());h.update(array_digest(value.detach().cpu().numpy()).encode())
        elif isinstance(value,np.ndarray):h.update(b'numpy:');h.update(array_digest(value).encode())
        elif isinstance(value,np.generic):h.update(('numpy_scalar:'+str(value.dtype)).encode());visit(value.item())
        elif isinstance(value,dict):
            h.update(b'dict:')
            for key in sorted(value,key=lambda x:(type(x).__name__,repr(x))):visit(key);visit(value[key])
            h.update(b':enddict')
        elif isinstance(value,(list,tuple)):
            h.update(type(value).__name__.encode()+b':')
            for item in value:visit(item)
            h.update(b':endsequence')
        elif isinstance(value,type):h.update(metadata_class_tag(value))
        elif value is None or isinstance(value,(str,int,float,bool)):
            h.update(type(value).__name__.encode()+b':'+json.dumps(value,allow_nan=False).encode()+b';')
        else:raise TypeError('Unsupported boundary metadata type: '+str(type(value)))
    visit(tree);return h.hexdigest()


def select_rows(selection,mode,shard_index,shard_count):
    require(selection['schema']=='native-joint-validation-selection-v1','Selection schema')
    rows=selection['records']
    require(len(rows)==selection['samples'] and rows,'Selection count')
    require(len({r['sample_token'] for r in rows})==len(rows),'Duplicate token')
    indices=[r['official_index'] for r in rows]
    require(all(type(i) is int for i in indices) and indices==sorted(set(indices)),'Official indices must be unique/increasing')
    require(all(r['split']=='validation' for r in rows),'Native validation split required')
    require(all(len(r['sample_token'])==32 and all(c in '0123456789abcdef' for c in r['sample_token']) for r in rows),'Unsafe token/path')
    require(len({r['scene_token'] for r in rows})==selection['scenes'],'Scene count mismatch')
    if mode=='full':
        require(selection['scope']=='full_native_validation' and indices==list(range(5119)) and selection['scenes']==150,'Full selection must be every native5119/150scene')
    else:
        require(selection['scope']=='engineering_only_two_cached_validation_anchors' and len(rows)==2 and selection['scenes']==2,'Pilot is exactly two frozen scenes')
    require(shard_count in (1,2) and 0<=shard_index<shard_count,'One or two mutually exclusive shards required')
    return [dict(row,selection_ordinal=i) for i,row in enumerate(rows) if i%shard_count==shard_index]


def frozen_arms(runs,protocol):
    hp=protocol['training'];updates=hp['passes']*hp['train_samples']//hp['accumulate']
    result={}
    for arm in ARMS:
        directory=Path(runs)/arm
        names=['manifest.json','training_complete.json','complete.json','latest.pth','development_records.jsonl']
        files={name:sha(directory/name) for name in names}
        manifest,train,done=[read(directory/name) for name in names[:3]]
        require(not (directory/'failed.json').exists(),'Failed arm: '+arm)
        require(manifest['arm']==arm and manifest['seed']==11 and manifest['protocol_sha256']==PROTOCOL_SHA,'Arm/seed/protocol mismatch')
        require(manifest['m0_sha256']==protocol['m0_sha256'] and manifest['numerical_policy']==protocol['numerical_policy'],'Arm source/precision mismatch')
        require(set(manifest['source_sha256'])=={'memory_experiment.py','native_state_cache.py','observation_memory.py'},'Missing/extra trained source bindings')
        for name,digest in manifest['source_sha256'].items():require(digest==protocol['source_sha256'][name],'Arm source mismatch')
        require(manifest['migration']['mode']==arm,'Wrong trained memory policy')
        require(train['status']=='TRAINED_FIXED_FINAL' and done['status']=='TRAINED_AND_DEVELOPMENT_EVALUATED','Require completed fixed final endpoint')
        for receipt in [train,done]:
            require(receipt['updates']==updates and receipt['examples']==hp['passes']*hp['train_samples'],'Training budget mismatch')
            require(receipt['checkpoint_sha256']==files['latest.pth'],'Checkpoint hash mismatch')
        require(done['evaluation']['sha256']==files['development_records.jsonl'] and done['evaluation']['samples']==200,'Original development completion mismatch')
        result[arm]=dict(directory=str(directory.resolve()),files_sha256=files,manifest=manifest)
    require(len({r['manifest']['train_index_sha256'] for r in result.values()})==1 and
            len({r['manifest']['dev_index_sha256'] for r in result.values()})==1,'Trained arms used different caches')
    return result


def evaluator_record(row,expected,counts):
    import numpy as np
    require(row['sample_token']==expected['sample_token'] and row['scene_token']==expected['scene_token'],'Evaluator identity mismatch')
    require(list(row['horizon_seconds'])==HORIZONS,'Evaluator horizon mismatch')
    hist=np.asarray(row['hist_by_horizon'])
    require(hist.shape==(5,2,2) and np.issubdtype(hist.dtype,np.integer) and (hist>=0).all(),'Invalid native confusion')
    require(np.array_equal(hist.sum(2),counts[2:,:2]),'Native GT row counts mismatch')
    return dict(row,hist_by_horizon=hist.tolist())


def pilot_reference(cache,arm_receipts,protocol):
    root=Path(cache).resolve();done,index=read(root/'complete.json'),read(root/'index.json')
    require(done['status']=='COMPLETE_NATIVE_STATE_CACHE' and done['index_sha256']==sha(root/'index.json'),'Pilot cache not complete/bound')
    require(index['schema']=='m0-native-state-cache-v1' and index['status']=='COMPLETE' and index['split']=='development','Wrong pilot reference cache')
    require(index['config_sha256']==protocol['config_sha256'] and index['selection_sha256']==protocol['selection_sha256'],'Pilot cache config/selection')
    require(index['script_sha256']==protocol['source_sha256']['native_state_cache.py'] and index['m0_sha256']==protocol['m0_sha256'],'Pilot cache source')
    require(index['source_model_sha256']=='67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5' and
            done['all_samples_bitwise_replayed'] is True,'Pilot native source/parity contract')
    lookup={r['sample_token']:r for r in index['records']}
    require(len(lookup)==200==done['samples'],'Pilot reference needs the completed development200 cache')
    require(all(r['parity']['status']=='PASS' and r['parity']['bitwise_all_five_horizons_three_layers'] and
                r['parity']['exact_native_confusion'] for r in lookup.values()),'Pilot per-sample parity receipts')
    references={model:{} for model in MODELS}
    for token,row in lookup.items():references['M0'][token]=row['native_hist']
    sources={str(root/'complete.json'):sha(root/'complete.json'),str(root/'index.json'):sha(root/'index.json')}
    for arm in ARMS:
        receipt=arm_receipts[arm]
        require(receipt['manifest']['dev_index_sha256']==done['index_sha256'],'Reference evaluation used different cache')
        path=Path(receipt['directory'])/'development_records.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require([r['sample_token'] for r in rows]==list(lookup),'Reference evaluation order differs')
        references[arm]={r['sample_token']:r['hist_by_horizon'] for r in rows};sources[str(path)]=sha(path)
    receipt=arm_receipts['native1'];path=Path(receipt['directory'])/'frozen_native_fp32_records.jsonl'
    bound=receipt['manifest']['frozen_precision_reference']
    require(bound['sha256']==sha(path) and bound['optimizer_updates']==0 and bound['source_m0_sha256']==protocol['m0_sha256'],'Frozen FP32 reference hash/source')
    require(bound['matmul_tf32'] is False and bound['cudnn_tf32'] is True,'Frozen reference precision')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require([r['sample_token'] for r in rows]==list(lookup),'Frozen reference coverage/order')
    references['M0_fp32']={r['sample_token']:r['hist_by_horizon'] for r in rows};sources[str(path)]=sha(path)
    return root,lookup,references,sources


class Evaluation:
    def __init__(self,a):
        self.a=a;self.package=Path(__file__).resolve().parent;self.start=time.monotonic()
        self.end=self.start+a.max_seconds
        if a.deadline_unix is not None:self.end=min(self.end,self.start+max(0,a.deadline_unix-time.time()))
        self.out=Path(a.out).resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.shard=self.out/('shard_%02d_of_%02d'%(a.shard_index,a.shard_count));self.shard.mkdir(exist_ok=False)
        self.lock=(self.shard/'owner.lock').open('x');fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.results=[]

    def check(self):
        require(time.monotonic()<self.end,'Evaluation wall-time bound reached')
        if self.a.deadline_unix is not None:require(time.time()<self.a.deadline_unix,'Global deadline reached')
        if self.a.mode=='full':
            for index in range(self.a.shard_count):
                if index!=self.a.shard_index:
                    failed=self.out/('shard_%02d_of_%02d'%(index,self.a.shard_count))/'failed.json'
                    require(not failed.exists(),'Peer shard failed; stop without retry or process termination: '+str(failed))

    def contracts(self):
        a=self.a
        require(a.protocol_sha256==PROTOCOL_SHA and sha(a.protocol)==PROTOCOL_SHA,'Frozen training protocol mismatch')
        require(a.selection_sha256==SELECTION_SHA[a.mode] and sha(a.selection)==a.selection_sha256,'Frozen evaluation selection mismatch')
        protocol,selection=read(a.protocol),read(a.selection)
        require(sha(a.config)==protocol['config_sha256']==selection['config_sha256'],'Config contract mismatch')
        require(sha(a.checkpoint)==protocol['m0_sha256'],'Original M0 checkpoint mismatch')
        for name,digest in protocol['source_sha256'].items():
            require(Path(name).name==name and sha(self.package/name)==digest,'Frozen helper/source changed: '+name)
        for name,digest in protocol['runtime_source_sha256'].items():
            require(Path(name).is_absolute() and sha(name)==digest,'Native runtime source changed: '+name)
        require(protocol['numerical_policy']['future_matmul_tf32'] is False and
                protocol['numerical_policy']['future_cudnn_tf32'] is True,'Unexpected future precision')
        return protocol,selection

    def run(self):
        a=self.a;self.check();protocol,selection=self.contracts()
        selected=select_rows(selection,a.mode,a.shard_index,a.shard_count)
        arm_receipts=frozen_arms(a.runs_root,protocol)
        common=dict(schema='native-joint-evaluation-contract-v1',mode=a.mode,models=list(MODELS),
            protocol_sha256=PROTOCOL_SHA,selection_sha256=a.selection_sha256,script_sha256=sha(__file__),
            config_sha256=protocol['config_sha256'],source_sha256=protocol['source_sha256'],
            runtime_source_sha256=protocol['runtime_source_sha256'],m0_sha256=protocol['m0_sha256'],
            final_checkpoints_sha256={arm:r['files_sha256']['latest.pth'] for arm,r in arm_receipts.items()},
            runs_root=str(Path(a.runs_root).resolve()),shard_count=a.shard_count,selected_samples=selection['samples'],
            numerical_policy=protocol['numerical_policy'],horizons=HORIZONS,optimizer_steps=0,
            sample_partition='selection_ordinal modulo shard_count',historical_validation_exposure=True)
        path=self.out/'contract.json'
        try:write(path,common,exclusive=True)
        except FileExistsError:require(read(path)==common,'Other shard has a different fixed contract')
        write(self.shard/'manifest.json',dict(common,created_at=utc(),shard_index=a.shard_index,
            selected_records=selected,arm_source_receipts=arm_receipts,pid=os.getpid(),pgid=os.getpgid(0),
            max_seconds=a.max_seconds,deadline_unix=a.deadline_unix),exclusive=True)
        event('IMPORT_AND_DATASET',mode=a.mode,shard=a.shard_index,samples=len(selected))
        import numpy as np
        import torch
        from mmcv import Config
        from mmcv.parallel import MMDataParallel,collate
        from mmdet3d.datasets import build_dataset
        from native_state_cache import (_prepare_repo,_selected_rows,_identity_from_metas,
            _capture_native,_tree_map,_tensor_digest,_verify_file,_sample_directory,
            validate_inputs,build_native_model,replay)
        from observation_memory import install_observation_memory
        from memory_experiment import state_digest
        torch.set_num_threads(2)
        try:
            import cv2
            cv2.setNumThreads(2)
        except ImportError:pass
        require(str(a.device).startswith('cuda') and torch.cuda.is_available(),'Native image forward requires CUDA')
        device=torch.device(a.device);torch.cuda.set_device(device)
        device_index=device.index if device.index is not None else torch.cuda.current_device()
        random.seed(11);np.random.seed(11);torch.manual_seed(11);torch.cuda.manual_seed_all(11)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
        _prepare_repo(a.repo)
        cfg=Config.fromfile(a.config);dc=copy.deepcopy(cfg.data.test);dc.test_mode=True;dc.pop('samples_per_gpu',None)
        require(dc.radar_cfg.cache_readonly is True and dc.radar_cfg.nsweeps==5,'Require unchanged read-only five-sweep radar')
        require(dc.get('radar_observation_cfg') is None and not dc.get('allow_dual_radar_inputs'),'Unexpected alternate radar interface')
        require(dc.future_metadata_only is True,'Require native metadata-only future loading')
        require(sha(dc.ann_file)==selection['ann_sha256'],'Native validation annotation hash mismatch')
        dataset=build_dataset(dc)
        require(len(dataset.usable_index)==5119,'Unexpected native usable validation count')
        all_rows=_selected_rows(selection,'validation',dataset)
        by_index={r['official_index']:r for r in all_rows}
        selected=[dict(by_index[r['official_index']],selection_ordinal=r['selection_ordinal']) for r in selected]
        native=build_native_model(a.config,a.checkpoint,a.device,repo=a.repo)
        variants={};migrations={};head_digests={}
        for name in MODELS[1:]:
            model=copy.deepcopy(native);mode='native1' if name=='M0_fp32' else name
            migrations[name]=install_observation_memory(model,mode)
            if name in ARMS:
                receipt=arm_receipts[name];checkpoint=Path(receipt['directory'])/'latest.pth'
                payload=torch.load(str(checkpoint),map_location='cpu')
                require(payload['arm']==name and payload['metadata']==receipt['manifest'],'Checkpoint metadata differs from manifest')
                hp=protocol['training'];require(payload['pass_index']==hp['passes'] and payload['update']==hp['passes']*hp['train_samples']//hp['accumulate'],'Not the fixed final checkpoint')
                rng=np.random.RandomState(11)
                require(payload['sample_orders']==[rng.permutation(hp['train_samples']).tolist() for _ in range(hp['passes'])],'Fixed seed11 recorded training order differs')
                model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True)
                for key,value in model.future_pred_head.state_dict().items():
                    require(torch.equal(value.detach().cpu(),payload['future_pred_head'][key]),'Loaded tensor mismatch: '+key)
                del payload
            variants[name]=model
        for name,model in [('M0',native)]+list(variants.items()):
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
            head_digests[name]=state_digest(model.future_pred_head)
        wrapped=MMDataParallel(native,device_ids=[device_index])
        write(self.shard/'loaded_models.json',dict(migrations=migrations,head_state_sha256=head_digests,
            source=native._native_state_provenance,torch_version=torch.__version__,cuda_version=torch.version.cuda,
            device=str(device),gpu_name=torch.cuda.get_device_name(device),training=False,optimizer_created=False),exclusive=True)
        reference=pilot_reference(a.parity_cache,arm_receipts,protocol) if a.mode=='pilot' else None
        if reference:write(self.shard/'pilot_reference_sources.json',reference[3],exclusive=True)
        for ordinal,row in enumerate(selected):
            self.check();begin=time.monotonic();token=row['sample_token']
            directory=self.out/'samples'/token;directory.mkdir(parents=True,exist_ok=False)
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
            torch.cuda.reset_peak_memory_stats(device);event('SAMPLE_BEGIN',official_index=row['official_index'],sample_token=token)
            example=dataset._prepare_data_info(row['data_info_index'],rand_interval=None)
            require(example is not None,'Selected native sample failed; never replace')
            identity=_identity_from_metas(example['img_metas'].data)
            require(all(identity[k]==row[k] for k in ['sample_token','scene_token']),'Prepared identity differs')
            data=collate([example],samples_per_gpu=1);data.pop('instance',None);context={}
            with _capture_native(native,context),torch.no_grad():wrapped(return_loss=False,rescale=True,**data)
            torch.cuda.synchronize(device);native_seconds=time.monotonic()-begin
            del data,example
            validate_inputs(context['inputs'])
            require(context['preds'].shape==(5,3,1,1,40000,16,2) and context['targets'].shape==(1,7,512,512,40),'Native shape contract')
            require(context['preds'].dtype==np.float32 and np.isfinite(context['preds']).all(),'Nonfinite/wrong native prediction')
            counts=np.asarray([[int((gt==c).sum()) for c in [0,1,255]] for gt in context['targets'][0]],dtype=np.int64)
            require(np.all(counts.sum(1)==512*512*40),'Unknown GT class')
            input_hash=tree_digest(context['inputs']);target_hash=array_digest(context['targets'])
            model_rows={'M0':evaluator_record(context['native_record'],row,counts)}
            logit_hashes={'M0':array_digest(context['preds'])};timings={'M0':native_seconds};parity={}
            if reference:
                cache,lookup,expected,_=reference;require(token in lookup,'Pilot token absent from frozen cache')
                old=lookup[token];old_directory=_sample_directory(cache,old)
                for key in ['inputs','targets','preds']:_verify_file(old_directory/old['files'][key]['file'],old['files'][key])
                old_inputs=torch.load(str(old_directory/old['files']['inputs']['file']),map_location='cpu')
                require(tree_digest(old_inputs)==input_hash,'Fresh observed boundary differs from frozen pilot cache')
                del old_inputs
                old_gt=np.load(old_directory/old['files']['targets']['file'],mmap_mode='r',allow_pickle=False)
                old_pred=np.load(old_directory/old['files']['preds']['file'],mmap_mode='r',allow_pickle=False)
                require(np.array_equal(context['targets'],old_gt),'Fresh GT differs from pilot cache')
                require(np.array_equal(context['preds'],old_pred),'Fresh native M0 5h x 3layer logits not bitwise equal')
                parity.update(full_boundary_hash_equal=True,exact_gt_equal=True,native_all_5h_3layer_logits_bitwise=True)
                del old_gt,old_pred
            del context['preds']
            sample=dict(inputs=_tree_map(context['inputs'],lambda x:x.to(device)),
                        targets=torch.from_numpy(context['targets']).to(device=device,dtype=torch.long),record=row)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
            for name,model in variants.items():
                self.check();tick=time.monotonic()
                with torch.no_grad():
                    pred=replay(model,sample,training=False)[0]
                    require(tuple(pred.shape)==(5,3,1,1,40000,16,2) and pred.dtype==torch.float32 and bool(torch.isfinite(pred).all()),'Variant logit contract')
                    result=model.evaluate_occ_records(pred,sample['targets'],sample['inputs']['img_metas'])
                    require(len(result)==1,'Variant evaluator record count')
                    model_rows[name]=evaluator_record(result[0],row,counts)
                    logit_hashes[name]=array_digest(pred.detach().cpu().numpy())
                torch.cuda.synchronize(device);timings[name]=time.monotonic()-tick;del pred,result
            require(tree_digest(context['inputs'])==input_hash,'CPU boundary mutated between models')
            require(tree_digest(sample['inputs'])==input_hash,'Device replay input mutated between models')
            if reference:
                for name in MODELS:
                    require(model_rows[name]['hist_by_horizon']==reference[2][name][token],'Pilot fullGT confusion differs: '+name)
                parity['all_5_models_5_horizons_full_GT_confusion_equal']=True
            self.check()
            result=dict(schema='native-joint-evaluation-sample-v1',identity=row,models=model_rows,
                input_tree_sha256=input_hash,targets_array_sha256=target_hash,
                observed_bev_tensor_sha256=_tensor_digest(context['inputs']['prev_bev_input']),
                native_radar_tensor_sha256=_tensor_digest(context['inputs']['radar_bev']),
                logits_array_sha256=logit_hashes,target_counts_0_1_255_by_frame=counts.tolist(),
                model_seconds=timings,total_seconds=time.monotonic()-begin,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                contract_sha256=sha(self.out/'contract.json'),optimizer_steps=0,engineering_parity=parity or None)
            write(directory/'records.json',result,exclusive=True)
            completion=dict(status='COMPLETE_SAMPLE',sample_token=token,official_index=row['official_index'],
                selection_ordinal=row['selection_ordinal'],shard_index=a.shard_index,
                files_sha256={'records.json':sha(directory/'records.json')})
            write(directory/'complete.json',completion,exclusive=True)
            self.results.append(dict(identity=row,directory=str(directory.relative_to(self.out)),
                records_sha256=sha(directory/'records.json'),complete_sha256=sha(directory/'complete.json'),
                total_seconds=result['total_seconds']))
            write(self.shard/'index.json',dict(status='RUNNING',records=self.results))
            write(self.shard/'progress.json',dict(status='RUNNING',completed=len(self.results),planned=len(selected),seconds=time.monotonic()-self.start))
            event('SAMPLE_COMPLETE',official_index=row['official_index'],seconds=result['total_seconds'],engineering_parity=parity)
            del context,sample,result
        self.check();self.contracts()
        require(frozen_arms(a.runs_root,protocol)==arm_receipts,'Fixed arm artifacts changed during evaluation')
        for name,model in [('M0',native)]+list(variants.items()):
            require(state_digest(model.future_pred_head)==head_digests[name],'Evaluation changed future-head state')
        write(self.shard/'index.json',dict(status='COMPLETE',records=self.results))
        write(self.shard/'complete.json',dict(status='COMPLETE_JOINT_NATIVE_EVALUATION_SHARD',
            mode=a.mode,shard_index=a.shard_index,shard_count=a.shard_count,samples=len(self.results),
            official_indices=[r['identity']['official_index'] for r in self.results],
            index_sha256=sha(self.shard/'index.json'),contract_sha256=sha(self.out/'contract.json'),
            manifest_sha256=sha(self.shard/'manifest.json'),loaded_models_sha256=sha(self.shard/'loaded_models.json'),seconds=time.monotonic()-self.start,
            optimizer_steps=0,threshold_selection=False,all_models_same_inputs=True,
            all_pilot_parity_passed=a.mode=='pilot'),exclusive=True)
        event('SHARD_COMPLETE',shard=a.shard_index,samples=len(self.results))


def self_test():
    import numpy as np
    # Axes/GT rows tested using an asymmetric integer confusion.
    counts=np.array([[11,7,2]]*7);hist=np.array([[[8,3],[2,5]]]*5)
    row=dict(sample_token='a'*32,scene_token='s',horizon_seconds=HORIZONS,hist_by_horizon=hist)
    evaluator_record(row,row,counts)
    try:evaluator_record(dict(row,hist_by_horizon=hist.transpose(0,2,1)),row,counts)
    except RuntimeError:pass
    else:raise AssertionError('Transposed GT/prediction axes accepted')
    root=Path(__file__).resolve().parent
    for mode,name in [('full','full_validation_selection_v1.json'),('pilot','joint_preflight_selection_v1.json')]:
        selection=read(root/name);require(sha(root/name)==SELECTION_SHA[mode],'Selection source')
        a,b=[select_rows(selection,mode,i,2) for i in range(2)]
        assert not {r['sample_token'] for r in a}&{r['sample_token'] for r in b}
        assert sorted([r['official_index'] for r in a+b])==[r['official_index'] for r in selection['records']]
    assert array_digest(np.array([1,2],dtype=np.int32))!=array_digest(np.array([1,2],dtype=np.int64))
    print(json.dumps(dict(status='PASS_STATIC_NUMPY_TESTS',checks=['frozen_selections','complete_disjoint_shards','pilot_shards_nonempty','GT_confusion_axes','typed_array_hash'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['config','checkpoint','selection','selection-sha256','runs-root','protocol','protocol-sha256','out','repo','parity-cache']:p.add_argument('--'+field)
    p.add_argument('--mode',choices=['pilot','full'],default='pilot')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--device',default='cuda:0');p.add_argument('--max-seconds',type=int,default=1200)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return 0
    require(all(getattr(a,k) for k in ['config','checkpoint','selection','selection_sha256','runs_root','protocol','protocol_sha256','out']),'Missing required fixed evaluation input')
    require(0<a.max_seconds<=43200 and (a.deadline_unix is None or math.isfinite(a.deadline_unix)),'Invalid evaluation time bound')
    require(a.mode!='pilot' or a.parity_cache,'Pilot requires frozen development200 --parity-cache')
    require(a.mode!='full' or a.parity_cache is None,'Full validation does not reuse pilot cache as model inputs')
    runner=Evaluation(a)
    def interrupted(number,_frame):raise InterruptedError('Joint evaluation signal '+str(number))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupted)
    signal.alarm(max(1,int(min(a.max_seconds,max(0,runner.end-time.monotonic())))))
    try:runner.run();return 0
    except BaseException as exc:
        write(runner.shard/'failed.json',dict(status='FAILED_OR_INTERRUPTED',error=repr(exc),
            traceback=traceback.format_exc(),completed=len(runner.results),optimizer_steps=0,automatic_retry=False),exclusive=True)
        traceback.print_exc();return 1
    finally:signal.alarm(0);runner.lock.close()


if __name__=='__main__':raise SystemExit(main())
