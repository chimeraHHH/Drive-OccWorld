"""Post-hoc fixed-final train-learning diagnostic; no training or selection.

Select the first 16 distinct scenes in the frozen train-cache order and the
first anchor from each. Evaluate M0_fp32 and the three completed fixed-final
heads on those same inputs. Native five-horizon confusion and the unchanged
M0 loss are computed in eval/no_grad. Existing development JSONL is reused;
development is not inferred again. This diagnoses learning/generalization,
never selects checkpoints, thresholds, or a replacement training policy.

Expected GPU scope: four sequential models x16 anchors, one forward/loss each.
No backward, optimizer construction, parameter update, or automatic retry.
Default wall deadline is 600 seconds; model/data imports count against it.
"""
import argparse
import datetime
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import types

MODELS = ('M0_fp32', 'native1', 'persistent2', 'rolling2')
ARMS = MODELS[1:]
TRAIN_SCENES = 16
M0_SHA = '0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def event(name,**values):
    print(json.dumps(dict(event=name,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          **values),allow_nan=False),flush=True)


def select_training_diagnostic(records):
    """Identity only, first occurrence in existing frozen order; no scores/GT."""
    selected=[];seen=set()
    for row in records:
        require(row['split']=='train','Training diagnostic cannot use a development row')
        if row['scene_token'] not in seen:
            selected.append(row);seen.add(row['scene_token'])
        if len(selected)==TRAIN_SCENES:
            break
    require(len(selected)==TRAIN_SCENES,'Fewer than 16 distinct training scenes')
    return selected


def verify_final_payload(payload,manifest,protocol,arm,expected_digest,model,state_digest):
    """Strict complete head load, fixed final counters, matching source metadata."""
    import torch
    import numpy as np
    hp=protocol['training'];updates=hp['passes']*hp['train_samples']//hp['accumulate']
    require(payload['arm']==arm and payload['pass_index']==hp['passes'] and payload['update']==updates,
            'Checkpoint is not the registered fixed final endpoint: '+arm)
    require(json.dumps(payload['metadata'],sort_keys=True)==json.dumps(manifest,sort_keys=True),
            'Checkpoint metadata differs from completed run manifest: '+arm)
    require(len(payload['sample_orders'])==hp['passes'],'Wrong final sample-order pass count')
    order_rng=np.random.RandomState(11)
    for order in payload['sample_orders']:
        require(order==order_rng.permutation(hp['train_samples']).tolist(),
                'Training sample order differs from fixed seed11 contract')
    head=payload['future_pred_head']
    expected_state=model.future_pred_head.state_dict()
    require(set(head)==set(expected_state),'Final future head key mismatch: '+arm)
    for name,value in head.items():
        require(torch.is_tensor(value) and value.shape==expected_state[name].shape and
                value.dtype==expected_state[name].dtype,'Final tensor shape/dtype mismatch: '+name)
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()),
                'Nonfinite final checkpoint tensor: '+name)
    source_digest=state_digest(types.SimpleNamespace(state_dict=lambda:head))
    result=model.future_pred_head.load_state_dict(head,strict=True)
    require(not result.missing_keys and not result.unexpected_keys,'Incomplete strict head load')
    require(state_digest(model.future_pred_head)==source_digest,'Loaded head tensor SHA differs')
    return dict(checkpoint_sha256=expected_digest,loaded_head_tensor_sha256=source_digest,
                pass_index=payload['pass_index'],update=payload['update'],strict=True,
                state_tensor_count=len(head),optimizer_loaded=False)


def _summarize_hist(hist):
    import numpy as np
    from aggregate_memory import metric_arrays
    h=np.asarray(hist,dtype=np.int64);scores=metric_arrays(h)
    return dict(hist_by_horizon=h.tolist(),horizon_seconds=[0.,.5,1.,1.5,2.],
        gmo_iou_by_horizon=scores['gmo_iou_by_horizon'].tolist(),
        binary_miou_by_horizon=scores['binary_miou_by_horizon'].tolist(),
        future_macro_gmo=float(scores['future_macro_gmo']),
        future_pooled_gmo=float(scores['future_pooled_gmo']),
        future_macro_binary_miou=float(scores['future_macro_binary_miou']),
        future_FP=int(h[1:,0,1].sum()),future_FN=int(h[1:,1,0].sum()),unit='ratio')


def _scene_gmo(hist):
    """Undefined local scene IoU is disclosed, never silently averaged away."""
    import numpy as np
    h=np.asarray(hist,dtype=np.int64)[1:]
    union=h[:,1,1]+h[:,0,1]+h[:,1,0]
    absent=np.flatnonzero(union==0).tolist()
    return (None if absent else float((h[:,1,1]/union).mean()),
            [.5*(i+1) for i in absent])


def evaluate_training(model,cache,records,path,device):
    import numpy as np
    import torch
    from memory_experiment import state_digest
    from native_state_cache import load_sample,replay
    before=state_digest(model.future_pred_head)
    rows=[];started=time.monotonic()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad(),Path(path).open('x') as stream:
        for ordinal,record in enumerate(records):
            begin=time.monotonic()
            sample=load_sample(cache,record,device=device)
            output=replay(model,sample,training=False)
            raw=model.evaluate_occ_records(output[0],sample['targets'],sample['inputs']['img_metas'])
            require(len(raw)==1,'Expected one native evaluation record')
            row=raw[0]
            require(row['sample_token']==record['sample_token'] and row['scene_token']==record['scene_token'],
                    'Training diagnostic identity mismatch')
            hist=np.asarray(row['hist_by_horizon'],dtype=np.int64)
            require(hist.shape==(5,2,2) and row['horizon_seconds']==[0.,.5,1.,1.5,2.],
                    'Training diagnostic native shape/horizon mismatch')
            require(np.array_equal(hist.sum(2),np.asarray(record['native_hist'],dtype=np.int64).sum(2)),
                    'Training diagnostic GT class counts changed')
            # Original loss retains current+four future, all three intermediate
            # heads, special GT pooling, class weights and all 12 components.
            losses=model.compute_occ_loss(output[0],sample['targets'])
            expected={f'loss_voxel_{kind}_inter_{layer}' for kind in
                      ('ce','sem_scal','lovasz','geo_scal') for layer in range(3)}
            require(set(losses)==expected,'Original model did not return the 12 registered loss components')
            values={name:float(value.detach()) for name,value in losses.items()}
            nonfinite={name:repr(value) for name,value in values.items() if not math.isfinite(value)}
            total=sum(values.values())
            finite=not nonfinite and math.isfinite(total)
            clean={name:value if math.isfinite(value) else None for name,value in values.items()}
            row=dict(row,hist_by_horizon=hist.tolist(),
                     original_loss_components=clean,original_loss_total=total if finite else None,
                     original_loss_status='FINITE' if finite else 'NONFINITE',
                     nonfinite_components=nonfinite,seconds=time.monotonic()-begin)
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
            event('TRAIN_DIAGNOSTIC_SAMPLE',ordinal=ordinal+1,sample_token=record['sample_token'],
                  loss_status=row['original_loss_status'],seconds=row['seconds'])
            del sample,output,losses,raw
    require(state_digest(model.future_pred_head)==before,'Evaluation changed future-head tensors')
    summed=np.asarray([r['hist_by_horizon'] for r in rows],dtype=np.int64).sum(0)
    metrics=_summarize_hist(summed)
    all_finite=all(r['original_loss_status']=='FINITE' for r in rows)
    metrics.update(samples=len(rows),scenes=len({r['scene_token'] for r in rows}),
        original_loss=dict(status='ALL_FINITE' if all_finite else 'NONFINITE_SAMPLES_PRESENT',
            arithmetic_mean_total=(sum(r['original_loss_total'] for r in rows)/len(rows)) if all_finite else None,
            arithmetic_mean_components={name:sum(r['original_loss_components'][name] for r in rows)/len(rows)
                for name in rows[0]['original_loss_components']} if all_finite else None,
            nonfinite_sample_tokens=[r['sample_token'] for r in rows if r['original_loss_status']!='FINITE'],
            mode='eval_no_grad',frame_scope='current plus four future, all three intermediate heads',
            excluded_nonfinite_from_mean=False),seconds=time.monotonic()-started,
        head_tensor_sha256=before,parameters_unchanged=True,record_sha256=sha(path))
    return metrics,rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','config','checkpoint','train-cache','dev-cache','runs-root','protocol','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-seconds',type=int,default=600)
    a=p.parse_args(argv)
    require(a.max_seconds>0,'Positive wall deadline required')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def stop(number,frame):
        raise InterruptedError('Post-hoc diagnostic interrupted/deadline: '+str(number)+'; no retry')
    for number in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):
        signal.signal(number,stop)
    signal.alarm(a.max_seconds)
    try:
        protocol=read(a.protocol);pkg=Path(__file__).resolve().parent
        require(protocol['status']=='FROZEN_BEFORE_CANDIDATE_TRAINING' and protocol['revision']==2,
                'Require the frozen protocol_v2 contract')
        require(protocol['m0_sha256']==M0_SHA and sha(a.checkpoint)==M0_SHA,
                'M0 checkpoint identity mismatch')
        require(sha(a.config)==protocol['config_sha256'],'Native config SHA mismatch')
        for name,expected in protocol['source_sha256'].items():
            require(sha(pkg/name)==expected,'Frozen protocol source changed: '+name)
        source_before={name:sha(pkg/name) for name in protocol['source_sha256']}
        for remote,expected in protocol['runtime_source_sha256'].items():
            relative=remote.split('/projects/',1)
            require(len(relative)==2,'Unexpected runtime source path')
            local=Path(a.repo)/'projects'/relative[1]
            require(sha(local)==expected,'Native runtime source SHA mismatch: '+str(local))
        sys.path.insert(0,str(Path(a.repo).resolve()))
        event('DEPENDENCY_IMPORT_BEGIN')
        import numpy as np
        import torch
        from memory_experiment import cached_records,prepare_model,seed_all,state_digest
        from aggregate_memory import load_cache,load_arm,load_precision_reference,NATIVE_INITIAL_HEAD_SHA
        require(str(a.device).startswith('cuda') and torch.cuda.is_available(),'Native diagnostic requires CUDA')
        torch.cuda.set_device(torch.device(a.device));torch.set_num_threads(2);seed_all(11)
        event('DEPENDENCY_IMPORT_COMPLETE',torch=torch.__version__)
        train=Path(a.train_cache);train_records=cached_records(train);train_index=read(train/'index.json')
        require(train_index['split']=='train' and len(train_records)==protocol['training']['train_samples'],
                'Not the completed original training cache')
        require(train_index['config_sha256']==protocol['config_sha256'] and
                train_index['selection_sha256']==protocol['selection_sha256'] and
                train_index['extractor_sha256']==protocol['source_sha256']['native_state_cache.py'],
                'Training cache provenance mismatch')
        require(train_index['inputs_targets_separate'] is True and train_index['future_occupancy_is_input'] is False,
                'Training cache input/target separation changed')
        require(train_index['native_model']['radar_contract']=='native_loader_B_without_common_source_override' and
                train_index['numerical_policy']['tf32_matmul'] is True and
                train_index['numerical_policy']['tf32_cudnn'] is True,
                'Original native observer cache contract changed')
        selection_path=Path(a.protocol).with_name('selection_v1.json')
        require(sha(selection_path)==protocol['selection_sha256'],'Frozen selection SHA mismatch')
        selection=read(selection_path)
        keys=('sample_token','scene_token','official_index','split')
        planned=[r for r in selection['records'] if r['split']=='train']
        require([[r[k] for k in keys] for r in train_records]==[[r[k] for k in keys] for r in planned],
                'Training cache order or identities differ from frozen selection')
        selected=select_training_diagnostic(train_records)
        write(out/'selection.json',dict(schema='posthoc-train-diagnostic-selection-v1',
            rule='first 16 distinct scenes in frozen training cache order, first anchor in each',
            source_train_index_sha256=sha(train/'index.json'),source_selection_sha256=protocol['selection_sha256'],
            model_scores_used=False,labels_used_for_selection=False,
            records=[{k:r[k] for k in keys} for r in selected]))
        # Reuse only completed development records, including frozen same-policy M0.
        dev,index,dev_records,m0_hist,_=load_cache(a.dev_cache,protocol,a.protocol)
        dev_sha=sha(dev/'index.json');development={};receipts={}
        for arm in ARMS:
            hist,receipt=load_arm(a.runs_root,arm,dev_records,m0_hist,protocol,a.protocol,dev_sha)
            require(receipt['train_index_sha256']==sha(train/'index.json'),'Run used a different training cache')
            development[arm]=_summarize_hist(hist.sum(0));receipts[arm]=receipt
        hist,receipts['M0_fp32']=load_precision_reference(a.runs_root,dev_records,m0_hist,protocol,a.protocol,dev_sha)
        development['M0_fp32']=_summarize_hist(hist.sum(0))
        training={};loaded={};all_rows={}
        for name in MODELS:
            seed_all(11);mode='native1' if name=='M0_fp32' else name
            model,migration,_=prepare_model(a.config,a.checkpoint,mode)
            if name=='M0_fp32':
                require(state_digest(model.future_pred_head)==NATIVE_INITIAL_HEAD_SHA,
                        'Frozen native future head initial tensor SHA mismatch')
                loaded[name]=dict(source_m0_sha256=M0_SHA,head_tensor_sha256=NATIVE_INITIAL_HEAD_SHA,
                                  optimizer_updates=0,strict=True)
            else:
                directory=Path(a.runs_root)/name;checkpoint=directory/'latest.pth'
                expected=receipts[name]['checkpoint_sha256']
                require(sha(checkpoint)==expected,'Final checkpoint file SHA mismatch')
                before=checkpoint.stat();payload=torch.load(str(checkpoint),map_location='cpu')
                after=checkpoint.stat()
                require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),
                        'Checkpoint changed while reading')
                # Serialized optimizer is never constructed/restored for this run.
                payload.pop('optimizer',None)
                loaded[name]=verify_final_payload(payload,read(directory/'manifest.json'),protocol,
                                                  name,expected,model,state_digest)
                del payload
            seed_all(11);event('POSTHOC_MODEL_BEGIN',model=name)
            training[name],all_rows[name]=evaluate_training(model,train,selected,out/(name+'_records.jsonl'),a.device)
            write(out/'progress.json',dict(status='RUNNING',completed_models=list(training),
                                          seconds=time.monotonic()-started))
            del model;gc.collect();torch.cuda.empty_cache()
        comparisons={}
        for name in ARMS:
            train_delta=training[name]['future_macro_gmo']-training['M0_fp32']['future_macro_gmo']
            dev_delta=development[name]['future_macro_gmo']-development['M0_fp32']['future_macro_gmo']
            left=training[name]['original_loss']['arithmetic_mean_total'];right=training['M0_fp32']['original_loss']['arithmetic_mean_total']
            per_scene=[]
            for current,reference in zip(all_rows[name],all_rows['M0_fp32']):
                require(current['sample_token']==reference['sample_token'],'Diagnostic model pair misaligned')
                cur,cur_undefined=_scene_gmo(current['hist_by_horizon'])
                ref,ref_undefined=_scene_gmo(reference['hist_by_horizon'])
                per_scene.append(dict(sample_token=current['sample_token'],scene_token=current['scene_token'],
                    gmo_delta_pp=None if cur is None or ref is None else 100*(cur-ref),
                    undefined_GMO_horizons=dict(model=cur_undefined,reference=ref_undefined)))
            comparisons[name+' - M0_fp32']=dict(train16_delta_ratio=train_delta,train16_delta_pp=100*train_delta,
                development_delta_ratio=dev_delta,development_delta_pp=100*dev_delta,
                train_minus_dev_improvement_pp=100*(train_delta-dev_delta),
                training_eval_mode_original_loss_delta=None if left is None or right is None else left-right,
                per_training_scene=per_scene)
        require({name:sha(pkg/name) for name in source_before}==source_before,'Diagnostic changed frozen source files')
        require(sha(a.checkpoint)==M0_SHA,'Diagnostic changed M0 checkpoint')
        result=dict(schema='posthoc-train-diagnostic-v1',status='COMPLETE_POSTHOC_DIAGNOSTIC',
            protocol_sha256=sha(a.protocol),script_sha256=sha(__file__),selection_sha256=sha(out/'selection.json'),
            frozen_sources_sha256=source_before,train_cache_index_sha256=sha(train/'index.json'),
            development_cache_index_sha256=dev_sha,loaded_checkpoints=loaded,source_receipts=receipts,
            train_samples=16,train_scenes=16,development_samples=len(dev_records),
            training=training,development=development,comparisons=comparisons,
            semantics=dict(inference_mode='eval_no_grad',optimizer_created=False,parameter_updates=0,
                checkpoint_selection=False,threshold_selection=False,training_policy_changed=False,
                future_matmul_tf32=False,future_cudnn_tf32=True,native_observer_cache_unchanged=True,
                posthoc=True,training_set_is_in_sample=True,development_inference_reused=True,
                loss='original 3-layer current+future composite loss, evaluated with dropout disabled',
                no_development_loss_comparison='Original development loss was not recorded and is not invented',
                no_causal_overfitting_claim='Different train/dev scene populations and posthoc 16-scene subset; descriptive diagnosis only'),
            seconds=time.monotonic()-started)
        write(out/'summary.json',result)
        lines=['# 固定最终模型的训练集学习诊断','',
            '固定取训练缓存顺序前16个场景，各首个anchor。四模型均eval/no_grad；开发集复用已完成的同精度记录。所有差值相对冻结M0_fp32。','',
            '| 模型 | Train16 future macro GMO % | Dev future macro GMO % | Train16 原损失均值 |','|---|---:|---:|---:|']
        for name in MODELS:
            loss=training[name]['original_loss']['arithmetic_mean_total'];loss_text='非有限，未求均值' if loss is None else f'{loss:.6f}'
            lines.append(f"| {name} | {100*training[name]['future_macro_gmo']:.4f} | {100*development[name]['future_macro_gmo']:.4f} | {loss_text} |")
        lines+=['','| 对照M0_fp32 | Train16 Δ pp | Dev Δ pp |','|---|---:|---:|']
        for contrast,row in comparisons.items():
            lines.append(f"| {contrast} | {row['train16_delta_pp']:+.4f} | {row['development_delta_pp']:+.4f} |")
        lines+=['','这些是训练后诊断，训练集结果属于in-sample，16场景与开发集人群也不同；不能仅据差值断言过拟合或机制成立。原损失在eval模式评估，与训练日志中的dropout开启损失不可直接等同。开发集原损失未记录，本报告不补造。',
                '不选择checkpoint或阈值，不修改训练策略。逐样本原生五时域混淆矩阵、12项原损失和完整源身份保留在JSONL/summary.json。','']
        (out/'report.md').write_text('\n'.join(lines))
        write(out/'complete.json',dict(status='COMPLETE',summary_sha256=sha(out/'summary.json'),
            report_sha256=sha(out/'report.md'),records_sha256={name:sha(out/(name+'_records.jsonl')) for name in MODELS},
            seconds=time.monotonic()-started,optimizer_updates=0,posthoc_only=True))
        event('POSTHOC_DIAGNOSTIC_COMPLETE',seconds=time.monotonic()-started)
    except BaseException as exc:
        write(out/'failed.json',dict(status='FAILED_OR_INTERRUPTED',error=repr(exc),
                                    seconds=time.monotonic()-started,automatic_retry=False))
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    main()
