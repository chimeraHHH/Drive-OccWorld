"""Merge complete joint-native evaluation shards, with a fail-closed hash chain.

Pilot (two fixed scenes) produces engineering parity/resource evidence only.
Full (all 5119 anchors/150 scenes) uses aggregate_memory.aggregate mathematics
for five fixed models/seven paired contrasts. Neither mode performs inference,
training, fitting, checkpoint selection, or source-result modification.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

PRODUCER_SHA='280a032a95070f4a618664f5e47788f00a04d8c048865b5e6000f80b724d1578'
PROTOCOL_SHA='071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
SELECTION_SHA={'pilot':'e616af491e77ad977dfddfd931c1110f9580921bc38ed6ad84281dfe23a9754d',
               'full':'60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391'}
M0_SHA='0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
INITIAL_HEAD_SHA='6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'
SOURCE_MODEL_SHA='67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5'
MODELS=('M0','M0_fp32','native1','persistent2','rolling2')
ARMS=MODELS[2:]
HORIZONS=[0.,.5,1.,1.5,2.]
VOXELS=512*512*40


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(data):return hashlib.sha256(data).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path,expected_sha=None):
    path=Path(path);require(path.is_file(),'Missing complete real input: '+str(path))
    data=path.read_bytes();observed=digest(data)
    if expected_sha is not None:require(observed==expected_sha,'Hash mismatch: '+str(path))
    return json.loads(data),observed


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def is_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def integer_array(value,shape,label):
    array=np.asarray(value)
    require(array.shape==shape and np.issubdtype(array.dtype,np.integer) and
            np.all(array>=0) and np.all(array<=np.iinfo(np.int64).max),label+' invalid shape/integer counts')
    return array.astype(np.int64)


def validate_selection(selection,mode):
    require(selection['schema']=='native-joint-validation-selection-v1','Selection schema')
    rows=selection['records'];indices=[r['official_index'] for r in rows]
    require(len(rows)==selection['samples'] and indices==sorted(set(indices)) and
            all(type(i) is int for i in indices),'Selection count/order')
    require(len({r['sample_token'] for r in rows})==len(rows) and
            len({r['scene_token'] for r in rows})==selection['scenes'],'Selection identity coverage')
    require(all(r['split']=='validation' for r in rows),'Selection split must be validation')
    require(all(isinstance(r['sample_token'],str) and len(r['sample_token'])==32 and
                all(c in '0123456789abcdef' for c in r['sample_token']) for r in rows),'Unsafe sample token')
    require(selection['historical_validation_exposure'] is True and selection['training_seed']==11 and
            selection['training_repeats_added']==0,'Selection scientific scope changed')
    if mode=='full':
        require(indices==list(range(5119)) and selection['scenes']==150 and
                selection['scope']=='full_native_validation','Full must cover exact official5119/150scenes')
    else:
        require(len(rows)==2 and selection['scenes']==2 and
                selection['scope']=='engineering_only_two_cached_validation_anchors','Pilot is exactly two frozen scenes')
    return rows


def validate_arm_receipts(receipts,contract,protocol):
    require(set(receipts)==set(ARMS),'Expected all three fixed-final arm receipts')
    training_indices=set();development_indices=set()
    for arm,receipt in receipts.items():
        files=receipt['files_sha256'];manifest=receipt['manifest']
        require(set(files)=={'manifest.json','training_complete.json','complete.json','latest.pth','development_records.jsonl'} and
                all(is_sha(v) for v in files.values()),'Incomplete final source file hash ledger')
        require(files['latest.pth']==contract['final_checkpoints_sha256'][arm],'Final checkpoint differs across contract/manifest')
        require(manifest['arm']==arm and manifest['seed']==11 and manifest['protocol_sha256']==PROTOCOL_SHA and
                manifest['m0_sha256']==M0_SHA,'Wrong arm/seed/initial checkpoint/protocol')
        require(manifest['numerical_policy']==protocol['numerical_policy'],'Final arm numerical policy differs')
        require(manifest['migration']['mode']==arm,'Wrong final arm memory policy')
        require(manifest['passes']==protocol['training']['passes'] and
                manifest['examples_per_pass']==protocol['training']['train_samples'],'Final arm training scope differs')
        require(set(manifest['source_sha256'])=={'memory_experiment.py','native_state_cache.py','observation_memory.py'},
                'Missing/extra trained source bindings')
        for name,value in manifest['source_sha256'].items():
            require(value==protocol['source_sha256'][name],'Final arm source differs: '+name)
        require(is_sha(manifest['train_index_sha256']) and is_sha(manifest['dev_index_sha256']),
                'Missing training/development index identities')
        training_indices.add(manifest['train_index_sha256']);development_indices.add(manifest['dev_index_sha256'])
    require(len(training_indices)==len(development_indices)==1,'Arms used different training/development caches')
    require(receipts['persistent2']['manifest']['trainable_parameters']==receipts['rolling2']['manifest']['trainable_parameters'],
            'Persistent/rolling capacity control differs')
    native=receipts['native1']['manifest'];ref=native['frozen_precision_reference']
    require(ref['initial_head_sha256']==native['initial_head_sha256']==INITIAL_HEAD_SHA and
            ref['source_m0_sha256']==M0_SHA and ref['optimizer_updates']==0 and
            ref['matmul_tf32'] is False and ref['cudnn_tf32'] is True and is_sha(ref['sha256']),
            'Frozen M0_fp32 identity/precision reference differs')


def validate_loaded_models(loaded,arm_receipts,protocol):
    require(loaded['training'] is False and loaded['optimizer_created'] is False,'Evaluation unexpectedly trained')
    heads=loaded['head_state_sha256'];require(set(heads)==set(MODELS) and all(is_sha(v) for v in heads.values()),
                                            'Missing actual loaded head tensor SHA')
    require(heads['M0']==heads['M0_fp32']==INITIAL_HEAD_SHA,'Frozen M0 head tensor identity changed')
    source=loaded['source']
    require(source['checkpoint_sha256']==M0_SHA and source['source_model_sha256']==SOURCE_MODEL_SHA and
            source['config_sha256']==protocol['config_sha256'] and
            source['radar_contract']=='native_loader_B_without_common_source_override' and
            source['checkpoint_epoch']==24 and source['model_only'] is True and source['optimizer_loaded'] is False,
            'Loaded native M0 model/source/input contract differs')
    require(set(loaded['migrations'])==set(MODELS[1:]),'Missing replay migration contracts')
    require(loaded['migrations']['M0_fp32']['mode']=='native1','Frozen precision reference changed architecture')
    for arm in ARMS:
        require(loaded['migrations'][arm]==arm_receipts[arm]['manifest']['migration'],
                'Evaluated migration differs from trained checkpoint: '+arm)


def validate_sample(result,expected,contract_sha,mode):
    require(result['schema']=='native-joint-evaluation-sample-v1' and
            result['contract_sha256']==contract_sha and result['optimizer_steps']==0,'Sample contract mismatch')
    identity=result['identity']
    require(identity==expected,'Sample records/index exact identity mismatch')
    require(set(result['models'])==set(MODELS) and set(result['logits_array_sha256'])==set(MODELS),
            'Incomplete five-model sample')
    for key in ['input_tree_sha256','targets_array_sha256','observed_bev_tensor_sha256','native_radar_tensor_sha256']:
        require(is_sha(result[key]),'Missing typed source hash: '+key)
    require(all(is_sha(v) for v in result['logits_array_sha256'].values()),'Missing full native logit array hash')
    counts=integer_array(result['target_counts_0_1_255_by_frame'],(7,3),'GT counts')
    require(np.all(counts.sum(1)==VOXELS),'GT 0/1/255 counts do not exhaust native grid')
    histograms={}
    for name in MODELS:
        row=result['models'][name]
        require(row['sample_token']==identity['sample_token'] and row['scene_token']==identity['scene_token'] and
                row['horizon_seconds']==HORIZONS,'Model identity/horizon mismatch')
        hist=integer_array(row['hist_by_horizon'],(5,2,2),'Native confusion '+name)
        require(np.array_equal(hist.sum(2),counts[2:,:2]),'GT rows/pred columns or GT counts differ: '+name)
        histograms[name]=hist
    require(set(result['model_seconds'])==set(MODELS) and
            all(math.isfinite(t) and t>=0 for t in result['model_seconds'].values()),'Invalid model resource timing')
    require(math.isfinite(result['total_seconds']) and result['total_seconds']>=0 and
            type(result['peak_cuda_allocated_bytes']) is int and result['peak_cuda_allocated_bytes']>=0,
            'Invalid sample resource metadata')
    if mode=='pilot':
        parity=result['engineering_parity']
        expected_flags={'full_boundary_hash_equal','exact_gt_equal','native_all_5h_3layer_logits_bitwise',
                        'all_5_models_5_horizons_full_GT_confusion_equal'}
        require(isinstance(parity,dict) and expected_flags<=set(parity) and
                all(parity[k] is True for k in expected_flags),'Incomplete/failed pilot engineering parity')
    else:
        require(result['engineering_parity'] is None,'Full evaluation must not substitute pilot-cache inference')
    return histograms,counts


def collect(evaluation_root,selection_path,protocol_path,mode):
    root=Path(evaluation_root).resolve()
    protocol,_=read(protocol_path,PROTOCOL_SHA);selection,_=read(selection_path,SELECTION_SHA[mode])
    rows=validate_selection(selection,mode)
    require(protocol['m0_sha256']==M0_SHA and protocol['training']['seed']==11,'Protocol M0/seed mismatch')
    require(selection['config_sha256']==protocol['config_sha256'],'Selection/config mismatch')
    require(not (root/'failed.json').exists(),'Evaluation root has failure receipt')
    contract,contract_sha=read(root/'contract.json')
    require(contract['schema']=='native-joint-evaluation-contract-v1' and contract['mode']==mode and
            contract['models']==list(MODELS) and contract['horizons']==HORIZONS,'Joint evaluation contract schema/scope mismatch')
    require(contract['script_sha256']==PRODUCER_SHA and contract['protocol_sha256']==PROTOCOL_SHA and
            contract['selection_sha256']==SELECTION_SHA[mode] and contract['m0_sha256']==M0_SHA,
            'Wrong producer/protocol/selection/checkpoint version')
    require(contract['config_sha256']==protocol['config_sha256'] and
            contract['source_sha256']==protocol['source_sha256'] and
            contract['runtime_source_sha256']==protocol['runtime_source_sha256'] and
            contract['numerical_policy']==protocol['numerical_policy'],'Joint source/config/precision contract mismatch')
    require(contract['optimizer_steps']==0 and contract['historical_validation_exposure'] is True and
            contract['sample_partition']=='selection_ordinal modulo shard_count' and
            contract['selected_samples']==len(rows),'Joint scientific scope mismatch')
    require(set(contract['final_checkpoints_sha256'])==set(ARMS) and
            all(is_sha(v) for v in contract['final_checkpoints_sha256'].values()),'Missing fixed final checkpoint SHA')
    shard_count=contract['shard_count'];require(type(shard_count) is int and shard_count in (1,2),'Invalid shard count')
    expected_shards={'shard_%02d_of_%02d'%(i,shard_count) for i in range(shard_count)}
    require({p.name for p in root.glob('shard_*') if p.is_dir()}==expected_shards,'Missing or extra shard directory')
    require((root/'samples').is_dir(),'Missing sample records')
    require({p.name for p in (root/'samples').iterdir()}=={r['sample_token'] for r in rows},
            'Sample directory set is incomplete or contains undeclared extras')
    joined={};shards=[];base_arm_receipts=None;base_loaded=None;base_pilot_sources=None
    for shard_index in range(shard_count):
        directory=root/('shard_%02d_of_%02d'%(shard_index,shard_count))
        require(not (directory/'failed.json').exists(),'Refuse failed/incomplete shard')
        done,done_sha=read(directory/'complete.json')
        require(done['status']=='COMPLETE_JOINT_NATIVE_EVALUATION_SHARD' and done['mode']==mode and
                done['shard_index']==shard_index and done['shard_count']==shard_count and
                done['contract_sha256']==contract_sha,'Shard completion scope/hash mismatch')
        index,index_sha=read(directory/'index.json',done['index_sha256'])
        manifest,manifest_sha=read(directory/'manifest.json',done['manifest_sha256'])
        loaded,loaded_sha=read(directory/'loaded_models.json',done['loaded_models_sha256'])
        require(index['status']=='COMPLETE' and done['optimizer_steps']==0 and
                done['threshold_selection'] is False and done['all_models_same_inputs'] is True and
                done['all_pilot_parity_passed'] is (mode=='pilot'),'Shard engineering/optimization scope mismatch')
        require(all(manifest.get(k)==v for k,v in contract.items()) and manifest['shard_index']==shard_index,
                'Shard manifest differs from common contract')
        expected=[dict(row,selection_ordinal=i) for i,row in enumerate(rows) if i%shard_count==shard_index]
        require(manifest['selected_records']==expected,'Shard selection order differs from frozen modulo partition')
        require(done['samples']==len(index['records'])==len(expected) and
                done['official_indices']==[r['official_index'] for r in expected],'Shard complete sample/order mismatch')
        validate_arm_receipts(manifest['arm_source_receipts'],contract,protocol)
        validate_loaded_models(loaded,manifest['arm_source_receipts'],protocol)
        loaded_signature={k:loaded[k] for k in ('head_state_sha256','source','torch_version','cuda_version','gpu_name','migrations')}
        if base_arm_receipts is None:
            base_arm_receipts=manifest['arm_source_receipts'];base_loaded=loaded_signature
        else:
            require(manifest['arm_source_receipts']==base_arm_receipts,'Fixed final artifacts differ across shards')
            require(loaded_signature==base_loaded,'Actual loaded tensors/source/runtime differ across shards')
        if mode=='pilot':
            pilot_sources,pilot_sources_sha=read(directory/'pilot_reference_sources.json')
            require(len(pilot_sources)==6 and all(is_sha(v) for v in pilot_sources.values()),'Missing pilot reference identities')
            for arm in ARMS:
                receipt=manifest['arm_source_receipts'][arm]
                source_path=str(Path(receipt['directory'])/'development_records.jsonl')
                require(pilot_sources.get(source_path)==receipt['files_sha256']['development_records.jsonl'],
                        'Pilot reference differs from completed development evaluation')
            native_receipt=manifest['arm_source_receipts']['native1']
            source_path=str(Path(native_receipt['directory'])/'frozen_native_fp32_records.jsonl')
            require(pilot_sources.get(source_path)==native_receipt['manifest']['frozen_precision_reference']['sha256'],
                    'Pilot frozen-precision reference differs')
            cache_indices=[v for k,v in pilot_sources.items() if Path(k).name=='index.json']
            require(cache_indices==[native_receipt['manifest']['dev_index_sha256']],
                    'Pilot reference cache differs from training arms development cache')
            if base_pilot_sources is None:base_pilot_sources=pilot_sources
            else:require(pilot_sources==base_pilot_sources,'Pilot reference sources differ across shards')
        else:pilot_sources_sha=None
        for descriptor,planned in zip(index['records'],expected):
            identity=descriptor['identity'];token=planned['sample_token'];ordinal=planned['selection_ordinal']
            require(all(identity.get(k)==v for k,v in planned.items()),'Shard index/sample frozen identity mismatch')
            require(set(identity)<=set(planned)|{'data_info_index'} and
                    type(identity.get('data_info_index')) is int and identity['data_info_index']>=0,
                    'Unexpected runtime sample identity fields')
            require(ordinal not in joined,'Overlapping shard ordinal')
            require(descriptor['directory']=='samples/'+token,'Unsafe/wrong sample directory')
            sample=(root/descriptor['directory']).resolve();require(root in sample.parents,'Sample path escapes evaluation root')
            sample_done,sample_done_sha=read(sample/'complete.json',descriptor['complete_sha256'])
            require(sample_done['status']=='COMPLETE_SAMPLE' and sample_done['sample_token']==token and
                    sample_done['official_index']==planned['official_index'] and
                    sample_done['selection_ordinal']==ordinal and sample_done['shard_index']==shard_index,
                    'Sample completion identity mismatch')
            require(set(sample_done['files_sha256'])=={'records.json'} and
                    sample_done['files_sha256']['records.json']==descriptor['records_sha256'],
                    'Sample records hash chain differs')
            result,records_sha=read(sample/'records.json',descriptor['records_sha256'])
            hist,counts=validate_sample(result,identity,contract_sha,mode)
            require(math.isclose(descriptor['total_seconds'],result['total_seconds'],rel_tol=1e-12,abs_tol=1e-12),
                    'Sample timing differs across index/records')
            joined[ordinal]=dict(identity=identity,hist=hist,counts=counts,result=result,
                complete_sha256=sample_done_sha,records_sha256=records_sha)
        require(math.isfinite(done['seconds']) and done['seconds']>=0,'Invalid shard timing')
        shards.append(dict(shard_index=shard_index,samples=len(expected),seconds=done['seconds'],
            complete_sha256=done_sha,index_sha256=index_sha,manifest_sha256=manifest_sha,
            loaded_models_sha256=loaded_sha,pilot_reference_sources_sha256=pilot_sources_sha))
    require(sorted(joined)==list(range(len(rows))),'Partial/overlapping full coverage')
    ordered=[joined[i] for i in range(len(rows))]
    require([r['identity']['official_index'] for r in ordered]==[r['official_index'] for r in rows],
            'Merged official order differs')
    return protocol,selection,contract,contract_sha,ordered,shards,base_arm_receipts,base_loaded


def resource_summary(ordered,shards):
    times=np.asarray([r['result']['total_seconds'] for r in ordered],dtype=float)
    model_times={name:np.asarray([r['result']['model_seconds'][name] for r in ordered]) for name in MODELS}
    return dict(samples=len(ordered),shards=len(shards),
        sum_shard_process_elapsed_seconds=sum(s['seconds'] for s in shards),
        maximum_shard_elapsed_seconds=max(s['seconds'] for s in shards),
        measured_sample_seconds=dict(minimum=float(times.min()),mean=float(times.mean()),maximum=float(times.max())),
        model_mean_seconds={name:float(v.mean()) for name,v in model_times.items()},
        maximum_observed_cuda_allocation_bytes=max(r['result']['peak_cuda_allocated_bytes'] for r in ordered),
        accounting='process residence and measured forward timings, not GPU kernel busy time',
        pilot_cost_is_not_full_run_guarantee=True)


def full_outputs(out,summary,ordered):
    arrays=np.stack([np.stack([row['hist'][name] for row in ordered]) for name in MODELS]).astype(np.uint64)
    np.savez_compressed(out/'raw_confusions.npz',models=np.asarray(MODELS),hist_by_model_sample_horizon=arrays,
        target_counts_0_1_255_by_sample_frame=np.stack([row['counts'] for row in ordered]).astype(np.uint64),
        sample_tokens=np.asarray([row['identity']['sample_token'] for row in ordered]),
        scene_tokens=np.asarray([row['identity']['scene_token'] for row in ordered]),
        official_indices=np.asarray([row['identity']['official_index'] for row in ordered],dtype=np.int64),
        horizon_seconds=np.asarray(HORIZONS))
    with (out/'metrics.csv').open('x',newline='') as f:
        names=['model','horizon_seconds','GMO_ratio','GMO_percent','binary_mIoU_ratio','binary_mIoU_percent','TN','FP','FN','TP']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for model in MODELS:
            for row in summary['models'][model]['horizons']:
                writer.writerow(dict(model=model,horizon_seconds=row['horizon_seconds'],GMO_ratio=row['gmo_iou'],
                    GMO_percent=100*row['gmo_iou'],binary_mIoU_ratio=row['binary_miou'],binary_mIoU_percent=100*row['binary_miou'],
                    **{key:row[key] for key in ['TN','FP','FN','TP']}))
    with (out/'contrasts.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['contrast','metric','delta_ratio','delta_pp','CI95_lower_pp','CI95_upper_pp'])
        for comparison in summary['comparisons']:
            for name,row in comparison['metrics'].items():
                writer.writerow([comparison['contrast'],name,row['delta_ratio'],row['delta_pp'],*row['ci95_pp']])
    lines=['# 五模型原生全量验证集联合评估','',
        '完整5,119 anchors /150 scenes，严格按原始official index排序。五模型使用同一原生图像/雷达观测及给定未来ego/action；当前模型固定单个训练seed11，没有新增训练。150场景历史验证暴露仍然存在，本结果不是新盲测。','',
        '主指标：每个未来时域先汇总全部样本混淆矩阵，再计算GMO前景IoU，最后对0.5/1/1.5/2秒算术平均。Pooled副指标先合并四时域混淆矩阵；二者不可互换。binary mIoU仅为GMO/non-GMO两类均值。','',
        '| 模型 | 0.5s GMO % | 1s % | 1.5s % | 2s % | Future macro GMO % [95% CI] | Future pooled GMO % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        m=summary['models'][name];h=[100*r['gmo_iou'] for r in m['horizons'][1:]];main=m['metrics']['future_macro_gmo'];pool=m['metrics']['future_pooled_gmo']['estimate']
        lines.append(f"| {name} | {h[0]:.4f} | {h[1]:.4f} | {h[2]:.4f} | {h[3]:.4f} | {100*main['estimate']:.4f} [{100*main['ci95'][0]:.4f}, {100*main['ci95'][1]:.4f}] | {100*pool:.4f} |")
    lines+=['','| 配对比较 | Future macro GMO Δ pp | 95% CI pp |','|---|---:|---:|']
    for c in summary['comparisons']:
        m=c['metrics']['future_macro_gmo'];lines.append(f"| {c['contrast']} | {m['delta_pp']:+.4f} | [{m['ci95_pp'][0]:+.4f}, {m['ci95_pp'][1]:+.4f}] |")
    lines+=['','区间为10,000次场景配对bootstrap、固定seed11、双侧percentile95%。同场景全部anchors一同重采样，模型保持固定；不包含训练随机性或模型选择不确定性，未作多重比较校正。',
        'M0保留原生TF32预测；M0_fp32仅关闭冻结未来head的matmul TF32，观测编码缓存和cuDNN TF32不变，三个训练臂使用同样的未来head精度。没有阈值拟合或R4校准混入。',
        '合并器不自动判定论文/总体科研目标完成。完整原生整数混淆矩阵与GT类别计数在raw_confusions.npz，计数轴为模型、样本、时域、GT行、预测列。','']
    (out/'report.md').write_text('\n'.join(lines))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-root',required=True);p.add_argument('--selection',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['pilot','full'],default='pilot')
    p.add_argument('--out',required=True);a=p.parse_args(argv);started=time.monotonic()
    out=Path(a.out).resolve();require(not out.exists(),'Output must be a new directory')
    pkg=Path(__file__).resolve().parent
    require(sha(pkg/'joint_native_evaluation.py')==PRODUCER_SHA,'Local producer differs from frozen audited evaluator')
    protocol,selection,contract,contract_sha,ordered,shards,arms,loaded=collect(
        a.evaluation_root,a.selection,a.protocol,a.mode)
    require(sha(pkg/'aggregate_memory.py')==protocol['source_sha256']['aggregate_memory.py'],
            'Frozen aggregation mathematics source changed')
    sources=dict(producer_sha256=PRODUCER_SHA,protocol_sha256=PROTOCOL_SHA,
        selection_sha256=SELECTION_SHA[a.mode],contract_sha256=contract_sha,shards=shards,
        final_checkpoints_sha256=contract['final_checkpoints_sha256'],actual_loaded_head_sha256=loaded['head_state_sha256'],
        native_helper_sha256=protocol['source_sha256'],runtime_source_sha256=protocol['runtime_source_sha256'],
        checkpoint_audit='producer strictly loaded and tensor-checked checkpoints; merger verifies consistent bound receipts, without rereading large checkpoint files')
    identities=[row['identity'] for row in ordered];resources=resource_summary(ordered,shards)
    # No output and no metric computation until complete hash/identity validation.
    out.mkdir(parents=True,exist_ok=False)
    if a.mode=='pilot':
        summary=dict(schema='native-joint-pilot-merge-v1',status='PILOT_ENGINEERING_PARITY_RESOURCE_PASS',
            samples=2,scenes=2,ordered_identities=identities,sources=sources,resources=resources,
            parity=dict(all_sample_hash_chains_valid=True,full_official_pilot_order=True,
                all_five_models_five_horizon_GT_counts_match=True,
                all_native_boundary_GT_logits_and_reference_confusion_checks_passed=True),
            performance_metrics_computed=False,performance_success_claim=False,optimizer_steps=0,
            interpretation='Two-scene engineering checks only; no effectiveness or full-run timing guarantee')
        (out/'report.md').write_text('# 联合评估工程pilot\n\n两条冻结场景记录的完整哈希链、互斥分片、五模型五时域GT计数和既有参考一致性检查通过。仅报告工程parity与资源，不计算或汇报性能成功。两样本时延不能保证全量运行耗时。\n')
    else:
        from aggregate_memory import aggregate
        hist={name:np.stack([row['hist'][name] for row in ordered]) for name in MODELS}
        models,comparisons,bootstrap=aggregate(hist,[r['scene_token'] for r in identities],
            protocol['evaluation'].get('minimum_meaningful_effect_pp'))
        summary=dict(schema='native-joint-full-summary-v1',status='COMPLETE_FULL_NATIVE_COMPARISON',
            samples=5119,scenes=150,ordered_identities=identities,sources=sources,resources=resources,
            models=models,comparisons=comparisons,bootstrap=bootstrap,
            primary='arithmetic mean of four future independently pooled GMO IoUs',
            secondary='GMO IoU after pooling all four future confusion matrices',
            units=dict(scores='ratio',differences='ratio and pp=100*ratio'),
            historical_validation_exposure=True,fixed_training_seed=11,training_seeds_added=0,
            full_validation=True,new_blind_test=False,threshold_fitting=False,R4_calibration_reused=False,
            goal_complete_decision='NOT_MADE_BY_MERGER')
        full_outputs(out,summary,ordered)
    summary.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   merger_sha256=sha(__file__),merge_seconds=time.monotonic()-started)
    write(out/'summary.json',summary)
    ledger=[dict(sample_token=row['identity']['sample_token'],official_index=row['identity']['official_index'],
        selection_ordinal=row['identity']['selection_ordinal'],records_sha256=row['records_sha256'],
        complete_sha256=row['complete_sha256']) for row in ordered]
    write(out/'sample_hash_ledger.json',ledger)
    names=['summary.json','report.md','sample_hash_ledger.json']+([] if a.mode=='pilot' else ['raw_confusions.npz','metrics.csv','contrasts.csv'])
    write(out/'complete.json',dict(status='COMPLETE',mode=a.mode,samples=len(ordered),
        summary_sha256=sha(out/'summary.json'),files_sha256={name:sha(out/name) for name in names},
        merge_seconds=time.monotonic()-started,source_result_files_modified=False))
    print(json.dumps(dict(status=summary['status'],out=str(out),samples=len(ordered))),flush=True)


if __name__=='__main__':main()
