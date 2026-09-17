"""Privileged history-centre CV versus recorded D on identical sparse points.

CPU/NumPy only; no learned forward, no GPU, no training, no new bootstrap.
CV has only t0 and the nearest valid past centre plus t0's coordinate basis.
Future boxes never enter cv_displacements; sparse future targets define error.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
D_COMPLETE='f0bf075fa36622b38c0c5136b187a184d16d3e07ed00857dfe21a0545f1c5f56'
D_PROTOCOL='e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
SPARSE_MANIFEST='cdb11231cbc3213d213e1f3f718fe9d3c90665dccea4154ecf3372ad1ce91efe'
SPARSE_COMPLETE='3d14a03cd61761a1ebc26313463dbff06854907c19c49057667a0ffd257e05bd'
RAW_MANIFEST='4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
RAW_COMPLETE='e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69'
GROUPS=('stationary','ambiguous','moving');HORIZONS=(.5,1.,1.5,2.)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def source(name):
    for p in (HERE/name,HERE.parent/'m0_improvement_20260915'/name):
        if p.is_file():return p
    raise FileNotFoundError(name)

def cv_displacements(past_centers, past_valid, past_timestamps_us, global_to_R_rotation):
    """Arguments contain ONLY frames -2,-1,0; no future dt/pose/velocity.

    Translation CV preserves current orientation; no angular-velocity fit.
    Full global xyz translation, rotated as a vector (without translation).
    All four slots use fixed nominal h, matching the D prediction slots.
    """
    require(len(past_centers)==len(past_valid)==len(past_timestamps_us)==3,'Only three past/current frames allowed')
    require(past_valid[2],'CV requires t0 annotation')
    chosen=1 if past_valid[1] else 0 if past_valid[0] else None
    if chosen is None:return None,dict(status='no_valid_history',relative_frame_index=None)
    dt=(int(past_timestamps_us[2])-int(past_timestamps_us[chosen]))/1e6
    c0=np.asarray(past_centers[2],np.float64);cp=np.asarray(past_centers[chosen],np.float64)
    rot=np.asarray(global_to_R_rotation,np.float64)
    require(dt>0 and math.isfinite(dt) and c0.shape==cp.shape==(3,) and rot.shape==(3,3) and
            np.isfinite(c0).all() and np.isfinite(cp).all() and np.isfinite(rot).all(),'Invalid historical state')
    velocity=(c0-cp)/dt
    # No future value occurs on the prediction side of this expression.
    displacement=np.asarray([rot@(velocity*h) for h in HORIZONS],np.float64)
    return displacement,dict(status='history_available',relative_frame_index=chosen-2,
        historical_dt_seconds=dt,global_translation_velocity_mps=velocity.tolist())


def stats(values):
    a=np.asarray(values,np.float64);require(np.isfinite(a).all(),'Nonfinite metric')
    return dict(count=len(a),mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None)


def grouped(rows,models):
    output=[]
    for h in HORIZONS:
        for group in ('all',*GROUPS):
            chosen=[r for r in rows if r['horizon_seconds']==h and (group=='all' or r['group']==group)]
            output.append(dict(horizon_seconds=h,group=group,objects=len(chosen),
                source_points=sum(r['source_points'] for r in chosen),
                models={model:{dim:stats([r[model][dim] for r in chosen]) for dim in ('xy_m','xyz_m')} for model in models}))
    return output


def small_chain(root,done,skip=()):
    for name,digest in done['files_sha256'].items():
        require(Path(name).name==name,'Unsafe file descriptor')
        if name not in skip:require(sha(root/name)==digest,'Completed file changed: '+name)


def run(a):
    start=time.monotonic();base=Path(a.root).resolve();training=base/'server_results/training/connected_motion_train_v2'
    sparse=base/'sparse_motion_v1';rawroot=base/'motion_targets_v1'
    require(sha(training/'complete.json')==D_COMPLETE and sha(base/'connected_motion_protocol_v2.json')==D_PROTOCOL,'D provenance changed')
    done=read(training/'complete.json');manifest=read(training/'manifest.json');protocol=read(base/'connected_motion_protocol_v2.json')
    require(done['status']=='COMPLETE_CONNECTED_MOTION_TRAINING' and (done['updates'],done['examples'],done['evaluated_samples'])==(512,2048,200),'Incomplete D final')
    small_chain(training,done)
    for name,digest in manifest['sources']['sources_sha256'].items():require(sha(source(name))==digest,'Training source changed: '+name)
    require(manifest['sources']['protocol_sha256']==D_PROTOCOL and manifest['sources']['sources_sha256']==protocol['sources_sha256'],'Source/protocol mismatch')
    for arm in ('J','D'):
        ar=training/'runs'/arm;ac=read(ar/'complete.json')
        require(sha(ar/'complete.json')==done['arm_complete_sha256'][arm],'Arm chain changed')
        small_chain(ar,ac,skip=('final.pth',))
        require(done['final_checkpoints'][arm]['sha256']==ac['files_sha256']['final.pth'],'Final pointer mismatch')
    # Existing actual D tensor/optimizer-load receipt, not a new tensor load.
    common=base/'server_results/training/common_connected_motion_dev200_v2';cc=read(common/'complete.json')
    require(cc['status']=='COMPLETE_FINAL_DEVELOPMENT_EVALUATION' and cc['training_complete_sha256']==D_COMPLETE,'Wrong actual-D load proof')
    small_chain(common,cc);dproof=read(common/'loaded_models.json')['arms']['D']['training_loader_receipt']
    require(dproof['checkpoint_sha256']==done['final_checkpoints']['D']['sha256'] and
            dproof['actual_motion_state_sha256']==done['final_checkpoints']['D']['motion_state_sha256'] and
            dproof['actual_optimizer_parameter_steps_all512'] and dproof['fixed_final_update']==512,'Wrong actual D state receipt')
    require(sha(sparse/'manifest.json')==SPARSE_MANIFEST and sha(sparse/'complete.json')==SPARSE_COMPLETE and
            sha(rawroot/'manifest.json')==RAW_MANIFEST and sha(rawroot/'complete.json')==RAW_COMPLETE,'Target provenance changed')
    sm=read(sparse/'manifest.json');sc=read(sparse/'complete.json');rm=read(rawroot/'manifest.json');rc=read(rawroot/'complete.json')
    require(sc['manifest_sha256']==SPARSE_MANIFEST and sc['raw_manifest_sha256']==RAW_MANIFEST and sc['raw_complete_sha256']==RAW_COMPLETE and
            rc['manifest_sha256']==RAW_MANIFEST and protocol['labels']==dict(manifest_sha256=SPARSE_MANIFEST,complete_sha256=SPARSE_COMPLETE),'Training label link changed')
    require(sha(source('build_sparse_motion_supervision_v1.py'))==sc['script_sha256'] and sha(source('motion_geometry.py'))==sc['geometry_sha256'],'Sparse geometry source changed')
    helperpath=source('train_source_motion_v1.py');spec=importlib.util.spec_from_file_location('history_sparse_helper',helperpath)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    require(tuple(helper.GROUPS)==GROUPS,'D original groups changed')
    descriptors=[x for x in sm['records'] if x['identity']['split']=='development'];rawdesc={x['identity']['sample_token']:x for x in rm['records']}
    require(len(descriptors)==200 and len({x['identity']['scene_token'] for x in descriptors})==100,'Wrong dev200 selection')
    fullobjects=jsonl(training/'development_objects.jsonl');Drows=[r for r in fullobjects if r['arm']=='D']
    require(len(Drows)==16074,'Unexpected recorded D support')
    Dmap={};Dorder=[]
    for row in Drows:
        key=(row['sample_token'],row['instance_token'],row['horizon_seconds']);require(key not in Dmap,'Duplicate D object-horizon');Dmap[key]=row;Dorder.append(key)
    # No train16 D object EPE exists in the preceding partial-oracle artifacts.
    partial=base/'server_results/training/partial_oracle_D_train16_v1';partialc=read(partial/'complete.json');small_chain(partial,partialc)
    partialrows=jsonl(partial/'records.jsonl');require(len(partialrows)==16 and all('epe_records' not in r and 'physical_objects' not in r for r in partialrows),'Train16 record schema changed; reconsider requested priority')
    rows=[];full=[];excluded=[];coverage=[];inputs=[];expectedkeys=set()
    for desc in descriptors:
        require(time.monotonic()-start<a.max_seconds,'CPU time ceiling')
        identity=desc['identity'];token=identity['sample_token'];rd=rawdesc[token]
        require(rd['identity']==identity and desc['label_source_sha256']==rd['sha256'],'Raw/sparse identity/source mismatch')
        path=rawroot/rd['file'];require(path.stat().st_size==rd['bytes'] and sha(path)==rd['sha256'],'Raw label file changed')
        rawbytes=gzip.decompress(path.read_bytes());require(hashlib.sha256(rawbytes).hexdigest()==rd['uncompressed_json_sha256'],'Decoded raw bytes differ')
        raw=json.loads(rawbytes);require(raw['identity']==identity and raw['ordinal']==desc['ordinal'],'Raw identity changed')
        label=helper.load_sparse(sparse,desc,identity)
        frames=raw['frames'];require([f['relative_frame_index'] for f in frames]==list(range(-2,5)) and frames[2]['sample_token']==token,'Wrong frame order')
        G0=np.asarray(frames[2]['lidar_to_global_column_matrix'],np.float64)
        require(np.allclose(G0[3],[0,0,0,1],rtol=0,atol=1e-12) and np.allclose(G0[:3,:3].T@G0[:3,:3],np.eye(3),rtol=0,atol=1e-6),'Nonrigid t0 basis')
        rotation=np.linalg.inv(G0)[:3,:3]
        tracks={t['instance_token']:t for t in raw['tracks']};retained=set(label['instance_tokens'].tolist())
        t0tracks=[t for t in raw['tracks'] if t['valid_mask'][2]]
        cv={};history={};histcounts={'nearest_minus1':0,'fallback_minus2':0,'no_history':0}
        for t in t0tracks:
            pred,info=cv_displacements(t['global_centers_m'][:3],t['valid_mask'][:3],[f['timestamp_us'] for f in frames[:3]],rotation)
            cv[t['instance_token']]=pred;history[t['instance_token']]=info
            histcounts['no_history' if pred is None else 'nearest_minus1' if info['relative_frame_index']==-1 else 'fallback_minus2']+=1
        K=len(label['instance_tokens']);N=len(label['source_flat_indices'])
        require(K==desc['counts']['retained_objects'] and N==desc['counts']['source_points'] and
                len(t0tracks)==desc['counts']['t0_present_boxes'] and len(t0tracks)-K==desc['counts']['t0_boxes_without_unique_grid_point'],'Original source support counts differ')
        rowcov=dict(identity=identity,raw_union_instances=len(tracks),t0_present_instances=len(t0tracks),retained_unique_point_objects=K,
            t0_objects_without_unique_source_point=len(t0tracks)-K,source_points=N,overlapping_grid_points=desc['counts']['overlapping_grid_points'],
            raw_t0_history=histcounts,retained_history={key:sum(('no_history' if cv[t] is None else 'nearest_minus1' if history[t]['relative_frame_index']==-1 else 'fallback_minus2')==key for t in retained) for key in histcounts},horizons=[])
        for h,nominal in enumerate(HORIZONS):
            future_index=h+3;timestampdt=(frames[future_index]['timestamp_us']-frames[2]['timestamp_us'])/1e6
            require(timestampdt==float(label['dt_future_seconds'][h]),'Target timestamp differs')
            validraw=[t for t in t0tracks if t['valid_mask'][future_index]]
            hcov=dict(horizon_seconds=nominal,raw_t0_present_future_valid=len(validraw),raw_t0_present_future_missing=len(t0tracks)-len(validraw),
                future_present_t0_missing=sum(t['valid_mask'][future_index] and not t['valid_mask'][2] for t in tracks.values()),
                future_valid_no_unique_source_points=sum(t['instance_token'] not in retained for t in validraw),
                retained_future_valid_objects=0,retained_future_missing_objects=0,retained_future_valid_points=0,retained_future_missing_points=0,
                common_objects=0,common_points=0,future_valid_without_history_objects=0,future_valid_without_history_points=0)
            for k,instance in enumerate(label['instance_tokens']):
                instance=str(instance);t=tracks[instance];require(t['valid_mask'][2],'Retained object absent at t0')
                ids=np.flatnonzero(label['object_index']==k);chosen=label['valid'][h]&(label['object_index']==k)
                futurevalid=bool(label['object_future_valid'][h,k]);require(futurevalid==bool(t['valid_mask'][future_index]),'Target future validity differs')
                require(np.count_nonzero(chosen)==(len(ids) if futurevalid else 0),'Unexpected object-specific valid point selection')
                if not futurevalid:
                    hcov['retained_future_missing_objects']+=1;hcov['retained_future_missing_points']+=len(ids);continue
                n=len(ids);key=(token,instance,nominal);require(key in Dmap,'D row missing from exact sparse support');expectedkeys.add(key);dr=Dmap[key]
                target=label['target_displacement_m'][h,chosen].astype(np.float64)
                zero3=float(np.linalg.norm(target,axis=1).mean());zeroxy=float(np.linalg.norm(target[:,:2],axis=1).mean())
                group=GROUPS[int(label['object_speed_group'][h,k])]
                require(dr['scene_token']==identity['scene_token'] and dr['group']==group and dr['source_points']==n and dr['dt_seconds']==timestampdt,'Recorded D identity/group/source/dt mismatch')
                require(dr['zero_epe_3d_m']==zero3 and dr['zero_epe_xy_m']==zeroxy,'Actual identical-point zero reference differs')
                hcov['retained_future_valid_objects']+=1;hcov['retained_future_valid_points']+=n
                fr=dict(sample_token=token,scene_token=identity['scene_token'],instance_token=instance,horizon_seconds=nominal,dt_seconds=timestampdt,
                    group=group,source_points=n,D=dict(xy_m=dr['epe_xy_m'],xyz_m=dr['epe_3d_m']),zero=dict(xy_m=zeroxy,xyz_m=zero3))
                full.append(fr)
                if cv[instance] is None:
                    hcov['future_valid_without_history_objects']+=1;hcov['future_valid_without_history_points']+=n
                    excluded.append(dict(fr,reason='no_valid_history_minus1_or_minus2'));continue
                delta=cv[instance][h]-target
                hcov['common_objects']+=1;hcov['common_points']+=n
                rows.append(dict(fr,CV=dict(xy_m=float(np.linalg.norm(delta[:,:2],axis=1).mean()),xyz_m=float(np.linalg.norm(delta,axis=1).mean())),
                    historical_source=history[instance],CV_displacement_R_m=cv[instance][h].tolist(),sparse_label_sha256=desc['sha256'],raw_label_sha256=rd['sha256']))
            require(hcov['retained_future_valid_objects']==desc['counts']['valid_objects_by_horizon'][h] and hcov['retained_future_valid_points']==desc['counts']['valid_points_by_horizon'][h],'Native support mismatch')
            require(hcov['common_objects']+hcov['future_valid_without_history_objects']==hcov['retained_future_valid_objects'] and
                    hcov['common_points']+hcov['future_valid_without_history_points']==hcov['retained_future_valid_points'],'History support partition differs')
            rowcov['horizons'].append(hcov)
        coverage.append(rowcov);inputs.append(dict(identity=identity,sparse_file=desc['file'],sparse_sha256=desc['sha256'],raw_file=rd['file'],raw_sha256=rd['sha256'],raw_uncompressed_sha256=rd['uncompressed_json_sha256']))
    require(expectedkeys==set(Dmap) and len(full)==len(Drows),'Extra or missing D support keys')
    allsummary=grouped(full,('D','zero'));commonsummary=grouped(rows,('D','CV','zero'));excludedsummary=grouped(excluded,('D','zero'))
    original=read(training/'summary.json')['physical_epe']['D']['groups']
    for new,old in zip(allsummary,original):
        require((new['horizon_seconds'],new['group'],new['objects'],new['source_points'])==(old['horizon_seconds'],old['group'],old['objects'],old['source_points']),'D original grouped support differs')
        for model,prefix in (('D',''),('zero','zero_')):
            for dim,suffix in (('xy_m','xy_m'),('xyz_m','3d_m')):
                require(new['models'][model][dim]==old['metrics'][prefix+'epe_'+suffix],'D original mean/median/p90 differs')
    totals=dict(samples=200,scenes=100,raw_union_anchor_instances=sum(x['raw_union_instances'] for x in coverage),t0_present_anchor_instances=sum(x['t0_present_instances'] for x in coverage),
        retained_unique_point_anchor_instances=sum(x['retained_unique_point_objects'] for x in coverage),t0_objects_without_unique_source_point=sum(x['t0_objects_without_unique_source_point'] for x in coverage),
        empty_source_anchors=sum(x['source_points']==0 for x in coverage),anchors_without_any_D_valid_future=sum(not any(h['retained_future_valid_objects'] for h in x['horizons']) for x in coverage),
        raw_t0_history={k:sum(x['raw_t0_history'][k] for x in coverage) for k in coverage[0]['raw_t0_history']},
        retained_history={k:sum(x['retained_history'][k] for x in coverage) for k in coverage[0]['retained_history']},horizons=[])
    for h,nominal in enumerate(HORIZONS):
        totals['horizons'].append({k:nominal if k=='horizon_seconds' else sum(x['horizons'][h][k] for x in coverage) for k in coverage[0]['horizons'][h]})
    require('torch' not in sys.modules,'This diagnostic must remain CPU NumPy-only')
    output=dict(schema='history-state-predictability-diagnostic-v1',status='COMPLETE_READ_ONLY_DIAGNOSTIC',elapsed_seconds=time.monotonic()-start,
        selection='previously exposed dev200/100scenes; train16 existing artifacts have no D object EPE; no selection by performance',
        prediction_contract='v_global=(center_t0-center_nearest_valid_past)/actual_historical_dt; d_R(h)=inv(G0).rotation @ (v_global*nominal_h); translation only; no future center/pose/dt/velocity supplied to CV',
        error_contract='exact original sparse source points, valid future support, float32 stored rigid targets promoted to float64; Euclidean XY/3D distance mean over points then equally weighted anchor-instance errors',
        D_contract='authenticated actual final D scalar EPE records, not rerun D vectors; original stationary<=.1, ambiguous(.1,.5], moving>.5 m/s endpoint-globalXY groups; future grouping only describes errors',
        privileged_diagnostic_not_deployable=True,model_forward_performed=False,optimizer_updates=0,GPU_used=False,bootstrap_performed=False,
        no_O_hard_positive_mask_stratification='Not saved for the same sparse points; no new inference or GT replacement for prediction support',
        no_dense_flow_or_O_native_EPE_claim=True,no_independence_or_novel_baseline_claim=True,
        checks=dict(D_all_record_keys_covered=True,D_original_full_group_stats_exact=True,identical_source_zero_EPE_exact=True,sparse_and_raw200_all_SHA_verified=True,all_missing_support_explicit=True),
        sources=dict(d_training_complete_sha256=D_COMPLETE,d_protocol_sha256=D_PROTOCOL,d_checkpoint_declaration=done['final_checkpoints']['D'],d_records_sha256=sha(training/'development_objects.jsonl'),
            d_existing_actual_load_complete_sha256=sha(common/'complete.json'),d_existing_loaded_models_sha256=sha(common/'loaded_models.json'),
            sparse_manifest_sha256=SPARSE_MANIFEST,sparse_complete_sha256=SPARSE_COMPLETE,raw_manifest_sha256=RAW_MANIFEST,raw_complete_sha256=RAW_COMPLETE,
            geometry_sha256=sc['geometry_sha256'],sparse_builder_sha256=sc['script_sha256'],D_helper_sha256=sha(helperpath),
            train16_partial_complete_sha256=sha(partial/'complete.json'),prior_different_support_oracle_report_sha256=sha(base/'oracle_transport_train16_结果与决策.md'),
            prior_train512_center_CV_report_sha256=sha(base/'train_motion_distribution_v1.md'),diagnostic_sha256=sha(__file__)),
        coverage_totals=totals,D_full_original_support=allsummary,common_support_D_CV_zero=commonsummary,D_valid_but_no_history=excludedsummary,
        paired_common_records=rows,excluded_no_history_records=excluded,per_anchor_coverage=coverage,input_sources=inputs)
    return output


def markdown(result):
    lines=['# 已知历史状态时，CV 与当前 D 的同源点可预测性诊断','',
     '本次是**特权状态、事后开发集诊断**，不是可部署 baseline、训练新成果或新泛化测试。旧 train16 oracle 只记录不同支持的 occupancy，未保存相同点的 D 逐对象 EPE；因此使用已暴露的 dev200 / 100 场景。没有重跑旧 train512 中心 CV 表。','',
     'CV 只取 t0 和最近有效历史中心（−0.5s 帧优先，否则 −1s 帧）及真实历史时间差，得到 global xyz 平移速度；按名义 0.5/1/1.5/2s 外推，再以 t0 global→LiDAR 旋转转换向量。没有未来 ego pose、future dt、未来中心或框角速度进入 CV。Future sparse 刚体目标只用于评分和原组别描述。XY 是原 D 口径的固定 t0 LiDAR XY；速度分组依据未来 global XY 端点平均速度。','',
     '每条误差先在原对象的同一批 sparse source points 上求欧氏距离均值，再对 anchor×instance 等权计算 mean / median / p90；不是中心 EPE、点加权总体或按场景均值。D 使用原认证标量，不声称重建其预测向量。没有 O hard-positive 的同点保存掩码，未新增该分层，也未用 GT 冒充预测支持。','',
     '## 支持与缺失','',json.dumps(result['coverage_totals'],ensure_ascii=False,indent=2),'',
     '完整 D 支持与同支持零位移的全部分组统计精确重现原正式开发 summary。以下完整表与共同支持表必须分开看：缺历史不会填零，也不会把不同支持的均值相减解释成 CV 增益。每个 anchor 的缺失/无唯一点/未来无标注账本与逐对象配对结果保存在 JSON。','',
     '## 原完整 D 支持：mean / median / p90（m）','',
     '| h(s) | 原组别 | 对象数 | D XY | D 3D | zero XY | zero 3D |','|---:|---|---:|---:|---:|---:|---:|']
    def cell(s):return '—' if s['mean'] is None else '/'.join(f"{s[k]:.5f}" for k in ('mean','median','p90'))
    for row in result['D_full_original_support']:
        lines.append('| '+str(row['horizon_seconds'])+' | '+row['group']+' | '+str(row['objects'])+' | '+' | '.join(cell(row['models'][m][d]) for m in ('D','zero') for d in ('xy_m','xyz_m'))+' |')
    lines+=['','## 同 future-valid ∩ history 支持：mean / median / p90（m）','','| h(s) | 原组别 | 对象数 | D XY | CV XY | zero XY | D 3D | CV 3D | zero 3D |','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['common_support_D_CV_zero']:
        lines.append('| '+str(row['horizon_seconds'])+' | '+row['group']+' | '+str(row['objects'])+' | '+' | '.join(cell(row['models'][m][d]) for d in ('xy_m','xyz_m') for m in ('D','CV','zero'))+' |')
    lines+=['','## 解释边界','',
     '若同支持 CV 明显低于 D，说明在给定可靠历史对象状态/关联的条件下，简单平移动力学已有可利用的预测结构；不能唯一定位为 BEV 表示、关联、目标分配或优化哪一项出错。历史标注可能有平滑或抖动，模型实际输入也未必能恢复这些特权状态。低速组与高运动组需分别看；刚体源点目标含旋转，纯平移 CV 仍有模型失配。','',
     '结果不保证加入显式状态或训练新读出后会降低全域 FP/FN，更不直接证明占据提升；同一对象跨 anchor / horizon 重复，未做独立性或显著性声明。无模型加载/预测、训练、GPU 或 bootstrap。已有 D actual-load 收据被重新认证，本次没有再次载入大权重。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=str(HERE));parser.add_argument('--out-prefix',default=str(HERE/'history_state_predictability_diagnostic_v1'));parser.add_argument('--max-seconds',type=float,default=120)
    a=parser.parse_args();require(math.isfinite(a.max_seconds) and 0<a.max_seconds<=600,'Bounded CPU ceiling required')
    prefix=Path(a.out_prefix);jp=prefix.with_suffix('.json');mp=prefix.with_suffix('.md');require(not jp.exists() and not mp.exists(),'Do not overwrite existing diagnostic')
    result=run(a);jp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');mp.write_text(markdown(result))
    print(json.dumps(dict(status=result['status'],seconds=result['elapsed_seconds'],json_sha256=sha(jp),coverage=result['coverage_totals'],common_all_and_moving=[x for x in result['common_support_D_CV_zero'] if x['group'] in ('all','moving')]),indent=2))


if __name__=='__main__':main()
