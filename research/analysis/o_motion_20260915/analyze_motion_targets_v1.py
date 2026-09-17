"""Train512 raw-box distribution and GT-past-only constant-velocity diagnostic.

Only train/*.json.gz label files are opened. Complete/manifest are read solely
for source identity/hash binding; no development label, model or prediction
file is accessed. This is not an error estimate for O or any learned model.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
COMPLETE_SHA = 'e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69'
SPEED_BINS = ('speed_le_0.1', 'speed_gt_0.1_le_0.5', 'speed_gt_0.5_le_5', 'speed_gt_5')
DISTANCE_BINS = ('distance_lt_10', 'distance_10_to_lt_30', 'distance_30_to_lt_50', 'distance_ge_50')
VISIBILITY = ('1','2','3','4','unknown')
FUTURE = (3,4,5,6)


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    x = np.asarray(values,dtype=np.float64)
    require(x.ndim == 1 and np.isfinite(x).all(), 'Nonfinite distribution values')
    if not len(x): return dict(n=0,mean=None,median=None,p90=None)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p90=float(np.percentile(x,90)))


def speed_bin(value):
    return SPEED_BINS[0] if value<=.1 else SPEED_BINS[1] if value<=.5 else SPEED_BINS[2] if value<=5 else SPEED_BINS[3]


def distance_bin(value):
    return DISTANCE_BINS[0] if value<10 else DISTANCE_BINS[1] if value<30 else DISTANCE_BINS[2] if value<50 else DISTANCE_BINS[3]


def blank():
    return dict(t0_instance_pairs=0,future_missing=0,future_valid_history_missing=0,
                future_valid_history_present=0,history_minus1=0,history_minus2=0,
                zero_all_available_xy=[],zero_all_available_3d=[],
                paired_zero_xy=[],paired_cv_xy=[],paired_zero_3d=[],paired_cv_3d=[],
                actual_dt_seconds=[],endpoint_average_global_xy_speed_mps=[])


def add(cell, item):
    cell['t0_instance_pairs']+=1
    if not item['future_valid']:
        cell['future_missing']+=1;return
    for key in ('actual_dt_seconds','endpoint_average_global_xy_speed_mps','zero_all_available_xy','zero_all_available_3d'):
        cell[key].append(item[key])
    if not item['history_valid']:
        cell['future_valid_history_missing']+=1;return
    cell['future_valid_history_present']+=1;cell['history_minus'+str(item['history_offset'])]+=1
    for key in ('paired_zero_xy','paired_cv_xy','paired_zero_3d','paired_cv_3d'):
        cell[key].append(item[key])


def finish(cell):
    result={k:v for k,v in cell.items() if not isinstance(v,list)}
    result['distributions']={k:describe(v) for k,v in cell.items() if isinstance(v,list)}
    paired=cell['future_valid_history_present']
    require(len(cell['paired_zero_xy'])==len(cell['paired_cv_xy'])==paired,'Unequal paired populations')
    for space in ('xy','3d'):
        z=np.asarray(cell['paired_zero_'+space]);cv=np.asarray(cell['paired_cv_'+space]);delta=z-cv
        result['paired_zero_minus_cv_'+space+'_m']=describe(delta)
        result['fraction_cv_strictly_better_'+space]=float(np.mean(cv<z)) if paired else None
    require(cell['future_missing']+cell['future_valid_history_missing']+paired==cell['t0_instance_pairs'],'Coverage does not exhaust population')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets',required=True);parser.add_argument('--out-prefix',required=True)
    args=parser.parse_args(argv);started=time.monotonic();root=Path(args.targets).resolve();prefix=Path(args.out_prefix).resolve()
    json_out=prefix.with_suffix('.json');md_out=prefix.with_suffix('.md')
    require(not json_out.exists() and not md_out.exists(),'Refuse to overwrite analysis outputs')
    require(sha(root/'manifest.json')==MANIFEST_SHA and sha(root/'complete.json')==COMPLETE_SHA,'Wrong frozen label source')
    complete=json.loads((root/'complete.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    require(complete['status']=='COMPLETE' and complete['manifest_sha256']==MANIFEST_SHA,'Incomplete label source')
    # Manifest descriptors only; never open any development payload.
    selected=[r for r in manifest['records'] if r['identity']['split']=='train']
    require(len(selected)==512 and [r['ordinal'] for r in selected]==list(range(512)),'Expected exact train512 order')
    require(len({r['identity']['scene_token'] for r in selected})==256,'Expected256 train scenes')
    groups=[]
    for _ in FUTURE:
        groups.append({'all':{'all':blank()},'speed':{k:blank() for k in SPEED_BINS+('future_missing_undefined_speed',)},
                       'distance':{k:blank() for k in DISTANCE_BINS},'visibility':{k:blank() for k in VISIBILITY},
                       't0_center_ROI':{'inside':blank(),'outside':blank()}})
    population=Counter();file_hashes={};observed_instances=set();observed_frames=set();scene_counts=Counter()
    past_speeds=[];past_dts=[];anchor_motion_shares=[]
    for desc in selected:
        rel=Path(desc['file']);require(rel.parts[0]=='train' and len(rel.parts)==2 and rel.suffix=='.gz','Non-train label path refused')
        file=root/rel;require(file.resolve().parent==root/'train','Unsafe label path')
        digest=sha(file);require(digest==desc['sha256'],'Train label SHA differs');file_hashes[str(rel)]=digest
        raw=gzip.decompress(file.read_bytes());require(hashlib.sha256(raw).hexdigest()==desc['uncompressed_json_sha256'],'Decompressed label SHA differs')
        value=json.loads(raw);require(value['identity']==desc['identity'] and value['ordinal']==desc['ordinal'],'Wrong label identity')
        require(value['schema']=='raw-nuscenes-motion-target-v1' and value['old_refine_applied'] is False and value['missing_fill_applied'] is False,'Refined or filled labels refused')
        frames=value['frames'];require(len(frames)==7 and frames[2]['sample_token']==value['identity']['sample_token'],'Invalid frame contract')
        ts=np.asarray([f['timestamp_us'] for f in frames],dtype=np.int64);dt=(ts-ts[2])/1e6
        require(np.all(np.diff(ts)>0) and np.array_equal(dt,np.asarray([f['dt_seconds'] for f in frames])),'Timestamp/dt contract differs')
        G=np.asarray(frames[2]['lidar_to_global_column_matrix'],dtype=np.float64)
        require(G.shape==(4,4) and np.isfinite(G).all() and np.allclose(G[:3,:3].T@G[:3,:3],np.eye(3),rtol=0,atol=1e-12),'Invalid t0 rotation')
        Rinv=G[:3,:3].T;scene_counts[value['identity']['scene_token']]+=1;population['anchors']+=1
        population['union_anchor_instance_pairs']+=len(value['tracks']);anchor_valid_future=anchor_static_future=0
        for track in value['tracks']:
            valid=track['valid_mask'];require(len(valid)==7 and all(type(v)is bool for v in valid),'Invalid mask')
            if not valid[2]:population['t0_absent_union_pairs']+=1;continue
            population['t0_present_instance_pairs']+=1;observed_instances.add(track['instance_token'])
            positions=track['global_centers_m'];require(all((p is not None)==v for p,v in zip(positions,valid)),'Missing center/mask conflict')
            p0=np.asarray(positions[2],dtype=np.float64);p0R=Rinv@(p0-G[:3,3]);distance=float(np.linalg.norm(p0R[:2]));dbin=distance_bin(distance)
            vis=track['visibility_tokens'][2] or 'unknown';require(vis in VISIBILITY,'Unknown visibility category')
            roi='inside' if np.all(p0R>=np.asarray([-51.2,-51.2,-5.0])) and np.all(p0R<np.asarray([51.2,51.2,3.0])) else 'outside'
            population['t0_center_ROI_'+roi]+=1
            past=1 if valid[1] else 0 if valid[0] else None
            history_valid=past is not None;velocityR=None
            if history_valid:
                past_dt=float(-dt[past]);require(past_dt>0,'Non-past CV source')
                velocity_global=(p0-np.asarray(positions[past],dtype=np.float64))/past_dt
                velocityR=Rinv@velocity_global;past_speeds.append(float(np.linalg.norm(velocity_global[:2])));past_dts.append(past_dt)
                population['history_minus'+str(2-past)]+=1
            else:population['no_available_history']+=1
            if not any(valid[h] for h in FUTURE):population['no_future_valid']+=1
            for hidx,h in enumerate(FUTURE):
                item={'future_valid':valid[h],'history_valid':history_valid}
                if not valid[h]:speed_group='future_missing_undefined_speed'
                else:
                    target_global=np.asarray(positions[h],dtype=np.float64)-p0;targetR=Rinv@target_global
                    speed=float(np.linalg.norm(target_global[:2])/dt[h]);speed_group=speed_bin(speed)
                    zero_xy=float(np.linalg.norm(targetR[:2]));zero_3d=float(np.linalg.norm(targetR))
                    item.update(actual_dt_seconds=float(dt[h]),endpoint_average_global_xy_speed_mps=speed,
                                zero_all_available_xy=zero_xy,zero_all_available_3d=zero_3d)
                    observed_frames.add((track['instance_token'],frames[h]['sample_token']))
                    anchor_valid_future+=1;anchor_static_future+=speed<=.1
                    if history_valid:
                        residualR=velocityR*dt[h]-targetR
                        item.update(history_offset=2-past,paired_zero_xy=zero_xy,paired_zero_3d=zero_3d,
                                    paired_cv_xy=float(np.linalg.norm(residualR[:2])),paired_cv_3d=float(np.linalg.norm(residualR)))
                for family,label in [('all','all'),('speed',speed_group),('distance',dbin),('visibility',vis),('t0_center_ROI',roi)]:
                    add(groups[hidx][family][label],item)
        if anchor_valid_future:anchor_motion_shares.append(anchor_static_future/anchor_valid_future)
    require(population['anchors']==512 and set(scene_counts.values())=={2},'Training scenes/sample order changed')
    results=[]
    for hidx,h in enumerate(FUTURE):
        result={family:{k:finish(v) for k,v in values.items()} for family,values in groups[hidx].items()}
        for family in ('speed','distance','visibility','t0_center_ROI'):
            require(sum(v['t0_instance_pairs'] for v in result[family].values())==population['t0_present_instance_pairs'],'Strata do not exhaust same population')
        results.append(dict(relative_frame_index=h-2,nominal_horizon_seconds=(h-2)*.5,**result))
    pooled={space:{} for space in ('xy','3d')}
    for space in pooled:
        for method in ('zero','cv'):
            vals=[x for groups_h in groups for x in groups_h['all']['all']['paired_'+method+'_'+space]]
            pooled[space][method+'_m']=describe(vals)
    source=dict(script_sha256=sha(__file__),manifest_sha256=MANIFEST_SHA,complete_sha256=COMPLETE_SHA,
                selection_sha256=manifest['selection']['sha256'],targets_directory=str(root),train_file_sha256=file_hashes,
                opened_train_label_files=512,opened_development_label_files=0,predictions_or_models_read=False,
                manifest_read_for_provenance_only=True)
    summary=dict(schema='train-motion-distribution-v1',status='COMPLETE_TRAIN_ONLY_LABEL_DIAGNOSTIC',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=source,population=dict(population),
        unique_t0_present_instance_tokens=len(observed_instances),unique_scored_future_instance_sample_pairs=len(observed_frames),
        scene_count=256,by_horizon=results,pooled_future_instance_horizon_pairs=pooled,
        past_global_xy_speed_mps=describe(past_speeds),past_dt_seconds=describe(past_dts),
        per_anchor_fraction_valid_future_endpoints_speed_le_0_1=describe(anchor_motion_shares),
        definition=dict(t0_index=2,history_choice='index1 if valid, else index0 if valid, else unavailable',
            cv='(global_center_t0-global_center_past)/(timestamp_t0-timestamp_past), rotated by t0 LiDAR R inverse; no future sample in velocity',
            target='R0_inverse @ (global_center_h-global_center_t0), meters; future prediction=CV_velocity_R*actual_dt_h',
            zero_motion='zero displacement in fixed t0 LiDAR R, not zero relative-to-moving-ego displacement',
            primary_speed='per horizon: norm(global_center_h_xy-global_center_t0_xy)/actual_dt_h; endpoint average speed magnitude, not accumulated-path speed',
            speed_bins={'speed_le_0.1':'[0,0.1]','speed_gt_0.1_le_0.5':'(0.1,0.5]','speed_gt_0.5_le_5':'(0.5,5]','speed_gt_5':'(5,infinity)'},
            distance='t0 fixed LiDAR xy Euclidean norm, meters; [0,10),[10,30),[30,50),[50,infinity)',
            visibility='raw t0 annotation visibility token; no visibility filtering',
            ROI='t0 center in half-open original box [-51.2,51.2)x[-51.2,51.2)x[-5,3); center test only, not fine occupancy coverage',
            paired='Both baseline errors use same t0-present, future-present, past-present instance/horizon set',
            zero_all_available='Additional zero baseline uses every t0/future valid pair; do not compare to CV with different denominator',
            delta='zero error minus CV error: positive favors CV',
            pooling='Each eligible anchor-instance-horizon equally weighted; instance sizes/voxel counts not weighted',
            quantiles='NumPy linear percentiles; p90 is a descriptive quantile, not a confidence bound'),
        limitations=dict(not_O_motion_error=True,GT_past_centers_are_privileged=True,
            future_GT_used_only_for_targets_and_descriptive_speed_strata=True,training_or_threshold_selection=False,
            repeated_anchor_instance_horizon_pairs_are_not_independent=True,
            no_future_imputation=True,no_model_or_GPU_computation=True,no_development_or_prediction_analysis=True,
            boxes_outside_fine_GT_support_retained=True,annotation_noise_not_actual_micro_motion=True,
            box_center_motion_not_point_scene_flow=True),seconds=time.monotonic()-started)
    def number(v):return '—' if v is None else f'{v:.4f}'
    lines=['# train512 原始运动标签分布与历史真值CV诊断','',
        '只打开512份训练标签；未打开development标签或任何预测。CV速度只由t0与最近有效历史帧（优先−1，否则−2）中心差/真实dt构造。它使用GT历史中心，属于有特权观测的运动学参照，不是O的运动预测，也不是可直接部署的检测器。',
        '',f"t0存在的anchor×instance对：{population['t0_present_instance_pairs']:,}；t0缺失并集对：{population['t0_absent_union_pairs']:,}；优先历史−1：{population['history_minus1']:,}，回退−2：{population['history_minus2']:,}，无历史：{population['no_available_history']:,}。",'',
        '全部误差单位m，固定t0 LiDAR坐标；零/CV主表严格使用相同有效集合。速度分层按每h的global xy端点位移/真实dt，不是累计行程速度。同一实例可随h落入不同速度组。',
        '', '| nominal h(s) | 分层 | 配对数 | future缺失 | 无历史但future有效 | zero xy mean/median/p90 | CV xy mean/median/p90 | zero 3D mean/median/p90 | CV 3D mean/median/p90 |',
        '|---:|---|---:|---:|---:|---|---|---|---|']
    def triple(x):return '/'.join(number(x[k]) for k in ('mean','median','p90'))
    for horizon in results:
        for family in ('all','speed','distance','visibility','t0_center_ROI'):
            for label,cell in horizon[family].items():
                d=cell['distributions']
                lines.append(f"| {horizon['nominal_horizon_seconds']:g} | {family}:{label} | {cell['future_valid_history_present']} | {cell['future_missing']} | {cell['future_valid_history_missing']} | {triple(d['paired_zero_xy'])} | {triple(d['paired_cv_xy'])} | {triple(d['paired_zero_3d'])} | {triple(d['paired_cv_3d'])} |")
    lines += ['', 'JSON另保留全部可评零运动分布、各组zero−CV误差改善、CV逐例优于零运动的比例、实际时间和速度分布。无历史的零运动数值只作覆盖补充，不能拿不同集合的均值与CV比较。',
        '', '解释边界：anchor、实例和时域重复，不是独立样本或新的泛化评测；不作显著性/多seed声明。动态阈值固定为0.1/0.5/5m/s，distance边界预设10/30/50m。域外中心及低可见对象保留；ROI中心分层也不等于fine occupancy有效体素覆盖。标签框的小位移可能含标注噪声。单看总体位移回归误差可能受静止目标占比影响；需要分别检查moving组与静止组，不能据此称模型超过O。',
        '',f"来源manifest SHA `{MANIFEST_SHA}`；分析源码SHA `{source['script_sha256']}`。完整512个输入SHA在JSON中。"]
    for rel,digest in file_hashes.items():require(sha(root/rel)==digest,'Input changed during analysis')
    require(sha(root/'manifest.json')==MANIFEST_SHA and sha(root/'complete.json')==COMPLETE_SHA,'Source receipts changed')
    with json_out.open('x') as stream:json.dump(summary,stream,indent=2,allow_nan=False);stream.write('\n')
    with md_out.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=summary['status'],json=str(json_out),md=str(md_out),seconds=summary['seconds'],
                          population=dict(population),json_sha256=sha(json_out),script_sha256=source['script_sha256'])))


if __name__=='__main__':
    main()
