"""Prespecified CPU analysis of authenticated radar and fixed-D train512 exports.

Uses frozen prior weighted_auc/continuous helpers only; no prior main/collector,
Torch, forward, fit, gate, threshold search, neighborhood or development data.
Original full physical EPE is hash-bound and referenced, not newly evaluated.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = 'radar-evidence-discrimination-train-analysis-v1'
RADAR_SCHEMA = 'radar-evidence-train-extraction-v1'
MOTION_SCHEMA = 'motion-evidence-train-extraction-v1'
RADAR_SOURCE_SHA = 'ee835ccb9accf3ae0d0e49c82466ee1504ee349e2f5de633967ba5d08946bd51'
MOTION_SOURCE_SHA = '3bf1cd4c2cadca9d84d2220371f4eaa046db48cd48cd7ccc6fb25f222c110342'
MOTION_ANALYSIS_SHA = '38bb878bffc98c67715adb84f4a1a2983e5425c1e2a07d0fea1509576117b518'
MATH_SOURCE_SHA = '5b8d36a60bba62f81b3e8d494f5b8a591bc540a3cda68fe311d434fbeda1139e'
MATH_RULES_SHA = 'ed1747708515931557dbaa2812abea642267737ec370d2bd8946989602dbc52e'
EXPECTED = dict(index='1dc14643df86f85c5ae87b17b83c60cada99f189a025675d150062acb046e7b1',
    complete='69e49dbd8c7667e9c6e17c2ce70b4f879e8f40c33a0018251b9de2ae978d01bc',
    selection='5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d',
    native='41953909ddfdfd08d51d98385a450640e5314b775ab01458cb55ff1b18e4bfb0',
    runtime='5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c')
FEATURES = ('radar_mean_speed_mps','radar_relative_zero_margin_mps','radar_velocity_residual_mps',
            'radar_log_count_normalized','radar_mean_lag_seconds','lsq_speed_xy_mps')
DIRECTIONS = (1,1,-1,1,-1,1)
ROLES = ('primary','primary','secondary_control','exploratory','exploratory','matched_population_D_control')
HORIZONS = (.5,1.,1.5,2.)
GROUPS = ('all','stationary','ambiguous','moving')
STRATA = ('all','no_radar','radar_present','radar_present_nonclip','radar_present_velocity_clip')
AUC_STRATA = STRATA[2:]
SPEED_EDGES = np.asarray([0.,.1,.5,1.,2.,5.,10.,np.inf])
RULES = dict(schema=SCHEMA,radar_extractor_sha256=RADAR_SOURCE_SHA,motion_extractor_sha256=MOTION_SOURCE_SHA,
    fixed_motion_analysis_sha256=MOTION_ANALYSIS_SHA,pure_math_source_sha256=MATH_SOURCE_SHA,
    pure_math_reuse=['weighted_auc','continuous'],
    support=dict(samples=512,scenes=256,source_points=1740053,object_horizon_rows=43189,
                 objects_by_horizon=[11254,10967,10636,10332],valid_points_by_horizon=[1709467,1675202,1639983,1608197]),
    horizons_seconds=list(HORIZONS),population='current owner < 0 AND original future valid',
    radar_mapping='XYZ C-order: x=source_flat_index//3200, y=(source_flat_index//16)%200; radar[0,:,y,x]',
    no_nearest_return_or_neighborhood=True,velocity='sum_h(h*D_h)/sum_h(h*h), nominal h only',
    features=[dict(name=n,direction=d,role=r) for n,d,r in zip(FEATURES,DIRECTIONS,ROLES)],
    definitions=dict(radar_mean_speed_mps='20*c6 (not norm of mean velocity)',
        radar_relative_zero_margin_mps='norm(20*[c4,c5])-norm(vD_xy-20*[c4,c5])',
        radar_velocity_residual_mps='norm(vD_xy-20*[c4,c5])',radar_log_count_normalized='c1',
        radar_mean_lag_seconds='0.5*c7',lsq_speed_xy_mps='norm(vD_xy)'),
    missing=dict(radar_features='undefined when c0=0; no zero imputation in scoring',
                 physical_support='retained, with own full-denominator benefit/cost contribution'),
    strata=list(STRATA),auc_strata=list(AUC_STRATA),
    velocity_clip='presence AND (abs(c4)==1 OR abs(c5)==1 OR c6==1)',
    other_saturation='c1==1 count and c7==1 age are separately counted, never included in velocity_clip',
    clipped_scores='computed from clipped summaries, not exact latent sensor velocities; no removal from physical support',
    auc=dict(positive='benefit_xy>0',negative='benefit_xy<0',zero='excluded only from binary AUC, always counted',
             score='fixed direction times feature',weight='1/n_ALL_original_valid_object_points',ties='0.5',
             undefined='null when either weighted class absent',fit=False,direction_selection=False),
    speed_bins=dict(edges_mps=[0.,.1,.5,1.,2.,5.,10.,None],final_edge='positive infinity',
                    convention='left closed, right open',same_bins_in_every_radar_stratum=True),
    continuous=dict(benefit='norm(gt_xy)-norm(D_xy-gt_xy)',
        conditional='sum(w*benefit)/sum(w) in named subset',
        full_contribution='sum(w*benefit)/ALL original object count of that horizon/GT group',
        positive_gain='same full denominator, positive benefit only',negative_cost='same full denominator, -negative benefit only',
        empty_subset='conditional null, contribution zero if original object denominator exists'),
    scene_summary='all 256 scene AUCs, null retained; equal-defined-scene median/q25/q75/IQR, not a confidence interval',
    repeated_z='same (sample,x,y) is one shared radar cell; retain original point weights and report distinct cell counts',
    physical_reference='all four arms / four horizons / four groups XY and XYZ full-object EPE copied from hash-bound prior analysis, no new physical evaluation',
    restrictions=dict(no_fit=True,no_threshold_search=True,no_gate=True,no_bootstrap=True,no_development=True,
                      no_model_forward=True,source_clipping_not_quality_filter=True),
    limitations=['train512 was seen by D; no unseen-scene generalization',
        'D already indirectly consumed radar; this is not independent new sensor information',
        'five-sweep/five-sensor aggregates have no raw LOS or covariance and no object-motion position compensation',
        'absence is missing measurement, not stationary evidence; clipping is censored magnitude',
        'coarse speed stratification does not establish causal information or remove all amplitude confounding',
        'sparse box-derived material-point support is not measured full-space flow',
        'AUC for tiny gains and conditional subset gains do not establish routing or occupancy success'])
RULES_PATH = HERE/'radar_evidence_train_analysis_rules_v1.json'


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def old_math():
    path=HERE/'analyze_motion_evidence_train_v1.py'
    require(sha(path)==MATH_SOURCE_SHA and sha(HERE/'motion_evidence_discrimination_analysis_rules_v1.json')==MATH_RULES_SHA,
            'Frozen pure-math dependency changed')
    spec=importlib.util.spec_from_file_location('frozen_motion_evidence_math',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def identity(row):return {k:row[k] for k in ('sample_token','scene_token','official_index','split')}


def artifact(root,relative):
    rel=Path(relative);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe artifact path')
    result=(root/rel).resolve();require(root.resolve() in result.parents,'Artifact escapes run root')
    return result


def ledger(root,schema,status):
    require(not (root/'failed.json').exists(),'Failed extraction rejected')
    done=read(root/'complete.json')
    require(done['schema']==schema and done['status']==status and done['samples']==512 and
            set(done['files_sha256'])=={'manifest.json','index.json','summary.json'},'Incomplete full extraction')
    for name,digest in done['files_sha256'].items():require(sha(root/name)==digest,'Extraction ledger changed: '+name)
    return done,read(root/'manifest.json'),read(root/'index.json'),read(root/'summary.json')


def authenticate(a):
    require(read(RULES_PATH)==RULES,'Prospective radar analysis rules changed')
    require(sha(HERE/'extract_radar_evidence_train_v1.py')==RADAR_SOURCE_SHA and
            sha(HERE/'extract_motion_evidence_train_v1.py')==MOTION_SOURCE_SHA,'Frozen extractor source changed')
    require(sha(a.train_index)==EXPECTED['index'] and
            sha(a.train_index.parent/'complete.json')==EXPECTED['complete'],'Wrong original native train index/complete')
    native=read(a.train_index);native_done=read(a.train_index.parent/'complete.json')
    require(native['schema']=='m0-native-state-cache-v1' and native['status']=='COMPLETE' and
            native['split']=='train' and native_done['index_sha256']==EXPECTED['index'] and
            len(native['records'])==native_done['samples']==512,'Original cache metadata incomplete')
    require(sha(a.motion_analysis)==MOTION_ANALYSIS_SHA,'Wrong completed prior motion analysis')
    prior=read(a.motion_analysis)
    require(prior['schema']=='motion-evidence-discrimination-analysis-v1' and
            prior['status']=='COMPLETE_CPU_FIXED_TRAIN_EVIDENCE_ANALYSIS' and
            (prior['samples'],prior['scenes'],prior['source_points'],prior['object_horizon_rows'])==(512,256,1740053,43189) and
            prior['sources']['source_sha256']==MATH_SOURCE_SHA and prior['sources']['rules_sha256']==MATH_RULES_SHA and
            prior['sources']['native_train_index_sha256']==EXPECTED['index'],'Prior motion analysis binding differs')
    rd,rm,ri,rs=ledger(a.radar_run,RADAR_SCHEMA,'COMPLETE_CPU_RADAR_TRAIN_EXPORT')
    md,mm,mi,ms=ledger(a.motion_run,MOTION_SCHEMA,'COMPLETE_FIXED_D_TRAIN_FEATURES')
    require(rd['source_sha256']==rm['source_sha256']==RADAR_SOURCE_SHA and
            rm['schema']==ri['schema']==rs['schema']==RADAR_SCHEMA and
            rm['original_contract_sha256']==EXPECTED and
            rm['shape']==[1,8,200,200] and rm['layout']=='B,C,Y,X' and rm['frame']=='current LIDAR_TOP' and
            rm['channel_names']==['presence','clipped_log_count','normalized_mean_z','normalized_mean_RCS',
                'normalized_mean_vx_comp_R','normalized_mean_vy_comp_R','normalized_mean_speed_R','normalized_mean_lag'] and
            rm['cell_width_m']==.512 and rm['split']=='train' and rm['samples']==rs['samples']==512 and
            rm['scenes']==rs['scenes']==256 and all(rm[k] is False for k in ('inference','target_files_opened','model_constructed')),
            'Radar manifest source/layout differs')
    require(rs['status']==rd['status'] and rs['optimizer_updates']==0 and
            rs['model_forward'] is False and rs['CUDA_initialized'] is False and
            rs['all_input_files_and_radar_digests_verified'] is True and rs['all_saved_arrays_roundtrip_exact'] is True,
            'Radar extraction did not complete its no-model contract')
    require(md['source_sha256']==MOTION_SOURCE_SHA and md['optimizer_updates']==0 and md['scenes']==256 and
            md['protocol_sha256']==prior['sources']['protocol_sha256'] and
            sha(a.motion_run/'complete.json')==prior['sources']['extraction_complete_sha256'] and
            md['files_sha256']==prior['sources']['extraction_files_sha256'] and
            mm['schema']==mi['schema']==ms['schema']==MOTION_SCHEMA and
            mm['sources_sha256']==prior['sources']['dependency_sources_sha256'] and
            mm['train_cache']['index_sha256']==EXPECTED['index'] and
            ms['D_state_unchanged'] is True and ms['all_saved_arrays_roundtrip_exact'] is True and
            ms['source_points']==mi['source_points']==1740053 and ms['object_horizon_rows']==43189,
            'Motion extraction no longer matches prior authenticated analysis')
    nr=native['records'];rr=ri['records'];mr=mi['records']
    require(len(rr)==len(mr)==512,'Incomplete paired collection')
    scenes=sorted({r['scene_token'] for r in nr})
    require(len(scenes)==256 and len({r['sample_token'] for r in nr})==512 and prior['scene_tokens']==scenes,
            'Scene/sample set differs')
    for root,rows in ((a.radar_run,rr),(a.motion_run,mr)):
        require({p.relative_to(root).as_posix() for p in (root/'samples').glob('*.npz')}=={r['file'] for r in rows},
                'Complete NPZ file set differs')
    for i,(n,r,m) in enumerate(zip(nr,rr,mr)):
        expected=identity(n)
        require(expected['split']=='train' and r['ordinal']==m['ordinal']==i and r['identity']==m['identity']==expected and
                r['file']==m['file']=='samples/%04d_%s.npz'%(i,n['sample_token']) and
                r['input_sha256']==m['input_sha256']==n['files']['inputs']['sha256'] and
                r['native_radar_tensor_sha256']==n['native_radar_tensor_sha256'],'Pair identity/input/radar digest differs')
    require(sum(r['occupied_xy_cells'] for r in rr)==rs['occupied_xy_cells'],'Radar occupied cell total differs')
    sources=dict(analysis_source_sha256=sha(__file__),rules_sha256=sha(RULES_PATH),
        radar_complete_sha256=sha(a.radar_run/'complete.json'),radar_files_sha256=rd['files_sha256'],radar_source_sha256=RADAR_SOURCE_SHA,
        motion_complete_sha256=sha(a.motion_run/'complete.json'),motion_files_sha256=md['files_sha256'],motion_source_sha256=MOTION_SOURCE_SHA,
        prior_motion_analysis_sha256=MOTION_ANALYSIS_SHA,pure_math_source_sha256=MATH_SOURCE_SHA,
        pure_math_rules_sha256=MATH_RULES_SHA,native_train_index_sha256=EXPECTED['index'],
        native_train_complete_sha256=EXPECTED['complete'],selection_sha256=EXPECTED['selection'])
    return rr,mr,nr,scenes,prior,sources


def radar_digest(array):
    h=hashlib.sha256();h.update(b'torch.float32');h.update(json.dumps(list(array.shape)).encode())
    h.update(np.ascontiguousarray(array).tobytes());return h.hexdigest()


def load_npz(root,row):
    path=artifact(root,row['file'])
    require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'NPZ SHA/bytes changed')
    with np.load(path,allow_pickle=False) as z:values={k:z[k] for k in z.files}
    require(values['identity_json'].shape==() and values['identity_json'].dtype.kind=='U' and
            json.loads(values['identity_json'].item())==row['identity'],'NPZ embedded identity differs')
    return values


def gather_radar(radar,indices):
    """Input-only exact cell lookup, never a nearest-return operation."""
    require(radar.shape==(1,8,200,200) and radar.dtype==np.float32,'Radar shape/dtype differs')
    require(indices.ndim==1 and indices.dtype==np.int64 and
            np.all((indices>=0)&(indices<640000)),'Invalid source grid index')
    x=indices//3200;y=(indices//16)%200;cells=y*200+x
    return radar[0].reshape(8,-1)[:,cells].T.astype(np.float64),cells


def features_at_sources(channels,velocity):
    """Missing radar features are NaN internally and never passed to an AUC."""
    require(channels.ndim==2 and channels.shape[1]==8 and velocity.shape==(len(channels),3) and
            np.isfinite(channels).all() and np.isfinite(velocity).all(),'Invalid feature inputs')
    present=channels[:,0]>0;vR=20.*channels[:,4:6];vD=velocity[:,:2]
    residual=np.linalg.norm(vD-vR,axis=1)
    mean_norm=np.linalg.norm(vR,axis=1);speed=np.linalg.norm(vD,axis=1)
    f=np.stack([20.*channels[:,6],mean_norm-residual,residual,channels[:,1],.5*channels[:,7],speed],axis=1)
    f[~present,:5]=np.nan
    clip=present&((np.abs(channels[:,4])==1)|(np.abs(channels[:,5])==1)|(channels[:,6]==1))
    require(np.all(np.abs(f[present,1])<=speed[present]+1e-10),'Relative-zero triangle bound violated')
    return f,present,clip,present&(channels[:,1]==1),present&(channels[:,7]==1)


def collect(a,rr,mr,nr,scenes,prior):
    scene_ids={s:i for i,s in enumerate(scenes)}
    keys=('features','benefit','weight','scene','group','object','cell','present','clip','count_sat','age_sat')
    parts=[{k:[] for k in keys} for _ in HORIZONS]
    object_counts=np.zeros((4,3),dtype=np.int64);valid_points=np.zeros(4,dtype=np.int64)
    original_points=0;object_offset=0;all_source_cells=[];occupied_xy=0
    labels=('source_flat_indices','target_displacement_m','valid','object_index','instance_tokens',
            'object_speed_group','object_future_valid','dt_future_seconds')
    for ordinal,(r,m,n) in enumerate(zip(rr,mr,nr)):
        rv=load_npz(a.radar_run,r);v=load_npz(a.motion_run,m)
        require(set(rv)=={'identity_json','radar_bev'} and set(v)==set(m['arrays']), 'NPZ keys differ')
        radar=rv['radar_bev']
        require(radar.shape==(1,8,200,200) and radar.dtype==np.float32 and np.isfinite(radar).all() and
                np.isin(radar[:,0],[0,1]).all() and np.max(np.abs(radar))<=1 and
                np.all(radar[:,[1,6,7]]>=0),'Radar encoding outside contract')
        require(radar_digest(radar)==r['native_radar_tensor_sha256']==n['native_radar_tensor_sha256'],
                'Original native radar typed digest differs')
        absent=radar[0,0]==0
        require(np.all(radar[0,:,absent]==0) and int((~absent).sum())==r['occupied_xy_cells'],'Presence/empty values differ')
        occupied_xy+=int((~absent).sum())
        for name,x in v.items():require(m['arrays'][name]==dict(shape=list(x.shape),dtype=str(x.dtype)),'Motion descriptor differs')
        require(set(v)==set(labels)|{'identity_json','owner','owner_original_indices','observable_features','D_displacement_m','CV_velocity_mps'},
                'Motion NPZ field set differs')
        idx=v['source_flat_indices'];obj=v['object_index'];k=len(v['instance_tokens']);size=len(idx)
        require(idx.dtype==obj.dtype==np.int64 and idx.shape==obj.shape==(size,) and
                (size==0 or (idx[0]>=0 and idx[-1]<640000 and np.all(np.diff(idx)>0))) and
                np.array_equal(np.unique(obj),np.arange(k)),'Original sparse object/index support differs')
        require(v['valid'].shape==(4,size) and v['valid'].dtype==np.bool_ and
                v['object_future_valid'].shape==(4,k) and v['object_future_valid'].dtype==np.bool_ and
                np.array_equal(v['valid'],v['object_future_valid'][:,obj]) and
                v['object_speed_group'].shape==(4,k) and np.isin(v['object_speed_group'],[-1,0,1,2]).all() and
                np.array_equal(v['object_speed_group']>=0,v['object_future_valid']),'Future validity/group support differs')
        d=v['D_displacement_m'];target=v['target_displacement_m']
        require(d.dtype==target.dtype==np.float32 and d.shape==target.shape==(4,size,3) and
                np.isfinite(d).all() and np.isfinite(target).all() and np.all(target[~v['valid']]==0),'Motion/target arrays differ')
        require(v['owner'].shape==v['owner_original_indices'].shape==(size,) and
                v['owner'].dtype==v['owner_original_indices'].dtype==np.int64 and
                np.all(v['owner']>=-1) and np.all(v['owner_original_indices']>=-1) and
                np.array_equal(v['owner']<0,v['owner_original_indices']<0),'Original owner map differs')
        channels,cells=gather_radar(radar,idx)
        cells=ordinal*40000+cells;all_source_cells.append(cells)
        h=np.asarray(HORIZONS)[:,None,None];d=d.astype(np.float64)
        velocity=np.sum(h*d,axis=0)/np.sum(h*h)
        f,present,clip,count_sat,age_sat=features_at_sources(channels,velocity)
        require(v['observable_features'].shape==(size,7) and
                np.allclose(f[:,5],v['observable_features'][:,0],rtol=1e-13,atol=1e-14),'Original D magnitude differs')
        counts=np.bincount(obj,minlength=k);scene=scene_ids[r['identity']['scene_token']]
        require(size==m['source_points'],'Source point count changed')
        original_points+=size;valid_points+=v['valid'].sum(axis=1)
        for hi in range(4):
            future=v['object_future_valid'][hi]
            for group in range(3):object_counts[hi,group]+=int(np.sum(future&(v['object_speed_group'][hi]==group)))
            take=v['valid'][hi]&(v['owner']<0)
            y=target[hi].astype(np.float64)
            benefit=np.linalg.norm(y[:,:2],axis=1)-np.linalg.norm((d[hi]-y)[:,:2],axis=1)
            e=parts[hi]
            values=dict(features=f[take],benefit=benefit[take],weight=1./counts[obj[take]],
                scene=np.full(int(take.sum()),scene,dtype=np.int32),group=v['object_speed_group'][hi,obj[take]],
                object=obj[take]+object_offset,cell=cells[take],present=present[take],clip=clip[take],
                count_sat=count_sat[take],age_sat=age_sat[take])
            for key,value in values.items():e[key].append(value)
        object_offset+=k
    require(original_points==1740053 and valid_points.tolist()==RULES['support']['valid_points_by_horizon'] and
            object_counts.sum(axis=1).tolist()==RULES['support']['objects_by_horizon'] and
            int(object_counts.sum())==43189,'Complete original support changed')
    arrays=[{k:np.concatenate(v,axis=0) for k,v in e.items()} for e in parts]
    return arrays,object_counts,dict(source_points=original_points,
        distinct_sample_XY_at_all_original_source_points=int(len(np.unique(np.concatenate(all_source_cells)))),
        all_grid_occupied_sample_XY_cells=occupied_xy,valid_points_by_horizon=valid_points.tolist())


def strata_masks(e):
    return dict(all=np.ones(len(e['benefit']),dtype=bool),no_radar=~e['present'],radar_present=e['present'],
        radar_present_nonclip=e['present']&~e['clip'],radar_present_velocity_clip=e['present']&e['clip'])


def group_mask(e,name):return np.ones(len(e['benefit']),dtype=bool) if name=='all' else e['group']==GROUPS.index(name)-1


def support(helper,e,mask,original_objects):
    value=helper.continuous(e,mask,original_objects)
    value.update(distinct_sample_XY_cells=int(len(np.unique(e['cell'][mask]))),
        scenes_with_points=int(len(np.unique(e['scene'][mask]))),
        count_saturated_points=int(e['count_sat'][mask].sum()),age_saturated_points=int(e['age_sat'][mask].sum()),
        velocity_clipped_points=int(e['clip'][mask].sum()),
        count_saturated_distinct_sample_XY_cells=int(len(np.unique(e['cell'][mask&e['count_sat']]))),
        age_saturated_distinct_sample_XY_cells=int(len(np.unique(e['cell'][mask&e['age_sat']]))))
    return value


def feature_auc(helper,e,mask,scenes=None):
    require(np.all(e['present'][mask]),'Radar scoring cannot include missing cells')
    result=[]
    for fi,(name,direction,role) in enumerate(zip(FEATURES,DIRECTIONS,ROLES)):
        score=direction*e['features'][:,fi]
        item=dict(feature=name,direction=direction,role=role,zero_benefit_points=int((e['benefit'][mask]==0).sum()),
                  **helper.weighted_auc(score[mask],e['benefit'][mask],e['weight'][mask]))
        if scenes is not None:
            rows=[]
            for si,scene in enumerate(scenes):
                sm=mask&(e['scene']==si)
                rows.append(dict(scene_token=scene,zero_benefit_points=int((e['benefit'][sm]==0).sum()),
                    **helper.weighted_auc(score[sm],e['benefit'][sm],e['weight'][sm])))
            defined=[r['auc'] for r in rows if r['auc'] is not None]
            q=np.quantile(defined,[.25,.5,.75]).tolist() if defined else [None,None,None]
            item['per_scene']=rows
            item['scene_distribution']=dict(defined_scenes=len(defined),undefined_scenes=len(scenes)-len(defined),
                q25=q[0],median=q[1],q75=q[2],IQR=q[2]-q[0] if defined else None,is_confidence_interval=False)
        result.append(item)
    return result


def close(a,b,message):
    if a is None or b is None:require(a is b,'Null support mismatch: '+message)
    else:require(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),'Aggregation mismatch: '+message)


def summarize(helper,arrays,object_counts,scenes,prior):
    result=[]
    for hi,h in enumerate(HORIZONS):
        e=arrays[hi];masks=strata_masks(e)
        counts={name:int(object_counts[hi].sum()) if name=='all' else int(object_counts[hi,GROUPS.index(name)-1]) for name in GROUPS}
        pe=next(r for r in prior['evidence'] if r['horizon_seconds']==h)
        item=dict(horizon_seconds=h,strata={},radar_auc={},
                  prior_all_uncovered_lsq_control=next(r for r in pe['features'] if r['feature']=='lsq_speed_xy_mps'))
        for stratum in STRATA:
            item['strata'][stratum]={name:support(helper,e,masks[stratum]&group_mask(e,name),counts[name]) for name in GROUPS}
        for name in GROUPS:
            p=next(r for r in prior['physical'] if r['horizon_seconds']==h and r['group']==name)
            complete=item['strata']['all'][name]
            require(counts[name]==p['original_objects'] and complete['points']==p['uncovered_points'],'Original horizon/group denominator changed')
            original=p['uncovered_D_versus_zero_benefit_xy']
            for key in ('weight_sum','conditional_original_weight_mean_benefit_xy_m',
                        'FULL_denominator_benefit_contribution_xy_m','FULL_denominator_positive_gain_xy_m','FULL_denominator_negative_cost_xy_m'):
                close(complete[key],original[key],'prior all-uncovered '+key)
            for key in ('points','nonempty_objects','positive_points','negative_points','zero_points'):
                require(complete[key]==original[key],'Original point/object count differs: '+key)
            for key in ('points','weight_sum','FULL_denominator_benefit_contribution_xy_m',
                        'FULL_denominator_positive_gain_xy_m','FULL_denominator_negative_cost_xy_m'):
                close(complete[key],item['strata']['no_radar'][name][key]+item['strata']['radar_present'][name][key],'presence decomposition')
                close(item['strata']['radar_present'][name][key],item['strata']['radar_present_nonclip'][name][key]+
                      item['strata']['radar_present_velocity_clip'][name][key],'clip decomposition')
        bins=np.searchsorted(SPEED_EDGES,e['features'][:,5],side='right')-1
        require(np.isin(bins,np.arange(7)).all(),'D speed fell outside fixed bins')
        for stratum in AUC_STRATA:
            mask=masks[stratum];value=dict(features=feature_auc(helper,e,mask,scenes),speed_bins=[])
            contributions=[]
            for bi in range(7):
                bm=mask&(bins==bi)
                by_group={name:support(helper,e,bm&group_mask(e,name),counts[name]) for name in GROUPS}
                contributions.append(by_group['all']['FULL_denominator_benefit_contribution_xy_m'])
                value['speed_bins'].append(dict(bin_index=bi,left_inclusive_mps=float(SPEED_EDGES[bi]),
                    right_exclusive_mps=None if bi==6 else float(SPEED_EDGES[bi+1]),right_unbounded=(bi==6),
                    by_group=by_group,features=feature_auc(helper,e,bm)))
            close(sum(contributions),item['strata'][stratum]['all']['FULL_denominator_benefit_contribution_xy_m'],'speed bins preserve original denominator')
            item['radar_auc'][stratum]=value
        result.append(item)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('radar-run','motion-run','motion-analysis','train-index','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.out.exists(),'New output required; never overwrite')
    helper=old_math();rr,mr,nr,scenes,prior,sources=authenticate(a)
    arrays,counts,coverage=collect(a,rr,mr,nr,scenes,prior)
    horizons=summarize(helper,arrays,counts,scenes,prior)
    require('torch' not in sys.modules,'Pure CPU analysis must not import Torch')
    result=dict(schema=SCHEMA,status='COMPLETE_CPU_RADAR_TRAIN_EVIDENCE_ANALYSIS',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,rules=RULES,
        samples=512,scenes=256,source_points=1740053,object_horizon_rows=43189,scene_tokens=scenes,
        sampling_coverage=coverage,horizons=horizons,
        unchanged_full_physical_reference=dict(source_sha256=MOTION_ANALYSIS_SHA,physical=prior['physical'],
            recomputed_by_this_analysis=False,description='original complete four-arm XY/XYZ EPE; hash-bound copy'),
        verification=dict(all_radar_motion_NPZ_SHA_identity_descriptors_checked=True,
            native_typed_radar_digest_recomputed_from_full_arrays=True,same_original_input_hash_per_pair=True,
            original_four_horizon_support_and_uncovered_benefit_reproduced=True,
            raw_checkpoint_or_original_label_tensors_reloaded=False,model_forward=False,optimizer_updates=0),
        runtime=dict(python=sys.version,numpy=np.__version__,Torch_imported=False,model_forward=False,optimizer_updates=0))
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(out=str(a.out),sha256=sha(a.out),samples=512,object_horizon_rows=43189)))


if __name__=='__main__':main()
