"""CPU-only prespecified analysis; no images/model/fit or real-data access in QA.

Reuse frozen radar authentication and motion weighted_auc/continuous. GT joins
occur only here, after the separate input-only correspondence extraction.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = 'history-camera-evidence-train-analysis-v1'
CAMERA_SCHEMA = 'history-camera-evidence-train-v1'
INPUT_SCHEMA = 'history-camera-inputs-train-v1'
RADAR_ANALYZER_SHA = '75df9873f97ca442e925d260876c124a9115b626edf35af2eb7ab1315f7d80b3'
RADAR_RULES_SHA = 'a518b7a0fecaa4e5fb528c20edc69e5f782ae554ff73fcf5ce5e5907dab22d8b'
INPUT_COMPLETE_SHA = '040d62242a541b9685f3f39f1534b3663b0e1142b4cd7d9737cdbfd15d94314d'
RULES_SHA = '166c9ea6ae2054983e81ddb6ef2513c8743875e787282aa44e99ee23de290944'
RULES_PATH = HERE/'history_camera_evidence_train_analysis_rules_v1.json'
FEATURES = ('score_true', 'score_broken', 'lsq_speed_xy_mps', 'current_zero_texture_std', 'past_hypothesis_pixel_separation')
REASONS = ('valid', 'no_current_zero_view', 'missing_previous_keyframe', 'hypothesis_projection_invalid', 'low_current_zero_texture')
GROUPS = ('all', 'stationary', 'ambiguous', 'moving')
STRATA = ('all', 'radar_present', 'no_radar')
HORIZONS = (.5, 1., 1.5, 2.)
EDGES = np.asarray([0., .1, .5, 1., 2., 5., 10., np.inf])


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def module(name, digest):
    path = HERE/(name+'.py'); require(sha(path) == digest, 'Dependency changed: '+name)
    spec = importlib.util.spec_from_file_location('_history_analysis_'+name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def dependencies():
    require(sha(RULES_PATH) == RULES_SHA, 'Frozen analysis rules changed')
    rules = read(RULES_PATH)
    require(rules['schema'] == SCHEMA and rules['features'] == list(FEATURES) and rules['directions'] == [1]*5 and
            rules['reason_codes'] == {str(i):n for i,n in enumerate(REASONS)} and
            rules['speed_edges_mps'] == [0., .1, .5, 1., 2., 5., 10., None], 'Prespecified analysis rules differ')
    require(sha(HERE/'radar_evidence_train_analysis_rules_v1.json') == RADAR_RULES_SHA, 'Radar dependency rules changed')
    radar = module('analyze_radar_evidence_train_v1', RADAR_ANALYZER_SHA)
    return rules, radar, radar.old_math()


def pose(ego, calibrated):
    """Independent column SE3 reconstruction from the preserved raw wxyz poses."""
    def matrix(record):
        q = np.asarray(record['rotation'], dtype=np.float64); require(q.shape == (4,) and np.isfinite(q).all(), 'Quaternion')
        require(abs(np.linalg.norm(q)-1.) < 1e-3, 'Quaternion normalization'); w,x,y,z = q/np.linalg.norm(q)
        t = np.eye(4); t[:3,:3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
        t[:3,3] = record['translation']; require(np.isfinite(t).all(), 'Nonfinite pose'); return t
    return matrix(ego) @ matrix(calibrated)


def history_inputs(a, core, nr):
    root = a.history_inputs; require(sha(root/'complete.json') == INPUT_COMPLETE_SHA, 'Wrong history metadata complete')
    done = read(root/'complete.json'); require(done['schema'] == INPUT_SCHEMA and done['samples'] == 512 and
        done['status'] == 'COMPLETE_AUTHENTICATED_HISTORY_CAMERA_INPUTS' and not (root/'failed.json').exists() and
        set(done['files_sha256']) == {'manifest.json','records.json','summary.json'}, 'Incomplete history metadata')
    for name,digest in done['files_sha256'].items(): require(sha(root/name) == digest, 'History metadata ledger changed')
    manifest, summary, rows = read(root/'manifest.json'), read(root/'summary.json'), read(root/'records.json')['records']
    require(manifest['schema'] == summary['schema'] == INPUT_SCHEMA and manifest['source_sha256'] == done['source_sha256'] and
        summary['status'] == done['status'] and (summary['samples'],summary['scenes'],summary['image_occurrences'],summary['unique_images']) == (512,256,6144,6144) and
        manifest['camera_order'] == list(core.CAMERA_ORDER) and summary['header_and_all_bytes_checked'] is True and
        all(summary[k] is False for k in ('future_images_read','annotations_read','model_forward','source_images_transformed')), 'History input contract differs')
    require(sha(HERE/'extract_history_camera_inputs_train_v1.py') == manifest['source_sha256'] and
        sha(HERE/'build_full_motion_targets_v1.py') == manifest['reader_sha256'], 'History metadata producer/reader source changed')
    images = {r['file']:r for r in manifest['images']}; require(len(images) == 6144 and len(rows) == len(nr), 'Image or sample ledger incomplete')
    used = []; timestamps = []
    for i,(r,n) in enumerate(zip(rows,nr)):
        require(r['ordinal'] == i and r['identity'] == {k:n[k] for k in ('sample_token','scene_token','official_index','split')} and
            r['lidar_sample_data']['token'] == n['lidar_token'] and r['lidar_sample_data']['sample_token'] == n['sample_token'], 'History/native identity differs')
        g = pose(r['lidar_ego_pose'],r['lidar_calibrated_sensor']); t0 = r['t0_lidar_us']; avail = r['input_availability_us']
        require(np.allclose(g,r['lidar_to_global'],rtol=0,atol=1e-9) and t0 == r['lidar_sample_data']['timestamp'] and avail >= t0, 'LiDAR geometry/time')
        require([c['channel'] for c in r['cameras']] == list(core.CAMERA_ORDER), 'Camera order differs')
        per_camera = []
        for c in r['cameras']:
            ts = []
            for when in ('current','past'):
                f = c[when]; sd,cs,ep = f['sample_data'],f['calibrated_sensor'],f['ego_pose']; image = f['image']
                require(f['channel'] == c['channel'] and sd['is_key_frame'] is True and sd['sample_token'] ==
                    (n['sample_token'] if when == 'current' else r['previous_sample_token']) and sd['timestamp'] <= avail and
                    sd['ego_pose_token'] == ep['token'] and sd['calibrated_sensor_token'] == cs['token'], 'Camera sensor identity/time')
                gc = pose(ep,cs); require(np.allclose(gc,f['camera_to_global'],rtol=0,atol=1e-9) and
                    np.allclose(np.linalg.inv(gc)@g,f['R_to_camera'],rtol=0,atol=1e-9) and
                    np.array_equal(f['K'],cs['camera_intrinsic']), 'Camera sensor matrix/K')
                require(image == images[sd['filename']] and image['file'] == sd['filename'] and
                    image['size_wh'] == [sd['width'],sd['height']] and image['mode'] == 'RGB' and
                    len(image['sha256']) == 64 and image['bytes'] > 0, 'Camera image SHA ledger')
                ts.append(sd['timestamp']); used.append(image['file'])
            require(ts[1] < min(ts[0],t0), 'Previous image not observed history'); per_camera.append(ts)
        timestamps.append(np.asarray(per_camera,dtype=np.int64))
    require(len(used) == 6144 and set(used) == set(images), 'Image occurrence support differs')
    return done,manifest,rows,timestamps


def authenticate(a, rules, radar):
    rr,mr,nr,scenes,prior,sources = radar.authenticate(a)
    require(rules['camera_extractor_sha256'] != 'PENDING_ROOT_FREEZE', 'Collector source not frozen')
    require(sha(HERE/'extract_history_camera_evidence_train_v1.py') == rules['camera_extractor_sha256'], 'Camera collector source changed')
    core = module('history_camera_evidence_v1',rules['camera_module_sha256'])
    proto = read(a.camera_protocol); require(sha(a.camera_protocol) == rules['camera_protocol_sha256'] and
        proto['schema'] == CAMERA_SCHEMA and proto['status'] == 'FROZEN', 'Camera protocol changed')
    constants = dict(camera_order=list(core.CAMERA_ORDER),horizons_seconds=list(core.HORIZONS_SECONDS),patch_radius=core.PATCH_RADIUS,
        patch_size=core.PATCH_SIZE,zncc_eps=core.ZNCC_EPS,texture_std_min=core.TEXTURE_STD_MIN,reason_codes={str(k):v for k,v in core.REASON_CODES.items()},axes=core.AXES)
    require(constants == proto['module_constants'], 'Loaded core constants differ')
    done,manifest,index,summary = radar.ledger(a.camera_run,CAMERA_SCHEMA,'COMPLETE_HISTORY_CAMERA_TRAIN_EVIDENCE')
    hd,hm,hr,times = history_inputs(a,core,nr)
    require(done['source_sha256'] == manifest['source_sha256'] == rules['camera_extractor_sha256'] and
        done['extraction_protocol_sha256'] == manifest['extraction_protocol_sha256'] == sha(a.camera_protocol) and
        done['module_sha256'] == manifest['module_sha256'] == rules['camera_module_sha256'] == proto['module_sha256'] and
        manifest['module_constants'] == proto['module_constants'] and manifest['history_inputs_complete_sha256'] == INPUT_COMPLETE_SHA and
        manifest['history_inputs_files_sha256'] == hd['files_sha256'] and manifest['motion_complete_sha256'] == sources['motion_complete_sha256'] and
        manifest['selection_sha256'] == hm['selection_sha256'] == radar.EXPECTED['selection'] and
        manifest['geometry_complete_sha256'] == proto['geometry_complete_sha256'] and
        manifest['points_R_sha256'] == '91bcc80882e41ffd5234cecce4b517ef4081839c89dbb6a048c4226b723ca6fd' and
        manifest['motion_files_sha256'] == sources['motion_files_sha256'] and manifest['scope'] == proto['scope'], 'Camera source/input/protocol binding differs')
    require(manifest['schema'] == summary['schema'] == index['schema'] == CAMERA_SCHEMA and summary['status'] == done['status'] and
        summary['samples'] == 512 and summary['scenes'] == 256 and summary['optimizer_updates'] == 0 and
        summary['source_points'] == index['source_points'] == 1740053 and summary['images_decoded'] == 6144 and
        summary['all_query_arrays_roundtrip_exact'] is True and summary['images_authenticated_again_before_decode'] is True and
        all(summary[k] is False for k in ('model_forward','future_GT_arrays_opened','threshold_selection','fitting')) and
        len(index['records']) == 512, 'Camera extraction not a full input-only result')
    cr = index['records']; require({p.relative_to(a.camera_run).as_posix() for p in (a.camera_run/'samples').glob('*.npz')} == {r['file'] for r in cr}, 'Camera NPZ file set differs')
    for i,(c,m,n) in enumerate(zip(cr,mr,nr)):
        require(c['ordinal'] == i and c['identity'] == m['identity'] and c['file'] == m['file'] and
            c['input_sha256'] == n['files']['inputs']['sha256'] and c['source_motion_npz_sha256'] == m['sha256'] and c['source_points'] == m['source_points'], 'Camera/motion pair differs')
    totals = {str(i):sum(r['reason_counts'][str(i)] for r in cr) for i in range(5)}
    require(totals == index['reason_counts'] == summary['reason_counts'] and sum(totals.values()) == 1740053, 'Camera total reason support differs')
    sources.update(analysis_source_sha256=sha(__file__),rules_sha256=sha(RULES_PATH),camera_complete_sha256=sha(a.camera_run/'complete.json'),
        camera_files_sha256=done['files_sha256'],camera_source_sha256=manifest['source_sha256'],camera_module_sha256=manifest['module_sha256'],
        camera_protocol_sha256=sha(a.camera_protocol),history_inputs_complete_sha256=INPUT_COMPLETE_SHA,history_inputs_files_sha256=hd['files_sha256'])
    return rr,mr,nr,cr,scenes,prior,sources,core,times


def camera_arrays(v, row, motion, core, times):
    n = len(motion['source_flat_indices']); shapes = dict(camera_index=(n,),reason_code=(n,),valid=(n,),optical_axis_distance=(n,),timestamps_us=(n,2),
        uv=(n,2,2,2),depth_m=(n,2,2),patch_std=(n,2,2),broken_past_patch_std=(n,2),correlation=(n,2),broken_correlation=(n,2),
        residual=(n,2),broken_residual=(n,2),score=(n,),broken_score=(n,),hypothesis_pixel_separation=(n,2),nominal_lsq_velocity_mps=(n,3))
    require(set(v) == set(row['arrays']) == set(shapes)|{'identity_json','source_flat_indices'} and np.array_equal(v['source_flat_indices'],motion['source_flat_indices']) and
        v['source_flat_indices'].dtype == np.int64, 'Camera source support or fields differ')
    for name,value in v.items():
        require(row['arrays'][name] == dict(shape=list(value.shape),dtype=str(value.dtype)), 'Full NPZ descriptor differs: '+name)
    for name,shape in shapes.items():
        dtype = {'camera_index':'int8','reason_code':'uint8','valid':'bool','timestamps_us':'int64'}.get(name,'float64')
        require(v[name].shape == shape and str(v[name].dtype) == dtype and row['arrays'][name] == dict(shape=list(shape),dtype=dtype), 'Camera array schema: '+name)
    reason, valid, ci = v['reason_code'],v['valid'],v['camera_index']; require(np.isin(reason,np.arange(5)).all() and
        np.array_equal(valid,reason==0) and np.isin(ci,np.arange(-1,6)).all() and np.array_equal(ci<0,reason==1), 'Camera missing reason or view differs')
    require(row['reason_counts'] == {str(i):int((reason==i).sum()) for i in range(5)}, 'Per-sample reason counts differ')
    d = motion['D_displacement_m'].astype(np.float64); h = np.asarray(HORIZONS)
    velocity = np.einsum('h,hnc->nc',h,d)/np.dot(h,h)
    require(v['nominal_lsq_velocity_mps'].tobytes() == velocity.tobytes() and
        np.allclose(np.linalg.norm(velocity[:,:2],axis=1),motion['observable_features'][:,0],rtol=1e-13,atol=1e-14), 'Nominal LSQ does not match original D')
    selected = ci >= 0; require(np.array_equal(v['timestamps_us'][selected],times[ci[selected]]), 'Selected camera timestamps differ')
    require(np.all(v['patch_std'][valid,0,0] >= core.TEXTURE_STD_MIN) and np.all(v['patch_std'][reason==4,0,0] < core.TEXTURE_STD_MIN), 'Fixed texture support differs')
    for name in ('correlation','broken_correlation','residual','broken_residual','score','broken_score'):
        require(np.isfinite(v[name][valid]).all() and np.isnan(v[name][~valid]).all(), 'Invalid score imputation: '+name)
    require(np.array_equal(v['residual'][valid],1-v['correlation'][valid]) and np.array_equal(v['broken_residual'][valid],1-v['broken_correlation'][valid]) and
        np.array_equal(v['score'][valid],v['residual'][valid,0]-v['residual'][valid,1]) and
        np.array_equal(v['broken_score'][valid],v['broken_residual'][valid,0]-v['broken_residual'][valid,1]), 'Residual/score definitions differ')
    # Preserve the old sum-reduction speed/bins; einsum above checks the collector's exact velocity bytes.
    old_velocity = np.sum(h[:,None,None]*d,axis=0)/np.sum(h*h)
    f = np.stack([v['score'],v['broken_score'],np.linalg.norm(old_velocity[:,:2],axis=1),v['patch_std'][:,0,0],v['hypothesis_pixel_separation'][:,1]],axis=1)
    require(np.isfinite(f[valid]).all(), 'Nonfinite common-population features'); return f


def collect(a, radar, rr,mr,nr,cr,scenes,prior,core,times):
    # Frozen collector validates every original motion/radar array and preserves all object denominators.
    arrays,counts,coverage = radar.collect(a,rr,mr,nr,scenes,prior)
    offsets = [0]*4
    for ci,(c,m) in enumerate(zip(cr,mr)):
        v = radar.load_npz(a.camera_run,c); motion = radar.load_npz(a.motion_run,m)
        f = camera_arrays(v,c,motion,core,times[ci])
        for hi,e in enumerate(arrays):
            if 'camera_features' not in e:
                e['camera_features'] = np.empty((len(e['benefit']),5)); e['reason'] = np.empty(len(e['benefit']),dtype=np.uint8)
            take = motion['valid'][hi] & (motion['owner']<0); n = int(take.sum()); sl = slice(offsets[hi],offsets[hi]+n)
            require(np.all(e['scene'][sl] == scenes.index(m['identity']['scene_token'])) and
                np.array_equal(e['features'][sl,5],f[take,2]), 'Camera/old evidence ordering mismatch')
            e['camera_features'][sl] = f[take]; e['reason'][sl] = v['reason_code'][take]; offsets[hi] += n
    require(offsets == [len(e['benefit']) for e in arrays], 'Camera evidence incomplete'); return arrays,counts,coverage


def auc_features(helper,e,mask):
    require(np.all(e['reason'][mask] == 0), 'All scores must share valid population')
    return {name:helper.weighted_auc(e['camera_features'][mask,fi],e['benefit'][mask],e['weight'][mask]) for fi,name in enumerate(FEATURES)}


def difference(a,b): return None if a is None or b is None else a-b


def scene_distribution(helper,e,mask,scenes):
    rows = []
    for si,scene in enumerate(scenes):
        m = mask & (e['scene']==si); values = auc_features(helper,e,m)
        auc = {k:v['auc'] for k,v in values.items()}
        rows.append(dict(scene_token=scene,points=int(m.sum()),positive_points=int((e['benefit'][m]>0).sum()),negative_points=int((e['benefit'][m]<0).sum()),
            auc=auc,paired_true_minus_broken_auc=difference(auc['score_true'],auc['score_broken'])))
    distribution = {}
    for name in list(FEATURES)+['paired_true_minus_broken_auc']:
        vals = [r[name] if name=='paired_true_minus_broken_auc' else r['auc'][name] for r in rows]
        values = [x for x in vals if x is not None]; q = np.quantile(values,[.25,.5,.75]).tolist() if values else [None]*3
        distribution[name] = dict(defined_scenes=len(values),undefined_scenes=len(scenes)-len(values),q25=q[0],median=q[1],q75=q[2],is_confidence_interval=False)
    return dict(per_scene=rows,equal_defined_scene_distribution=distribution,paired_difference_is_significance_test=False)


def summarize(helper,radar,arrays,counts,scenes,prior):
    output = []
    for hi,h in enumerate(HORIZONS):
        e = arrays[hi]; bins = np.searchsorted(EDGES,e['camera_features'][:,2],side='right')-1
        require(np.isin(bins,np.arange(7)).all(), 'Fixed speed bins'); masks = radar.strata_masks(e); item = dict(horizon_seconds=h,strata={})
        for group in GROUPS:
            gm = radar.group_mask(e,group); denom = int(counts[hi].sum() if group=='all' else counts[hi,GROUPS.index(group)-1])
            old = next(p for p in prior['physical'] if p['horizon_seconds']==h and p['group']==group)
            current = helper.continuous(e,gm,denom)
            require(denom == old['original_objects'] and current['points'] == old['uncovered_points'], 'Original physical support differs')
            for key,value in current.items(): radar.close(value,old['uncovered_D_versus_zero_benefit_xy'][key],'Prior full-denominator '+key)
        for stratum in STRATA:
            groups = {}; item['strata'][stratum] = groups
            for group in GROUPS:
                gm = radar.group_mask(e,group); denom = int(counts[hi].sum() if group=='all' else counts[hi,GROUPS.index(group)-1])
                base = masks[stratum]&gm; total = helper.continuous(e,gm,denom); rs = helper.continuous(e,base,denom)
                records = []
                for bi in [-1]+list(range(7)):
                    bm = base if bi==-1 else base&(bins==bi); bstats = helper.continuous(e,bm,denom); by_reason = {}
                    for ri,reason in enumerate(REASONS):
                        sm = bm&(e['reason']==ri); z = helper.continuous(e,sm,denom)
                        for refname,ref in [('all_uncovered_group',total),('this_radar_group',rs),('this_radar_speed_bin',bstats)]:
                            cost = ref['FULL_denominator_negative_cost_xy_m']; gain = ref['FULL_denominator_positive_gain_xy_m']
                            z['negative_cost_fraction_of_'+refname] = z['FULL_denominator_negative_cost_xy_m']/cost if cost else None
                            z['positive_gain_fraction_of_'+refname] = z['FULL_denominator_positive_gain_xy_m']/gain if gain else None
                        by_reason[reason] = z
                    for key in ('points','weight_sum','FULL_denominator_benefit_contribution_xy_m','FULL_denominator_positive_gain_xy_m','FULL_denominator_negative_cost_xy_m'):
                        radar.close(sum(v[key] for v in by_reason.values()),bstats[key],'Missing reasons conserve '+key)
                    valid = bm&(e['reason']==0); features = auc_features(helper,e,valid)
                    records.append(dict(bin_index=None if bi==-1 else bi,all_speeds=bi==-1,support=bstats,by_camera_reason=by_reason,
                        features_same_valid_population=features,paired_true_minus_broken_auc=difference(features['score_true']['auc'],features['score_broken']['auc'])))
                for key in ('points','weight_sum','FULL_denominator_benefit_contribution_xy_m','FULL_denominator_positive_gain_xy_m','FULL_denominator_negative_cost_xy_m'):
                    radar.close(sum(r['support'][key] for r in records[1:]),records[0]['support'][key],'Seven bins conserve '+key)
                groups[group] = dict(original_object_count=denom,bins=records,scenes=scene_distribution(helper,e,base&(e['reason']==0),scenes))
        output.append(item)
    return output


def analytic_qa(helper,radar):
    e = dict(benefit=np.array([2.,-4.,0.,-2.]),weight=np.array([.25,.25,.5,1.]),object=np.array([0,0,0,1]),
        reason=np.array([0,4,0,3]),camera_features=np.array([[2.,-2.,1.,.1,2.],[np.nan,np.nan,1.,.001,np.nan],[0.,0.,0.,.1,0.],[np.nan,np.nan,2.,np.nan,np.nan]]))
    allmask = np.ones(4,dtype=bool); whole = helper.continuous(e,allmask,2)
    require(whole['FULL_denominator_negative_cost_xy_m']==1.5 and whole['FULL_denominator_positive_gain_xy_m']==.25, 'QA full denominator')
    parts = [helper.continuous(e,e['reason']==i,2) for i in range(5)]
    radar.close(sum(p['FULL_denominator_negative_cost_xy_m'] for p in parts),1.5,'QA missing harm retained')
    require(auc_features(helper,e,e['reason']==0)['score_true']['auc'] is None, 'QA one-sided AUC null')
    a=helper.weighted_auc(np.array([1.,1.,0.]),np.array([1.,-1.,-1.]),np.array([1.,2.,1.]))
    require(abs(a['auc']-2/3)<1e-15, 'QA weighted tie credit')
    h=np.asarray(HORIZONS); v=np.array([[2.,-3.,.5]]); d=h[:,None,None]*v
    require(np.array_equal(np.einsum('h,hnc->nc',h,d)/np.dot(h,h),v), 'QA nominal LSQ')
    require(np.array_equal(np.searchsorted(EDGES,np.array([0.,.1,.5,1.,2.,5.,10.]),side='right')-1,np.arange(7)), 'QA bin edges')
    sx=dict(benefit=np.array([1.,-1.]),weight=np.ones(2),reason=np.zeros(2,dtype=np.uint8),scene=np.zeros(2,dtype=np.int32),
        camera_features=np.array([[1.,-1.,1.,.1,1.],[0.,0.,0.,.2,2.]]))
    sd=scene_distribution(helper,sx,np.ones(2,dtype=bool),['synthetic_A','synthetic_empty'])
    require(sd['per_scene'][0]['paired_true_minus_broken_auc']==1. and sd['per_scene'][1]['auc']['score_true'] is None and
        sd['equal_defined_scene_distribution']['score_true']['is_confidence_interval'] is False, 'QA paired scene descriptive/null semantics')
    return dict(status='PASS_SYNTHETIC_ANALYTIC_QA_ONLY',tests=['full-object denominators','all missing costs retained','one-sided AUC null','weighted tie credit','nominal LSQ','fixed seven bin boundaries','paired scene delta and empty scene null'],real_data_opened=False,model_forward=False,optimizer_updates=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('camera-run','camera-protocol','history-inputs','radar-run','motion-run','motion-analysis','train-index'): p.add_argument('--'+key,type=Path)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--qa-only',action='store_true'); a=p.parse_args()
    require(not a.out.exists(),'Fresh output required'); rules,radar,helper=dependencies()
    if a.qa_only: result=analytic_qa(helper,radar)
    else:
        require(all(getattr(a,k.replace('-','_')) is not None for k in ('camera-run','camera-protocol','history-inputs','radar-run','motion-run','motion-analysis','train-index')), 'All input paths required')
        rr,mr,nr,cr,scenes,prior,sources,core,times=authenticate(a,rules,radar)
        arrays,counts,coverage=collect(a,radar,rr,mr,nr,cr,scenes,prior,core,times)
        result=dict(schema=SCHEMA,status='COMPLETE_CPU_HISTORY_CAMERA_TRAIN_EVIDENCE_ANALYSIS',sources=sources,rules=rules,samples=512,scenes=256,
            source_points=1740053,object_horizon_rows=43189,scene_tokens=scenes,sampling_coverage=coverage,horizons=summarize(helper,radar,arrays,counts,scenes,prior),
            unchanged_full_physical_reference=dict(source_sha256=radar.MOTION_ANALYSIS_SHA,physical=prior['physical'],recomputed_by_this_analysis=False),
            verification=dict(original_support_join=True,all_sample_npz_SHA_checked=True,nominal_LSQ_bytes_checked=True,metadata_image_ledger_and_matrices_checked=True,
                original_source_images_reopened=False,original_labels_reloaded=False,all_640000_grid_points_evaluated=False,model_forward=False,optimizer_updates=0))
    require('torch' not in sys.modules,'CPU NumPy-only analysis'); result.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),analysis_source_sha256=sha(__file__),rules_sha256=sha(RULES_PATH))
    with a.out.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(dict(out=str(a.out),sha256=sha(a.out),status=result['status'])))


if __name__=='__main__': main()
