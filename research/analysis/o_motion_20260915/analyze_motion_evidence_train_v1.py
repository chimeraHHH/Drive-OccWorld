"""CPU-only, prespecified analysis of COMPLETE fixed-D train512 extraction.

No model/helper imports, fitting, threshold selection, bootstrap or gate.
The collection must be complete and every original sparse label is checked.
Zero benefit is excluded ONLY from binary AUC; physical support is unchanged.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = 'motion-evidence-discrimination-analysis-v1'
COLLECTOR_SCHEMA = 'motion-evidence-train-extraction-v1'
COLLECTOR_SHA = '3bf1cd4c2cadca9d84d2220371f4eaa046db48cd48cd7ccc6fb25f222c110342'
PROTOCOL_SHA = '7e52515ec303570d6e1e64c50866120af3012bbdcdd76ac2e14b2f3a21f6d969'
SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
TRAIN_INDEX_SHA = '1dc14643df86f85c5ae87b17b83c60cada99f189a025675d150062acb046e7b1'
SPARSE_MANIFEST_SHA = 'cdb11231cbc3213d213e1f3f718fe9d3c90665dccea4154ecf3372ad1ce91efe'
SPARSE_COMPLETE_SHA = '3d14a03cd61761a1ebc26313463dbff06854907c19c49057667a0ffd257e05bd'
D_CHECKPOINT_SHA = '7e2750d56ade7fb36e334b6431e83f51aafa997357744780a507924a696951e3'
D_STATE_SHA = '4abd5bf8aab3fff045d4ea9f4c5d7bc3579f7265f795b4aef87eb5774afadb61'
HORIZONS = (.5, 1., 1.5, 2.)
FEATURES = ('lsq_speed_xy_mps', 'min_horizon_speed_xy_mps',
            'max_horizon_speed_xy_mps', 'non_cv_velocity_rms_xy_mps',
            'relative_non_cv_velocity_rms_xy', 'local_3x3_velocity_rms_xy_mps',
            'local_3x3_predicted_coverage_fraction')
DIRECTIONS = (1, 1, 1, -1, -1, -1, 1)
ARMS = ('D', 'CRN_CV', 'fixed_blend', 'zero')
GROUPS = ('all', 'stationary', 'ambiguous', 'moving')
LABEL_KEYS = ('source_flat_indices', 'target_displacement_m', 'valid', 'object_index',
              'instance_tokens', 'object_speed_group', 'object_future_valid', 'dt_future_seconds')
ARRAY_KEYS = set(LABEL_KEYS) | {'identity_json', 'owner', 'owner_original_indices',
                              'observable_features', 'D_displacement_m', 'CV_velocity_mps'}
SPEED_EDGES = np.asarray([0., .1, .5, 1., 2., 5., 10., np.inf])
RULES = dict(
    schema=SCHEMA, collector_sha256=COLLECTOR_SHA, protocol_sha256=PROTOCOL_SHA,
    support=dict(split='train', samples=512, scenes=256, source_points=1740053,
                 object_horizon_rows=43189, objects_by_horizon=[11254,10967,10636,10332],
                 valid_points_by_horizon=[1709467,1675202,1639983,1608197]),
    horizons_seconds=list(HORIZONS), arms=list(ARMS), groups=list(GROUPS),
    evidence_population='current prediction owner < 0 AND original future label valid',
    benefit_xy_m='norm(gt_xy) - norm(D_xy - gt_xy)',
    auc=dict(positive='benefit > 0', negative='benefit < 0', zero='excluded only from binary AUC; count retained',
             weight='1 / number of ALL original valid points in the same anchor-object-horizon',
             formula='weighted positive-negative ranking probability; tied scores count 0.5',
             undefined='null if either weighted class has no support',
             directions=dict(zip(FEATURES, DIRECTIONS)), direction_selection=False,
             per_scene='same weights and classes, every one of 256 scenes; undefined is null',
             scene_summary='equal-scene median and 25/75 percentiles over defined scene AUCs; not a confidence interval'),
    speed_control=dict(feature=FEATURES[0], bin_edges_mps=[0., .1, .5, 1., 2., 5., 10., None],
                       final_edge='positive infinity', intervals='left closed, right open',
                       refit_bins=False, within_bin_auc_weight='original object weight, not renormalized within bin'),
    continuous=dict(conditional_mean='sum(point_weight * benefit) / sum(point_weight) in named subset',
                    full_contribution='sum(point_weight * benefit) / ALL original object count in named horizon/group',
                    positive_gain='same full denominator, include only benefit > 0',
                    negative_cost='same full denominator, sum -benefit for benefit < 0',
                    empty_subset='conditional mean null; full contribution zero when full denominator exists'),
    physical=dict(metrics=['XY_EPE_m', 'XYZ_EPE_m'],
                  mean='point Euclidean error within original object, then equal original-object mean',
                  parts='covered/uncovered point error sums / original object point count, then original-object mean',
                  fixed_blend='CRN_CV on owner>=0; D on owner<0; no GT routing',
                  missing_future='original label valid only; do not invent zero-error observations',
                  zero_benefit_points_remain=True),
    inference=dict(fitting=False, threshold_search=False, new_gate=False, model_forward=False,
                   bootstrap=False, training_seed_uncertainty=False, scene_AUC_IQR_is_CI=False),
    limitations=['D saw every train512 scene; no held-out D generalization claim',
                 'four horizons are outputs of one head, not independent observations',
                 'smoothness is not certainty; magnitude and relative residual are coupled',
                 'large AUC for tiny gains does not establish a useful routing rule',
                 'rigid-box sparse material-point proxy omits unlabeled and ambiguous source regions',
                 'no occupancy result, no O flow and no claim of exceeding O'])
RULES_PATH = HERE / 'motion_evidence_discrimination_analysis_rules_v1.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound_file(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact path')
    path = (root / relative).resolve()
    require(root.resolve() in path.parents, 'Artifact escapes root')
    return path


def digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def authenticate(a):
    require(read(RULES_PATH) == RULES, 'Prospective rules changed')
    require(sha(a.protocol) == PROTOCOL_SHA and sha(a.selection) == SELECTION_SHA and
            sha(a.train_index) == TRAIN_INDEX_SHA, 'Wrong protocol/selection/native index')
    p = read(a.protocol)
    require(p['schema'] == COLLECTOR_SCHEMA and p['status'] == 'FROZEN' and
            p['sources_sha256']['extract_motion_evidence_train_v1.py'] == COLLECTOR_SHA and
            p['selection_sha256'] == SELECTION_SHA, 'Wrong frozen collection contract')
    for name, digest in p['sources_sha256'].items():
        require(Path(name).name == name and digest_string(digest), 'Invalid source binding')
        paths = [a.source_root / name, a.source_root.parent / 'm0_improvement_20260915' / name]
        path = next((path for path in paths if path.is_file()), None)
        require(path is not None and sha(path) == digest, 'Source bytes differ: ' + name)
    require(not (a.run / 'failed.json').exists(), 'Incomplete/failed extraction rejected')
    done = read(a.run / 'complete.json')
    require(done['schema'] == COLLECTOR_SCHEMA and done['status'] == 'COMPLETE_FIXED_D_TRAIN_FEATURES' and
            (done['samples'], done['scenes'], done['optimizer_updates']) == (512,256,0) and
            done['source_sha256'] == COLLECTOR_SHA and done['protocol_sha256'] == PROTOCOL_SHA and
            set(done['files_sha256']) == {'manifest.json','index.json','summary.json'}, 'Wrong complete endpoint')
    for name, digest in done['files_sha256'].items():
        require(sha(a.run / name) == digest, 'Collection ledger differs: ' + name)
    m, ix, summary = (read(a.run / name) for name in ('manifest.json','index.json','summary.json'))
    require(m['schema'] == ix['schema'] == summary['schema'] == COLLECTOR_SCHEMA and
            m['protocol_sha256'] == PROTOCOL_SHA and m['sources_sha256'] == p['sources_sha256'] and
            m['recipe'] == p['recipe'] and m['selection_sha256'] == SELECTION_SHA and
            m['sparse_manifest_sha256'] == SPARSE_MANIFEST_SHA and
            m['train_cache']['index_sha256'] == TRAIN_INDEX_SHA and
            m['geometry_cache']['complete_sha256'] == p['geometry_complete_sha256'], 'Manifest source identity differs')
    d = m['D_loader']
    require(d['arm'] == 'D' and d['checkpoint_sha256'] == D_CHECKPOINT_SHA and
            d['actual_motion_state_sha256'] == D_STATE_SHA and
            d['top_complete_sha256'] == p['d_complete_sha256'] and d['protocol_sha256'] == p['connected_protocol_sha256'] and
            (d['fixed_final_update'],d['fixed_examples'],d['fixed_development_samples']) == (512,2048,200) and
            d['actual_optimizer_parameter_steps_all512'] is True and d['optimizer_restored'] is False and
            d['optimizer_updates_in_this_evaluation'] == 0, 'Wrong frozen D loading receipt')
    require(summary['status'] == done['status'] and
            (summary['samples'],summary['scenes'],summary['source_points'],summary['object_horizon_rows']) ==
            (512,256,1740053,43189) and summary['optimizer_updates'] == 0 and
            summary['D_state_unchanged'] is True and summary['all_saved_arrays_roundtrip_exact'] is True and
            all(summary[k] is False for k in ('threshold_selection','model_fitted','development_forward','O_model_loaded')),
            'Collection did not finish the fixed support')
    require(math.isfinite(summary['seconds']) and 0 < summary['seconds'] < p['resources']['max_seconds'] and
            0 <= summary['peak_allocated_gib'] <= p['resources']['max_allocated_gib'], 'Resource receipt differs')
    require(ix['feature_names'] == list(FEATURES) and ix['source_points'] == 1740053 and
            ix['valid_points_by_horizon'] == RULES['support']['valid_points_by_horizon'] and
            ix['valid_objects_by_horizon'] == RULES['support']['objects_by_horizon'], 'Index support differs')
    selection = read(a.selection)['records']
    train = [r for r in selection if r['split'] == 'train']
    dev = [r for r in selection if r['split'] == 'development']
    scenes = sorted({r['scene_token'] for r in train})
    require(len(train) == 512 and len(dev) == 200 and len(scenes) == 256 and
            len({r['sample_token'] for r in selection}) == 712 and
            not set(scenes).intersection(r['scene_token'] for r in dev), 'Frozen identity support differs')
    native = read(a.train_index)
    require(native['schema'] == 'm0-native-state-cache-v1' and native['status'] == 'COMPLETE' and
            native['split'] == 'train' and len(native['records']) == 512, 'Native training index differs')
    require(sha(a.sparse_labels / 'manifest.json') == SPARSE_MANIFEST_SHA and
            sha(a.sparse_labels / 'complete.json') == SPARSE_COMPLETE_SHA, 'Original sparse labels changed')
    labels, label_done = read(a.sparse_labels / 'manifest.json'), read(a.sparse_labels / 'complete.json')
    require(labels['status'] == label_done['status'] == 'COMPLETE' and
            label_done['manifest_sha256'] == SPARSE_MANIFEST_SHA and
            labels['samples'] == 712 and labels['train_samples'] == 512 and
            labels['train_dev_scenes_disjoint'] is True, 'Original labels incomplete')
    rows = ix['records']; require(len(rows) == 512, 'Incomplete collected samples')
    require({x.relative_to(a.run).as_posix() for x in (a.run/'samples').glob('*.npz')} ==
            {r['file'] for r in rows}, 'Sample file set differs from complete index')
    for i, (r, identity, nr, desc) in enumerate(zip(rows, train, native['records'], labels['records'][:512])):
        require(r['ordinal'] == desc['ordinal'] == i and r['identity'] == desc['identity'] == identity and
                {k:nr[k] for k in identity} == identity and
                r['file'] == 'samples/%04d_%s.npz' % (i,identity['sample_token']) and
                r['input_sha256'] == nr['files']['inputs']['sha256'] and r['sparse_sha256'] == desc['sha256'] and
                r['source_points'] == desc['counts']['source_points'], 'Per-sample data/order binding differs')
        g = r['geometry']['cache']
        require(g['split'] == 'train' and g['ordinal'] == g['raw_global_ordinal'] == i and
                g['identity'] == identity and g['complete_sha256'] == p['geometry_complete_sha256'] and
                digest_string(g['npz_sha256']), 'Per-sample geometry provenance differs')
    sources = dict(extraction_complete_sha256=sha(a.run/'complete.json'),
                   extraction_files_sha256=done['files_sha256'], protocol_sha256=PROTOCOL_SHA,
                   source_sha256=sha(__file__), rules_sha256=sha(RULES_PATH),
                   collector_sha256=COLLECTOR_SHA, dependency_sources_sha256=p['sources_sha256'],
                   selection_sha256=SELECTION_SHA, native_train_index_sha256=TRAIN_INDEX_SHA,
                   sparse_manifest_sha256=SPARSE_MANIFEST_SHA, sparse_complete_sha256=SPARSE_COMPLETE_SHA,
                   D_checkpoint_sha256=D_CHECKPOINT_SHA, D_state_sha256=D_STATE_SHA)
    return rows, labels['records'][:512], scenes, sources


def load_sample(a, row, desc):
    path = bound_file(a.run, row['file'])
    require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Sample NPZ SHA/size differs')
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == ARRAY_KEYS == set(row['arrays']), 'Sample array set differs')
        v = {k:z[k] for k in z.files}
    for name, x in v.items():
        require(row['arrays'][name] == dict(shape=list(x.shape),dtype=str(x.dtype)), 'Array descriptor differs: '+name)
    require(v['identity_json'].shape == () and v['identity_json'].dtype.kind == 'U' and
            json.loads(v['identity_json'].item()) == row['identity'], 'NPZ identity differs')
    original = bound_file(a.sparse_labels, desc['file'])
    require(original.stat().st_size == desc['bytes'] and sha(original) == desc['sha256'], 'Original label NPZ differs')
    with np.load(original, allow_pickle=False) as z:
        for name in LABEL_KEYS:
            require(same_bytes(v[name],z[name]), 'Original label values/support differ: '+name)
    n, k = len(v['source_flat_indices']), len(v['instance_tokens'])
    shapes = dict(source_flat_indices=(n,), object_index=(n,), instance_tokens=(k,),
                  target_displacement_m=(4,n,3), D_displacement_m=(4,n,3), CV_velocity_mps=(n,3),
                  valid=(4,n), object_future_valid=(4,k), object_speed_group=(4,k),
                  dt_future_seconds=(4,), owner=(n,), owner_original_indices=(n,), observable_features=(n,7))
    dtypes = dict(source_flat_indices='int64',object_index='int64',instance_tokens='<U32',
                  target_displacement_m='float32',D_displacement_m='float32',CV_velocity_mps='float64',
                  valid='bool',object_future_valid='bool',object_speed_group='int8',dt_future_seconds='float64',
                  owner='int64',owner_original_indices='int64',observable_features='float64')
    for name, shape in shapes.items():
        require(v[name].shape == shape and str(v[name].dtype) == dtypes[name], 'Wrong array shape/dtype: '+name)
    idx, obj = v['source_flat_indices'], v['object_index']
    require(n == row['source_points'] and (n == 0 or
            (idx[0] >= 0 and idx[-1] < 640000 and np.all(np.diff(idx)>0))), 'Invalid original source indices')
    require(np.array_equal(np.unique(obj),np.arange(k)) and
            len(set(v['instance_tokens'].tolist())) == k, 'Object mapping not complete/compact')
    require(np.array_equal(v['valid'],v['object_future_valid'][:,obj]) and
            np.isin(v['object_speed_group'],[-1,0,1,2]).all() and
            np.array_equal(v['object_speed_group']>=0,v['object_future_valid']) and
            np.all(v['target_displacement_m'][~v['valid']] == 0), 'Invalid original future support')
    for name in ('target_displacement_m','D_displacement_m','CV_velocity_mps','observable_features','dt_future_seconds'):
        require(np.isfinite(v[name]).all(), 'Nonfinite stored values: '+name)
    require(np.all(v['dt_future_seconds']>0) and np.all(np.diff(v['dt_future_seconds'])>0), 'Invalid label timestamps')
    uncovered = v['owner']<0
    require(np.all(v['owner']>=-1) and np.all(v['owner_original_indices']>=-1) and
            np.array_equal(uncovered,v['owner_original_indices']<0) and
            np.all(v['CV_velocity_mps'][uncovered] == 0) and
            int((~uncovered).sum()) == row['covered_points'], 'Invalid current prediction coverage/CV zero')
    f = v['observable_features']
    require(np.all(f>=0) and np.all(f[:,6]<=1+1e-12), 'Feature range differs')
    h = np.asarray(HORIZONS)[:,None,None]; d = v['D_displacement_m'].astype(np.float64)
    velocity = (h*d).sum(axis=0)/np.sum(h*h)
    speeds = np.linalg.norm((d/h)[:,:,:2],axis=2)
    speed = np.linalg.norm(velocity[:,:2],axis=1)
    rms = np.sqrt(np.mean(np.sum(((d/h)[:,:,:2]-velocity[None,:,:2])**2,axis=2),axis=0))
    reconstructed = np.stack([speed,speeds.min(axis=0),speeds.max(axis=0),rms,rms/(speed+.1)],axis=1)
    require(np.allclose(f[:,:5],reconstructed,rtol=1e-13,atol=1e-14), 'Temporal observable fields do not match saved D')
    require(v['valid'].sum(axis=1).tolist() == desc['counts']['valid_points_by_horizon'] and
            v['object_future_valid'].sum(axis=1).tolist() == desc['counts']['valid_objects_by_horizon'], 'Original label counts differ')
    return v


def weighted_auc(score, benefit, weight):
    """Weighted P(score_positive > score_negative), with half credit for ties."""
    require(score.shape == benefit.shape == weight.shape and
            np.isfinite(score).all() and np.isfinite(benefit).all() and
            np.isfinite(weight).all() and np.all(weight>0), 'Invalid AUC arrays')
    chosen = benefit != 0
    score, benefit, weight = score[chosen], benefit[chosen], weight[chosen]
    positive, negative = benefit>0, benefit<0
    wp, wn = float(weight[positive].sum()), float(weight[negative].sum())
    info = dict(auc=None,positive_weight=wp,negative_weight=wn,
                positive_points=int(positive.sum()),negative_points=int(negative.sum()))
    if wp == 0 or wn == 0:
        return info
    order = np.argsort(score,kind='stable')
    score, positive, weight = score[order], positive[order], weight[order]
    starts = np.r_[0,np.flatnonzero(score[1:]!=score[:-1])+1]
    pos = np.add.reduceat(weight*positive,starts)
    neg = np.add.reduceat(weight*(~positive),starts)
    below = np.cumsum(neg)-neg
    result = float(np.dot(pos,below+.5*neg)/(wp*wn))
    require(-1e-12 <= result <= 1+1e-12, 'Weighted AUC out of range')
    info['auc'] = min(1.,max(0.,result))
    return info


def collect(a, records, labels, scenes):
    scene_lookup = {s:i for i,s in enumerate(scenes)}
    physical = [[] for _ in HORIZONS]
    evidence = [{k:[] for k in ('features','benefit','weight','scene','group','object')} for _ in HORIZONS]
    total_points = 0; valid_counts = np.zeros(4,dtype=np.int64); object_base = 0
    for row, desc in zip(records,labels):
        v = load_sample(a,row,desc); n = len(v['source_flat_indices']); k = len(v['instance_tokens'])
        total_points += n; valid_counts += v['valid'].sum(axis=1)
        obj = v['object_index']; counts = np.bincount(obj,minlength=k)
        scene = scene_lookup[row['identity']['scene_token']]; uncovered = v['owner']<0
        for hi,h in enumerate(HORIZONS):
            valid = v['valid'][hi]; target = v['target_displacement_m'][hi].astype(np.float64)
            d = v['D_displacement_m'][hi].astype(np.float64); cv = h*v['CV_velocity_mps']
            blend = np.where(uncovered[:,None],d,cv)
            errors = np.stack([np.stack([np.linalg.norm((p-target)[:,:2],axis=1),
                                         np.linalg.norm(p-target,axis=1)],axis=1)
                               for p in (d,cv,blend,np.zeros_like(d))],axis=1)  # N,arm,axis
            sums = np.stack([np.bincount(obj,weights=errors[:,arm,axis],minlength=k)
                             for arm in range(4) for axis in range(2)],axis=1).reshape(k,4,2)
            outside = np.stack([np.bincount(obj,weights=errors[:,arm,axis]*uncovered,minlength=k)
                                for arm in range(4) for axis in range(2)],axis=1).reshape(k,4,2)
            uc = np.bincount(obj,weights=uncovered.astype(np.int64),minlength=k).astype(np.int64)
            for oi in np.flatnonzero(v['object_future_valid'][hi]):
                require(counts[oi]>0, 'Valid original object has no source points')
                physical[hi].append(dict(object=object_base+int(oi),scene=scene,
                    group=int(v['object_speed_group'][hi,oi]),points=int(counts[oi]),uncovered_points=int(uc[oi]),
                    full=sums[oi]/counts[oi],uncovered=outside[oi]/counts[oi]))
            take = valid & uncovered
            e = evidence[hi]
            e['features'].append(v['observable_features'][take])
            e['benefit'].append(errors[take,3,0]-errors[take,0,0])
            e['weight'].append(1./counts[obj[take]])
            e['scene'].append(np.full(int(take.sum()),scene,dtype=np.int32))
            e['group'].append(v['object_speed_group'][hi,obj[take]])
            e['object'].append(obj[take]+object_base)
        object_base += k
    require(total_points == 1740053 and valid_counts.tolist() == RULES['support']['valid_points_by_horizon'] and
            [len(rows) for rows in physical] == RULES['support']['objects_by_horizon'] and
            sum(map(len,physical)) == 43189, 'Incomplete original analysis support')
    return physical,[{key:np.concatenate(parts,axis=0) for key,parts in h.items()} for h in evidence]


def group_mask(groups, name):
    return np.ones(len(groups),dtype=bool) if name == 'all' else groups == GROUPS.index(name)-1


def continuous(e, mask, original_objects):
    b,w = e['benefit'][mask],e['weight'][mask]
    mass = float(w.sum()); signed = float(np.dot(w,b))
    gain = float(np.dot(w,np.maximum(b,0))); cost = float(np.dot(w,np.maximum(-b,0)))
    require(math.isclose(signed,gain-cost,rel_tol=1e-11,abs_tol=1e-10), 'Benefit decomposition differs')
    return dict(points=len(b),nonempty_objects=int(len(np.unique(e['object'][mask]))),
                original_object_count=original_objects,weight_sum=mass,
                positive_points=int((b>0).sum()),negative_points=int((b<0).sum()),zero_points=int((b==0).sum()),
                conditional_original_weight_mean_benefit_xy_m=signed/mass if mass else None,
                FULL_denominator_benefit_contribution_xy_m=signed/original_objects if original_objects else None,
                FULL_denominator_positive_gain_xy_m=gain/original_objects if original_objects else None,
                FULL_denominator_negative_cost_xy_m=cost/original_objects if original_objects else None)


def feature_results(e, mask, scenes=None):
    result = []
    for fi,(name,direction) in enumerate(zip(FEATURES,DIRECTIONS)):
        score = direction*e['features'][:,fi]
        item = dict(feature=name,direction=direction,
                    **weighted_auc(score[mask],e['benefit'][mask],e['weight'][mask]))
        if scenes is not None:
            per_scene = []
            for si,scene in enumerate(scenes):
                m = mask & (e['scene']==si)
                per_scene.append(dict(scene_token=scene,zero_benefit_points=int((e['benefit'][m]==0).sum()),
                    **weighted_auc(score[m],e['benefit'][m],e['weight'][m])))
            values = [row['auc'] for row in per_scene if row['auc'] is not None]
            quantiles = np.quantile(values,[.25,.5,.75]).tolist() if values else [None]*3
            item['per_scene'] = per_scene
            item['scene_distribution'] = dict(defined_scenes=len(values),undefined_scenes=len(scenes)-len(values),
                q25=quantiles[0],median=quantiles[1],q75=quantiles[2],
                IQR=None if not values else quantiles[2]-quantiles[0],is_confidence_interval=False)
        result.append(item)
    return result


def summarize(physical,evidence,scenes):
    all_physical = []; all_evidence = []
    for hi,h in enumerate(HORIZONS):
        rows,e = physical[hi],evidence[hi]
        groups = np.asarray([r['group'] for r in rows]); full = np.stack([r['full'] for r in rows])
        outside = np.stack([r['uncovered'] for r in rows]); original_counts = {}
        for name in GROUPS:
            mask = group_mask(groups,name); count = int(mask.sum()); original_counts[name] = count
            item = dict(horizon_seconds=h,group=name,original_objects=count,
                original_points=sum(r['points'] for r,m in zip(rows,mask) if m),
                uncovered_points=sum(r['uncovered_points'] for r,m in zip(rows,mask) if m),metrics={})
            emask = group_mask(e['group'],name)
            item['uncovered_D_versus_zero_benefit_xy'] = continuous(e,emask,count)
            for ai,axis in enumerate(('xy','xyz')):
                arms = {}
                for mi,arm in enumerate(ARMS):
                    mean = float(full[mask,mi,ai].mean()) if count else None
                    u = float(outside[mask,mi,ai].mean()) if count else None
                    c = float((full[mask,mi,ai]-outside[mask,mi,ai]).mean()) if count else None
                    arms[arm] = dict(full_original_object_mean_epe_m=mean,
                        uncovered_FULL_denominator_contribution_m=u,covered_FULL_denominator_contribution_m=c)
                item['metrics'][axis] = arms
                if count:
                    require(np.allclose(full[mask,2,ai],full[mask,1,ai]-outside[mask,1,ai]+outside[mask,0,ai],rtol=1e-12,atol=1e-12),
                            'Fixed blend decomposition differs')
                    require(np.allclose(outside[mask,1,ai],outside[mask,3,ai],rtol=0,atol=0), 'Uncovered CV is not zero')
            if count:
                value = item['uncovered_D_versus_zero_benefit_xy']['FULL_denominator_benefit_contribution_xy_m']
                require(math.isclose(value,float((outside[mask,3,0]-outside[mask,0,0]).mean()),rel_tol=1e-11,abs_tol=1e-12),
                        'Point and object benefit aggregation differ')
            all_physical.append(item)
        all_mask = np.ones(len(e['benefit']),dtype=bool)
        item = dict(horizon_seconds=h,population='original valid points with current owner<0',
                    support=continuous(e,all_mask,len(rows)),features=feature_results(e,all_mask,scenes),speed_bins=[])
        bins = np.searchsorted(SPEED_EDGES,e['features'][:,0],side='right')-1
        require(np.isin(bins,np.arange(7)).all(), 'Nonnegative finite speed fell outside fixed bins')
        contributions = []
        for bi in range(7):
            mask = bins == bi
            by_group = {name:continuous(e,mask & group_mask(e['group'],name),original_counts[name]) for name in GROUPS}
            contributions.append(by_group['all']['FULL_denominator_benefit_contribution_xy_m'])
            item['speed_bins'].append(dict(bin_index=bi,left_inclusive_mps=float(SPEED_EDGES[bi]),
                right_exclusive_mps=None if bi==6 else float(SPEED_EDGES[bi+1]),right_unbounded=(bi==6),
                by_group=by_group,features=feature_results(e,mask)))
        require(math.isclose(sum(contributions),item['support']['FULL_denominator_benefit_contribution_xy_m'],rel_tol=1e-11,abs_tol=1e-12),
                'Speed bins do not preserve complete original denominator contribution')
        all_evidence.append(item)
    return all_physical,all_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','protocol','selection','sparse-labels','train-index','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=HERE)
    a = parser.parse_args()
    require(not a.out.exists(), 'Output must be new; no overwrite')
    rows,labels,scenes,sources = authenticate(a)
    physical,evidence = collect(a,rows,labels,scenes)
    p,e = summarize(physical,evidence,scenes)
    require('torch' not in sys.modules, 'CPU-only analysis imported Torch')
    result = dict(schema=SCHEMA,status='COMPLETE_CPU_FIXED_TRAIN_EVIDENCE_ANALYSIS',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,rules=RULES,
        samples=512,scenes=256,source_points=1740053,object_horizon_rows=43189,scene_tokens=scenes,
        physical=p,evidence=e,
        verification=dict(all_512_sample_npz_SHA_and_array_descriptors_checked=True,
            all_original_sparse_label_bytes_and_native_input_hash_bindings_checked=True,
            first_five_features_recomputed_from_saved_D=True,
            spatial_features_bound_by_collection_SHA_not_recomputed_without_dense_field=True,
            checkpoint_tensors_reloaded=False,model_inference_performed=False),
        runtime=dict(python=sys.version,numpy=np.__version__,Torch_imported=False,optimizer_updates=0))
    with a.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(out=str(a.out),sha256=sha(a.out),samples=512,object_horizon_rows=43189)))


if __name__ == '__main__':
    main()
