"""Independent completed V/G audit: stdlib/NumPy, no producer imports or model calls."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

N = Path(__file__).resolve().parent
P = N.parent / 'm0_improvement_20260915'
ROOT = N/'server_results/training/object_state_forecast_train_v1'
OUT = N/'object_state_final_independent_statistics_audit_v1.json'
ARMS = ('O','V','G')
GROUPS = ('speed_le_0.1','speed_gt_0.1_le_0.5','speed_gt_0.5_le_5','speed_gt_5',
          'annotated_future_only','t0_missing_with_history','unknown_no_current_box','overlap_current_boxes')
PG = ('stationary','ambiguous','moving')
H = (.5,1.,1.5,2.)
ENDPOINT = '64ae1ec73fd69ead7a9342d1352ea45e9f9f7621ca23859f82dde7cc134a9452'
hashes = {}
maxdiff = dict(common_points=0.,common_intervals=0.,physical_points=0.,physical_intervals=0.,zero_from_labels=0.)


def sha(p):
    value=hashlib.sha256(Path(p).read_bytes()).hexdigest();hashes[str(p.relative_to(N.parent.parent))]=value
    return value


def read(p):return json.loads(Path(p).read_bytes())
def lines(p):return [json.loads(x) for x in Path(p).read_bytes().splitlines() if x.strip()]


def close(x,y,bucket,tol=5e-12):
    if y is None:
        assert not np.isfinite(x);return
    assert np.isfinite(x) and np.isfinite(y)
    delta=abs(float(x)-float(y));maxdiff[bucket]=max(maxdiff[bucket],delta)
    assert delta<=tol,(bucket,x,y,delta)


def ratio(n,d):
    n,d=np.broadcast_arrays(np.asarray(n,float),np.asarray(d,float))
    return np.divide(n,d,out=np.full(n.shape,np.nan),where=d!=0)


def values(o,c,g):
    iou=100*ratio(o[...,1,1],o[...,1,1]+o[...,0,1]+o[...,1,0])
    v={'future_GMO_IoU_percent':iou[...,1:].mean(-1),'t0_GMO_IoU_percent':iou[...,0],
       'future_FP':o[...,1:,0,1].sum(-1),'future_FN':o[...,1:,1,0].sum(-1)}
    for h in range(4):v[f'h{h+1}_GMO_IoU_percent']=iou[...,h+1]
    for k,name in enumerate(('00_background','01_arrival','10_vacating','11_persistent')):
        a=100*ratio(c[...,k,k],c[...,k,:].sum(-1)+c[...,:,k].sum(-1)-c[...,k,k])
        v[f'future_{name}_IoU_percent']=a.mean(-1)
        for h in range(4):v[f'h{h+1}_{name}_IoU_percent']=a[...,h]
    for k,name in enumerate(GROUPS):
        a=100*ratio(g[...,k,1],g[...,k,0]);v[f'future_{name}_recall_percent']=a.mean(-1)
        for h in range(4):v[f'h{h+1}_{name}_recall_percent']=a[...,h]
    m=g[...,2:4,:].sum(-2);a=100*ratio(m[...,1],m[...,0]);v['future_speed_gt_0.5_recall_percent']=a.mean(-1)
    for h in range(4):v[f'h{h+1}_speed_gt_0.5_recall_percent']=a[...,h]
    return v


def ledger(root,done,skip=()):
    for name,digest in done['files_sha256'].items():
        if name not in skip:assert sha(root/name)==digest


def main():
    tick=time.monotonic();assert not OUT.exists()
    assert sha(ROOT/'complete.json')==ENDPOINT
    done=read(ROOT/'complete.json');assert done['updates']==512 and done['examples']==2048 and done['evaluated_samples']==200
    assert done['status']=='COMPLETE_OBJECT_STATE_FORECAST_TRAINING';ledger(ROOT,done)
    state=read(ROOT/'_job/state.json');assert state['status']=='EXITED_ZERO' and state['returncode']==0
    manifest=read(ROOT/'manifest.json');proto=N/'object_state_forecast_training_protocol_v1.json';p=read(proto)
    assert sha(proto)==manifest['sources']['protocol_sha256']
    for name,digest in {**p['sources_sha256'],**p['analysis_sources_sha256']}.items():
        path=N/name if (N/name).is_file() else P/name;assert sha(path)==digest
    order=np.random.RandomState(11);expected_orders=np.concatenate([order.permutation(512) for _ in range(4)]).tolist()
    for arm in ('V','G'):
        folder=ROOT/'runs'/arm;assert sha(folder/'complete.json')==done['arm_complete_sha256'][arm]
        ac=read(folder/'complete.json');ledger(folder,ac,('final.pth',));am=read(folder/'manifest.json')
        assert am['common_manifest_sha256']==sha(ROOT/'manifest.json') and am['sources']==manifest['sources']
        assert ac['files_sha256']['final.pth']==done['final_checkpoints'][arm]['sha256']
        logs=lines(folder/'training.jsonl');assert len(logs)==512
        assert [x['ordinal'] for log in logs for x in log['samples']]==expected_orders
        assert all(log['update']==i+1 and log['examples']==4*(i+1) and len(log['samples'])==4 for i,log in enumerate(logs))
    cpu=N/'object_state_final_cpu_audit_v1';cpudone=read(cpu/'complete.json');sha(cpu/'complete.json');ledger(cpu,cpudone)
    cpua=read(cpu/'audit.json');assert cpua['status']=='PASS_ACTUAL_CPU_FINAL_TENSORS' and cpua['complete_sha256']==ENDPOINT
    assert cpua['metadata']['final_checkpoints']==done['final_checkpoints'] and not cpua['cuda_initialized']
    selection=P/'selection_v1.json';assert sha(selection)==p['selection_sha256']
    selected=[x for x in read(selection)['records'] if x['split']=='development']
    scenes=sorted({x['scene_token'] for x in selected});si={s:i for i,s in enumerate(scenes)}
    assert len(selected)==200 and len(scenes)==100
    rows=lines(ROOT/'development_records.jsonl');assert len(rows)==200
    Oref=P/'server_results/campaign_objective_v1/runs/O/development_records.jsonl'
    assert sha(Oref)==p['o_development_records_sha256'];oldO=lines(Oref)
    sparse=N/'sparse_motion_v1';sm=read(sparse/'manifest.json');sha(sparse/'manifest.json')
    raw=N/'motion_targets_v1';rm=read(raw/'manifest.json');sha(raw/'manifest.json')
    sd={x['identity']['sample_token']:x for x in sm['records'] if x['identity']['split']=='development'}
    rd={x['identity']['sample_token']:x for x in rm['records'] if x['identity']['split']=='development'}
    cs=read(N/'object_state_forecast_dev200_common_v1.json');ps=read(N/'object_state_forecast_dev200_physical_v1.json')
    sha(N/'object_state_forecast_dev200_common_v1.json');sha(N/'object_state_forecast_dev200_physical_v1.json')
    assert cs['scene_tokens']==ps['scene_tokens']==scenes
    occ={a:np.zeros((100,5,2,2),np.int64) for a in ARMS}
    change={a:np.zeros((100,4,4,4),np.int64) for a in ARMS}
    groups={a:np.zeros((100,4,8,3),np.int64) for a in ARMS}
    for i,(row,ident,old) in enumerate(zip(rows,selected,oldO,strict=True)):
        token=ident['sample_token'];sid=si[ident['scene_token']]
        assert row['ordinal']==i and all(row[k]==v for k,v in ident.items())
        assert old['sample_token']==token and old['scene_token']==ident['scene_token'] and row['hist_by_arm']['O']==old['hist_by_horizon']
        assert row['sparse_label_sha256']==sd[token]['sha256'] and row['raw_label_sha256']==sd[token]['label_source_sha256']==rd[token]['sha256']
        previous=None
        for a in ARMS:
            m=row['metrics_by_arm'][a];assert m['identity']==ident and m['shape_hxyz']==[5,512,512,40]
            o=np.array(row['hist_by_arm'][a]);c=np.array([h['confusion'] for h in m['transitions']])
            g=np.array([[[h['motion_positive_attribution']['groups'][k][v] for v in ('GT','TP','FN')] for k in GROUPS] for h in m['horizons'][1:]])
            assert o.shape==(5,2,2) and c.shape==(4,4,4) and g.shape==(4,8,3)
            assert all(x.dtype.kind in 'iu' and (x>=0).all() for x in (o,c,g))
            assert o.tolist()==[h['occupancy']['confusion'] for h in m['horizons']]
            assert np.array_equal(g[...,0],g[...,1]+g[...,2])
            assert np.array_equal(g.sum(1)[:,0],o[1:,1].sum(-1)) and np.array_equal(g.sum(1)[:,1],o[1:,1,1])
            for h,hr in enumerate(m['horizons']):
                assert hr['horizon_index']==h and hr['occupancy']['valid_voxels']==int(o[h].sum())
                assert hr['global_negative']==dict(GT=int(o[h,0].sum()),FP=int(o[h,0,1]),TN=int(o[h,0,0]))
            for h,tr in enumerate(m['transitions']):
                d=tr['domain'];assert tr['horizon_index']==h+1 and int(c[h].sum())==d['both_valid']
                assert sum(d[k] for k in ('both_valid','t0_valid_h_ignored','t0_ignored_h_valid','both_ignored'))==d['grid_voxels']
            t0=m['t0_boundary'];assert t0['valid_voxels']==int(o[0].sum()) and t0['predicted_foreground_on_valid']==int(o[0,:,1].sum())
            denom=(o.sum(-1),c.sum(-1),g[...,0])
            if previous is not None:
                assert all(np.array_equal(x,y) for x,y in zip(denom,previous))
                assert t0['gt_valid_mask_sha256']==row['metrics_by_arm']['O']['t0_boundary']['gt_valid_mask_sha256']
                assert [x['domain'] for x in m['transitions']]==[x['domain'] for x in row['metrics_by_arm']['O']['transitions']]
            previous=denom;occ[a][sid]+=o;change[a][sid]+=c;groups[a][sid]+=g
    draws=np.random.RandomState(11).randint(0,100,(10000,100))
    weights=np.stack([np.bincount(draw,minlength=100) for draw in draws]).astype(np.int64)
    whash=hashlib.sha256(weights.tobytes()).hexdigest();assert whash==cs['bootstrap_weights_sha256']==ps['bootstrap_weights_sha256']
    vp={};vb={};counts={}
    for a in ARMS:
        totals=[x[a].sum(0) for x in (occ,change,groups)]
        counts[a]=dict(zip(('occupancy_5x2x2','transition_4x4x4','groups_4x8x3_GT_TP_FN'),[x.tolist() for x in totals]))
        assert counts[a]==cs['pooled_counts'][a]
        vp[a]=values(*totals)
        resampled=[(weights@x[a].reshape(100,-1)).reshape((10000,)+x[a].shape[1:]) for x in (occ,change,groups)]
        vb[a]=values(*resampled)
        assert set(vp[a])==set(cs['values'][a])
        for k,v in vp[a].items():close(v,cs['values'][a][k],'common_points')
    comparisons={}
    for a,b in (('V','O'),('G','O'),('V','G')):
        pair=a+'-minus-'+b;comparisons[pair]={}
        for k,v in vp[a].items():
            diff=float(v-vp[b][k]);boot=vb[a][k]-vb[b][k];f=boot[np.isfinite(boot)];ci=np.quantile(f,[.025,.975]) if len(f) else [np.nan,np.nan]
            expected=cs['comparisons'][pair][k];close(diff,expected['difference'],'common_points')
            close(ci[0],expected['lower95'],'common_intervals',1e-10);close(ci[1],expected['upper95'],'common_intervals',1e-10)
            assert expected['finite_bootstrap_repetitions']==len(f)
            comparisons[pair][k]=dict(difference=diff,ci95=ci)
    # Raw object rows, independent original D and corrected CRN-CV complete ledgers.
    droot=N/'server_results/training/connected_motion_train_v2';ddone=read(droot/'complete.json')
    assert sha(droot/'complete.json')==ps['reference_authentication']['D_complete_sha256']
    assert sha(droot/'development_objects.jsonl')==ddone['files_sha256']['development_objects.jsonl']
    cvroot=N/'crn_object_state_cv_dev200_v2';cvdone=read(cvroot/'complete.json');ledger(cvroot,cvdone)
    assert sha(cvroot/'complete.json')==ps['reference_authentication']['CRN_complete_sha256']
    cv=read(cvroot/'summary.json');assert sha(cvroot/'summary.json')==ps['reference_authentication']['CRN_summary_sha256']
    objects=lines(ROOT/'development_objects.jsonl');oldD=[r for r in lines(droot/'development_objects.jsonl') if r['arm']=='D']
    key=lambda r:(r['sample_token'],r['instance_token'],r['horizon_seconds'])
    indexed={a:{} for a in ('D','V','G','CRN_CV')}
    for r in objects+[r for r in cv['physical_object_records'] if r['arm']=='CRN_CV']:
        a=r['arm'];k=key(r);assert k not in indexed[a];indexed[a][k]=r
    keys=sorted(indexed['D']);assert len(keys)==16074 and all(set(z)==set(keys) for z in indexed.values())
    for r in oldD:
        actual=indexed['D'][key(r)];assert all(actual[k]==v for k,v in r.items() if k not in ('arm','ordinal'))
    expected_keys=set();label_zero_count=0
    for token,descriptor in sd.items():
        path=sparse/descriptor['file'];assert sha(path)==descriptor['sha256']
        with np.load(path,allow_pickle=False) as z:
            assert str(z['sample_token'])==token
            tokens=z['instance_tokens'];idx=z['object_index'];future=z['object_future_valid'];valid=z['valid']
            assert np.array_equal(valid,future[:,idx]);assert np.isfinite(z['target_displacement_m']).all()
            for h,horizon in enumerate(H):
                assert int(future[h].sum())==descriptor['counts']['valid_objects_by_horizon'][h]
                assert int(valid[h].sum())==descriptor['counts']['valid_points_by_horizon'][h]
                for obj in np.flatnonzero(future[h]):
                    k=(token,str(tokens[obj]),horizon);expected_keys.add(k);base=z['target_displacement_m'][h,idx==obj].astype(np.float64)
                    n=len(base);group=PG[int(z['object_speed_group'][h,obj])];dtr=float(z['dt_future_seconds'][h])
                    zx=np.sqrt((base[:,:2]**2).sum(1)).mean();zz=np.sqrt((base**2).sum(1)).mean()
                    for a in indexed:
                        r=indexed[a][k];assert r['scene_token']==descriptor['identity']['scene_token'] and r['source_points']==n and r['group']==group and r['dt_seconds']==dtr
                        for field,v in (('zero_epe_xy_m',zx),('zero_epe_3d_m',zz)):close(v,r[field],'zero_from_labels')
                        assert np.isfinite(r['epe_xy_m']) and np.isfinite(r['epe_3d_m']) and r['epe_3d_m']>=0 and r['epe_xy_m']>=0
                    label_zero_count+=1
    assert expected_keys==set(keys)
    reports=[]
    for expected in ps['physical']:
        h,g=expected['horizon_seconds'],expected['group'];kk=[k for k in keys if k[2]==h and (g=='all' or indexed['D'][k]['group']==g)]
        ids=np.array([si[indexed['D'][k]['scene_token']] for k in kk]);nc=np.bincount(ids,minlength=100);den=weights@nc
        assert len(kk)==expected['object_anchor_pairs'] and sum(indexed['D'][k]['source_points'] for k in kk)==expected['source_point_occurrences']
        assert int((nc>0).sum())==expected['scenes_with_support'];item=dict(horizon_seconds=h,group=g,objects=len(kk),metrics={})
        for field,e in expected['metrics'].items():
            vals={a:np.array([indexed[a][k][field] for k in kk]) for a in indexed}
            vals['zero']=np.array([indexed['D'][k]['zero_'+field] for k in kk]);boots={};desc={}
            for a,v in vals.items():
                desc[a]=dict(count=len(v),mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)))
                assert desc[a]['count']==e['values'][a]['count']
                for name in ('mean','median','p90'):close(desc[a][name],e['values'][a][name],'physical_points')
                # Different accumulation path from producer's weighted bincount.
                sums=np.array([v[ids==s].sum() for s in range(100)])
                boots[a]=ratio(weights@sums,den)
            contrasts={}
            for name,value in e['comparisons'].items():
                a,b=name.split('-minus-');diff=desc[a]['mean']-desc[b]['mean'];sample=boots[a]-boots[b];finite=sample[np.isfinite(sample)]
                interval=np.quantile(finite,[.025,.975]);close(diff,value['difference'],'physical_points')
                close(interval[0],value['lower95'],'physical_intervals');close(interval[1],value['upper95'],'physical_intervals')
                assert len(finite)==value['defined_draws'];contrasts[name]=dict(difference=diff,ci95=interval)
            item['metrics'][field]=dict(values=desc,comparisons=contrasts)
        reports.append(item)
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
        if isinstance(x,np.integer):return int(x)
        return x
    sha(Path(__file__))
    result=dict(schema='object-state-final-independent-statistics-audit-v1',status='PASS_RAW_RECORDS_LABEL_SUPPORT_AND_INDEPENDENT_NUMPY_STATISTICS',
        sources_sha256=hashes,endpoint=ENDPOINT,samples=200,scenes=100,physical_object_horizon_rows_per_arm=16074,
        assertions=dict(original_O_all200_five_horizon_hist_exact=True,all_integer_pooled_counts_exact=True,
            all_common_point_metrics_recomputed=True,all_common_scene_intervals_recomputed=True,
            all_physical_groups_xy_xyz_mean_median_p90_recomputed=True,all_physical_scene_intervals_recomputed=True,
            all_D_reference_objects_exact=True,all_source_NPZ_identity_validmask_object_group_dt_zero_recomputed=True,
            source_NPZ_files_hashed=200,label_zero_object_horizons_checked=label_zero_count,paired_training_order_512_updates_checked=True,
            model_or_tensor_forward=False,SSH=False,Torch_imported=False,new_training=False,
            checkpoint_bytes_locally_loaded=False,remote_actual_CPU_tensor_audit_receipt_crosslinked=True),
        bootstrap=dict(repetitions=10000,seed=11,unit='paired whole scene',weights_sha256=whash),
        maximum_absolute_difference_from_frozen_statistics=maxdiff,pooled_counts=counts,common_values=vp,
        common_comparisons=comparisons,physical=reports,elapsed_seconds=time.monotonic()-tick)
    OUT.write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],out=str(OUT),maximum_differences=maxdiff,seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
