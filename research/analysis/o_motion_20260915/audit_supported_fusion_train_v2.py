"""Independent local CPU audit of completed supported-fusion v2 ledgers.

Does not import trainer metrics, load Torch/checkpoints, connect to servers,
or evaluate a model. Final checkpoint bytes/tensors remain an explicit gap;
only mutually bound checkpoint receipts are checked here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import struct

import numpy as np

N = Path(__file__).resolve().parent
P = N.parent/'m0_improvement_20260915'
PROTOCOL_SHA = '0752e433d9f37dbd88406099bee4ca4d3af5298b70dafb82d4cf308ec2df30b3'
DEV_O_SHA = '3f426891f6811a595542a1e9694783521ae89cf18d6385896012e5f52a8e9b9a'
SCHEMA = 'supported-fusion-training-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def jhash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def rate(u):
    if u < 25:
        return .001*(.1+.9*u/25)
    return .001*(.1+.9*.5*(1+math.cos(math.pi*(u-25)/486)))


def all_finite(value):
    if isinstance(value,dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value,list):
        return all(all_finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def ordered_float32(value):
    bits=struct.unpack('>I',struct.pack('>f',value))[0]
    return 0x80000000-(bits&0x7fffffff) if bits&0x80000000 else 0x80000000+bits


def field_checks(field):
    require(field['covered_voxels']+field['uncovered_voxels']==640000, 'Coarse support size')
    gate=field['gate_on_covered']; n=field['covered_voxels']
    require(gate['count']==n, 'Gate count vs coverage')
    for key in ('mean','p10','median','p90','fraction_below_point1','fraction_above_point9'):
        require((gate[key] is None) if n==0 else (gate[key] is not None and 0<=gate[key]<=1), 'Invalid gate summary')
    if n:
        require(gate['p10']<=gate['median']<=gate['p90'], 'Local gate quantile ordering')
    require(all_finite(field), 'Nonfinite diagnostic')


def audit():
    root=N/'server_results/training/supported_fusion_train_v2'
    pre=N/'server_results/training/supported_fusion_preflight_v2'
    cache=P/'server_results/cache/campaign_cache_v1'
    protocol=N/'supported_fusion_protocol_v2.json'
    paths={}; checks={}
    def bound(path,expected=None):
        digest=sha(path)
        if expected is not None: require(digest==expected,'SHA mismatch: '+str(path))
        paths[str(path.relative_to(N.parent))]=digest
        return digest
    bound(protocol,PROTOCOL_SHA); p=read(protocol)
    require(p['status']=='FROZEN' and p['schema']==SCHEMA,'Protocol status/schema')
    m=read(root/'manifest.json'); done=read(root/'complete.json'); summary=read(root/'summary.json')
    top_sha=bound(root/'complete.json')
    require(done['status']=='COMPLETE_SUPPORTED_FUSION_TRAINING' and done['mode']=='train'
            and done['schema']==SCHEMA and done['updates']==512 and done['examples']==2048
            and done['evaluated_samples']==200,'Formal completion')
    require(set(done['files_sha256'])=={'manifest.json','summary.json','development_records.jsonl'},'Top file set')
    for f,s in done['files_sha256'].items():bound(root/f,s)
    require(not (root/'failed.json').exists(),'Formal failed file')
    state_path=N/'server_results/jobs/supported_fusion_train_v2/state.json'
    state=read(state_path);bound(state_path)
    require(state['status']=='EXITED_ZERO' and state['returncode']==0,'Real job terminal status')
    require(m['training']==p['training'] and m['sources']['protocol_sha256']==PROTOCOL_SHA
            and m['sources']['sources_sha256']==p['sources_sha256'] and m['sources']['motion']==p['motion'], 'Manifest recipe/source')
    for f,s in p['sources_sha256'].items():bound(N/f if (N/f).exists() else P/f,s)
    require(m['numerical_policy']==p['numerical_policy'],'Numerical policy')
    require(m['single_training_seed']==m['training']['seed']==11 and m['same_initial_parameters']
            and m['independent_optimizers'] and m['planned_updates']==512
            and m['trainable_names']==['gate.0.weight','gate.0.bias','gate.2.weight','gate.2.bias']
            and m['training']['parameters_per_arm']==97,'Initial/optimizer/scope declarations')
    require(m['boxes_or_motion_labels_read'] is False and m['t0_and_uncovered_logits_preserved_bytewise']
            and m['final_decoder_futures_only'],'Predictor/label boundary declarations')
    require(summary['initial_gate_sha256']==m['initial_gate_sha256']
            and summary['zero_readout_changed'] and summary['all_gradients_finite']
            and summary['O_and_motion_parameters_buffers_unchanged'] and summary['weights_persisted']
            and summary['updates']==512 and summary['examples']==2048 and summary['evaluated_samples']==200,'Producer completion gates')
    checks['top_source_and_terminal_receipts']='PASS'
    indexes={};records={}
    for role in ('train','development'):
        folder=cache/role;index=read(folder/'index.json');cc=read(folder/'complete.json')
        bound(folder/'index.json',p['cache_index_sha256'][role]);bound(folder/'complete.json',m['sources'][role+'_cache']['complete_sha256'])
        require(cc['index_sha256']==p['cache_index_sha256'][role] and cc['status']=='COMPLETE_NATIVE_STATE_CACHE','Cache completion')
        require(index['status']=='COMPLETE' and index['split']==role and len(index['records'])==(512 if role=='train' else 200),'Cache split/count')
        records[role]=index['records'];indexes[role]=index
    require(len({r['sample_token'] for r in records['train']})==512 and len({r['sample_token'] for r in records['development']})==200,'Unique anchors')
    require(not ({r['scene_token'] for r in records['train']} & {r['scene_token'] for r in records['development']}),'Train/dev scenes')
    rng=np.random.RandomState(11);orders=[rng.permutation(512).tolist() for _ in range(4)];flat=[x for a in orders for x in a]
    require(jhash(orders)==m['sample_orders_sha256'],'Fixed RandomState11 four-pass order')
    pre_done=read(pre/'complete.json');pre_m=read(pre/'manifest.json');pre_s=read(pre/'summary.json')
    bound(pre/'complete.json')
    for f,s in pre_done['files_sha256'].items():bound(pre/f,s)
    require(pre_done['status']=='PASS_SUPPORTED_FUSION_PREFLIGHT' and pre_done['updates']==4
            and pre_done['examples']==16 and pre_done['evaluated_samples']==0
            and pre_done['final_checkpoints']=={} and pre_s['weights_persisted'] is False,'Preflight did not persist candidate weights')
    require(pre_m['initial_gate_sha256']==m['initial_gate_sha256'] and pre_m['sources']['protocol_sha256']==PROTOCOL_SHA,'Fresh formal initialization receipt')
    require(all(summary['final_gate_sha256'][a]!=m['initial_gate_sha256'] and summary['parameter_max_abs_change'][a]>0 for a in ('A','Z')),'Nonzero parameter update receipts')
    checks['matched_seed_initialization_and_no_preflight_weight_reuse']='PASS_SOURCE_AND_RECEIPTS_NOT_LOCAL_TENSOR_REHASH'
    logs={};arm_devs={};checkpoints={};grad_stats={};ce_delta=[];ce_ulps=[];lr_deltas=[]
    optkeys=[f'loss_voxel_{family}_inter_{layer}' for layer in range(3) for family in ('ce','lovasz')]
    allkeys={f'loss_voxel_{family}_inter_{layer}' for layer in range(3) for family in ('ce','sem_scal','lovasz','geo_scal')}
    for arm in ('A','Z'):
        folder=root/'runs'/arm;comp=read(folder/'complete.json');am=read(folder/'manifest.json')
        bound(folder/'complete.json',done['arm_complete_sha256'][arm])
        require(comp['schema']==SCHEMA and comp['status']==done['status'] and comp['mode']=='train'
                and comp['arm']==arm and comp['updates']==512 and comp['examples']==2048,'Arm complete')
        require(set(comp['files_sha256'])=={'manifest.json','training.jsonl','final.pth','development_records.jsonl'},'Arm complete file set')
        for f,s in comp['files_sha256'].items():
            if f!='final.pth':bound(folder/f,s)
        require(am==dict(m,arm=arm,transport='predicted' if arm=='A' else 'zero',common_manifest_sha256=sha(root/'manifest.json')),'Arm/common manifest')
        c=done['final_checkpoints'][arm]
        require(c['file']==f'runs/{arm}/final.pth' and c['sha256']==comp['files_sha256']['final.pth']
                and c['gate_state_sha256']==comp['final_gate_sha256']==summary['final_gate_sha256'][arm],'Cross-receipt final state/checkpoint consistency')
        checkpoints[arm]=dict(**c,local_checkpoint_present=(folder/'final.pth').exists(),
            checkpoint_bytes_verified_here=False,actual_tensor_rehashed_here=False)
        logs[arm]=lines(folder/'training.jsonl');arm_devs[arm]=lines(folder/'development_records.jsonl')
        require(len(logs[arm])==512,'512 actual update records')
        for u,log in enumerate(logs[arm]):
            require(all_finite(log),'Nonfinite actual training log')
            require(log['update']==u+1 and log['examples']==4*(u+1) and log['pass_index']==u//128,'Actual update/exposure')
            expected_lr=rate(u)
            # Linux/macOS libm cosine last bits differ. This checks formula
            # reproduction only; actual paired A/Z LRs below remain EXACT.
            lr_ulps=abs(log['lr']-expected_lr)/float(np.spacing(expected_lr))
            require(lr_ulps<=4.,'Actual LR schedule beyond cross-platform roundoff')
            if arm=='A' and lr_ulps:
                lr_deltas.append(dict(update=u+1,actual=log['lr'],local_recomputed=expected_lr,float64_ULPs=lr_ulps))
            require(log['preclip_grad_norm']>=0 and log['clip_factor']==min(1.,35./(log['preclip_grad_norm']+1e-6)),'Clip formula')
            require(len(log['samples'])==4,'Accumulation size')
            require(log['sample_group_sha256']==jhash([x['sample_token'] for x in log['samples']]),'Group digest')
            for item,j in zip(log['samples'],flat[4*u:4*u+4]):
                cr=records['train'][j]
                require(item['ordinal']==j and item['sample_token']==cr['sample_token']
                    and item['scene_token']==cr['scene_token'] and item['inputs_sha256']==cr['files']['inputs']['sha256']
                    and item['targets_sha256']==cr['files']['targets']['sha256'],'Actual input/target/order')
                loss=item['loss'];terms=loss['original_twelve']
                require(set(terms)==allkeys and loss['optimized_keys']==optkeys,'Original12/optimized6 keys')
                scalar=np.float32(0.)
                for k in optkeys:scalar=np.float32(scalar+np.float32(terms[k]))
                require(float(scalar)==loss['objective_sum'],'Native FP32 sequential selected-loss sum')
                const=[k for k in optkeys if k.endswith(('_inter_0','_inter_1'))]
                require(loss['constant_intermediate_keys']==const
                        and loss['constant_intermediate_sum']==sum(terms[k] for k in const)
                        and loss['last_decoder_sum']==sum(terms[k] for k in optkeys if k.endswith('_inter_2'))
                        and loss['omitted_semgeo_sum']==sum(v for k,v in terms.items() if k not in optkeys)
                        and loss['last_decoder_pools_fixed_t0_and_future'],'Loss audit sub-sums')
                require([f['horizon_slot'] for f in item['fields']]==[1,2,3,4],'Field horizon order')
                for field in item['fields']:field_checks(field)
            require(log['loss']==float(np.mean([x['loss']['objective_sum'] for x in log['samples']])),'Update loss mean')
        norms=np.array([x['preclip_grad_norm'] for x in logs[arm]])
        require(np.any(norms>0) and summary['at_least_one_nonzero_task_gradient'][arm],'Nonzero logged task gradient')
        grad_stats[arm]=dict(nonzero_updates=int((norms>0).sum()),zero_updates=int((norms==0).sum()),
            minimum=float(norms.min()),median=float(np.median(norms)),p90=float(np.quantile(norms,.9)),
            maximum=float(norms.max()),clipped_updates=sum(x['clip_factor']<1 for x in logs[arm]),
            initial_gate_sha256=m['initial_gate_sha256'],final_gate_sha256=comp['final_gate_sha256'],
            parameter_max_abs_change_receipt=summary['parameter_max_abs_change'][arm])
    for left,right in zip(logs['A'],logs['Z']):
        require(all(left[k]==right[k] for k in ('update','examples','pass_index','lr','sample_group_sha256')),'Paired optimizer exposure')
        # Each arm's elapsed time is sampled while serializing that arm;
        # those wall-clock floats are not supposed to be bitwise identical.
        require(left['shared_update_wall_seconds']>0 and right['shared_update_wall_seconds']>0,'Positive observed update duration')
        for la,lz in zip(left['samples'],right['samples']):
            require(la['constant_loss_comparison']==lz['constant_loss_comparison'],'Same comparison receipts')
            comparison=la['constant_loss_comparison'];require(set(comparison)==set(optkeys[:4]),'Constant comparison key set')
            for key,record in comparison.items():
                a=la['loss']['original_twelve'][key];z=lz['loss']['original_twelve'][key]
                ce='_ce_' in key;expected=math.isclose(a,z,rel_tol=1e-5,abs_tol=1e-7) if ce else a==z
                ulp=abs(ordered_float32(a)-ordered_float32(z))
                require(record==dict(A=a,Z=z,Z_minus_A=z-a,float32_ULPs=ulp,
                    comparison='math.isclose' if ce else 'exact',rel_tol=1e-5 if ce else 0.,
                    abs_tol=1e-7 if ce else 0.,passed=expected) and expected,'CE tolerance / Lovasz exact gate')
                if ce:ce_delta.append(abs(z-a));ce_ulps.append(ulp)
    checks['actual_512_updates_2048_examples_both_arms_order_LR_losses_gradients']='PASS'
    checks['constant_intermediate_CE_tolerance_and_Lovasz_exact']='PASS'
    dev=lines(root/'development_records.jsonl');ref_path=P/'server_results/campaign_objective_v1/runs/O/development_records.jsonl'
    bound(ref_path,DEV_O_SHA);ref=lines(ref_path)
    require(len(dev)==len(ref)==len(arm_devs['A'])==len(arm_devs['Z'])==200,'Complete dev200')
    hist={a:[] for a in ('O','A','Z')}
    for i,(row,old,cr) in enumerate(zip(dev,ref,records['development'])):
        require(all_finite(row),'Nonfinite dev records')
        require(row['ordinal']==i and row['split']=='development' and row['official_index']==cr['official_index']
            and all(row[k]==cr[k]==old[k] for k in ('sample_token','scene_token')),'Dev identity and fixed order')
        require(row['inputs_sha256']==cr['files']['inputs']['sha256'] and row['targets_sha256']==cr['files']['targets']['sha256'],'Dev input/GT bytes')
        require(set(row['hist_by_arm'])=={'O','A','Z'} and set(row['diagnostics'])=={'A','Z'},'Dev arm set')
        require(row['hist_by_arm']['O']==old['hist_by_horizon'] and row['O_frozen_reference_hist_exact']
                and row['t0_all_arms_byte_equal'],'Native O parity / t0 runtime gate')
        for arm in hist:
            h=np.asarray(row['hist_by_arm'][arm]);require(h.shape==(5,2,2) and h.dtype.kind in 'iu' and np.all(h>=0),'Histogram')
            require(np.array_equal(h.sum(-1),np.asarray(old['hist_by_horizon']).sum(-1)),'Per-frame GT row support')
            require(np.array_equal(h[0],np.asarray(old['hist_by_horizon'])[0]),'t0 exact confusion')
            hist[arm].append(h)
            if arm!='O':
                ar=arm_devs[arm][i]
                require(ar['sample_token']==row['sample_token'] and ar['scene_token']==row['scene_token']
                        and ar['hist_by_horizon']==row['hist_by_arm'][arm] and ar['diagnostics']==row['diagnostics'][arm]
                        and ar['horizon_seconds']==[0.,.5,1.,1.5,2.],'Arm/common dev ledger')
                for f in row['diagnostics'][arm]:field_checks(f)
    checks['dev200_O_reference_GT_rows_t0_hist_and_all_arm_ledger_parity']='PASS'
    scene_names=sorted({r['scene_token'] for r in dev});require(len(scene_names)==100,'100 dev scenes')
    scenes={s:i for i,s in enumerate(scene_names)}
    scene_hist={a:np.zeros((100,5,2,2),np.int64) for a in hist}
    for a in hist:
        hist[a]=np.stack(hist[a])
        for i,r in enumerate(dev):scene_hist[a][scenes[r['scene_token']]]+=hist[a][i]
    def values(array):
        tp=array[...,1,1];fp=array[...,0,1];fn=array[...,1,0]
        require(np.all(tp+fp+fn>0),'Undefined original GMO IoU')
        iou=tp/(tp+fp+fn)*100
        return dict(iou_by_horizon_percent=iou,future_mean_iou_percent=iou[...,1:].mean(-1),
            FP_by_horizon=fp,FN_by_horizon=fn,future_FP=fp[...,1:].sum(-1),future_FN=fn[...,1:].sum(-1),
            precision_by_horizon=tp/(tp+fp),recall_by_horizon=tp/(tp+fn))
    totals={a:hist[a].sum(0) for a in hist};points={a:values(v) for a,v in totals.items()}
    for a in hist:
        old=summary['scores'][a]
        require(np.array_equal(totals[a],old['hist_by_horizon']),'Producer pooled histogram')
        require(np.allclose(points[a]['iou_by_horizon_percent'],old['iou_by_horizon_percent'],rtol=0,atol=1e-12)
                and abs(points[a]['future_mean_iou_percent']-old['future_mean_iou_percent'])<1e-12,'Independent point scores')
    draws=np.random.RandomState(11).randint(100,size=(10000,100))
    weights=np.zeros((10000,100),np.int64);np.add.at(weights,(np.arange(10000)[:,None],draws),1)
    boots={a:values((weights@scene_hist[a].reshape(100,-1)).reshape(10000,5,2,2)) for a in hist}
    contrasts={}
    for a,b in (('A','O'),('A','Z'),('Z','O')):
        stats={}
        for key in ('future_mean_iou_percent','iou_by_horizon_percent','future_FP','future_FN','FP_by_horizon','FN_by_horizon'):
            diff=boots[a][key]-boots[b][key]
            stats[key]=dict(difference=points[a][key]-points[b][key],lower95=np.quantile(diff,.025,axis=0),
                upper95=np.quantile(diff,.975,axis=0),unit='percentage_points' if 'percent' in key else 'voxel_occurrences')
        contrasts[a+'-minus-'+b]=stats
    checks['independent_pooled_scores_and_scene_paired_bootstrap']='PASS'
    gate_summary={}
    for arm in ('A','Z'):
        hs=[]
        for h in range(4):
            items=[r['diagnostics'][arm][h] for r in dev]
            count=sum(d['covered_voxels'] for d in items)
            entry=dict(horizon_index=h+1,nominal_seconds=(h+1)*.5,covered_voxel_occurrences=count,
                evaluated_coarse_voxel_occurrences=200*640000,coverage_fraction=count/(200*640000))
            for k in ('mean','fraction_below_point1','fraction_above_point9'):
                entry['coverage_weighted_'+k]=sum(d['covered_voxels']*(d['gate_on_covered'][k] or 0.) for d in items)/count if count else None
            hs.append(entry)
        n=sum(h['covered_voxel_occurrences'] for h in hs)
        whole=dict(covered_voxel_occurrences=n,coverage_fraction=n/(200*4*640000))
        for k in ('mean','fraction_below_point1','fraction_above_point9'):
            whole['coverage_weighted_'+k]=sum(h['covered_voxel_occurrences']*(h['coverage_weighted_'+k] or 0.) for h in hs)/n if n else None
        gate_summary[arm]=dict(by_horizon=hs,all_futures=whole,
            pooled_median=None,pooled_quantiles_available=False,
            numerical_scope='Coverage-weighted reconstruction from logged FP32 local means/fractions; no raw pooled gate histogram; rounding remains')
    checks['gate_means_fractions_weighted_by_actual_predicted_coverage']='PASS'
    def convert(x):
        if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
        if isinstance(x,list):return [convert(v) for v in x]
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        return x
    result=convert(dict(schema='supported-fusion-v2-independent-local-audit-v1',
        status='PASS_LOCAL_RECEIPTS_LOGS_AND_COUNTS',source_sha256=sha(__file__),
        python=platform.python_version(),numpy=np.__version__,verified_local_files_sha256=paths,
        checks=checks,complete_sha256=top_sha,final_checkpoint_receipts=checkpoints,
        training=dict(seed=11,updates_per_arm=512,examples_per_arm=2048,passes=4,accumulate=4,
            train_anchors=512,development_anchors=200,development_scenes=100,
            sample_orders_sha256=jhash(orders),gradient_statistics=grad_stats,
            CE_max_absolute_difference=max(ce_delta),CE_max_float32_ULPs=max(ce_ulps),
            CE_comparisons_nonexact=sum(x>0 for x in ce_delta),CE_comparisons=len(ce_delta),
            LR_actual_A_Z_exact=True,LR_formula_local_cosine_roundoff=lr_deltas,
            LR_formula_recheck_max_float64_ULPs=4,
            Lovasz_intermediate_exact=True,all_original_twelve_losses_finite=True),
        scores=points,pooled_histograms=totals,contrasts=contrasts,gate_statistics=gate_summary,
        bootstrap=dict(repetitions=10000,seed=11,unit='scene',scene_order=scene_names,
            draw_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
            paired_same_scene_multiplicities_all_arms=True,preserve_all_anchors_within_scene=True,
            statistic='sum raw histograms; per-horizon IoU; mean four future IoUs; NOT mean-per-scene IoU',
            interval='percentile95',multiplicity_adjusted=False,training_seed_uncertainty=False),
        resources=dict(job_seconds=state['seconds'],producer_seconds=done['elapsed_seconds'],
            peak_allocated_gib=summary['peak_allocated_bytes']/2**30,peak_reserved_gib=summary['peak_reserved_bytes']/2**30),
        limitations=['Final pth bytes and actual tensor state not locally loaded; receipts only. Common evaluator independently authenticates them.',
            'Logit byte equality/frozen weights/no gradient leakage are source-bound producer runtime assertions; local histogram equality cannot independently prove all tensor bytes.',
            'Historically exposed development200, single seed11, no multiplicity adjustment; no new full validation or motion EPE.',
            'Gate statistics pool predicted coarse support, not fine-GT valid-domain voxels or actual moving-object support.',
            'A and Z were separately optimized, so their difference is a matched trained-pipeline contrast, not an isolated same-gate displacement intervention.',
            'Positive small development difference does not establish the overall goal or motion accuracy.']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();target=Path(args.out)
    require(not target.exists(),'No overwrite of prior audit')
    result=audit();target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],contrasts=result['contrasts'],training=result['training'],
        gate_statistics=result['gate_statistics'],resources=result['resources']),indent=2,allow_nan=False))
