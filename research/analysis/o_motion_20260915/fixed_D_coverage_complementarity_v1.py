"""Fixed-D/CV coverage diagnostic, no optimization or native occupancy replay.

Construct all dense fields from inputs before sparse future labels are opened.
Coverage is fixed by current predicted-box ownership, never GT association.
The CV-covered/D-uncovered field is a prespecified diagnostic, not a new model.
Source-root can be an existing read-only frozen package; this script is separate.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

SCHEMA = 'fixed-D-coverage-complementarity-v1'
HORIZONS = (.5, 1., 1.5, 2.)
ARMS = ('D', 'CRN_CV', 'CV_covered_D_uncovered')
IDENTITY = ('sample_token', 'scene_token', 'split', 'official_index')
CONNECTED_SHA = 'e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
D_COMPLETE_SHA = 'f0bf075fa36622b38c0c5136b187a184d16d3e07ed00857dfe21a0545f1c5f56'
GEOMETRY_SHA = 'a2504a0389cb2756032531961e305e22287ec890c96085e5ea29b2e61e4f1c99'
GEOMETRY_SOURCE_SHA = 'd47431242dff6d9dba10fbd791e6516739b410282e60bf7f73470186c90fcaaa'
REFERENCE_COMPLETE_SHA = 'edfdf0b38145ad0b1d5fb69370ba8e8d91fda973f33540c1f5149917f58077f6'
REFERENCE_SUMMARY_SHA = 'ba67afc75b02ef481b0846914207fc01c6d04a4d0c675d3a7c00d1129b81141e'
RUNTIME_SHA = '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c'
SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
RECIPE = dict(samples=200, scenes=100, object_horizon_rows=16074, optimizer_updates=0,
    horizons_seconds=list(HORIZONS), coverage='current_prediction_owner>=0',
    diagnostic_field='CRN_CV on covered; fixed D on uncovered; no fitting',
    forward_order='full D/CV/blend fields before load_sparse and gather_sparse',
    point_order='XYZ C-order; native D [1,4,3,200,200,16]',
    aggregation='point errors within object; equal object means; partition contributions divide original full object count',
    reference_rtol=1e-12, reference_atol_m=1e-10,
    numerical_policy=dict(dtype='float32', matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False))


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): digest.update(b)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source_path(name, source_root):
    require(Path(name).name == name and name.endswith('.py'), 'Source must be basename')
    if name == Path(__file__).name: return Path(__file__).resolve()
    root = Path(source_root).resolve()
    for p in (root/name, root.parent/'m0_improvement_20260915'/name):
        if p.is_file(): return p.resolve()
    raise FileNotFoundError(name)


def bind(name, bindings, source_root):
    path = source_path(name+'.py', source_root)
    require(sha(path) == bindings[path.name], 'Bound source SHA differs: '+name)
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path, 'Imported source path differs: '+name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path, 'Executed source differs')
    return module


def expected_sources(connected_protocol, source_root=None):
    """Root may call before freezing; only imports NumPy cache definitions."""
    root = Path(source_root or Path(__file__).parent)
    require(sha(connected_protocol) == CONNECTED_SHA, 'Wrong connected protocol')
    result = dict(read(connected_protocol)['sources_sha256'])
    result['shared_rigid_geometry_cache_v1.py'] = GEOMETRY_SOURCE_SHA
    cache = bind('shared_rigid_geometry_cache_v1', result, root)
    for name, digest in cache.SOURCES.items():
        require(name not in result or result[name] == digest, 'Source closure conflict')
        result[name] = digest
    result[Path(__file__).name] = sha(__file__)
    for name, digest in result.items(): require(sha(source_path(name, root)) == digest, 'Source changed: '+name)
    return result


def protocol_template():
    return dict(schema=SCHEMA, status='REVIEW_REQUIRED', recipe=RECIPE,
        sources_sha256={}, connected_protocol_sha256=CONNECTED_SHA,
        d_complete_sha256=D_COMPLETE_SHA, geometry_complete_sha256=GEOMETRY_SHA,
        reference_complete_sha256=REFERENCE_COMPLETE_SHA, reference_summary_sha256=REFERENCE_SUMMARY_SHA,
        runtime_contract_sha256=RUNTIME_SHA, selection_sha256=SELECTION_SHA,
        resources=dict(max_seconds=900, max_allocated_gib=8, cpu_threads=2))


def identity(row): return {k:row[k] for k in IDENTITY}
def key(row): return (row['sample_token'], row['instance_token'], row['horizon_seconds'])


def unique_map(rows):
    result = {key(r):r for r in rows}
    require(len(result) == len(rows) == 16074, 'Reference duplicate/missing object-horizon support')
    return result


def reference(root):
    root = Path(root)
    require(not (root/'failed.json').exists() and sha(root/'complete.json') == REFERENCE_COMPLETE_SHA,
            'Wrong/incomplete CRN-CV reference')
    done = read(root/'complete.json')
    require(done['status'] == 'COMPLETE_READ_ONLY_DIAGNOSTIC' and done['samples'] == 200 and done['scenes'] == 100,
            'Reference endpoint differs')
    for name, digest in done['files_sha256'].items(): require(sha(root/name) == digest, 'Reference artifact changed')
    require(sha(root/'summary.json') == REFERENCE_SUMMARY_SHA, 'Reference summary changed')
    summary = read(root/'summary.json')
    require(summary['samples'] == 200 and summary['scenes'] == 100 and summary['object_horizon_rows_per_arm'] == 16074,
            'Reference support differs')
    physical = {a:unique_map([r for r in summary['physical_object_records'] if r['arm'] == a]) for a in ARMS[:2]}
    require(set(physical['D']) == set(physical['CRN_CV']), 'Reference arms support differs')
    coverage = unique_map(summary['coverage_object_records'])
    require(set(coverage) == set(physical['D']) and len(summary['per_anchor_coverage']) == 200, 'Reference coverage differs')
    return physical, coverage, summary['per_anchor_coverage']


def close(actual, expected, label, maxima):
    if actual is None or expected is None:
        require(actual is expected, 'Null support differs: '+label); return
    actual, expected = float(actual), float(expected)
    require(math.isfinite(actual) and math.isfinite(expected), 'Nonfinite comparison: '+label)
    delta = abs(actual-expected); maxima[label.split(':')[0]] = max(maxima.get(label.split(':')[0], 0.), delta)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
            'Frozen tolerance failed '+label+': actual='+repr(actual)+' expected='+repr(expected)+' abs='+repr(delta))


def summarize(rows):
    result = []
    for h in HORIZONS:
        for group in ('all', 'stationary', 'ambiguous', 'moving'):
            chosen = [r for r in rows if r['horizon_seconds'] == h and (group == 'all' or r['group'] == group)]
            entry = dict(horizon_seconds=h, group=group, objects=len(chosen), source_points=sum(r['source_points'] for r in chosen),
                         covered_points=sum(r['covered_points'] for r in chosen), arms={})
            for arm in ARMS:
                metrics = {}
                for dimension in ('xy', '3d'):
                    values = np.asarray([r['arms'][arm]['epe_'+dimension+'_m'] for r in chosen], dtype=np.float64)
                    metrics[dimension] = dict(mean=float(values.mean()) if len(values) else None,
                        median=float(np.median(values)) if len(values) else None,
                        p90=float(np.percentile(values,90)) if len(values) else None,
                        covered_full_denominator_contribution_mean=float(np.mean([r['parts']['covered'][arm][dimension+'_contribution_m'] for r in chosen])) if chosen else None,
                        uncovered_full_denominator_contribution_mean=float(np.mean([r['parts']['uncovered'][arm][dimension+'_contribution_m'] for r in chosen])) if chosen else None)
                entry['arms'][arm] = metrics
            result.append(entry)
    return result


def run(a, out, started):
    p = read(a.protocol); expected = protocol_template()
    require(p['status'] == 'FROZEN' and p['schema'] == SCHEMA and p['recipe'] == RECIPE, 'Require exact frozen diagnostic recipe')
    for field in ('connected_protocol_sha256','d_complete_sha256','geometry_complete_sha256',
                  'reference_complete_sha256','reference_summary_sha256','runtime_contract_sha256','selection_sha256'):
        require(p[field] == expected[field], 'Frozen asset binding differs: '+field)
    require(p['resources'] == dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib,cpu_threads=2)
            and a.max_seconds > 0 and a.max_allocated_gib > 0, 'CLI resource caps must equal frozen protocol')
    sources = expected_sources(a.connected_protocol, a.source_root)
    require(p['sources_sha256'] == sources, 'Exact source closure differs')
    require(sha(a.runtime_contract) == RUNTIME_SHA and sha(a.selection) == SELECTION_SHA, 'Runtime/selection changed')
    for path, digest in read(a.runtime_contract)['runtime_source_sha256'].items(): require(sha(path) == digest, 'Runtime source changed: '+path)
    require(sha(Path(a.d_run)/'complete.json') == D_COMPLETE_SHA, 'Fixed D completion changed')
    # Ordinary imports from bound dependencies resolve inside the immutable package.
    sys.path.insert(0, str(Path(a.source_root).resolve()))
    trainer = bind('train_connected_motion_v2', sources, a.source_root)
    helper = bind('train_source_motion_v1', sources, a.source_root)
    native = bind('native_state_cache', sources, a.source_root)
    cache_module = bind('shared_rigid_geometry_cache_v1', sources, a.source_root)
    connected = read(a.connected_protocol)
    cache_root, selected, cache_receipt = helper.cache_index(a.dev_cache, 'development', connected)
    require(len({r['scene_token'] for r in selected}) == 100, 'Fixed 100 scenes required')
    geometry = cache_module.GeometryCache(a.geometry_cache, GEOMETRY_SHA, a.selection)
    require([identity(r) for r in selected] == geometry.chosen['development'], 'Native cache/geometry selection differs')
    label_root, descriptors = helper.labels_manifest(a.sparse_labels, connected)
    refs, covrefs, anchorrefs = reference(a.cv_reference)
    if a.check:
        require('torch' not in sys.modules, 'Metadata check must not import Torch')
        write(out/'check.json', dict(schema=SCHEMA,status='AUTHENTICATED_METADATA_ONLY',
            protocol_sha256=sha(a.protocol),sources_sha256=sources,
            source_paths={n:str(source_path(n,a.source_root)) for n in sources},
            geometry_cache=geometry.receipt,native_cache=cache_receipt,
            sparse_label_manifest_sha256=connected['labels']['manifest_sha256'],
            reference_complete_sha256=REFERENCE_COMPLETE_SHA,reference_summary_sha256=REFERENCE_SUMMARY_SHA,
            samples=200,scenes=100,reference_object_horizon_rows=16074,
            torch_imported=False,GPU_used=False,D_payload_loaded=False,D_loader_executed=False,
            actual_per_sample_geometry_and_input_bytes_checked=False,
            optimizer_updates=0,seconds=time.monotonic()-started))
        return
    import torch
    require(str(a.device).startswith('cuda') and torch.cuda.is_available(), 'Expected original CUDA D runtime')
    torch.set_num_threads(2); torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(min(1., a.max_allocated_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory), a.device)
    torch.cuda.reset_peak_memory_stats(a.device)
    native._prepare_repo(a.repo)  # Register only native pickle classes; never build O.
    D, unused_gate, loader_receipt = trainer.load_completed_arm(Path(a.d_run)/'runs/D', a.connected_protocol, helper, a.device)
    del unused_gate
    initial_digest = helper.state_digest(D)
    require(not D.training and all(not q.requires_grad for q in D.parameters()), 'D must be frozen eval')
    maxima, rows, anchors, seen = {}, [], [], set()
    manifest = dict(schema=SCHEMA, protocol_sha256=sha(a.protocol), sources_sha256=sources,
        source_paths={n:str(source_path(n,a.source_root)) for n in sources}, selection_sha256=SELECTION_SHA,
        runtime_contract_sha256=RUNTIME_SHA, connected_protocol_sha256=CONNECTED_SHA,
        D_loader=loader_receipt, geometry_cache=geometry.receipt, native_cache=cache_receipt,
        reference_complete_sha256=REFERENCE_COMPLETE_SHA, reference_summary_sha256=REFERENCE_SUMMARY_SHA,
        recipe=RECIPE, resources=p['resources'], optimizer_updates=0,
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,device=str(a.device),
                         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    write(out/'manifest.json',manifest)
    def budget():
        require(time.monotonic()-started < a.max_seconds, 'Wall-clock ceiling exceeded')
        require(torch.cuda.max_memory_allocated(a.device) <= a.max_allocated_gib*2**30, 'GPU allocated ceiling exceeded')
    for ordinal, record in enumerate(selected):
        budget(); tick=time.monotonic(); ident=identity(record)
        g=geometry.load('development',ordinal,ident)
        tokens=helper.load_tokens(cache_root,record,native,a.device)
        with torch.inference_mode(): prediction=D(tokens)
        require(tuple(prediction.shape)==(1,4,3,200,200,16) and prediction.dtype==torch.float32 and
                bool(torch.isfinite(prediction).all()), 'D dense field invalid')
        # All 640000 cells, nominal h only; never label dt or sparse support here.
        cv=np.stack([g['cv_velocity_R']*h for h in HORIZONS],axis=0)
        d_full=prediction[0].reshape(4,3,-1).permute(0,2,1).detach().cpu().numpy().astype(np.float64)
        blend=np.where((g['owner']>=0)[None,:,None],cv,d_full)
        require(cv.shape==blend.shape==(4,640000,3) and np.isfinite(cv).all() and np.isfinite(blend).all(), 'CV/blend dense fields invalid')
        cv_tensor=torch.from_numpy(cv).permute(0,2,1).reshape(1,4,3,200,200,16)
        blend_tensor=torch.from_numpy(blend).permute(0,2,1).reshape(1,4,3,200,200,16)
        # First future-label content read: all three fields already exist.
        desc=descriptors[record['sample_token']]; label=helper.load_sparse(label_root,desc,record)
        sparse={arm:helper.gather_sparse(field,label).detach().cpu().numpy().astype(np.float64)
                for arm,field in zip(ARMS,(prediction,cv_tensor,blend_tensor))}
        indices=label['source_flat_indices']; owners=g['owner'][indices]; original=g['owner_original_indices'][indices]
        aref=anchorrefs[ordinal]
        require(aref['ordinal']==ordinal and aref['identity']==ident and aref['source_points']==len(indices) and
            aref['covered_points']==int((owners>=0).sum()) and
            aref['boxes_owning_source_points']==len(np.unique(original[original>=0])) and
            aref['owners_original_export_index_sha256']==hashlib.sha256(original.tobytes()).hexdigest(), 'Anchor coverage/ownership differs')
        for hi,h in enumerate(HORIZONS):
            for k,instance in enumerate(label['instance_tokens'].tolist()):
                if not label['object_future_valid'][hi,k]: continue
                valid=label['valid'][hi] & (label['object_index']==k); n=int(valid.sum())
                require(n>0,'Empty valid object'); target=label['target_displacement_m'][hi,valid].astype(np.float64)
                covered=owners[valid]>=0; zero=np.linalg.norm(target,axis=1); zeroxy=np.linalg.norm(target[:,:2],axis=1)
                row=dict(sample_token=record['sample_token'],scene_token=record['scene_token'],instance_token=instance,
                    horizon_seconds=h,dt_seconds=float(label['dt_future_seconds'][hi]),group=helper.GROUPS[int(label['object_speed_group'][hi,k])],
                    source_points=n,covered_points=int(covered.sum()),zero_epe_3d_m=float(zero.mean()),zero_epe_xy_m=float(zeroxy.mean()),arms={},parts={})
                kk=key(row); require(kk not in seen and kk in covrefs,'Duplicate/extra support');seen.add(kk)
                cr=covrefs[kk]
                for field in ('scene_token','group','source_points','covered_points','dt_seconds'): require(row[field]==cr[field],'Coverage metadata differs: '+field)
                errors={arm:dict(xy=np.linalg.norm((sparse[arm][hi,valid]-target)[:,:2],axis=1),
                                 **{'3d':np.linalg.norm(sparse[arm][hi,valid]-target,axis=1)}) for arm in ARMS}
                for arm in ARMS:
                    row['arms'][arm]={'epe_'+dim+'_m':float(v.mean()) for dim,v in errors[arm].items()}
                    if arm in refs:
                        rr=refs[arm][kk]
                        for field in ('scene_token','group','source_points','dt_seconds'): require(row[field]==rr[field],'Physical metadata differs: '+field)
                        for field,value in row['arms'][arm].items():close(value,rr[field],arm+':'+str(kk)+':'+field,maxima)
                        for field in ('zero_epe_3d_m','zero_epe_xy_m'):close(row[field],rr[field],'zero:'+str(kk)+':'+field,maxima)
                for part,mask in (('covered',covered),('uncovered',~covered)):
                    count=int(mask.sum()); require(count==cr[part]['source_points'],'Partition counts differ')
                    row['parts'][part]=dict(source_points=count)
                    for arm in ARMS:
                        item={}
                        for dim,err in errors[arm].items():
                            total=float(err[mask].sum(dtype=np.float64));item[dim+'_error_sum_m']=total
                            item[dim+'_contribution_m']=total/n;item[dim+'_subset_mean_m']=total/count if count else None
                        row['parts'][part][arm]=item
                    for dim,refdim,err in (('xy','xy_m',zeroxy),('3d','xyz_m',zero)):
                        close(row['parts'][part]['CRN_CV'][dim+'_subset_mean_m'],cr[part]['CRN_CV'][refdim],'CV_subset:'+str(kk)+part+dim,maxima)
                        close(float(err[mask].mean()) if count else None,cr[part]['zero'][refdim],'zero_subset:'+str(kk)+part+dim,maxima)
                for arm in ARMS:
                    for dim in ('xy','3d'):
                        reconstructed=sum(row['parts'][part][arm][dim+'_contribution_m'] for part in ('covered','uncovered'))
                        close(reconstructed,row['arms'][arm]['epe_'+dim+'_m'],'partition_reconstruction:'+arm,maxima)
                rows.append(row)
                with (out/'objects.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        anchors.append(dict(ordinal=ordinal,identity=ident,geometry=g['receipt'],sparse_sha256=desc['sha256'],
            input_sha256=record['files']['inputs']['sha256'],source_points=len(indices),covered_points=int((owners>=0).sum()),
            seconds=time.monotonic()-tick))
        del tokens,prediction,cv,blend,d_full,cv_tensor,blend_tensor,sparse,label,g
        budget();print(json.dumps(dict(event='sample_complete',completed=ordinal+1,seconds=time.monotonic()-started)),flush=True)
    require(len(rows)==16074 and seen==set(covrefs)==set(refs['D'])==set(refs['CRN_CV']),'Incomplete original support')
    require(helper.state_digest(D)==initial_digest and all(q.grad is None for q in D.parameters()),'D state/gradient changed')
    for name,digest in sources.items():require(sha(source_path(name,a.source_root))==digest,'Source changed during evaluation')
    budget()
    summary=dict(schema=SCHEMA,status='COMPLETE_FIXED_D_COVERAGE_DIAGNOSTIC',samples=200,scenes=100,
        object_horizon_rows=16074,arms=list(ARMS),physical=summarize(rows),per_anchor=anchors,
        maximum_absolute_reference_differences_m=maxima,reference_parity_passed=True,
        denominator='all original supported points per object, then equal object mean; empty subset mean null',
        no_GT_in_forward=True,native_O_model_loaded=False,native_O_replay=False,occupancy_scored=False,
        optimizer_updates=0,D_state_unchanged=True,D_gradients_all_None=True,
        learned_candidate_or_model_selection=False,training_seed_uncertainty_measured=False,
        peak_allocated_gib=torch.cuda.max_memory_allocated(a.device)/2**30,seconds=time.monotonic()-started)
    write(out/'summary.json',summary);budget()
    write(out/'complete.json',dict(schema=SCHEMA,status=summary['status'],samples=200,scenes=100,
        object_horizon_rows=16074,optimizer_updates=0,script_sha256=sha(__file__),protocol_sha256=sha(a.protocol),
        files_sha256={n:sha(out/n) for n in ('manifest.json','objects.jsonl','summary.json')}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','connected-protocol','d-run','dev-cache','sparse-labels','geometry-cache','selection','cv-reference',
                 'repo','runtime-contract','source-root','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--max-seconds',type=int,required=True)
    parser.add_argument('--check',action='store_true',help='Authenticate metadata/sources only; no Torch or D payload loading; writes check.json, not complete.json')
    parser.add_argument('--max-allocated-gib',type=float,required=True);a=parser.parse_args(argv)
    out=Path(a.out).resolve();require(not out.exists(),'Fresh output required; no resume/retry')
    out.mkdir(parents=True);started=time.monotonic()
    def interrupted(signum,frame):raise TimeoutError('Terminated by signal '+str(signum))
    old={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    try:
        require(a.max_seconds>0,'Positive wall-clock cap required');signal.alarm(a.max_seconds)
        run(a,out,started)
    except BaseException as error:
        signal.alarm(0)
        write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(error),
            traceback=traceback.format_exc(),seconds=time.monotonic()-started,optimizer_updates=0,script_sha256=sha(__file__)))
        raise
    finally:
        signal.alarm(0)
        for s,handler in old.items():signal.signal(s,handler)


if __name__=='__main__':main()
