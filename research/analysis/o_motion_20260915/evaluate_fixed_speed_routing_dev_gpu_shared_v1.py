"""Bounded shared-GPU evaluation of all eight fixed speed routes on original dev200.

Input-only full-grid D/owner gates precede every sparse label payload read.
No fitting, policy selection, O model, or occupancy replay. Numerical agreement is checked at per-object EPE level against the frozen GPU
diagnostic, not claimed for every dense prediction. No CPU fallback is allowed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

SCHEMA = 'fixed-speed-routing-dev-gpu-v1'
STATUS = 'COMPLETE_SHARED_GPU_FIXED_SPEED_ROUTING_DEV_DIAGNOSTIC'
HORIZONS = (.5, 1., 1.5, 2.)
THRESHOLDS = [0., .1, .5, 1., 2., 5., 10., None]
GROUPS = ('all', 'stationary', 'ambiguous', 'moving')
BASELINES = ('D', 'CRN_CV', 'fixed_blend', 'zero')
GPU_ARMS = ('D', 'CRN_CV', 'CV_covered_D_uncovered')
CONNECTED_SHA = 'e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
D_COMPLETE_SHA = 'f0bf075fa36622b38c0c5136b187a184d16d3e07ed00857dfe21a0545f1c5f56'
CHECKPOINT_SHA = '7e2750d56ade7fb36e334b6431e83f51aafa997357744780a507924a696951e3'
GEOMETRY_SHA = 'a2504a0389cb2756032531961e305e22287ec890c96085e5ea29b2e61e4f1c99'
GEOMETRY_SOURCE_SHA = 'd47431242dff6d9dba10fbd791e6516739b410282e60bf7f73470186c90fcaaa'
RUNTIME_SHA = '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c'
SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
SCORER_SHA = '1e45065d38a968c4662080993f1d75e90c3d3b3e9594e774d119e770c8b3d96c'
GPU_COMPLETE_SHA = '310f1965bde29f2a0a30fae8bceaabb66de0ad97f2cc734ab4e4b46a1d2844ef'
GPU_OBJECTS_SHA = 'eefc72266ccaa3d15e6ddb2344c5a40949a9248f371c7807da55a30f7a9e86d5'
GPU_UUID = 'GPU-000b6236-3632-a001-9667-1f02cbb61c8b'
GPU_PREFLIGHT = dict(physical_index=1, uuid=GPU_UUID, minimum_free_MiB=12*1024,
    other_compute_processes_allowed=True, nonzero_utilization_allowed=True)
OPERATIONAL_AMENDMENT = dict(
    previous_protocol_sha256='616b28aefbf44a508165e1b9a6063e6c2ff0e6edea9d22cee49b822be2e0ef05',
    previous_worker_sha256='01f2873a681b101ceae0dd0bb9235839db48da81d983db3f913e976f7ef47ee3',
    user_authorization_semantics='User explicitly authorized running our experiment on H200 even while other tasks are running.',
    scope='Operational scheduling amendment only: shared execution of the existing fixed dev200 evaluation; no new training.',
    previous_physical_gpu=dict(index=0, uuid='GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe'),
    execution_physical_gpu=dict(index=1, uuid=GPU_UUID, logical_device='cuda:0'),
    change='Move physical GPU 0 to GPU 1 to separate this evaluation from RAFT on GPU 0; allow other compute PIDs and nonzero utilization; require exact GPU 1 UUID and at least 12 GiB free memory; record occupancy and processes.',
    unchanged='Eight thresholds, original samples/support/weights, scientific scoring, numerical policy/tolerances, source closure, 8 GiB allocator ceiling and all time/RSS budgets.')
RESOURCES = dict(max_seconds=1800, outer_seconds=1860, cpu_threads=2, max_allocated_gib=8, max_RSS_bytes=8*1024**3)
TOLERANCES = dict(per_object_D_or_blend_abs_EPE_m=1e-4, per_object_CV_abs_EPE_m=1e-10)
RECIPE = dict(samples=200, scenes=100, object_horizon_rows=16074, thresholds_mps=THRESHOLDS,
    horizons_seconds=list(HORIZONS), threshold_comparison='>=; None denotes positive infinity',
    coverage='owner>=0 retains CRN_CV; owner<0 uses D iff full-grid LSQ XY speed>=threshold, otherwise zero',
    LSQ='float64 sum(h*D[h],axis=0)/sum(h*h); XY norm; nominal four horizons',
    forward_order='complete 640000-cell D, float64 LSQ speed and all eight owner-aware gates before load_sparse',
    aggregation='all original legal source points equally within each object, then original objects equally',
    groups='original future-GT stationary/ambiguous/moving; grouping enters scoring only',
    numerical_reference='per-object XY/XYZ EPE versus original GPU reference only; no dense all-point or bitwise claim',
    numerical_policy=dict(dtype='float32', matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False),
    policy_selection=False, fitting=False, optimizer_updates=0, native_O_model=False, occupancy_result=False,
    comparison='train-prespecified amplitude thresholds; development diagnostic; no best policy selected')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps(dict(dtype=a.dtype.str, shape=a.shape), sort_keys=True).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def source_path(name, root):
    require(Path(name).name == name and name.endswith('.py'), 'Source must be a basename')
    root = Path(root).resolve()
    for path in (root/name, root.parent/'m0_improvement_20260915'/name):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(name)


def bind_path(name, path, digest):
    path = Path(path).resolve()
    require(sha(path) == digest, 'Bound source changed: '+name)
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path, 'Imported source path differs: '+name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bind(name, sources, root):
    return bind_path(name, source_path(name+'.py', root), sources[name+'.py'])


def expected_sources(connected_protocol, source_root):
    """Metadata-only closure for the parent dispatcher; excludes this worker."""
    require(sha(connected_protocol) == CONNECTED_SHA, 'Connected protocol differs')
    sources = dict(read(connected_protocol)['sources_sha256'])
    sources['shared_rigid_geometry_cache_v1.py'] = GEOMETRY_SOURCE_SHA
    cache = bind('shared_rigid_geometry_cache_v1', sources, source_root)
    for name, digest in cache.SOURCES.items():
        require(name not in sources or sources[name] == digest, 'Source closure conflict')
        sources[name] = digest
    for name, digest in sources.items():
        require(sha(source_path(name, source_root)) == digest, 'Source differs: '+name)
    return sources


def protocol_template():
    """Parent fills worker source hash and source closure, then sets FROZEN."""
    return dict(schema=SCHEMA, status='REVIEW_REQUIRED', source_sha256='PENDING_ROOT_FREEZE',
        sources_sha256={}, scorer_source_sha256=SCORER_SHA,
        connected_protocol_sha256=CONNECTED_SHA, d_complete_sha256=D_COMPLETE_SHA,
        checkpoint_sha256=CHECKPOINT_SHA, geometry_complete_sha256=GEOMETRY_SHA,
        runtime_contract_sha256=RUNTIME_SHA, selection_sha256=SELECTION_SHA,
        gpu_reference_complete_sha256=GPU_COMPLETE_SHA, gpu_reference_objects_sha256=GPU_OBJECTS_SHA,
        recipe=RECIPE, resources=RESOURCES, tolerances=TOLERANCES, gpu_preflight=GPU_PREFLIGHT,
        operational_amendment=OPERATIONAL_AMENDMENT)


def check_ledger(root, complete):
    for name, digest in complete['files_sha256'].items():
        require(Path(name).name == name and sha(Path(root)/name) == digest, 'Linked artifact differs: '+name)


def gpu_shared_preflight():
    """Observe shared occupancy before CUDA setup under explicit user authorization."""
    def query(arguments):
        return subprocess.run(['nvidia-smi']+arguments, check=True, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    gpu_text = query(['--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    rows = [[part.strip() for part in line.split(',')] for line in gpu_text.splitlines()]
    selected = [row for row in rows if row[1] == GPU_UUID]
    require(len(selected) == 1 and int(selected[0][0]) == GPU_PREFLIGHT['physical_index'], 'Physical GPU 1 UUID binding differs')
    require(int(selected[0][2]) >= GPU_PREFLIGHT['minimum_free_MiB'], 'Shared GPU has less than 12 GiB free memory')
    process_text = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    processes = [[part.strip() for part in line.split(',')] for line in process_text.splitlines() if line.strip()]
    target_processes = [dict(gpu_uuid=row[0], pid=int(row[1])) for row in processes if row[0] == GPU_UUID]
    return dict(physical_index=1, uuid=GPU_UUID, free_MiB=int(selected[0][2]), used_MiB=int(selected[0][3]),
                utilization_percent=int(selected[0][4]), compute_processes=len(target_processes),
                compute_process_records=target_processes, shared_execution_authorized=True,
                scope='pre-CUDA shared occupancy observation; free memory is not reserved by this check')


def reference(root):
    root = Path(root)
    require(not (root/'failed.json').exists() and sha(root/'complete.json') == GPU_COMPLETE_SHA, 'GPU reference changed')
    complete = read(root/'complete.json'); check_ledger(root, complete)
    require(complete['status'] == 'COMPLETE_FIXED_D_COVERAGE_DIAGNOSTIC' and complete['samples'] == 200 and
            complete['scenes'] == 100 and complete['object_horizon_rows'] == 16074 and
            sha(root/'objects.jsonl') == GPU_OBJECTS_SHA, 'GPU reference support differs')
    rows = [json.loads(line) for line in (root/'objects.jsonl').read_text().splitlines()]
    refs = {object_key(row): row for row in rows}
    require(len(refs) == len(rows) == 16074, 'Duplicate/missing GPU reference objects')
    return refs, read(root/'summary.json')['per_anchor']


def identity(row):
    return {k: row[k] for k in ('sample_token', 'scene_token', 'split', 'official_index')}


def object_key(row):
    return (row['sample_token'], row['instance_token'], row['horizon_seconds'])


def dense_decisions(d_full, owner, scorer):
    require(d_full.shape == (4, len(owner), 3) and np.isfinite(d_full).all() and owner.ndim == 1,
            'Dense D/owner layout differs')
    h = np.asarray(HORIZONS, dtype=np.float64)[:, None, None]
    velocity = np.sum(h*d_full.astype(np.float64), axis=0)/np.sum(h*h)
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    return speed, scorer.select_D(owner, speed)


def new_aggregates(scenes):
    counts = np.zeros((4, 3, len(scenes)), dtype=np.int64)
    totals = {name: np.zeros((4, 3, len(scenes))+shape, dtype=np.float64) for name, shape in
        [('epe', (8, 2)), ('gain', (8, 2)), ('cost', (8, 2)), ('baseline', (4, 2)),
         ('selected_count', (8,)), ('selected_mass', (8,)), ('uncovered_count', ()),
         ('uncovered_mass', ()), ('source_count', ())]}
    return counts, totals


def aggregate_sample(counts, totals, scene_index, label, scored):
    for hi in range(4):
        for gi in range(3):
            take = label['object_future_valid'][hi] & (label['object_speed_group'][hi] == gi)
            counts[hi, gi, scene_index] += int(take.sum())
            for name, out in totals.items():
                out[hi, gi, scene_index] += scored[hi][name][take].sum(axis=0)


def policy_rows(counts, totals):
    rows = []
    for hi, horizon in enumerate(HORIZONS):
        for group in GROUPS:
            chosen = [0, 1, 2] if group == 'all' else [GROUPS.index(group)-1]
            count = int(counts[hi, chosen].sum())
            require(count > 0, 'Expected original group has no objects')
            t = {name: value[hi, chosen].sum(axis=(0, 1)) for name, value in totals.items()}
            for ti, threshold in enumerate(THRESHOLDS):
                metrics = {}
                for di, dimension in enumerate(('xy', 'xyz')):
                    epe, gain, cost = (t[key][ti, di]/count for key in ('epe', 'gain', 'cost'))
                    cv, blend = t['baseline'][1, di]/count, t['baseline'][2, di]/count
                    require(np.isclose(epe-cv, cost-gain, rtol=1e-12, atol=1e-12), 'Full-denominator gain/cost identity')
                    metrics[dimension] = dict(epe_m=float(epe), minus_CV_m=float(epe-cv),
                        minus_full_blend_m=float(epe-blend), positive_gain_m=float(gain), negative_cost_m=float(cost),
                        CV_epe_m=float(cv), full_blend_epe_m=float(blend))
                rows.append(dict(horizon_seconds=horizon, group=group, threshold_mps=threshold,
                    infinite_threshold=threshold is None, original_objects=count, original_points=int(t['source_count']),
                    uncovered_points=int(t['uncovered_count']), selected_D_points=int(t['selected_count'][ti]),
                    selected_original_object_weight=float(t['selected_mass'][ti]),
                    routing_partition=dict(CV_retained=dict(points=int(t['source_count']-t['uncovered_count']),
                        original_object_weight=float(count-t['uncovered_mass'])),
                        D_selected=dict(points=int(t['selected_count'][ti]), original_object_weight=float(t['selected_mass'][ti])),
                        zero_fallback=dict(points=int(t['uncovered_count']-t['selected_count'][ti]),
                            original_object_weight=float(t['uncovered_mass']-t['selected_mass'][ti]))),
                    selected_fraction_of_uncovered_weight=float(t['selected_mass'][ti]/t['uncovered_mass']) if t['uncovered_mass'] else None,
                    metrics=metrics))
    require(len(rows) == 128, 'Expected all 128 policy/group/horizon cells')
    return rows


def compare_objects(record, label, scored, refs, seen, comparisons, stream):
    for hi, horizon in enumerate(HORIZONS):
        for k, instance in enumerate(label['instance_tokens'].tolist()):
            if not label['object_future_valid'][hi, k]:
                continue
            s = scored[hi]
            row = dict(sample_token=record['sample_token'], scene_token=record['scene_token'], instance_token=instance,
                horizon_seconds=horizon, dt_seconds=float(label['dt_future_seconds'][hi]),
                group=GROUPS[int(label['object_speed_group'][hi, k])+1], source_points=int(s['source_count'][k]),
                covered_points=int(s['source_count'][k]-s['uncovered_count'][k]))
            key = object_key(row)
            require(key in refs and key not in seen, 'Duplicate/extra object support')
            ref = refs[key]; seen.add(key)
            for name in row:
                require(row[name] == ref[name], 'Original GPU reference metadata differs: '+name)
            row['reevaluated_GPU_baselines'] = {}; row['GPU_reference_arms'] = ref['arms']
            row['GPU_reference_parts'] = ref['parts']; row['reevaluated_minus_reference_GPU_EPE_m'] = {}
            for ai, arm in enumerate(GPU_ARMS):
                row['reevaluated_GPU_baselines'][arm] = {}; row['reevaluated_minus_reference_GPU_EPE_m'][arm] = {}
                tolerance = TOLERANCES['per_object_CV_abs_EPE_m' if ai == 1 else 'per_object_D_or_blend_abs_EPE_m']
                for di, dim in enumerate(('xy', '3d')):
                    current = float(s['baseline'][k, ai, di]); gpu = float(ref['arms'][arm]['epe_'+dim+'_m'])
                    delta = current-gpu; error = abs(delta)
                    require(np.isfinite(current) and np.isfinite(gpu), 'Nonfinite per-object EPE')
                    entry = comparisons.setdefault(arm+'/'+dim, dict(count=0, sum_abs_m=0., sum_signed_m=0., max_abs_m=0., worst_object=None))
                    entry['count'] += 1; entry['sum_abs_m'] += error; entry['sum_signed_m'] += delta
                    if error > entry['max_abs_m']:
                        entry['max_abs_m'] = error; entry['worst_object'] = list(key)
                    require(error <= tolerance, 'Frozen GPU per-object reference tolerance failed: '+str(key)+' '+arm+'/'+dim+' '+str(error))
                    row['reevaluated_GPU_baselines'][arm]['epe_'+dim+'_m'] = current
                    row['reevaluated_minus_reference_GPU_EPE_m'][arm][dim] = delta
            row['routed_epe_xy_xyz_m'] = s['epe'][k].tolist()
            row['positive_gain_xy_xyz_m'] = s['gain'][k].tolist()
            row['negative_cost_xy_xyz_m'] = s['cost'][k].tolist()
            row['selected_D_points'] = s['selected_count'][k].astype(np.int64).tolist()
            row['selected_original_object_weight'] = s['selected_mass'][k].tolist()
            stream.write(json.dumps(row, allow_nan=False)+'\n')


def synthetic_qa(scorer):
    checks = scorer.synthetic_qa()
    # Run routing before creating the synthetic label indices. Covered cells
    # never use D; equality at a boundary is admitted; infinity rejects all D.
    owner = np.array([-1, 0, -1, -1], dtype=np.int64)
    velocity = np.array([[.5, 0., 0.], [10., 0., 0.], [0., 0., 2.], [2., 0., 0.]])
    d = np.asarray(HORIZONS)[:, None, None]*velocity
    speed, gates = dense_decisions(d, owner, scorer)
    indices = np.array([3, 0, 2])
    require(np.array_equal(gates[indices], scorer.select_D(owner[indices], speed[indices])), 'Dense/sparse decision mismatch')
    require(np.array_equal(speed, [.5, 10., 0., 2.]) and gates[0, 2] and not gates[1].any() and
            not gates[:, -1].any(), 'LSQ XY direction, ties or coverage differs')
    return checks+['full-grid float64 LSQ and gathered gates agree without sparse targets']


def run(a, out, started, progress):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_UUID, 'Require exact scheduled GPU UUID visibility')
    p = read(a.protocol); template = protocol_template()
    require(p['schema'] == SCHEMA and p['status'] == 'FROZEN' and p['source_sha256'] == sha(__file__), 'Worker/protocol not frozen')
    for name in template:
        if name not in ('status', 'source_sha256', 'sources_sha256'):
            require(p[name] == template[name], 'Frozen recipe/binding differs: '+name)
    sources = expected_sources(a.connected_protocol, a.source_root)
    require(p['sources_sha256'] == sources, 'Frozen source closure differs')
    require(sha(a.runtime_contract) == RUNTIME_SHA and sha(a.selection) == SELECTION_SHA, 'Runtime/selection differs')
    for path, digest in read(a.runtime_contract)['runtime_source_sha256'].items():
        require(sha(path) == digest, 'Runtime source changed: '+path)
    require(sha(a.d_run/'complete.json') == D_COMPLETE_SHA, 'Fixed D completion differs')
    scorer = bind_path('fixed_speed_routing_train_bound_scorer', a.scorer, SCORER_SHA)
    require(scorer.HORIZONS == HORIZONS and scorer.GROUPS == GROUPS and scorer.BASELINES == BASELINES and
            np.array_equal(scorer.THRESHOLDS, [0., .1, .5, 1., 2., 5., 10., np.inf]), 'Scorer conventions differ')
    qa = synthetic_qa(scorer)
    sys.path.insert(0, str(a.source_root.resolve()))
    trainer = bind('train_connected_motion_v2', sources, a.source_root)
    helper = bind('train_source_motion_v1', sources, a.source_root)
    native = bind('native_state_cache', sources, a.source_root)
    cache_module = bind('shared_rigid_geometry_cache_v1', sources, a.source_root)
    connected = read(a.connected_protocol)
    cache_root, selected, cache_receipt = helper.cache_index(a.dev_cache, 'development', connected)
    scenes = sorted({r['scene_token'] for r in selected})
    require(len(selected) == 200 and len(scenes) == 100, 'Expected fixed dev200/100 scenes')
    geometry = cache_module.GeometryCache(a.geometry_cache, GEOMETRY_SHA, a.selection)
    require([identity(r) for r in selected] == geometry.chosen['development'], 'Native/geometry selection differs')
    label_root, descriptors = helper.labels_manifest(a.sparse_labels, connected)
    refs, anchorrefs = reference(a.gpu_reference)
    require(len(anchorrefs) == 200, 'GPU reference anchor support differs')
    preflight_receipt = gpu_shared_preflight()
    import torch
    require(torch.cuda.is_available(), 'Original CUDA runtime required; CPU fallback forbidden')
    device = 'cuda:0'
    torch.set_num_threads(2); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(min(1., RESOURCES['max_allocated_gib']*2**30/
        torch.cuda.get_device_properties(device).total_memory), device)
    torch.cuda.reset_peak_memory_stats(device)
    runtime = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, device=device,
        threads=2, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_name=torch.cuda.get_device_name(device), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark)
    native._prepare_repo(a.repo)
    D, gate, loader = trainer.load_completed_arm(a.d_run/'runs/D', a.connected_protocol, helper, device)
    del gate
    require(loader['checkpoint_sha256'] == CHECKPOINT_SHA and not D.training and
            all(not q.requires_grad and q.device.type == 'cuda' for q in D.parameters()), 'D must be fixed GPU eval')
    initial_digest = helper.state_digest(D)
    counts, totals = new_aggregates(scenes)
    records = []; seen = set(); comparisons = {}; source_points = 0; valid_points = np.zeros(4, dtype=np.int64)
    manifest = dict(schema=SCHEMA, source_sha256=sha(__file__), protocol_sha256=sha(a.protocol), protocol=p,
        source_paths={n: str(source_path(n, a.source_root)) for n in sources}, scorer_path=str(a.scorer.resolve()),
        D_loader=loader, geometry_cache=geometry.receipt, native_cache=cache_receipt,
        sparse_label_manifest_sha256=connected['labels']['manifest_sha256'], runtime=runtime, gpu_preflight=preflight_receipt,
        GPU_reference_complete_sha256=GPU_COMPLETE_SHA, GPU_reference_objects_sha256=GPU_OBJECTS_SHA,
        reference_scope='original dev200 GPU per-object EPE, not dense all-point parity')
    write(out/'manifest.json', manifest); (out/'samples').mkdir()
    def budget():
        require(time.monotonic()-started < RESOURCES['max_seconds'], 'Internal 1800-second ceiling exceeded')
        require(rss_bytes() <= RESOURCES['max_RSS_bytes'], 'Host peak RSS exceeds 8 GiB')
        require(torch.cuda.max_memory_allocated(device) <= RESOURCES['max_allocated_gib']*2**30, 'GPU allocated ceiling exceeded')
    with (out/'objects.jsonl').open('x') as object_stream:
        for ordinal, record in enumerate(selected):
            budget(); tick = time.monotonic(); ident = identity(record)
            progress.update(phase='dense_input_only_forward', ordinal=ordinal, identity=ident)
            g = geometry.load('development', ordinal, ident)
            tokens = helper.load_tokens(cache_root, record, native, device)
            with torch.inference_mode():
                prediction = D(tokens)
            require(tuple(prediction.shape) == (1, 4, 3, 200, 200, 16) and prediction.dtype == torch.float32 and
                    bool(torch.isfinite(prediction).all()), 'Invalid D dense output')
            d_full = prediction[0].reshape(4, 3, -1).permute(0, 2, 1).detach().cpu().numpy().astype(np.float64)
            owner = g['owner']
            require(owner.shape == (640000,) and g['cv_velocity_R'].shape == (640000, 3) and
                    np.isfinite(g['cv_velocity_R']).all() and np.all(g['cv_velocity_R'][owner < 0] == 0), 'Full CV/owner layout differs')
            speed, full_gates = dense_decisions(d_full, owner, scorer)
            require(full_gates.shape == (640000, 8), 'All 640000 cells require eight fixed gates')
            dense_receipt = dict(D_float64_sha256=array_sha(d_full), LSQ_speed_sha256=array_sha(speed),
                owner_sha256=array_sha(owner), gates_sha256=array_sha(full_gates), full_grid_cells=640000,
                sparse_labels_opened=False, target_or_group_arrays_used=False)
            budget()
            # First sparse future-label payload read for this sample. All eight
            # dense gates above are fixed and cannot depend on support or labels.
            progress['phase'] = 'sparse_scoring_and_reference_check'
            desc = descriptors[record['sample_token']]
            label = helper.load_sparse(label_root, desc, record)
            indices = label['source_flat_indices']
            d_sparse = helper.gather_sparse(prediction, label).detach().cpu().numpy().copy()
            cv_sparse = g['cv_velocity_R'][indices].copy(); sparse_owner = owner[indices].copy()
            sparse_speed = speed[indices].copy(); gathered_gates = full_gates[indices].copy()
            require(np.array_equal(d_sparse.astype(np.float64), d_full[:, indices]) and
                    np.array_equal(gathered_gates, scorer.select_D(sparse_owner, sparse_speed)), 'Dense/sparse scoring decisions differ')
            v = dict(label, D_displacement_m=d_sparse, CV_velocity_mps=cv_sparse, owner=sparse_owner,
                     observable_features=sparse_speed[:, None])
            scored = scorer.score_sample(v)
            aref = anchorrefs[ordinal]
            require(aref['ordinal'] == ordinal and aref['identity'] == ident and aref['source_points'] == len(indices) and
                    aref['covered_points'] == int((sparse_owner >= 0).sum()) and aref['sparse_sha256'] == desc['sha256'] and
                    aref['input_sha256'] == record['files']['inputs']['sha256'], 'Original anchor support/input differs')
            compare_objects(record, label, scored, refs, seen, comparisons, object_stream)
            aggregate_sample(counts, totals, scenes.index(record['scene_token']), label, scored)
            source_points += len(indices); valid_points += label['valid'].sum(axis=1)
            payload = dict(label)
            added = dict(D_displacement_m=d_sparse, CV_velocity_mps=cv_sparse, owner=sparse_owner,
                owner_original_indices=g['owner_original_indices'][indices], LSQ_speed_xy_mps=sparse_speed,
                selected_D_mask=gathered_gates)
            require(not (set(payload) & set(added)), 'Do not overwrite any original label array')
            payload.update(added)
            relative = 'samples/%04d_%s.npz' % (ordinal, record['sample_token'])
            with (out/relative).open('xb') as stream:
                np.savez_compressed(stream, **payload)
            records.append(dict(ordinal=ordinal, identity=ident, file=relative, sha256=sha(out/relative),
                sparse_label_sha256=desc['sha256'], input_sha256=record['files']['inputs']['sha256'], geometry=g['receipt'],
                full_grid_before_labels=dense_receipt, gathered_gates_equal_scorer=True,
                original_label_arrays={name: dict(dtype=value.dtype.str, shape=list(value.shape), sha256=array_sha(value))
                                       for name, value in label.items()},
                source_points=len(indices), covered_points=int((sparse_owner >= 0).sum()), seconds=time.monotonic()-tick))
            progress.update(completed_samples=ordinal+1, object_horizon_rows=len(seen), comparisons=comparisons)
            del tokens, prediction, d_full, speed, full_gates, g, label, payload, added, v, scored
            budget()
            print(json.dumps(dict(event='sample_complete', completed=ordinal+1, seconds=time.monotonic()-started,
                                  peak_RSS_bytes=rss_bytes())), flush=True)
    require(len(records) == 200 and len(seen) == int(counts.sum()) == 16074 and seen == set(refs), 'Original complete support differs')
    require(helper.state_digest(D) == initial_digest and all(q.grad is None for q in D.parameters()), 'D state/gradients changed')
    for name, digest in sources.items():
        require(sha(source_path(name, a.source_root)) == digest, 'Source changed during evaluation')
    require(sha(a.scorer) == SCORER_SHA and sha(a.protocol) == manifest['protocol_sha256'] and
            sha(__file__) == p['source_sha256'], 'Worker/scorer/protocol changed during evaluation')
    for entry in comparisons.values():
        require(entry['count'] == 16074, 'Reference comparison support differs')
        entry['mean_abs_m'] = entry['sum_abs_m']/entry['count']
        entry['mean_signed_m'] = entry['sum_signed_m']/entry['count']
    policies = policy_rows(counts, totals)
    with (out/'scene_sums.npz').open('xb') as stream:
        np.savez_compressed(stream, counts=counts, scene_tokens=np.asarray(scenes), **totals)
    write(out/'index.json', dict(schema=SCHEMA, records=records))
    summary = dict(schema=SCHEMA, status=STATUS, samples=200, scenes=100, source_points=source_points,
        object_horizon_rows=16074, original_objects_by_horizon=counts.sum(axis=(1, 2)).tolist(),
        original_valid_points_by_horizon=valid_points.tolist(), policies=policies, recipe=RECIPE,
        GPU_per_object_EPE_reference_comparison=comparisons, tolerances=TOLERANCES, all_reference_tolerances_passed=True,
        scene_sums=dict(file='scene_sums.npz', sha256=sha(out/'scene_sums.npz'),
            axes='horizon,stationary/ambiguous/moving,scene,policy/arm,xy/xyz'),
        verification=dict(full_grid_gates_before_sparse_labels=True, all_gathered_gates_equal_scorer=True,
            all_original_label_arrays_preserved=True, complete_original_support_verified=True,
            training=False, optimizer_updates=0, threshold_selected=False, O_model_constructed=False,
            occupancy_result=False, D_state_unchanged=True,
            bitwise_reference_parity_claimed=False, all_point_reference_parity_claimed=False),
        synthetic_checks=qa, peak_RSS_bytes=rss_bytes(),
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device), seconds=time.monotonic()-started)
    write(out/'summary.json', summary); budget()
    write(out/'complete.json', dict(schema=SCHEMA, status=STATUS, samples=200, scenes=100, object_horizon_rows=16074,
        source_sha256=sha(__file__), scorer_source_sha256=SCORER_SHA, protocol_sha256=sha(a.protocol),
        files_sha256={name: sha(out/name) for name in ('manifest.json', 'index.json', 'objects.jsonl', 'scene_sums.npz', 'summary.json')}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'connected-protocol', 'source-root', 'scorer', 'd-run', 'dev-cache', 'sparse-labels',
                 'geometry-cache', 'selection', 'gpu-reference', 'runtime-contract', 'repo', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args(argv); out = a.out.resolve()
    require(not out.exists(), 'Fresh output required; no resume or overwrite')
    out.mkdir(parents=True); started = time.monotonic(); progress = dict(phase='authentication', completed_samples=0)
    def stop(signum, frame):
        raise TimeoutError('Bounded original-GPU evaluation signal '+str(signum))
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    try:
        signal.alarm(RESOURCES['max_seconds'])
        run(a, out, started, progress)
    except BaseException as error:
        signal.alarm(0)
        write(out/'failed.json', dict(schema=SCHEMA, status='FAILED_NO_RETRY', source_sha256=sha(__file__),
            error=repr(error), traceback=traceback.format_exc(), progress=progress,
            seconds=time.monotonic()-started, peak_RSS_bytes=rss_bytes(), training=False, optimizer_updates=0))
        raise
    finally:
        signal.alarm(0)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    main()
