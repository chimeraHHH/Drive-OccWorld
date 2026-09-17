"""Frozen-M0 numerical diagnosis; no optimizer, training, or source mutation.

Compare native1 and fresh persistent2 on two distinct cached train scenes under
the 2x2 CUDA matmul/cuDNN TF32 policies. A native repeat measures the numerical
floor. The original 1e-4 absolute/relative parity criterion is never relaxed.
All five horizons/three decoder outputs are recorded, but only t0 and the first
future step are native/duplicate equivalence claims: later memory differs.

First-step hooks observe (never replace) activations. Expanded projections are
unpacked in head/level/point order before comparing each level against native.
The input to output_proj is the native deformable-attention reduction, before
the shared output projection. Softmax is reconstructed from captured logits
for diagnosis only; its output is never supplied to the model. No large trace
arrays are written. Source: frozen observation_memory.py and native_state_cache.py.
"""
import argparse
import copy
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback


RTOL = ATOL = 1e-4
POLICIES = [('TT', True, True), ('FF', False, False),
            ('TF', True, False), ('FT', False, True)]
ADAPTER_SHA = '0f01a29dc79daf2f0abb42bd2a23f8d168606cd3928212229ee5aeb3f94a5733'
CACHE_CODE_SHA = '41953909ddfdfd08d51d98385a450640e5314b775ab01458cb55ff1b18e4bfb0'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def phase(name, **kwargs):
    print(json.dumps(dict(phase=name, utc=utc(), **kwargs), allow_nan=False), flush=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def difference(reference, candidate):
    """Bounded-memory FP64 CPU comparison, with torch.assert_close semantics."""
    import numpy as np
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    require(reference.shape == candidate.shape, 'Comparison shape mismatch')
    left, right = reference.reshape(-1), candidate.reshape(-1)
    count, changed, violations, absolute, squared, maximum = left.size, 0, 0, 0., 0., 0.
    for start in range(0, count, 262144):
        x = left[start:start + 262144].astype(np.float64)
        y = right[start:start + 262144].astype(np.float64)
        require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite activation')
        d = np.abs(y - x)
        changed += int(np.count_nonzero(d))
        violations += int(np.count_nonzero(d > ATOL + RTOL * np.abs(x)))
        absolute += float(d.sum()); squared += float((d * d).sum())
        maximum = max(maximum, float(d.max(initial=0.)))
    return dict(shape=list(reference.shape), elements=int(count), changed=changed,
                max_abs=maximum, mean_abs=absolute / max(1, count),
                rms=math.sqrt(squared / max(1, count)), violations=violations,
                violation_fraction=violations / max(1, count),
                rtol=RTOL, atol=ATOL, pass_original_tolerance=violations == 0)


def parts(tensor, kind, levels=1, heads=None, points=None):
    """Canonical slot slices; works with NumPy and torch without importing torch."""
    if kind == 'plain':
        return [tensor]
    if kind == 'memory':
        require(tensor.shape[1] == levels, 'Memory dimension')
        return [tensor[:, i] for i in range(levels)]
    if kind == 'references':
        require(tensor.shape[2] == levels, 'Reference dimension')
        return [tensor[:, :, i] for i in range(levels)]
    if kind == 'value':
        b, total, c = tensor.shape
        require(total % levels == 0, 'Value sequence dimension')
        reshaped = tensor.reshape(b, levels, total // levels, c)
        return [reshaped[:, i] for i in range(levels)]
    require(kind in ('offset', 'attention', 'softmax_twice'), 'Unknown trace kind')
    b, n, _ = tensor.shape
    if kind == 'offset':
        reshaped = tensor.reshape(b, n, heads, levels, points, 2)
    else:
        reshaped = tensor.reshape(b, n, heads, levels, points)
    return [reshaped[:, :, :, i] for i in range(levels)]


class FirstStepTrace:
    """One native CPU trace, then stream comparisons from the duplicate model."""
    def __init__(self):
        self.reference = {}
        self.order = []
        self.rows = []
        self.active = False
        self.handles = []
        self.seen = []

    def capture(self, tag, tensor, kind='plain', levels=1, heads=None, points=None):
        if not self.active:
            return
        slices = parts(tensor, kind, levels, heads, points)
        if self.native:
            require(tag not in self.reference and len(slices) == 1, 'Duplicate native trace event')
            self.reference[tag] = slices[0].detach().cpu().numpy().copy()
            self.order.append(tag)
        else:
            require(tag in self.reference and tag not in self.seen, 'Unmatched duplicate trace event')
            self.seen.append(tag)
            values = [x.detach().cpu().numpy() for x in slices]
            row = dict(event=tag, raw_shape=list(tensor.shape), kind=kind,
                       native_vs_slots=[difference(self.reference[tag], x) for x in values])
            if len(values) == 2:
                row['within_duplicate_slots'] = difference(values[0], values[1])
            self.rows.append(row)

    def attach(self, model, native):
        from observation_memory import _cross_attention_modules
        import torch
        self.native = native
        head = model.future_pred_head
        levels = head.memory_queue_len

        def head_pre(module, args, kwargs):
            def arg(i, key):
                return args[i] if len(args) > i else kwargs[key]
            self.active = int(arg(2, 'target_frame_index')) == 1
            self.capture('head.input.memory', arg(0, 'prev_feats'), 'memory', levels)
            self.capture('head.input.reference_points', arg(6, 'ref_points'), 'references', levels)
            self.capture('head.input.target_points', arg(5, 'tgt_points'))

        def head_post(module, args, output):
            self.capture('head.output.features', output[0])
            self.active = False

        self.handles.append(head.register_forward_pre_hook(head_pre, with_kwargs=True))
        self.handles.append(head.register_forward_hook(head_post))

        def neck_post(module, args, output):
            self.capture('prev_render_neck.output', output['bev_embed'], 'memory', levels)
        self.handles.append(head.prev_render_neck.register_forward_hook(neck_post))

        def post(tag, kind='plain', nlevels=1, heads=None, points=None):
            def hook(module, args, output):
                self.capture(tag, output, kind, nlevels, heads, points)
            return hook

        for i, layer in enumerate(head.transformer.decoder.layers):
            for j, attention in enumerate(layer.attentions):
                tag = 'layer%d.attention%d.output' % (i, j)
                self.handles.append(attention.register_forward_hook(post(tag)))
            self.handles.append(layer.register_forward_hook(post('layer%d.output' % i)))

        for name, module in _cross_attention_modules(head):
            nheads, npoints = module.num_heads, module.num_points
            for field, kind in [('value_proj', 'value'), ('sampling_offsets', 'offset'),
                                ('attention_weights', 'attention'), ('output_proj', 'plain')]:
                tag = name + '.' + field
                def pre(linear, args, tag=tag, field=field, nlevels=levels):
                    input_kind = 'value' if field == 'value_proj' else 'plain'
                    suffix = '.deformable_sum' if field == 'output_proj' else '.input'
                    self.capture(tag + suffix, args[0], input_kind, nlevels)
                self.handles.append(getattr(module, field).register_forward_pre_hook(pre))
                self.handles.append(getattr(module, field).register_forward_hook(
                    post(tag + '.output', kind, levels, nheads, npoints)))
                if field == 'attention_weights':
                    def softmax_hook(linear, args, output, tag=tag, h=nheads, p=npoints, m=levels):
                        if self.active:
                            b, n, _ = output.shape
                            # Native sums to one; each identical duplicate slot to one half.
                            weights = output.reshape(b, n, h, m * p).softmax(-1) * m
                            self.capture(tag + '.softmax_times_levels',
                                         weights.reshape(b, n, -1), 'softmax_twice', m, h, p)
                    self.handles.append(getattr(module, field).register_forward_hook(softmax_hook))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active = False

    def result(self):
        require(self.order == self.seen and self.rows, 'Incomplete or reordered trace')
        nonzero = next((r['event'] for r in self.rows
                        if any(x['changed'] for x in r['native_vs_slots'])), None)
        violation = next((r['event'] for r in self.rows
                          if any(x['violations'] for x in r['native_vs_slots'])), None)
        evidence = []
        by_name = {r['event']: r for r in self.rows}
        for row in self.rows:
            name = row['event']
            if name.endswith('.output') and any(row['native_vs_slots'][i]['changed']
                                               for i in range(len(row['native_vs_slots']))):
                before = by_name.get(name[:-7] + '.input')
                if before and all(x['changed'] == 0 for x in before['native_vs_slots']):
                    evidence.append(dict(projection=name, same_corresponding_input=True,
                        output_changed=True, interpretation='Shape-dependent projection arithmetic observed; does not identify a particular CUDA kernel.'))
        return dict(first_nonzero_event=nonzero, first_tolerance_violation=violation,
                    projection_localization=evidence, rows=self.rows,
                    limitation='Observed hook boundaries, not a replacement implementation or proof that all unobserved intermediates agree.')


def seed():
    import numpy as np
    import torch
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(11)


def run_forward(model, sample, device, evaluate=True):
    import numpy as np
    import torch
    from native_state_cache import replay
    seed()
    if device.type == 'cuda':
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    with torch.no_grad():
        output = replay(model, sample, training=False)[0]
        require(tuple(output.shape) == (5, 3, 1, 1, 40000, 16, 2), 'Unexpected native logits layout')
        pred = output.detach().cpu().numpy().copy()
        forward_seconds = time.monotonic() - start
        rows = None
        if evaluate:
            rows = model.evaluate_occ_records(output, sample['targets'], sample['inputs']['img_metas'])
            require(len(rows) == 1 and rows[0]['sample_token'] == sample['record']['sample_token'], 'Evaluator identity')
            rows = dict(rows[0]); rows['hist_by_horizon'] = np.asarray(rows['hist_by_horizon']).tolist()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    receipt = dict(forward_seconds=forward_seconds, total_seconds=time.monotonic() - start,
                   peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None,
                   evaluation=rows)
    return pred, receipt


def logits_report(reference, candidate):
    return [dict(horizon_index=h, seconds=h * .5, layer=layer,
                 equivalent_by_construction=h <= 1, **difference(reference[h, layer], candidate[h, layer]))
            for h in range(5) for layer in range(3)]


def self_test():
    import numpy as np
    x = np.array([0., 1., -2.], dtype=np.float32)
    require(difference(x, x)['changed'] == 0, 'Exact floor')
    y = x.copy(); y[0] = .001
    require(difference(x, y)['violations'] == 1, 'Tolerance detection')
    original = np.arange(2 * 3 * 2 * 4 * 2).reshape(2, 3, 2, 1, 4, 2)
    duplicate = np.repeat(original, 2, axis=3)
    native_parts = parts(original.reshape(2, 3, -1), 'offset', 1, 2, 4)
    require(all(np.array_equal(p, native_parts[0]) for p in
                parts(duplicate.reshape(2, 3, -1), 'offset', 2, 2, 4)), 'Head/level/point map')
    wrong = np.concatenate([original.reshape(2, 3, -1)] * 2, axis=-1)
    require(not np.array_equal(parts(wrong, 'offset', 2, 2, 4)[0], native_parts[0]), 'Reject naive whole-row duplication')
    memory = np.arange(2 * 5 * 3).reshape(2, 1, 5, 3)
    require(all(np.array_equal(p, memory[:, 0]) for p in
                parts(np.repeat(memory, 2, 1).reshape(2, 10, 3), 'value', 2)), 'Value sequence map')
    print(json.dumps(dict(status='PASS_CPU_HELPERS', tests=['tolerance', 'head_level_point', 'naive_repeat_counterexample', 'value_slots'])))


def diagnose(args):
    import numpy as np
    import torch
    from memory_experiment import cached_records
    from native_state_cache import build_native_model, load_sample
    from observation_memory import install_observation_memory
    out = Path(args.out)
    package = Path(__file__).resolve().parent
    source_files = [Path(__file__), package / 'observation_memory.py', package / 'native_state_cache.py',
                    package / 'memory_experiment.py', Path(args.config), Path(args.checkpoint),
                    Path(args.cache) / 'index.json', Path(args.cache) / 'complete.json']
    sources = {str(p.resolve()): sha(p) for p in source_files}
    require(sha(package / 'observation_memory.py') == ADAPTER_SHA, 'Frozen adapter version changed')
    require(sha(package / 'native_state_cache.py') == CACHE_CODE_SHA, 'Frozen cache implementation changed')
    records = cached_records(args.cache)[:2]
    require(len(records) == 2 and records[0]['scene_token'] != records[1]['scene_token'], 'Need two distinct preflight scenes')
    require(all(r['split'] == 'train' for r in records), 'Diagnostic anchors must be train split')
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda'), 'Only CPU/CUDA supported')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False  # Match the frozen preflight policy.
    policies = POLICIES if args.combinations == 'all' else POLICIES[:2]
    manifest = dict(schema='m0-memory-numerical-diagnosis-v1', started_at=utc(), sources_sha256=sources,
        selected_records=[{k:r[k] for k in ['sample_token','scene_token','official_index','split']} for r in records],
        torch_version=torch.__version__, cuda_version=torch.version.cuda, device=str(device),
        gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        policies=[dict(name=n,matmul_tf32=m,cudnn_tf32=c) for n,m,c in policies],
        trace_policies=['TT','FF'], seed=11, optimizer_steps=0,
        cuda_environment={k:os.environ.get(k) for k in ['NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG']},
        criterion=dict(rtol=RTOL,atol=ATOL,horizons_equivalent=[0,.5]),
        scope='Frozen cached t0 observation with native future ego/actions, no gradients or fitting.')
    write(out / 'manifest.json', manifest)
    phase('STRICT_NATIVE_LOAD')
    seed()
    native = build_native_model(args.config, args.checkpoint, device=str(device), repo=args.repo)
    duplicate = copy.deepcopy(native)
    migrations = dict(native1=install_observation_memory(native, 'native1'),
                      persistent2=install_observation_memory(duplicate, 'persistent2'))
    for model in [native, duplicate]:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    write(out / 'migration.json', migrations)
    results, traces, policy_comparisons = [], [], []
    tt_native = {}
    for policy_name, matmul, cudnn in policies:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        for ordinal, record in enumerate(records):
            phase('POLICY_SAMPLE', policy=policy_name, ordinal=ordinal, token=record['sample_token'])
            sample = load_sample(args.cache, record, device=str(device))
            p_native, r_native = run_forward(native, sample, device)
            p_repeat, r_repeat = run_forward(native, sample, device)
            p_duplicate, r_duplicate = run_forward(duplicate, sample, device)
            repeat_rows = logits_report(p_native, p_repeat)
            duplicate_rows = logits_report(p_native, p_duplicate)
            hist_native = np.asarray(r_native['evaluation']['hist_by_horizon'], dtype=np.int64)
            hist_repeat = np.asarray(r_repeat['evaluation']['hist_by_horizon'], dtype=np.int64)
            hist_duplicate = np.asarray(r_duplicate['evaluation']['hist_by_horizon'], dtype=np.int64)
            require(np.array_equal(hist_native.sum(-1), hist_duplicate.sum(-1)) and
                    np.array_equal(hist_native.sum(-1), hist_repeat.sum(-1)), 'Evaluator GT row counts changed')
            row = dict(policy=policy_name, ordinal=ordinal, sample_token=record['sample_token'],
                native_repeat=repeat_rows, native_vs_persistent=duplicate_rows,
                first_step_parity=all(x['pass_original_tolerance'] for x in duplicate_rows if x['horizon_index'] <= 1),
                receipts=dict(native=r_native,native_repeat=r_repeat,persistent2=r_duplicate),
                hist_repeat_minus_native=(hist_repeat-hist_native).tolist(),
                hist_persistent_minus_native=(hist_duplicate-hist_native).tolist())
            results.append(row)
            if policy_name == 'TT':
                tt_native[ordinal] = (p_native.copy(), hist_native.copy())
            if policy_name == 'FF':
                policy_comparisons.append(dict(ordinal=ordinal, sample_token=record['sample_token'],
                    contrast='native_FF_minus_native_TT', logits=logits_report(tt_native[ordinal][0],p_native),
                    hist_delta=(hist_native-tt_native[ordinal][1]).tolist()))
                del tt_native[ordinal]
            write(out / 'progress.json', dict(status='RUNNING',results=results,traces=traces,
                                              native_precision_policy_comparisons=policy_comparisons))
            phase('PARITY_RESULT', policy=policy_name, ordinal=ordinal, first_step_parity=row['first_step_parity'])
            if policy_name in ('TT', 'FF'):
                phase('FIRST_STEP_TRACE', policy=policy_name, ordinal=ordinal)
                trace = FirstStepTrace()
                trace.attach(native, native=True)
                try:
                    traced_native, _ = run_forward(native, sample, device, evaluate=False)
                finally:
                    trace.detach()
                trace.attach(duplicate, native=False)
                try:
                    traced_duplicate, _ = run_forward(duplicate, sample, device, evaluate=False)
                finally:
                    trace.detach()
                report = dict(policy=policy_name, ordinal=ordinal, sample_token=record['sample_token'],
                    trace_vs_unhooked_native=difference(p_native, traced_native),
                    trace_vs_unhooked_persistent=difference(p_duplicate, traced_duplicate), **trace.result())
                traces.append(report)
                write(out / ('trace_%s_%d.json' % (policy_name,ordinal)),report)
                del trace,traced_native,traced_duplicate
            del p_native,p_repeat,p_duplicate,sample
            gc.collect()
    by_policy = {name:all(r['first_step_parity'] for r in results if r['policy']==name) for name,_,_ in policies}
    numerical_floor_exact = all(x['changed']==0 for r in results for x in r['native_repeat'])
    result = dict(schema=manifest['schema'],status='COMPLETE_DIAGNOSIS',completed_at=utc(),
        first_step_parity_by_policy=by_policy,native_repeat_bitwise=numerical_floor_exact,
        results=results,traces=traces,native_precision_policy_comparisons=policy_comparisons,
        conclusions=dict(all_disabled_resolves_tested_parity=by_policy.get('FF'),
            projection_shape_localization=[dict(policy=t['policy'],ordinal=t['ordinal'],evidence=t['projection_localization']) for t in traces],
            interpretation='A resolved parity test is numerical engineering evidence on these two anchors, not a performance result. TF32 flag effects do not alone identify an internal CUDA kernel. Any production precision change must be applied to all arms and re-preflighted; this script changes no frozen configuration.'),
        optimizer_steps=0,source_manifest_sha256=sha(out/'manifest.json'))
    write(out/'result.json',result)
    write(out/'complete.json',dict(status='COMPLETE_DIAGNOSIS',result_sha256=sha(out/'result.json'),
                                  first_step_parity_by_policy=by_policy,optimizer_steps=0))
    phase('COMPLETE_DIAGNOSIS', first_step_parity_by_policy=by_policy)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config'); p.add_argument('--checkpoint'); p.add_argument('--cache')
    p.add_argument('--out'); p.add_argument('--repo'); p.add_argument('--device',default='cuda:0')
    p.add_argument('--combinations',choices=['all','joint'],default='all')
    p.add_argument('--max-seconds',type=int,default=900)
    p.add_argument('--self-test',action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.self_test:
        self_test(); return
    require(all(getattr(args,k) for k in ['config','checkpoint','cache','out']), 'Required: --config --checkpoint --cache --out')
    require(1 <= args.max_seconds <= 1800, 'Bounded wall-time required')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    def abort(signum, frame):
        raise TimeoutError('Bounded diagnostic interrupted by signal %d' % signum)
    old_handlers={s:signal.signal(s,abort) for s in [signal.SIGALRM,signal.SIGTERM,signal.SIGINT]}
    signal.alarm(args.max_seconds)
    try:
        diagnose(args)
    except BaseException as error:
        write(out/'failed.json',dict(status='FAILED_DIAGNOSIS',utc=utc(),error=repr(error),traceback=traceback.format_exc(),optimizer_steps=0))
        raise
    finally:
        signal.alarm(0)
        for s, handler in old_handlers.items():
            signal.signal(s,handler)


if __name__ == '__main__':
    main()
