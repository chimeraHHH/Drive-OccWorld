"""One-anchor, zero-update VJP diagnostic of frozen connected-motion V1.

Retains the authenticated V1 initialization, measured inputs, loss formulas and
precision. Original P/O/T VJPs are captured twice on each arm's same graph. A
read-only flow hook records the actual cotangent arriving at the shared head;
it returns None and never changes that gradient. Only the engineering boundary
is replaced in a private run namespace, and it always raises DiagnosticStop.
The failed sum gate remains an observed number, never a training authorization.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import types

import numpy as np

SCHEMA = 'connected-motion-vjp-diagnostic-v1'
TRAINER_SHA = '7b61c80b87e3a7c5940ba25288429b1b5ebc1696556f0dfab05512c4860ba1b9'
PROTOCOL_SHA = '9ccfa458a8b67b978959240b790e0e1b99b4c72e1771c0696900e41be6f3af2e'
MAX_SECONDS = 600
MAX_ALLOCATED_GIB = 32


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


class DiagnosticStop(RuntimeError):
    pass


def numeric_summary(values):
    x = np.asarray(values).reshape(-1); finite = np.isfinite(x)
    clean = x.astype(np.float64) if bool(finite.all()) else None
    return dict(elements=x.size, dtype=str(x.dtype), finite_count=int(finite.sum()), all_finite=bool(finite.all()),
        nnz=int(np.count_nonzero(x)), norm_l2=None if clean is None else float(np.linalg.norm(clean)),
        max_abs=None if clean is None or not x.size else float(np.abs(clean).max()),
        mean_abs=None if clean is None or not x.size else float(np.abs(clean).mean()))


def comparison(left, right, *, rtol, atol, top=8, descriptor=None):
    """CPU float64 arithmetic on the original float32 gradient values."""
    a = np.asarray(left).reshape(-1); b = np.asarray(right).reshape(-1)
    require(a.shape == b.shape, 'Gradient comparison shape changed')
    result = dict(left=numeric_summary(a), right=numeric_summary(b), rtol=rtol, atol=atol)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        result.update(all_finite=False, passed=False); return result
    a = a.astype(np.float64); b = b.astype(np.float64); delta = a-b; absolute = np.abs(delta)
    tolerance = atol+rtol*np.abs(b); failed = absolute > tolerance
    denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(np.float64).tiny)
    result.update(all_finite=True, max_abs=float(absolute.max()) if a.size else 0.,
        norm_l2=float(np.linalg.norm(delta)), relative_l2=float(np.linalg.norm(delta)/denominator),
        changed_elements=int(np.count_nonzero(delta)), failed_elements=int(failed.sum()), passed=not bool(failed.any()),
        max_tolerance_multiple=float(np.max(absolute/tolerance)) if a.size else 0.)
    worst = np.argsort(absolute)[-min(top, len(a)):][::-1]
    result['worst'] = [dict(index=int(i), location=None if descriptor is None else descriptor(int(i)),
        left=float(a[i]), right=float(b[i]), delta=float(delta[i]), allowed=float(tolerance[i]),
        left_hex=float(a[i]).hex(), right_hex=float(b[i]).hex()) for i in worst]
    return result


def additivity(physical, occupancy, total, *, rtol, atol, descriptor=None):
    p, o, t = [np.asarray(x).reshape(-1) for x in (physical, occupancy, total)]
    require(p.dtype == o.dtype == t.dtype == np.float32 and p.shape == o.shape == t.shape, 'Expected same FP32 VJPs')
    # This float32 sum is the original comparison RHS. A second CPU float64
    # sum measures how much of the discrepancy is just this final addition.
    with np.errstate(over='ignore', invalid='ignore'):
        sum32 = np.add(p, o, dtype=np.float32)
    sum64 = p.astype(np.float64)+o.astype(np.float64)
    amplitude = np.abs(p.astype(np.float64))+np.abs(o.astype(np.float64))
    cancellation = np.divide(np.abs(sum64), amplitude, out=np.ones_like(amplitude), where=amplitude > 0)
    failed = np.abs(t.astype(np.float64)-sum32.astype(np.float64)) > atol+rtol*np.abs(sum32.astype(np.float64))
    return dict(total_vs_float32_component_sum=comparison(t, sum32, rtol=rtol, atol=atol, descriptor=descriptor),
        total_vs_float64_component_sum=comparison(t, sum64, rtol=rtol, atol=atol, descriptor=descriptor),
        final_float32_addition_roundoff=comparison(sum32, sum64, rtol=rtol, atol=atol, descriptor=descriptor),
        cancellation_ratio_definition='abs(gP+gO)/(abs(gP)+abs(gO)); 1 when both are zero',
        opposing_nonzero_coordinates=int(((p.astype(np.float64)*o.astype(np.float64) < 0) & (amplitude > 0)).sum()),
        cancellation_ratio_min=float(cancellation.min()) if len(cancellation) else None,
        ratio_below_1e_3=int((cancellation < 1e-3).sum()),
        failed_coordinates_ratio_below_1e_3=int((failed & (cancellation < 1e-3)).sum()))


def parameter_layout(models, gates, arm):
    rows = []; offset = 0
    for prefix, network in (('motion', models[arm]), ('gate', gates[arm])):
        for name, p in network.named_parameters():
            group = 'motion_trunk' if prefix == 'motion' and name.startswith('trunk.') else 'motion_readout' if prefix == 'motion' else 'gate'
            rows.append(dict(name=prefix+'.'+name, shape=list(p.shape), dtype=str(p.dtype),
                requires_grad=p.requires_grad, group=group, start=offset, stop=offset+p.numel()))
            offset += p.numel()
    return rows


def parameter_location(layout, index):
    for row in layout:
        if row['start'] <= index < row['stop']:
            return dict(parameter=row['name'], tensor_index=[int(x) for x in np.unravel_index(index-row['start'], row['shape'])])
    raise ValueError('Out-of-range parameter coordinate')


def vector(grads, parameters):
    return np.concatenate([np.zeros(p.numel(), np.float32) if g is None else
        g.detach().cpu().contiguous().numpy().reshape(-1).copy() for g, p in zip(grads, parameters)])


def grouped_measurements(captures, layout, flow_shape, numerical):
    rtol, atol = numerical['gradient_rtol'], numerical['gradient_atol']
    result = dict(parameter_layout=layout, parameter_groups={})
    for group in ('motion', 'motion_trunk', 'motion_readout', 'gate'):
        selected = [r for r in layout if (r['name'].startswith('motion.') if group == 'motion' else r['group'] == group)]
        index = np.concatenate([np.arange(r['start'], r['stop']) for r in selected])
        describe = lambda i: parameter_location(layout, int(index[i]))
        data = {k: v['parameters'][index] for k, v in captures.items()}
        result['parameter_groups'][group] = dict(
            first_additivity=additivity(data['physical_1'], data['occupancy_1'], data['total_1'], rtol=rtol, atol=atol, descriptor=describe),
            second_additivity=additivity(data['physical_2'], data['occupancy_2'], data['total_2'], rtol=rtol, atol=atol, descriptor=describe),
            repeat={kind: comparison(data[kind+'_1'], data[kind+'_2'], rtol=rtol, atol=atol, descriptor=describe)
                    for kind in ('physical', 'occupancy', 'total')},
            component_stats={k: numeric_summary(v) for k, v in data.items()})
    location = lambda i: dict(flow_B_H_XYZ_X_Y_Z=[int(x) for x in np.unravel_index(i, flow_shape)])
    data = {k: v['flow'] for k, v in captures.items()}
    result['flow_node'] = dict(shape=list(flow_shape),
        first_additivity=additivity(data['physical_1'], data['occupancy_1'], data['total_1'], rtol=rtol, atol=atol, descriptor=location),
        second_additivity=additivity(data['physical_2'], data['occupancy_2'], data['total_2'], rtol=rtol, atol=atol, descriptor=location),
        repeat={kind: comparison(data[kind+'_1'], data[kind+'_2'], rtol=rtol, atol=atol, descriptor=location)
                for kind in ('physical', 'occupancy', 'total')},
        component_stats={k: numeric_summary(v) for k, v in data.items()},
        hook_calls={k: v['flow_hook_calls'] for k, v in captures.items()},
        missing_hook_represented_as_zero_only_in_statistics=True)
    result['None_by_parameter'] = {k: v['none_names'] for k, v in captures.items()}
    result['path_requirements_observed'] = dict(
        physical_gate_all_None=all(all(r['name'] in captures['physical_'+str(j)]['none_names'] for r in layout if r['group']=='gate') for j in (1,2)),
        occupancy_motion_all_None=all(all(r['name'] in captures['occupancy_'+str(j)]['none_names'] for r in layout if r['name'].startswith('motion.')) for j in (1,2)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-job', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(argv); started = time.monotonic()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    stage = Path(args.stage_job).resolve(); package = stage/'package'
    launch = read(stage/'launch.json'); state = read(stage/'state.json')
    require(state['status'] == 'FAILED' and state['returncode'] == 1, 'Require preserved terminal failed V1')
    for name, digest in launch['script_sha256'].items():
        require(Path(name).name == name and sha(package/name) == digest, 'Staged source changed: '+name)
    source = package/'train_connected_motion_v1.py'; require(sha(source) == TRAINER_SHA, 'Wrong frozen trainer')
    command = launch['command']; location = command.index(str(source)); old_argv = list(command[location+1:])
    require('--preflight' in old_argv and '--dev-cache' not in old_argv and '--o-development-records' not in old_argv, 'Require original train-only preflight')
    original_out = Path(old_argv[old_argv.index('--out')+1]); failed = read(original_out/'failed.json')
    require(failed['updates'] == failed['examples'] == failed['evaluated_samples'] == 0 and
            failed['actual_optimizer_updates_by_arm'] == {a: dict(motion=0, gate=0) for a in ('J','D')}, 'Original stage unexpectedly updated/evaluated')
    protocol_path = Path(old_argv[old_argv.index('--protocol')+1])
    require(sha(protocol_path) == PROTOCOL_SHA and float(old_argv[old_argv.index('--max-seconds')+1]) == MAX_SECONDS and
            float(old_argv[old_argv.index('--max-allocated-gib')+1]) == MAX_ALLOCATED_GIB, 'Original diagnostic budget/protocol differs')
    old_argv[old_argv.index('--out')+1] = str(out/'original_stage_execution')
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location('observed_connected_v1', source)
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    records = {}; identity = {}; unchanged = {}; call_count = 0
    def budget():
        import torch
        require(time.monotonic()-started < MAX_SECONDS, 'Diagnostic wall-clock limit')
        require(torch.cuda.max_memory_allocated() <= MAX_ALLOCATED_GIB*2**30, 'Diagnostic CUDA allocated-memory limit')
    def observed(shared, label, models, gates, module, helper, fusion, oracle, adapter, model, numerical):
        nonlocal call_count
        import torch
        call_count += 1; require(call_count == 1, 'Only first engineering anchor is allowed')
        sample, pred, base, future, tokens, probability, foreground = shared
        native = sys.modules['native_state_cache']; tree = sys.modules['joint_native_evaluation']
        def snapshot():
            return dict(O=helper.state_digest(model), motion={a: helper.state_digest(h) for a,h in models.items()},
                gate={a: helper.state_digest(h) for a,h in gates.items()},
                inputs=tree.tree_digest(sample['inputs']), targets=trainer.target_value_digest(sample['targets'], native),
                original_predictions=native._tensor_digest(pred))
        before = snapshot()
        identity.update(sample_token=str(label['sample_token'].item()), scene_token=str(label['scene_token'].item()),
            ordinal=int(label['ordinal']), split=str(label['split'].item()),
            input_tensor_shape=list(tokens.shape), boundary_before=before,
            precision=dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
                all_O_modules_eval=not any(x.training for x in model.modules())))
        require(identity['ordinal'] == helper.planned_orders()[0][0] and identity['split'] == 'train', 'Wrong fixed first anchor')
        require(not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32 and
                not torch.backends.cudnn.benchmark, 'Changed diagnostic precision')
        flows = {arm: models[arm](tokens) for arm in trainer.ARMS}
        flow_comparison = trainer.tensor_comparison(flows['J'], flows['D'],
            rtol=numerical['independent_forward_rtol'], atol=numerical['independent_forward_atol'])
        fields = module.supported_transport(probability, foreground, flows['J'], oracle.EXTENT)
        exact_fields = {'J': fields, 'D': {k: v.detach() for k,v in fields.items()}}
        initial_logits = {}; physical_vectors = {}; gate_vectors = {}; loss_audits = {}
        for arm in trainer.ARMS:
            budget(); tick = time.monotonic()
            params = list(models[arm].parameters())+list(gates[arm].parameters())
            layout = parameter_layout(models, gates, arm)
            fused = gates[arm](future, exact_fields[arm]['probability_mass'], exact_fields[arm]['support_weight'])
            require(bool(torch.all(fused['gate'] == .5)), 'Initial gate value changed')
            changed = fusion.compose_prediction(pred, base, fused, oracle)
            initial_logits[arm] = oracle.predictions_to_xyz(changed).detach()
            occupancy, loss_audits[arm] = fusion.loss_terms(model, changed, sample['targets'], adapter)
            physical, physical_audit = helper.object_group_loss(helper.gather_sparse(flows[arm], label), label)
            losses = dict(physical=physical, occupancy=occupancy, total=physical+occupancy)
            captures = {}; active = {}; flow_shape = tuple(flows[arm].shape)
            original_gpu_vectors = {}; original_gpu_gate = {}
            def flow_hook(gradient):
                active['calls'] += 1
                # Read only; return None is the identity hook contract.
                active['flow'] = gradient.detach().cpu().contiguous().numpy().reshape(-1).copy()
                return None
            handle = flows[arm].register_hook(flow_hook)
            try:
                for repeat in (1,2):
                    for kind in ('physical','occupancy','total'):
                        budget(); active.clear(); active.update(calls=0, flow=None)
                        final = repeat == 2 and kind == 'total'
                        grads = torch.autograd.grad(losses[kind], params, allow_unused=True, retain_graph=not final)
                        name = kind+'_'+str(repeat)
                        captures[name] = dict(parameters=vector(grads,params),
                            flow=np.zeros(int(np.prod(flow_shape)), np.float32) if active['flow'] is None else active['flow'],
                            flow_hook_calls=active['calls'], none_names=[r['name'] for r,g in zip(layout,grads) if g is None])
                        original_gpu_vectors[kind] = trainer.gradient_vector(grads, params)
                        if kind == 'total':
                            n_motion = sum(p.numel() for p in models[arm].parameters())
                            original_gpu_gate[str(repeat)] = trainer.tensor_comparison(
                                original_gpu_vectors['total'][:n_motion],
                                original_gpu_vectors['physical'][:n_motion]+original_gpu_vectors['occupancy'][:n_motion],
                                rtol=numerical['gradient_rtol'], atol=numerical['gradient_atol'])
                            original_gpu_vectors.clear()
                        del grads
                        write(out/'progress.json', dict(schema=SCHEMA, phase='VJP_CAPTURE', arm=arm, captured=name,
                            optimizer_updates=0, seconds=time.monotonic()-started))
            finally:
                handle.remove()
            torch.cuda.synchronize(); budget()
            records[arm] = grouped_measurements(captures, layout, flow_shape, numerical)
            records[arm]['original_gpu_motion_sum_gate_by_repeat'] = original_gpu_gate
            n = sum(v.numel() for v in models[arm].parameters())
            physical_vectors[arm] = captures['physical_1']['parameters'][:n].copy()
            gate_vectors[arm] = captures['occupancy_1']['parameters'][n:].copy()
            records[arm].update(losses={k: float(v.detach()) for k,v in losses.items()},
                losses_hex={k: float(v.detach()).hex() for k,v in losses.items()},
                original_loss_audit=loss_audits[arm], physical_loss_audit=physical_audit,
                no_parameter_grad_written=all(v.grad is None for v in params),
                retained_same_graph_for_repeat=True, final_total_repeat_released_graph=True,
                seconds=time.monotonic()-tick)
            require(records[arm]['no_parameter_grad_written'], 'autograd.grad wrote .grad')
            write(out/'observations.json', dict(schema=SCHEMA, identity=identity, arms=records, complete=False, optimizer_updates=0))
            del captures, losses, physical, occupancy, changed, fused
        unchanged.update(before=before, after=snapshot())
        unchanged['all_equal'] = unchanged['before'] == unchanged['after']
        require(unchanged['all_equal'], 'Diagnostic changed parameters/buffers/inputs/GT/O predictions')
        pair = dict(shared_value_initial_prediction_byte_equal=fusion.exact32(initial_logits['J'], initial_logits['D']),
            independent_flow=flow_comparison,
            physical_motion_VJP=comparison(physical_vectors['J'],physical_vectors['D'],rtol=numerical['gradient_rtol'],atol=numerical['gradient_atol']),
            occupancy_gate_VJP=comparison(gate_vectors['J'],gate_vectors['D'],rtol=numerical['gradient_rtol'],atol=numerical['gradient_atol']),
            fixed_losses_J_as_A_D_as_Z=fusion.compare_constant_losses(loss_audits['J'],loss_audits['D']))
        records['paired_initial_state'] = pair
        write(out/'observations.json', dict(schema=SCHEMA, identity=identity, arms=records, unchanged=unchanged,
            complete=True, optimizer_updates=0, original_gate_tolerances_unchanged=True))
        raise DiagnosticStop('Captured same-graph VJPs; stop before all optimizer/update loops')
    # Private namespace only. No original file or imported global function is replaced.
    run_namespace = dict(trainer.run.__globals__); run_namespace['engineering_probe'] = observed
    private_run = types.FunctionType(trainer.run.__code__, run_namespace, trainer.run.__name__, trainer.run.__defaults__, trainer.run.__closure__)
    main_namespace = dict(trainer.main.__globals__); main_namespace['run'] = private_run
    private_main = types.FunctionType(trainer.main.__code__, main_namespace, trainer.main.__name__, trainer.main.__defaults__, trainer.main.__closure__)
    def timeout(signum, frame): raise TimeoutError('Independent 600-second VJP diagnostic deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.01, MAX_SECONDS-(time.monotonic()-started)))
    try:
        try:
            private_main(old_argv)
            raise RuntimeError('Diagnostic unexpectedly returned into training')
        except DiagnosticStop:
            require(call_count == 1 and set(records) == {'J','D','paired_initial_state'}, 'Incomplete diagnostic capture')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        stopped = read(out/'original_stage_execution'/'failed.json')
        require(stopped['updates'] == stopped['examples'] == stopped['evaluated_samples'] == 0 and
                stopped['actual_optimizer_updates_by_arm'] == {a:dict(motion=0,gate=0) for a in ('J','D')} and
                not stopped['partial_checkpoints'], 'Diagnostic updated/evaluated/persisted model')
        require(all(sha(package/name) == digest for name,digest in launch['script_sha256'].items()), 'Stage source changed')
        import torch
        budget()
        result = dict(schema=SCHEMA, status='COMPLETE_VJP_DIAGNOSTIC_NO_UPDATE', optimizer_updates=0,
            examples_trained=0, development_evaluated=False, source_sha256=sha(__file__),
            trainer_sha256=TRAINER_SHA, protocol_sha256=PROTOCOL_SHA,
            stage_launch_sha256=sha(stage/'launch.json'), stage_state_sha256=sha(stage/'state.json'),
            original_failure_sha256=sha(original_out/'failed.json'), staged_sources_sha256=launch['script_sha256'],
            identity=identity, arms=records, unchanged=unchanged,
            original_allclose_gate_not_relaxed=True, original_training_never_entered=True,
            gradients_or_logits_persisted=False, same_forward_graph_per_arm=True,
            shared_transport_value_for_initial_VJP=True, flow_hook_never_modifies_gradient=True,
            interpretation='Observations only; no claim that discrepancies are harmless or that the training gate passes.',
            resources=dict(elapsed_seconds=time.monotonic()-started, max_seconds=MAX_SECONDS,
                max_allocated_gib=MAX_ALLOCATED_GIB, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        write(out/'result.json', result)
        write(out/'complete.json', dict(schema=SCHEMA, status=result['status'], optimizer_updates=0,
            result_sha256=sha(out/'result.json'), observations_sha256=sha(out/'observations.json'),
            original_boundary_stop_sha256=sha(out/'original_stage_execution'/'failed.json'), resources=result['resources']))
        print(json.dumps(dict(status=result['status'], optimizer_updates=0,
            original_gate_passed={a: records[a]['original_gpu_motion_sum_gate_by_repeat']['1']['passed'] for a in ('J','D')},
            seconds=time.monotonic()-started)), flush=True)
    except BaseException as error:
        write(out/'failed.json', dict(schema=SCHEMA, status='FAILED_DIAGNOSTIC_NO_RETRY', error=repr(error),
            optimizer_updates=0, seconds=time.monotonic()-started))
        raise


if __name__ == '__main__': main()
