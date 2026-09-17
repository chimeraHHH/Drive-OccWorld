"""Fixed-cotangent same-head TF32 backward intervention; one anchor, no update.

One original J motion forward and one original loss graph are evaluated under
the frozen policy. Their qP=dLphys/dd, qO=dLocc/dd, qT=d(Lphys+Locc)/dd are
then detached and fixed. Only cuDNN allow_tf32 changes for head-only VJPs:
H(qP), H(qO), H(qP+qO), H(qT). O/GT/flow/loss forwards are never recomputed
under another precision. The original flag is restored in finally.

PyTorch v2.1.2 convolution_backward reads allowTF32CuDNN from globalContext:
https://github.com/pytorch/pytorch/blob/v2.1.2/aten/src/ATen/native/Convolution.cpp#L1865-L1876
https://github.com/pytorch/pytorch/blob/v2.1.2/tools/autograd/derivatives.yaml#L2030-L2037
This diagnostic never authorizes training or modifies any frozen source/gate.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import types

import numpy as np

BASE_SHA = '6d19fb692a559f4c408fa2a4e3a99466d61197d1c7032d675fc0d6c2ec1383de'
TRAINER_SHA = '7b61c80b87e3a7c5940ba25288429b1b5ebc1696556f0dfab05512c4860ba1b9'
PROTOCOL_SHA = '9ccfa458a8b67b978959240b790e0e1b99b4c72e1771c0696900e41be6f3af2e'
PREVIOUS_RESULT_SHA = '4ef3f729c5c85e22574337ca189ee7a41d5fdc8a02ffb1194ac10e071318a64d'
SCHEMA = 'connected-motion-fixed-cotangent-diagnostic-v2'
MAX_SECONDS = 600
MAX_ALLOCATED_GIB = 32


def load_base():
    import hashlib
    path = Path(__file__).parent/'train_connected_motion_vjp_diagnostic_v1.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Fixed CPU statistics helper source changed')
    spec = importlib.util.spec_from_file_location('vjp_diagnostic_frozen_math', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class DiagnosticStop(RuntimeError):
    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-job', required=True)
    parser.add_argument('--previous-diagnostic', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv); started = time.monotonic()
    b = load_base(); require, sha, read, write = b.require, b.sha, b.read, b.write
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    previous = Path(args.previous_diagnostic).resolve()
    prior_done = read(previous/'complete.json')
    require(prior_done['status'] == 'COMPLETE_VJP_DIAGNOSTIC_NO_UPDATE' and prior_done['optimizer_updates'] == 0 and
            prior_done['result_sha256'] == sha(previous/'result.json') == PREVIOUS_RESULT_SHA and
            prior_done['observations_sha256'] == sha(previous/'observations.json'), 'Wrong completed V1 diagnostic evidence')
    stage = Path(args.stage_job).resolve(); package = stage/'package'
    launch, state = read(stage/'launch.json'), read(stage/'state.json')
    require(state['status'] == 'FAILED' and state['returncode'] == 1, 'Require terminal V1 failure')
    for name, digest in launch['script_sha256'].items():
        require(Path(name).name == name and sha(package/name) == digest, 'Staged source changed: '+name)
    source = package/'train_connected_motion_v1.py'; require(sha(source) == TRAINER_SHA, 'Wrong frozen trainer')
    command = launch['command']; old_argv = list(command[command.index(str(source))+1:])
    require('--preflight' in old_argv and '--dev-cache' not in old_argv and '--o-development-records' not in old_argv, 'Not train-only preflight')
    original_out = Path(old_argv[old_argv.index('--out')+1]); failed = read(original_out/'failed.json')
    require(failed['updates'] == failed['examples'] == failed['evaluated_samples'] == 0 and
            failed['actual_optimizer_updates_by_arm'] == {a:dict(motion=0,gate=0) for a in ('J','D')}, 'Original run updated/evaluated')
    protocol = Path(old_argv[old_argv.index('--protocol')+1])
    require(sha(protocol) == PROTOCOL_SHA and float(old_argv[old_argv.index('--max-seconds')+1]) == MAX_SECONDS and
            float(old_argv[old_argv.index('--max-allocated-gib')+1]) == MAX_ALLOCATED_GIB, 'Original source/resource contract changed')
    old_argv[old_argv.index('--out')+1] = str(out/'original_stage_execution')
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location('connected_v1_fixed_backward_observation', source)
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    record = {}; calls = 0
    def budget():
        import torch
        require(time.monotonic()-started < MAX_SECONDS, 'Diagnostic wall-clock limit')
        require(torch.cuda.max_memory_allocated() <= MAX_ALLOCATED_GIB*2**30, 'Diagnostic GPU allocation limit')
    def observed(shared, label, models, gates, module, helper, fusion, oracle, adapter, model, numerical):
        nonlocal calls
        import torch
        calls += 1; require(calls == 1, 'Only first fixed anchor allowed')
        sample, pred, base, future, tokens, probability, foreground = shared
        native = sys.modules['native_state_cache']; tree = sys.modules['joint_native_evaluation']
        def state_snapshot():
            return dict(O=helper.state_digest(model), motion={a:helper.state_digest(h) for a,h in models.items()},
                gate={a:helper.state_digest(g) for a,g in gates.items()}, inputs=tree.tree_digest(sample['inputs']),
                targets=trainer.target_value_digest(sample['targets'],native), O_predictions=native._tensor_digest(pred))
        before = state_snapshot(); budget()
        require(int(label['ordinal']) == helper.planned_orders()[0][0] and str(label['split'].item()) == 'train', 'Wrong first anchor')
        saved_tf32 = torch.backends.cudnn.allow_tf32
        require(saved_tf32 is True and torch.backends.cuda.matmul.allow_tf32 is False and
                torch.backends.cudnn.benchmark is False, 'Changed original forward precision')
        # Only J's motion and gate are exercised; D weights remain initialized,
        # untouched, and included in the before/after full state check.
        flow = models['J'](tokens)
        fields = module.supported_transport(probability,foreground,flow,oracle.EXTENT)
        fused = gates['J'](future,fields['probability_mass'],fields['support_weight'])
        require(bool(torch.all(fused['gate'] == .5)), 'Initial gate changed')
        changed = fusion.compose_prediction(pred,base,fused,oracle)
        occupancy, loss_audit = fusion.loss_terms(model,changed,sample['targets'],adapter)
        physical, physical_audit = helper.object_group_loss(helper.gather_sparse(flow,label),label)
        # Stop VJPs at d: no head-weight backward is performed here.
        q = {}
        for name, loss in (('physical',physical),('occupancy',occupancy),('total',physical+occupancy)):
            budget()
            value = torch.autograd.grad(loss,flow,retain_graph=True)[0]
            require(bool(torch.isfinite(value).all()), 'Nonfinite fixed cotangent')
            q[name] = value.detach().clone()
        q['component_sum'] = (q['physical']+q['occupancy']).detach().clone()
        require(not any(v.requires_grad for v in q.values()), 'Cotangents must be fixed constants')
        q_digest = {name:native._tensor_digest(value) for name,value in q.items()}
        flow_digest = native._tensor_digest(flow)
        layout = [row for row in b.parameter_layout(models,gates,'J') if row['name'].startswith('motion.')]
        parameters = list(models['J'].parameters()); full = {}; reports = {}
        rtol, atol = numerical['gradient_rtol'], numerical['gradient_atol']
        try:
            for flag in (True,False):
                torch.backends.cudnn.allow_tf32 = flag
                require(torch.backends.cudnn.allow_tf32 == flag and not torch.backends.cuda.matmul.allow_tf32,
                        'Backward precision intervention did not take effect')
                measured = {}; tensors = {}; none_by_q = {}
                for name in ('physical','occupancy','component_sum','total'):
                    budget(); tick = time.monotonic()
                    grads = torch.autograd.grad(flow,parameters,grad_outputs=q[name],
                        allow_unused=True,retain_graph=not (flag is False and name == 'total'))
                    tensors[name] = trainer.gradient_vector(grads,parameters)
                    measured[name] = b.vector(grads,parameters)
                    none_by_q[name] = [row['name'] for row,g in zip(layout,grads) if g is None]
                    torch.cuda.synchronize(); del grads
                    write(out/'progress.json',dict(schema=SCHEMA,phase='HEAD_ONLY_VJP',cudnn_tf32=flag,q=name,
                        optimizer_updates=0,seconds=time.monotonic()-started,last_vjp_seconds=time.monotonic()-tick))
                key = 'tf32_enabled' if flag else 'tf32_disabled'; full[key] = measured
                groups = {}
                for group in ('motion','motion_trunk','motion_readout'):
                    selected = layout if group == 'motion' else [row for row in layout if row['group']==group]
                    index = np.concatenate([np.arange(row['start'],row['stop']) for row in selected])
                    location = lambda i: b.parameter_location(layout,int(index[i]))
                    data = {name:values[index] for name,values in measured.items()}
                    groups[group] = dict(fixed_qsum_additivity=b.additivity(data['physical'],data['occupancy'],data['component_sum'],
                        rtol=rtol,atol=atol,descriptor=location),
                        upstream_qT_vs_qsum_effect=b.comparison(data['total'],data['component_sum'],rtol=rtol,atol=atol,descriptor=location),
                        original_total_additivity=b.additivity(data['physical'],data['occupancy'],data['total'],
                            rtol=rtol,atol=atol,descriptor=location),
                        norms={name:b.numeric_summary(value) for name,value in data.items()})
                reports[key] = dict(cudnn_tf32=flag,matmul_tf32=False,groups=groups,None_by_q=none_by_q,
                    original_GPU_rule_fixed_qsum=trainer.tensor_comparison(tensors['component_sum'],tensors['physical']+tensors['occupancy'],rtol=rtol,atol=atol),
                    original_GPU_rule_qT=trainer.tensor_comparison(tensors['total'],tensors['physical']+tensors['occupancy'],rtol=rtol,atol=atol))
                del tensors
                write(out/'observations.json',dict(schema=SCHEMA,head_backwards=reports,complete=False,optimizer_updates=0))
        finally:
            torch.backends.cudnn.allow_tf32 = saved_tf32
        require(torch.backends.cudnn.allow_tf32 == saved_tf32 and not torch.backends.cuda.matmul.allow_tf32,
                'Original precision was not restored')
        require(native._tensor_digest(flow) == flow_digest and
                all(native._tensor_digest(value) == q_digest[name] for name,value in q.items()), 'Fixed flow/cotangent values changed')
        after = state_snapshot(); require(before == after, 'State/input/GT/O prediction changed')
        require(all(v.grad is None for net in (model,*models.values(),*gates.values()) for v in net.parameters()), '.grad contamination')
        record.update(identity=dict(ordinal=int(label['ordinal']),sample_token=str(label['sample_token'].item()),
            scene_token=str(label['scene_token'].item()),split='train'),
            precision=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
                forward_cudnn_tf32=True,matmul_tf32=False,backward_cudnn_tf32_sequence=[True,False],restored_cudnn_tf32=True),
            actual_motion_forward_calls=1,actual_loss_forward_calls=1,O_forward_calls_from_original_shared=1,
            fixed_q_sha256=q_digest,flow_sha256=flow_digest,
            q_stats={name:b.numeric_summary(value.detach().cpu().numpy()) for name,value in q.items()},
            qT_vs_component_sum=b.comparison(q['total'].cpu().numpy(),q['component_sum'].cpu().numpy(),rtol=rtol,atol=atol),
            original_losses=dict(physical=float(physical.detach()),occupancy=float(occupancy.detach())),
            original_loss_audit=loss_audit,physical_loss_audit=physical_audit,head_backwards=reports,
            between_policy={name:b.comparison(full['tf32_enabled'][name],full['tf32_disabled'][name],rtol=rtol,atol=atol)
                            for name in full['tf32_enabled']},
            state_before=before,state_after=after,parameters_buffers_inputs_GT_predictions_unchanged=True,
            no_parameter_grad_written=True,no_optimizer_step=True,original_precision_restored=True)
        write(out/'observations.json',dict(schema=SCHEMA,record=record,complete=True,optimizer_updates=0))
        raise DiagnosticStop('Fixed-cotangent TF32 backward intervention captured; no training')
    private = dict(trainer.run.__globals__); private['engineering_probe'] = observed
    run = types.FunctionType(trainer.run.__code__,private,trainer.run.__name__,trainer.run.__defaults__,trainer.run.__closure__)
    entry = dict(trainer.main.__globals__); entry['run'] = run
    main_function = types.FunctionType(trainer.main.__code__,entry,trainer.main.__name__,trainer.main.__defaults__,trainer.main.__closure__)
    def timeout(signum,frame): raise TimeoutError('Independent 600-second diagnostic deadline')
    signal.signal(signal.SIGALRM,timeout); signal.setitimer(signal.ITIMER_REAL,max(.01,MAX_SECONDS-(time.monotonic()-started)))
    try:
        try:
            main_function(old_argv)
            raise RuntimeError('Diagnostic returned to training loop')
        except DiagnosticStop:
            require(calls == 1 and record['no_optimizer_step'], 'Incomplete diagnostic')
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
        stopped = read(out/'original_stage_execution'/'failed.json')
        require(stopped['updates'] == stopped['examples'] == stopped['evaluated_samples'] == 0 and
                stopped['actual_optimizer_updates_by_arm'] == {a:dict(motion=0,gate=0) for a in ('J','D')} and
                not stopped['partial_checkpoints'], 'Training/development unexpectedly ran')
        require(all(sha(package/name) == digest for name,digest in launch['script_sha256'].items()), 'Staged source changed')
        import torch
        budget()
        result = dict(schema=SCHEMA,status='COMPLETE_FIXED_COTANGENT_TF32_DIAGNOSTIC_NO_UPDATE',optimizer_updates=0,
            record=record,source_sha256=sha(__file__),statistics_helper_sha256=BASE_SHA,trainer_sha256=TRAINER_SHA,
            protocol_sha256=PROTOCOL_SHA,previous_diagnostic_result_sha256=PREVIOUS_RESULT_SHA,
            previous_diagnostic_complete_sha256=sha(previous/'complete.json'),stage_launch_sha256=sha(stage/'launch.json'),
            original_failure_sha256=sha(original_out/'failed.json'),staged_sources_sha256=launch['script_sha256'],
            no_training_gate_or_loss_changed=True,development_evaluated=False,no_gradients_or_logits_persisted=True,
            resources=dict(elapsed_seconds=time.monotonic()-started,max_seconds=MAX_SECONDS,max_allocated_gib=MAX_ALLOCATED_GIB,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        write(out/'result.json',result)
        write(out/'complete.json',dict(schema=SCHEMA,status=result['status'],optimizer_updates=0,
            result_sha256=sha(out/'result.json'),observations_sha256=sha(out/'observations.json'),
            boundary_stop_sha256=sha(out/'original_stage_execution'/'failed.json'),resources=result['resources']))
        print(json.dumps(dict(status=result['status'],optimizer_updates=0,
            fixed_qsum_gate={name:row['original_GPU_rule_fixed_qsum']['passed'] for name,row in record['head_backwards'].items()},
            seconds=time.monotonic()-started)),flush=True)
    except BaseException as error:
        write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_DIAGNOSTIC_NO_RETRY',error=repr(error),
            optimizer_updates=0,seconds=time.monotonic()-started))
        raise


if __name__ == '__main__': main()
