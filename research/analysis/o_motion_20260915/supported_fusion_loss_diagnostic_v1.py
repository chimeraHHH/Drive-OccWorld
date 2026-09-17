"""One-anchor observer of frozen fusion V1, stopped before any optimizer step.

Uses the already staged V1 command and sources. The only private-module hook
observes loss_terms: original A forward/backward, original Z loss forward,
then two no_grad forwards of the identical detached A prediction. It never
relaxes the failed constant-loss equality gate or writes a new trainer source.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import struct
import sys
import time

TRAINER_SHA = '54bd8747c534b5d0a74474651350398f3d304aade3a23d071a8d3458e44fe4b9'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class DiagnosticComplete(RuntimeError):
    pass


def ulp32(a,b):
    def ordered(v):
        bits=struct.unpack('>I',struct.pack('>f',v))[0]
        return 0x80000000-(bits&0x7fffffff) if bits&0x80000000 else 0x80000000+bits
    return abs(ordered(a)-ordered(b))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-job',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    stage=Path(args.stage_job).resolve();pkg=stage/'package'
    launch=json.loads((stage/'launch.json').read_text())
    state=json.loads((stage/'state.json').read_text())
    require(state['status']=='FAILED' and state['returncode']==1,'Require preserved terminal failed V1 job')
    for name,digest in launch['script_sha256'].items():
        require(Path(name).name==name and sha(pkg/name)==digest,'Staged source changed: '+name)
    source=pkg/'train_supported_fusion_v1.py'
    require(sha(source)==TRAINER_SHA,'Wrong frozen V1 trainer')
    spec=json.loads((stage/'spec.json').read_text())
    command=spec['command'];index=command.index(str(source));argv=command[index+1:]
    require('--preflight' in argv and '--dev-cache' not in argv and '--o-development-records' not in argv,
            'Original command is not train-only preflight')
    target_index=argv.index('--out')+1
    old_out=Path(argv[target_index]);argv[target_index]=str(out/'original_stage_execution')
    require(json.loads((old_out/'failed.json').read_text())['updates']==0,'Original failure had optimizer updates')
    sys.path.insert(0,str(pkg))
    module_spec=importlib.util.spec_from_file_location('observed_frozen_fusion_v1',source)
    trainer=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(trainer)
    original=trainer.loss_terms
    calls=[];held={}
    def observation(model,prediction,targets,adapter,label,repeat_no_grad=False):
        import torch
        native=sys.modules['native_state_cache']
        def fingerprints():
            return dict(prediction_sha256=native._tensor_digest(prediction),
                decoder_layers_sha256=[native._tensor_digest(prediction[:,i]) for i in range(3)],
                target_sha256=native._tensor_digest(targets),
                prediction_shape=list(prediction.shape),prediction_stride=list(prediction.stride()),
                prediction_requires_grad=prediction.requires_grad,grad_enabled=torch.is_grad_enabled(),
                target_shape=list(targets.shape),target_dtype=str(targets.dtype))
        before=fingerprints();tick=time.monotonic()
        loss,audit=original(model,prediction,targets,adapter)
        torch.cuda.synchronize()
        after=fingerprints()
        values=audit['original_twelve']
        item=dict(label=label,repeat_no_grad=repeat_no_grad,before=before,after=after,
            losses=values,losses_float_hex={k:float(v).hex() for k,v in values.items()},
            objective_tensor_dtype=str(loss.dtype),torch_version=torch.__version__,cuda_version=torch.version.cuda,
            constant_keys=audit['constant_intermediate_keys'],seconds=time.monotonic()-tick,
            prediction_unchanged=before['prediction_sha256']==after['prediction_sha256'],
            target_unchanged=before['target_sha256']==after['target_sha256'])
        calls.append(item)
        write(out/'observations.json',dict(calls=calls,optimizer_updates=0,complete=False))
        return loss,audit
    def observed(model,prediction,targets,adapter):
        import torch
        if len(calls)==0:
            held['prediction']=prediction.detach().clone()
            loss,audit=observation(model,prediction,targets,adapter,'A_original_grad_enabled')
            return loss,audit
        require(len(calls)==1,'Unexpected original loss call sequence')
        loss,audit=observation(model,prediction,targets,adapter,'Z_original_after_A_backward')
        del loss,audit
        # Do not backpropagate Z, update either arm, or enter sample2/dev.
        with torch.no_grad():
            for j in range(2):
                loss,audit=observation(model,held['prediction'],targets,adapter,
                    'A_identical_detached_no_grad_repeat_'+str(j+1),repeat_no_grad=True)
                del loss,audit
        del held['prediction']
        raise DiagnosticComplete('Collected A/Z and two same-A repeats; optimizer updates remain zero')
    trainer.loss_terms=observed
    def timeout(signum,frame):raise TimeoutError('Independent 180-second diagnostic deadline')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(.01,180-(time.monotonic()-started)))
    status='FAILED_DIAGNOSTIC'
    try:
        trainer.main(argv)
        raise RuntimeError('Observer did not stop original loop')
    except DiagnosticComplete:
        require(len(calls)==4,'Incomplete measurement')
        status='COMPLETE_LOSS_DIFFERENCE_DIAGNOSTIC'
    finally:
        trainer.loss_terms=original
        signal.setitimer(signal.ITIMER_REAL,0)
    failed=json.loads((out/'original_stage_execution'/'failed.json').read_text())
    require(failed['updates']==0 and failed['actual_optimizer_updates_by_arm']=={'A':0,'Z':0}
            and failed['evaluated_samples']==0,'Diagnostic unexpectedly updated/evaluated')
    diffs=[]
    for j in range(1,4):
        diffs.append(dict(left=calls[0]['label'],right=calls[j]['label'],
            losses={k:dict(a=calls[0]['losses'][k],b=calls[j]['losses'][k],
                difference=calls[j]['losses'][k]-calls[0]['losses'][k],
                float32_ULPs=ulp32(calls[0]['losses'][k],calls[j]['losses'][k]),
                a_hex=calls[0]['losses_float_hex'][k],b_hex=calls[j]['losses_float_hex'][k]) for k in calls[0]['losses']},
            fixed_decoder_input_hash_equal=[calls[0]['before']['decoder_layers_sha256'][i]==
                calls[j]['before']['decoder_layers_sha256'][i] for i in (0,1)],
            target_input_hash_equal=calls[0]['before']['target_sha256']==calls[j]['before']['target_sha256']))
    result=dict(schema='supported-fusion-loss-diagnostic-v1',status=status,calls=calls,differences_from_A=diffs,
        optimizer_updates=0,development_evaluated=False,original_A_backward_executed_before_Z_loss=True,
        repeated_A_prediction_is_detached=True,no_constant_gate_relaxed=True,
        original_stage_sources_unchanged=all(sha(pkg/n)==d for n,d in launch['script_sha256'].items()),
        source_sha256=sha(__file__),staged_trainer_sha256=TRAINER_SHA,stage_launch_sha256=sha(stage/'launch.json'),
        original_failed_sha256=sha(old_out/'failed.json'),
        original_preflight_first_seed11_ordinal=396,seconds=time.monotonic()-started)
    write(out/'result.json',result)
    write(out/'complete.json',dict(schema=result['schema'],status=status,optimizer_updates=0,
        result_sha256=sha(out/'result.json'),observations_sha256=sha(out/'observations.json'),
        original_observer_stop_sha256=sha(out/'original_stage_execution'/'failed.json'),seconds=result['seconds']))
    print(json.dumps(dict(status=status,optimizer_updates=0,seconds=result['seconds'],
        differing_constant_keys=[k for k in calls[0]['constant_keys'] if calls[0]['losses'][k]!=calls[1]['losses'][k]])),flush=True)


if __name__=='__main__':main()
