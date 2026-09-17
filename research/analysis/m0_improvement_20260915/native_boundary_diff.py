"""Observe an unchanged failed native-joint boundary gate on one frozen anchor.

Wrap only the private run's hash function, return its original value, and retain
the original rejection. Persist small difference statistics, never model data.
"""
import argparse
import inspect
import json
from pathlib import Path
import signal
import time
import traceback
from types import SimpleNamespace

from objective_joint_evaluation import ObjectiveEvaluation, frozen_module, OLD_PRODUCER_SHA, sha


def differences(left, right, path='inputs'):
    import numpy as np
    import torch
    rows = []
    if type(left) is not type(right):
        return [dict(path=path, kind='type', fresh=str(type(left)), cached=str(type(right)))]
    if isinstance(left, dict):
        if set(left) != set(right):
            return [dict(path=path, kind='dictionary_keys', fresh=sorted(map(str,left)), cached=sorted(map(str,right)))]
        for key in left:
            rows += differences(left[key], right[key], path+'.'+str(key))
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return [dict(path=path, kind='length', fresh=len(left), cached=len(right))]
        for index, (x,y) in enumerate(zip(left,right)):
            rows += differences(x,y,path+'['+str(index)+']')
    elif torch.is_tensor(left) or isinstance(left,np.ndarray):
        x=left.detach().cpu().numpy() if torch.is_tensor(left) else left
        y=right.detach().cpu().numpy() if torch.is_tensor(right) else right
        if x.shape != y.shape or x.dtype != y.dtype:
            rows.append(dict(path=path,kind='array_contract',fresh_shape=list(x.shape),cached_shape=list(y.shape),
                fresh_dtype=str(x.dtype),cached_dtype=str(y.dtype)))
        elif x.tobytes() != y.tobytes():
            row=dict(path=path,kind='array_values',shape=list(x.shape),dtype=str(x.dtype),
                unequal_elements=int(np.count_nonzero(x!=y)),elements=int(x.size),bitwise_equal=False)
            if np.issubdtype(x.dtype,np.number):
                delta=x.astype(np.float64)-y.astype(np.float64)
                row.update(max_abs=float(np.max(np.abs(delta))),mean_abs=float(np.mean(np.abs(delta))),
                    fresh_l2=float(np.linalg.norm(x.ravel())),cached_l2=float(np.linalg.norm(y.ravel())))
            rows.append(row)
    elif left != right:
        rows.append(dict(path=path,kind='scalar_value',fresh=repr(left),cached=repr(right)))
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation-protocol','objective-protocol','protocol','config','checkpoint','repo','out','parity-cache'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--shard-index',type=int,choices=(0,1),default=0)
    a=parser.parse_args(argv)
    a=SimpleNamespace(**vars(a),mode='pilot',selection=None,full_authorization=None,device='cuda:0',
        max_seconds=600,max_allocated_gib=64,deadline_unix=time.time()+600)
    base=frozen_module('joint_native_evaluation.py',OLD_PRODUCER_SHA)
    runner=ObjectiveEvaluation(a,base)
    original=runner.run.__func__.__globals__['tree_digest'];calls=0;fresh=None;record=None

    def observed_hash(tree):
        nonlocal calls,fresh,record
        calls+=1
        value=original(tree)
        if calls==1:
            fresh=tree
        elif calls==2:
            import numpy as np
            local=inspect.currentframe().f_back.f_locals
            context=local['context'];old=local['old'];directory=local['old_directory']
            targets=np.load(directory/old['files']['targets']['file'],mmap_mode='r',allow_pickle=False)
            preds=np.load(directory/old['files']['preds']['file'],mmap_mode='r',allow_pickle=False)
            delta=np.abs(context['preds'].astype(np.float64)-preds.astype(np.float64))
            record=dict(schema='native-joint-boundary-difference-v1',status='OBSERVED_BEFORE_ORIGINAL_GATE',
                diagnostic_source_sha256=sha(__file__),evaluation_protocol_sha256=sha(a.evaluation_protocol),
                identity=local['row'],fresh_boundary_sha256=local['input_hash'],cached_boundary_sha256=value,
                boundary_equal=local['input_hash']==value,differences=differences(fresh,tree),
                original_full_GT_bitwise_equal=bool(np.array_equal(context['targets'],targets)),
                native_full_GT_hist_exact=bool(np.array_equal(context['native_record']['hist_by_horizon'],old['native_hist'])),
                native_5h_3layer_logits_bitwise_equal=bool(np.array_equal(context['preds'],preds)),
                native_logits_max_abs=float(delta.max()),
                native_logits_unequal_elements=int(np.count_nonzero(context['preds']!=preds)),
                original_gate_not_relaxed=True,original_forward_and_inputs_unchanged=True,optimizer_steps=0,
                candidate_performance_computed=False,planning_enabled=bool(local['native'].turn_on_plan),
                native_module_training_flags=sorted(set(m.training for m in local['native'].modules())))
            base.write(runner.out/'difference.json',record,exclusive=True)
        return value

    runner.run.__func__.__globals__['tree_digest']=observed_hash
    def interrupted(*_):
        raise InterruptedError('Boundary diagnostic deadline')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,interrupted)
    signal.alarm(600)
    try:
        runner.run()
        raise RuntimeError('Previously observed boundary discrepancy did not reproduce')
    except BaseException as exc:
        expected=record is not None and str(exc)=='Fresh observed boundary differs from frozen pilot cache'
        base.write(runner.out/'complete.json',dict(status='DIFFERENCE_CAPTURED_ORIGINAL_GATE_REJECTED' if expected else 'DIAGNOSTIC_FAILED',
            error=repr(exc),traceback=traceback.format_exc(),difference_sha256=sha(runner.out/'difference.json') if record else None,
            optimizer_steps=0,original_gate_not_relaxed=True),exclusive=True)
        print(json.dumps(dict(status='DIFFERENCE_CAPTURED' if expected else 'FAILED',error=repr(exc))),flush=True)
        return 0 if expected else 1
    finally:
        signal.alarm(0);runner.lock.close()


if __name__=='__main__':
    raise SystemExit(main())
