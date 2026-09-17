"""One fixed train-anchor CPU/GPU-cache numerical probe; no labels or fitting."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

import numpy as np

CONNECTED_SHA = 'e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
RUNTIME_SHA = '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c'
TRAIN_COMPLETE_SHA = '3bd429effb1fbe50ac0aeb77330bc0100ee144bf8406fbd078097fb336301a73'
CHECKPOINT_SHA = '7e2750d56ade7fb36e334b6431e83f51aafa997357744780a507924a696951e3'
REFERENCE_SHA = '4ac2ffa9dc500e5e3eb03ef252db6ea6bc3f77606bcb296e30400b3db2bc0b07'
THRESHOLDS = np.asarray([0., .1, .5, 1., 2., 5., 10., np.inf])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def run(a, start):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU must be hidden')
    p = read(a.source_root/'connected_motion_protocol_v2.json')
    require(sha(a.source_root/'connected_motion_protocol_v2.json') == CONNECTED_SHA, 'Frozen D protocol differs')
    for name, digest in p['sources_sha256'].items():
        require(sha(a.source_root/name) == digest, 'Frozen source differs: '+name)
    require(sha(a.source_root/'runtime_source_contract.json') == RUNTIME_SHA, 'Runtime source contract differs')
    for name, digest in read(a.source_root/'runtime_source_contract.json')['runtime_source_sha256'].items():
        require(sha(name) == digest, 'Runtime source differs: '+name)
    sys.path.insert(0, str(a.source_root))
    def module(name):
        spec = importlib.util.spec_from_file_location(name, a.source_root/(name+'.py'))
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
    trainer = module('train_connected_motion_v2'); helper = module('train_source_motion_v1'); native = module('native_state_cache')
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA initialized before CPU probe')
    native._prepare_repo(a.repo)
    model, gate, receipt = trainer.load_completed_arm(a.d_run/'runs/D', a.source_root/'connected_motion_protocol_v2.json', helper, 'cpu')
    del gate
    require(receipt['checkpoint_sha256'] == CHECKPOINT_SHA, 'Wrong D checkpoint')
    initial = helper.state_digest(model)
    cache, records, cache_receipt = helper.cache_index(a.train_cache, 'train', p)
    record = records[0]
    require(record['sample_token'] == '464fe0be05a74ec9852573cb9e3afd89', 'Wrong fixed first train anchor')
    tokens = helper.load_tokens(cache, record, native, 'cpu')
    tick = time.monotonic()
    with torch.inference_mode():
        prediction = model(tokens)
    forward_seconds = time.monotonic()-tick
    require(tuple(prediction.shape) == (1,4,3,200,200,16) and prediction.dtype == torch.float32 and
            torch.isfinite(prediction).all(), 'Invalid full-grid CPU output')
    # Only after the complete input-only prediction: read cached prediction values
    # and indices for numerical comparison, never target_displacement or groups.
    require(sha(a.train_run/'complete.json') == TRAIN_COMPLETE_SHA, 'Wrong completed train reference')
    done = read(a.train_run/'complete.json')
    for name, digest in done['files_sha256'].items():
        require(sha(a.train_run/name) == digest, 'Train reference ledger differs')
    row = read(a.train_run/'index.json')['records'][0]
    require(row['ordinal'] == 0 and row['identity']['sample_token'] == record['sample_token'] and
            row['sha256'] == REFERENCE_SHA and sha(a.train_run/row['file']) == REFERENCE_SHA, 'Wrong cached anchor prediction')
    with np.load(a.train_run/row['file'], allow_pickle=False) as z:
        indices = z['source_flat_indices']; reference = z['D_displacement_m']; original_speed = z['observable_features'][:, 0]
    cpu = prediction[0].reshape(4,3,-1)[:,:,indices].permute(0,2,1).numpy()
    require(cpu.shape == reference.shape == (4,1210,3) and cpu.dtype == reference.dtype == np.float32, 'Sparse CPU/cache shape differs')
    error = np.abs(cpu.astype(np.float64)-reference.astype(np.float64))
    h = np.asarray([.5,1.,1.5,2.])[:,None,None]
    def speed(d):
        return np.linalg.norm(((h*d.astype(np.float64)).sum(axis=0)/np.sum(h*h))[:,:2], axis=1)
    refspeed, cpuspeed = speed(reference), speed(cpu)
    require(np.allclose(refspeed, original_speed, rtol=1e-13, atol=1e-14), 'Original LSQ speed convention differs')
    flips = np.sum((cpuspeed[:,None] >= THRESHOLDS) != (original_speed[:,None] >= THRESHOLDS), axis=0)
    numeric_pass = bool(error.max() <= 1e-4 and error.mean() <= 1e-6)
    gate_pass = bool(not flips.any())
    require(helper.state_digest(model) == initial and not model.training and
            all(not q.requires_grad and q.grad is None for q in model.parameters()) and
            not torch.cuda.is_initialized(), 'Model/GPU state changed')
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(peak_rss <= 8*1024**3, 'CPU peak RSS exceeds 8GiB')
    report = dict(status='PASS_FIXED_TRAIN_ANCHOR_CPU_PROBE' if numeric_pass and gate_pass else 'NUMERICAL_PROBE_REJECTED',
        source_sha256=sha(__file__), loader=receipt, cache=cache_receipt, ordinal=0, identity=row['identity'],
        original_prediction_npz_sha256=REFERENCE_SHA, points=1210,
        tolerances=dict(max_abs_displacement_m=1e-4,mean_abs_displacement_m=1e-6,max_gate_flips=0),
        max_abs_displacement_m=float(error.max()),mean_abs_displacement_m=float(error.mean()),
        max_abs_by_horizon_xyz_m=error.max(axis=1).tolist(),mean_abs_by_horizon_xyz_m=error.mean(axis=1).tolist(),
        max_abs_LSQ_speed_difference_mps=float(np.abs(cpuspeed-original_speed).max()),
        mean_abs_LSQ_speed_difference_mps=float(np.abs(cpuspeed-original_speed).mean()),
        threshold_mps=[0.,.1,.5,1.,2.,5.,10.,None],gate_flip_counts=flips.tolist(),
        numeric_pass=numeric_pass,gate_pass=gate_pass,CPU_forward_seconds=forward_seconds,
        peak_RSS_bytes=peak_rss,seconds=time.monotonic()-start,training=False,optimizer_updates=0,
        development_read=False,target_arrays_read=False,full_grid_inference_before_reference_read=True,
        CUDA_initialized=False,model_state_unchanged=True,comparison_scope='One fixed train anchor only; not dev200 equivalence',
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device='cpu',threads=2,
                     interop_threads=1,mkldnn_enabled=torch.backends.mkldnn.enabled,original_GPU_cudnn_tf32=True))
    write(a.out/'report.json', report)
    with (a.out/'predictions.npz').open('xb') as stream:
        np.savez_compressed(stream,source_flat_indices=indices,CPU_D=cpu,CPU_LSQ_speed=cpuspeed,reference_D=reference,reference_LSQ_speed=original_speed)
    write(a.out/'complete.json',dict(status=report['status'],source_sha256=sha(__file__),
        files_sha256={name:sha(a.out/name) for name in ('report.json','predictions.npz')}))
    print(json.dumps(report),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','d-run','train-cache','train-run','repo','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args(); a.out.mkdir(parents=True,exist_ok=False); start=time.monotonic()
    def stop(signum,frame):
        raise TimeoutError('Bounded CPU probe signal '+str(signum))
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(s,stop)
    signal.alarm(600)
    try:
        run(a,start)
    except BaseException as exc:
        write(a.out/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start,training=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
