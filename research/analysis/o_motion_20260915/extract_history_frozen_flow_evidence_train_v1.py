"""Frozen official image correspondence on the original train512 query support.

Images alone produce flow; cached GT-defined volume query locations are only
used afterwards. Thus this is a diagnostic, not a GT-free deployed predictor.
No target arrays, new checkpoint training, score fitting or development reads.
"""
import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
from PIL import Image

SCHEMA = 'history-frozen-flow-evidence-train-v1'
CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
HISTORY_SHA = '040d62242a541b9685f3f39f1534b3663b0e1142b4cd7d9737cdbfd15d94314d'
CAMERA_SHA = '0c0d2e640a12e3299e8dacc03189776669cd2e50036a65de823026304119b648'
ASSET_SHA = 'db378bcf49a348a8968d571f834a33e3f434582e7b0d551b805f49cefb320cdd'
PROBE_SOURCE_SHA = '60491da9cc3b0dcfabcf9c09d15ca6ac1b2f2e8175b657870c6d54fb6e65789e'
CHECKPOINT_SHA = 'fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def module(path, expected, name):
    require(sha(path) == expected, 'Frozen module changed: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def ledger(root, expected):
    require(sha(root/'complete.json') == expected and not (root/'failed.json').exists(), 'Wrong completed input')
    done = read(root/'complete.json')
    for name, digest in done['files_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe ledger path')
        require(sha(root/name) == digest, 'Input metadata changed: ' + name)
    return done


def array_fingerprint(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def model_fingerprint(model):
    return array_fingerprint({key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})


def load_model(a, probe):
    import torch
    assets, verified = probe.check_assets(a)
    require(assets['checkpoint']['sha256'] == CHECKPOINT_SHA, 'Different released checkpoint')
    gpu = probe.check_idle_gpu(a.gpu_uuid)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one CUDA GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = (a.asset_root/'source/core').resolve()
    for name in ('raft', 'update', 'extractor', 'corr', 'utils', 'utils.utils'):
        require(name not in sys.modules, 'Unexpected official module namespace: ' + name)
    sys.path.insert(0, str(source))
    raft = importlib.import_module('raft'); utils = importlib.import_module('utils.utils')
    for name, relative in (('raft', 'raft.py'), ('update', 'update.py'), ('extractor', 'extractor.py'),
                           ('corr', 'corr.py'), ('utils', 'utils/__init__.py'), ('utils.utils', 'utils/utils.py')):
        require(Path(sys.modules[name].__file__).resolve() == source/relative, 'Wrong official module origin')
    model = raft.RAFT(probe.RaftArguments(small=False, mixed_precision=False, alternate_corr=False, dropout=0.))
    state = torch.load(a.asset_root/assets['checkpoint']['path'], map_location='cpu', weights_only=True)
    require(isinstance(state, dict) and state and all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()),
            'Released weight must be a tensor state_dict')
    prefix = [key.startswith('module.') for key in state]
    require(all(prefix) or not any(prefix), 'Mixed checkpoint prefix')
    stripped = all(prefix)
    if stripped:
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, 'module.')
    require(all(not v.is_floating_point() or (v.dtype == torch.float32 and torch.isfinite(v).all()) for v in state.values()),
            'Weights must be finite float32')
    model.load_state_dict(state, strict=True); del state
    model.eval().requires_grad_(False)
    initial = model_fingerprint(model)
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
    model = model.to(device='cuda:0', dtype=torch.float32)
    return torch, model, utils, initial, gpu, assets, verified, stripped


def image_pair(row, pair, data_root, probe):
    images = []
    for when in ('current', 'past'):
        frame = pair[when]; desc = frame['image']; sd = frame['sample_data']
        require(sd['is_key_frame'] and sd['timestamp'] <= row['input_availability_us'] and
                sd['sample_token'] == (row['identity']['sample_token'] if when == 'current' else row['previous_sample_token']),
                'Original same-camera keyframe/availability changed')
        file = probe.beneath(data_root, desc['file'])
        require(desc['file'] == sd['filename'] and desc['size_wh'] == [1600, 900] and
                file.stat().st_size == desc['bytes'] and sha(file) == desc['sha256'], 'Original image SHA/size differs')
        with Image.open(file) as im:
            require(im.size == (1600, 900) and im.mode == 'RGB' and im.format == 'JPEG', 'Original raw RGB image required')
            images.append(np.array(im, dtype=np.uint8))
    require(pair['past']['sample_data']['timestamp'] < min(pair['current']['sample_data']['timestamp'], row['t0_lidar_us']),
            'Previous image not historical')
    return images


def three_flows(torch, model, utils, images):
    # Only this camera's three outputs are retained on CPU. No full-dataset flow cache.
    tensors = [torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to('cuda:0', dtype=torch.float32) for x in images]
    padder = utils.InputPadder(tensors[0].shape)
    current, past = padder.pad(*tensors)
    broken_raw = torch.roll(tensors[1], shifts=800, dims=3)
    broken_past = padder.pad(broken_raw)[0]
    require(padder._pad == [0, 0, 2, 2] and current.shape == past.shape == broken_past.shape == (1, 3, 904, 1600),
            'Official padding or original resolution changed')
    result, timings, fingerprints = {}, {}, {}
    with torch.no_grad():
        for name, first, second in (('current_to_past', current, past), ('past_to_current', past, current),
                                    ('current_to_broken_past', current, broken_past)):
            torch.cuda.synchronize(0); tick = time.monotonic()
            low, high = model(first, second, iters=20, flow_init=None, upsample=True, test_mode=True)
            torch.cuda.synchronize(0); timings[name] = time.monotonic()-tick
            require(high.shape == (1, 2, 904, 1600) and high.dtype == torch.float32, 'Flow output shape/dtype')
            flow = padder.unpad(high)[0].permute(1, 2, 0).contiguous().cpu().numpy().copy()
            require(flow.shape == (900, 1600, 2) and np.isfinite(flow).all(), 'Nonfinite or incorrectly unpadded flow')
            result[name] = flow; fingerprints[name] = hashlib.sha256(flow.tobytes()).hexdigest()
            del low, high
    return result, timings, fingerprints


def run(a, out, started, progress):
    sys.dont_write_bytecode = True
    protocol = read(a.protocol)
    require(protocol['schema'] == SCHEMA and protocol['status'] == 'FROZEN' and
            protocol['resources'] == dict(max_seconds=3600, cpu_threads=2, max_allocated_bytes=12*1024**3) and
            protocol['source_sha256'] == sha(__file__) and protocol['asset_manifest_sha256'] == ASSET_SHA and
            a.asset_manifest_sha256 == ASSET_SHA and protocol['probe_source_sha256'] == PROBE_SOURCE_SHA and
            protocol['history_inputs_complete_sha256'] == HISTORY_SHA and protocol['camera_complete_sha256'] == CAMERA_SHA and
            protocol['checkpoint_sha256'] == CHECKPOINT_SHA, 'Frozen protocol changed')
    core = module(a.module, protocol['module_sha256'], 'fixed_flow_evidence_core')
    probe = module(a.probe_module, PROBE_SOURCE_SHA, 'fixed_raft_probe_helpers')
    hc = ledger(a.history_inputs, HISTORY_SHA); cc = ledger(a.camera_run, CAMERA_SHA)
    rows = read(a.history_inputs/'records.json')['records']; camera_rows = read(a.camera_run/'index.json')['records']
    require(len(rows) == len(camera_rows) == 512 and
            len({r['identity']['scene_token'] for r in rows}) == 256, 'Complete train512/256 required')
    data_root = Path(read(a.history_inputs/'manifest.json')['source_data_root'])
    progress['phase'] = 'load_official_frozen_model'
    torch, model, utils, initial_state, gpu, assets, verified_assets, stripped = load_model(a, probe)
    require(all(not m.training for m in model.modules()), 'RAFT must remain eval')
    manifest = dict(schema=SCHEMA, source_sha256=sha(__file__), protocol_sha256=sha(a.protocol),
        module_sha256=sha(a.module), probe_source_sha256=PROBE_SOURCE_SHA, asset_manifest_sha256=ASSET_SHA,
        checkpoint_sha256=CHECKPOINT_SHA, official_commit=assets['commit'], verified_consumed_assets=verified_assets,
        history_inputs_complete_sha256=HISTORY_SHA, history_inputs_files_sha256=hc['files_sha256'],
        camera_complete_sha256=CAMERA_SHA, camera_files_sha256=cc['files_sha256'],
        initial_model_state_sha256=initial_state, uniform_module_prefix_removed=stripped,
        settings=protocol['settings'], scope=protocol['scope'], gpu=gpu,
        runtime=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda))
    write(out/'manifest.json', manifest); (out/'samples').mkdir()
    records = []; total = decoded = forwards = 0
    totals = dict(primary_valid=0, low_texture_extension=0, flow_valid=0, fb_zero_valid=0, fb_D_valid=0)
    total_reasons = {str(i): 0 for i in range(5)}
    forward_times = {key: [] for key in ('current_to_past', 'past_to_current', 'current_to_broken_past')}
    for ordinal, (row, old) in enumerate(zip(rows, camera_rows)):
        progress.update(phase='sample', ordinal=ordinal, completed=ordinal)
        require(time.monotonic()-started < 3600, 'Frozen batch time bound exceeded')
        require(row['ordinal'] == old['ordinal'] == ordinal and row['identity'] == old['identity'] and
                row['identity']['split'] == 'train' and tuple(p['channel'] for p in row['cameras']) == CAMERAS,
                'Original identity/camera order differs')
        path = probe.beneath(a.camera_run, old['file'])
        require(path.stat().st_size == old['bytes'] and sha(path) == old['sha256'], 'Original camera NPZ differs')
        with np.load(path, allow_pickle=False) as archive:
            cached = {key: archive[key] for key in archive.files}
        require(json.loads(cached['identity_json'].item()) == row['identity'] and len(cached['source_flat_indices']) == old['source_points'],
                'Original point/identity differs')
        before = array_fingerprint(cached)
        arrays = core.initialize_flow_evidence(cached)
        merged = np.zeros(len(cached['camera_index']), dtype=bool)
        timings = {}; camera_fingerprints = {}
        for ci, pair in enumerate(row['cameras']):
            progress.update(camera=ci, completed_model_forwards=forwards)
            images = image_pair(row, pair, data_root, probe); decoded += 2
            flows, current_times, fingerprints = three_flows(torch, model, utils, images); forwards += 3
            ix, result = core.evaluate_camera_flow_evidence(cached, ci, flows)
            require(np.array_equal(ix, np.flatnonzero(cached['camera_index'] == ci)) and not merged[ix].any(), 'Camera row merge changed')
            require(set(result) == set(arrays), 'Per-camera result schema differs')
            for key in arrays:
                arrays[key][ix] = result[key]
            merged[ix] = True
            timings[pair['channel']] = current_times; camera_fingerprints[pair['channel']] = fingerprints
            for key, value in current_times.items(): forward_times[key].append(value)
            require(torch.cuda.max_memory_allocated(0) <= 12*1024**3, 'Allocated memory limit exceeded')
            del images, flows, result
        require(np.array_equal(merged, cached['camera_index'] >= 0) and array_fingerprint(cached) == before,
                'Incomplete camera merge or changed cached input')
        require(np.array_equal(arrays['flow_valid'], np.isin(cached['reason_code'], [0, 4])) and
                np.array_equal(arrays['primary_valid'], cached['reason_code'] == 0), 'Primary/extension support changed')
        arrays['identity_json'] = cached['identity_json'].copy()
        name = old['file']
        with (out/name).open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(out/name, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays), 'Saved array schema changed')
            for key, value in arrays.items():
                require(np.array_equal(value, saved[key], equal_nan=True) if value.dtype.kind in 'fc' else np.array_equal(value, saved[key]),
                        'Saved array roundtrip differs: '+key)
        counts = dict(primary_valid=int(arrays['primary_valid'].sum()), low_texture_extension=int((arrays['reason_code'] == 4).sum()),
                      flow_valid=int(arrays['flow_valid'].sum()), fb_zero_valid=int(arrays['fb_valid'][:, 0].sum()),
                      fb_D_valid=int(arrays['fb_valid'][:, 1].sum()))
        reasons = {str(i): int((arrays['reason_code'] == i).sum()) for i in range(5)}
        for key, value in counts.items(): totals[key] += value
        for key, value in reasons.items(): total_reasons[key] += value
        records.append(dict(ordinal=ordinal, identity=row['identity'], file=name, bytes=(out/name).stat().st_size,
            sha256=sha(out/name), source_points=len(cached['source_flat_indices']), camera_npz_sha256=old['sha256'],
            counts=counts, reason_counts=reasons, forward_seconds=timings, flow_array_sha256_HWC=camera_fingerprints,
            arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()}))
        total += len(cached['source_flat_indices']); progress['completed'] = ordinal+1
        print(json.dumps(dict(event='sample_complete', completed=ordinal+1, seconds=time.monotonic()-started,
                             model_forwards=forwards, peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)))), flush=True)
        del arrays, cached
    progress['phase'] = 'finalize'
    require((total, decoded, forwards) == (1740053, 6144, 9216) and totals['primary_valid'] == 1397509 and
            totals['low_texture_extension'] == 261702 and totals['flow_valid'] == 1659211, 'Full support/call accounting differs')
    require(model_fingerprint(model) == initial_state and all(p.grad is None for p in model.parameters()) and
            all(not m.training for m in model.modules()), 'Frozen model state changed')
    probe.check_assets(a)
    write(out/'index.json', dict(schema=SCHEMA, records=records, source_points=total, counts=totals, reason_counts=total_reasons))
    elapsed = time.monotonic()-started
    require(elapsed < 3600, 'Extraction exceeded frozen time budget')
    write(out/'summary.json', dict(schema=SCHEMA, status='COMPLETE_HISTORY_FROZEN_FLOW_TRAIN_EVIDENCE',
        samples=512, scenes=256, source_points=total, images_decoded=decoded, model_forwards=forwards, seconds=elapsed,
        counts=totals, reason_counts=total_reasons, model_state_sha256=initial_state, frozen_model_state_unchanged=True,
        checkpoint_sha256=CHECKPOINT_SHA, peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
        forward_seconds={key: dict(calls=len(value), total=float(sum(value)), median=float(np.median(value)),
                                  maximum=float(max(value))) for key, value in forward_times.items()},
        all_original_cached_arrays_unchanged=True, all_query_arrays_roundtrip_exact=True, GT_target_arrays_read=False,
        original_query_population_is_GT_defined=True, image_flow_uses_GT=False, O_or_D_model_loaded=False,
        training=False, optimizer_updates=0, threshold_selection=False, fitting=False, development_used=False,
        flow_quality_claim=False, batch_accuracy_statistics_computed=False))
    write(out/'complete.json', dict(schema=SCHEMA, status='COMPLETE_HISTORY_FROZEN_FLOW_TRAIN_EVIDENCE', samples=512,
        source_sha256=sha(__file__), protocol_sha256=sha(a.protocol), module_sha256=sha(a.module),
        files_sha256={name: sha(out/name) for name in ('manifest.json', 'index.json', 'summary.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'module', 'probe-module', 'asset-root', 'history-inputs', 'camera-run', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--asset-manifest-sha256', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--max-seconds', type=int, default=3600)
    a = parser.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); progress = dict(phase='initial', completed=0)
    def stop(signum, frame):
        raise TimeoutError('Bounded batch signal '+str(signum))
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    signal.alarm(3600)
    try:
        require(a.max_seconds == 3600 and os.environ.get('CUDA_VISIBLE_DEVICES') == a.gpu_uuid, 'Fixed resource settings changed')
        run(a, a.out, started, progress)
    except BaseException as exc:
        signal.alarm(0)
        write(a.out/'failed.json', dict(schema=SCHEMA, error=repr(exc), traceback=traceback.format_exc(),
              progress=progress, seconds=time.monotonic()-started, optimizer_updates=0, completed_result=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
