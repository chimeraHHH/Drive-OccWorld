"""One fixed original CAM_FRONT pair: official full RAFT resource probe only.

No download, installation, O/D, annotation, optimizer, AUC or flow-quality test.
Run only after review, with one idle physical GPU exposed by its exact UUID.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


COMMIT = '2888e15a51fa41140771d3f498ed8023cff098d1'
HISTORY_COMPLETE_SHA = '040d62242a541b9685f3f39f1534b3663b0e1142b4cd7d9737cdbfd15d94314d'
SCHEMA = 'raft-official-pair-resource-probe-v1'


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
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def beneath(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe asset/image path')
    path = (root / relative).resolve()
    require(root.resolve() in path.parents, 'Path escapes authenticated root')
    return path


def check_assets(a):
    root = a.asset_root
    require(sha(root / 'manifest.json') == a.asset_manifest_sha256, 'Asset manifest SHA differs')
    manifest, complete = read(root / 'manifest.json'), read(root / 'complete.json')
    require(manifest['schema'] == 'raft-official-fixed-asset-v1' and
            complete['schema'] == 'raft-official-fixed-asset-complete-v1' and
            manifest['status'] == complete['status'] == 'COMPLETE_SOURCE_AND_REQUESTED_WEIGHT' and
            complete['manifest_sha256'] == a.asset_manifest_sha256 and
            manifest['repository'] == 'https://github.com/princeton-vl/RAFT' and manifest['commit'] == COMMIT and
            manifest['official_tree_gitblob_verified'] is True and
            manifest['requested_checkpoint'] == 'raft-things.pth', 'Incomplete or different official asset')
    source_names = sorted(name for name in manifest['files'] if name.startswith('source/'))
    require(len(source_names) == manifest['source_files'] == 35 and
            {p.relative_to(root).as_posix() for p in (root / 'source').rglob('*') if p.is_file()} == set(source_names),
            'Official source tree incomplete or contains extra files')
    verified = {}
    for name in source_names + ['weights/raft-things.pth']:
        desc = manifest['files'][name]; path = beneath(root, name)
        require(path.stat().st_size == desc['bytes'] and sha(path) == desc['sha256'], 'Asset bytes differ: ' + name)
        verified[name] = desc
    checkpoint = manifest['checkpoint']
    require(checkpoint['path'] == 'weights/raft-things.pth' and
            checkpoint['sha256'] == complete['checkpoint_sha256'] == verified[checkpoint['path']]['sha256'] and
            checkpoint['bytes'] == verified[checkpoint['path']]['bytes'] and
            checkpoint['archive_CRC_verified_on_full_read'] is True, 'Requested weight binding')
    return manifest, verified


def fixed_pair(a):
    """Only input metadata and two original JPEGs; never a label/GT file."""
    root = a.history_inputs
    require(sha(root / 'complete.json') == HISTORY_COMPLETE_SHA, 'Original history complete differs')
    done = read(root / 'complete.json')
    require(done['samples'] == 512 and done['status'] == 'COMPLETE_AUTHENTICATED_HISTORY_CAMERA_INPUTS' and
            set(done['files_sha256']) == {'manifest.json', 'records.json', 'summary.json'}, 'History endpoint incomplete')
    for name, digest in done['files_sha256'].items():
        require(sha(root / name) == digest, 'History metadata ledger differs: ' + name)
    metadata = read(root / 'manifest.json'); rows = read(root / 'records.json')['records']
    require(len(rows) == 512, 'Original train512 selection differs')
    row = rows[0]
    require(row['ordinal'] == 0 and row['identity']['split'] == 'train', 'Fixed first train anchor differs')
    selected = [p for p in row['cameras'] if p['channel'] == 'CAM_FRONT']
    require(len(selected) == 1, 'Fixed CAM_FRONT absent/duplicated')
    source_root = Path(metadata['source_data_root']); pair = selected[0]
    descriptions = {}; arrays = []
    import numpy as np
    from PIL import Image
    for when in ('current', 'past'):
        frame = pair[when]; desc = frame['image']; sd = frame['sample_data']
        require(sd['is_key_frame'] is True and sd['sample_token'] ==
                (row['identity']['sample_token'] if when == 'current' else row['previous_sample_token']) and
                sd['timestamp'] <= row['input_availability_us'], 'Fixed historical sensor identity/availability')
        path = beneath(source_root, desc['file'])
        require(sd['filename'] == desc['file'] and desc['size_wh'] == [1600, 900] and
                path.stat().st_size == desc['bytes'] and sha(path) == desc['sha256'], 'Original JPEG bytes/header contract')
        with Image.open(path) as image:
            require(image.size == (1600, 900) and image.format == 'JPEG' and image.mode == 'RGB', 'Original RGB JPEG required')
            rgb = np.array(image, dtype=np.uint8)
        require(rgb.shape == (900, 1600, 3), 'Raw image shape')
        arrays.append(rgb)
        descriptions[when] = dict(file=desc['file'], sha256=desc['sha256'], bytes=desc['bytes'],
            sample_data_token=sd['token'], sample_token=sd['sample_token'], timestamp_us=sd['timestamp'],
            decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), shape=list(rgb.shape))
    require(descriptions['past']['timestamp_us'] < min(descriptions['current']['timestamp_us'], row['t0_lidar_us']),
            'Previous keyframe is not historical')
    return arrays, dict(anchor_ordinal=0, identity=row['identity'], camera='CAM_FRONT',
        history_complete_sha256=HISTORY_COMPLETE_SHA, history_files_sha256=done['files_sha256'],
        t0_lidar_us=row['t0_lidar_us'], input_availability_us=row['input_availability_us'], images=descriptions)


def check_idle_gpu(uuid):
    require(uuid.startswith('GPU-') and os.environ.get('CUDA_VISIBLE_DEVICES') == uuid,
            'Expose exactly the requested physical GPU UUID')
    def query(args):
        return subprocess.run(['nvidia-smi'] + args, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    lines = query(['--id=' + uuid, '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits']).splitlines()
    require(len(lines) == 1, 'GPU UUID query ambiguous')
    fields = [x.strip() for x in lines[0].split(',')]
    require(len(fields) == 5 and fields[0] == '0' and fields[1] == uuid and
            int(fields[3]) == 0 and int(fields[4]) == 0, 'Physical GPU0 is not idle')
    processes = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    require(not any(line.split(',')[0].strip() == uuid for line in processes.splitlines()), 'Target GPU has a compute process')
    return dict(index=0, uuid=uuid, name=fields[2], prelaunch_memory_used_MiB=0, prelaunch_utilization_percent=0)


class RaftArguments(argparse.Namespace):
    # Official RAFT checks membership with "'dropout' not in args".
    def __contains__(self, key):
        return hasattr(self, key)


def run(a, out, started, progress):
    # Keep the staged official source tree read-only, including import caches.
    sys.dont_write_bytecode = True
    progress['phase'] = 'authenticate_assets_and_images'
    assets, verified = check_assets(a)
    images, pair = fixed_pair(a)
    progress['phase'] = 'check_gpu_idle'
    gpu = check_idle_gpu(a.gpu_uuid)
    import numpy as np
    import torch
    progress['phase'] = 'load_official_source_and_cpu_weights'
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one CUDA device required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    core = (a.asset_root / 'source/core').resolve()
    for name in ('raft', 'update', 'extractor', 'corr', 'utils', 'utils.utils'):
        require(name not in sys.modules, 'Official import namespace already occupied: ' + name)
    sys.path.insert(0, str(core))
    raft_module = importlib.import_module('raft')
    utilities = importlib.import_module('utils.utils')
    for name, relative in (('raft', 'raft.py'), ('update', 'update.py'), ('extractor', 'extractor.py'),
                           ('corr', 'corr.py'), ('utils', 'utils/__init__.py'), ('utils.utils', 'utils/utils.py')):
        require(Path(sys.modules[name].__file__).resolve() == core / relative, 'Official module origin differs: ' + name)
    settings = RaftArguments(small=False, mixed_precision=False, alternate_corr=False, dropout=0.)
    model = raft_module.RAFT(settings)
    checkpoint = beneath(a.asset_root, assets['checkpoint']['path'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(isinstance(state, dict) and state and all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()),
            'Expected a tensor state_dict, not a training payload')
    prefixes = [key.startswith('module.') for key in state]
    require(all(prefixes) or not any(prefixes), 'Mixed module. checkpoint prefixes')
    stripped = all(prefixes)
    if stripped:
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, 'module.')
    require(all(not value.is_floating_point() or (value.dtype == torch.float32 and torch.isfinite(value).all())
                for value in state.values()), 'Checkpoint is not finite float32')
    loaded = model.load_state_dict(state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, 'Strict checkpoint load failed')
    del state
    model.eval().requires_grad_(False)
    require(all(not m.training for m in model.modules()), 'Model must remain in eval mode')
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    model = model.to(device='cuda:0', dtype=torch.float32)
    tensors = [torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device='cuda:0', dtype=torch.float32) for rgb in images]
    padder = utilities.InputPadder(tensors[0].shape)
    first, second = padder.pad(*tensors)
    require(list(first.shape) == list(second.shape) == [1, 3, 904, 1600] and padder._pad == [0, 0, 2, 2],
            'Official padding differs')
    overall_allocated = int(torch.cuda.max_memory_allocated(0)); overall_reserved = int(torch.cuda.max_memory_reserved(0))
    directions, flows = {}, {}
    with torch.no_grad():
        for name, image1, image2 in (('current_to_past', first, second), ('past_to_current', second, first)):
            progress['phase'] = 'forward_' + name
            torch.cuda.synchronize(0); torch.cuda.reset_peak_memory_stats(0)
            tick = time.monotonic()
            low, high = model(image1, image2, iters=20, flow_init=None, upsample=True, test_mode=True)
            torch.cuda.synchronize(0); forward_seconds = time.monotonic() - tick
            require(high.shape == (1, 2, 904, 1600) and high.dtype == torch.float32, 'Official flow output contract')
            unpadded = padder.unpad(high).contiguous()
            require(unpadded.shape == (1, 2, 900, 1600) and torch.isfinite(unpadded).all(), 'Unpadded flow shape/finite')
            flow = unpadded.cpu().numpy().copy()
            flows['flow_' + name] = flow
            peak_allocated = int(torch.cuda.max_memory_allocated(0)); peak_reserved = int(torch.cuda.max_memory_reserved(0))
            overall_allocated = max(overall_allocated, peak_allocated); overall_reserved = max(overall_reserved, peak_reserved)
            directions[name] = dict(forward_seconds=forward_seconds, forward_validation_copy_seconds=time.monotonic()-tick,
                peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                shape=list(flow.shape), dtype=str(flow.dtype), finite=True,
                decoded_array_sha256=hashlib.sha256(flow.tobytes()).hexdigest())
            del low, high, unpadded
    require(all(p.grad is None for p in model.parameters()) and all(not m.training for m in model.modules()), 'Unexpected training/gradient state')
    progress['phase'] = 'save_and_roundtrip'
    with (out / 'flows.npz').open('xb') as stream:
        np.savez_compressed(stream, **flows)
    with np.load(out / 'flows.npz', allow_pickle=False) as saved:
        require(set(saved.files) == set(flows) and all(saved[key].dtype == np.float32 and
                np.array_equal(saved[key], flows[key]) for key in flows), 'Flow save roundtrip differs')
    # Recheck every consumed asset byte, including the weight, after the two calls.
    check_assets(a)
    elapsed = time.monotonic() - started
    require(elapsed < 180, 'Probe exceeded inner budget')
    report = dict(schema=SCHEMA, status='COMPLETE_RESOURCE_PROBE_ONLY', source_sha256=sha(__file__),
        asset_manifest_sha256=a.asset_manifest_sha256, asset_complete_sha256=sha(a.asset_root / 'complete.json'),
        official_repository=assets['repository'], official_commit=COMMIT, verified_consumed_assets=verified,
        checkpoint_sha256=assets['checkpoint']['sha256'], checkpoint_load=dict(weights_only=True, map_location='cpu',
            strict=True, removed_uniform_module_prefix=stripped), input=pair, gpu=gpu,
        settings=dict(model='full_RAFT_things', iterations=20, dtype='float32', input_range=[0, 255],
            input_RGB=True, official_internal_normalization=True, eval=True, no_grad=True, flow_init=None,
            mixed_precision=False, alternate_corr=False, matmul_allow_tf32=False, cudnn_allow_tf32=False,
            cudnn_benchmark=False, original_shape=[1, 3, 900, 1600], padded_shape=[1, 3, 904, 1600],
            pad_left_right_top_bottom=[0, 0, 2, 2], output_axes='B,dx_dy,H,W', output_units='pixels',
            reverse_is_independent_model_call=True, reverse_is_negated_forward=False, warmup_calls=0),
        directions=directions, elapsed_seconds=elapsed, peak_allocated_bytes=overall_allocated,
        peak_reserved_bytes=overall_reserved, max_seconds=180, optimizer_updates=0,
        weights_saved=False, GT_read=False, O_or_D_loaded=False, quality_metrics_computed=False,
        dependencies_installed=False, automatic_fetch=False, raw_RGB_images_unchanged=True,
        runtime=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), GPU_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='One predetermined image pair and two directions; resource/engineering evidence, not flow accuracy or full512 cost.')
    write(out / 'report.json', report)
    write(out / 'complete.json', dict(schema=SCHEMA, status=report['status'], source_sha256=report['source_sha256'],
        files_sha256={name: sha(out / name) for name in ('report.json', 'flows.npz')}))
    progress['phase'] = 'complete'
    print(json.dumps(dict(status=report['status'], seconds=elapsed, peak_allocated_bytes=overall_allocated)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('asset-root', 'history-inputs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--asset-manifest-sha256', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--max-seconds', type=int, default=180)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); progress = dict(phase='initial'); failure = None
    def stop(signum, frame):
        raise TimeoutError('Probe time/termination signal ' + str(signum))
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    signal.alarm(180)
    try:
        require(a.max_seconds == 180, 'Fixed inner budget is 180 seconds')
        run(a, a.out, started, progress)
    except BaseException as exc:
        failure = dict(schema=SCHEMA, status='FAILED_RESOURCE_PROBE', source_sha256=sha(__file__),
            phase=progress['phase'], error=repr(exc), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started, optimizer_updates=0, is_quality_result=False,
            gpu_uuid=a.gpu_uuid, asset_manifest_sha256=a.asset_manifest_sha256,
            peak_allocated_bytes=None, peak_reserved_bytes=None)
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            try:
                failure['peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated(0))
                failure['peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved(0))
            except Exception as memory_error:
                failure['memory_query_error'] = repr(memory_error)
        raise
    finally:
        signal.alarm(0)
        if failure is not None:
            write(a.out / 'failed.json', failure)


if __name__ == '__main__':
    main()
