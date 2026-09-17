"""CPU-only authenticated export of the existing native train512 radar input.

No model construction/forward, targets, future annotations, fitting or GPU.
Preserves full C,Y,X fields, including missing cells, without GT selection.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

SCHEMA = 'radar-evidence-train-extraction-v1'
EXPECTED = {
    'index': '1dc14643df86f85c5ae87b17b83c60cada99f189a025675d150062acb046e7b1',
    'complete': '69e49dbd8c7667e9c6e17c2ce70b4f879e8f40c33a0018251b9de2ae978d01bc',
    'selection': '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d',
    'native': '41953909ddfdfd08d51d98385a450640e5314b775ab01458cb55ff1b18e4bfb0',
    'runtime': '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def identity(record):
    return {k: record[k] for k in ('sample_token', 'scene_token', 'official_index', 'split')}


def run(a, out, started):
    root = Path(a.train_cache)
    paths = dict(index=root/'index.json', complete=root/'complete.json',
                 selection=Path(a.selection), native=Path(a.native), runtime=Path(a.runtime))
    require(all(sha(paths[k]) == v for k, v in EXPECTED.items()), 'Original input contract changed')
    for path, digest in read(a.runtime)['runtime_source_sha256'].items():
        require(sha(path) == digest, 'Native runtime source changed: ' + path)
    index = read(paths['index'])
    records = index['records']
    selected = [r for r in read(a.selection)['records'] if r['split'] == 'train']
    require([identity(r) for r in records] == selected and len(records) == 512, 'Training identities differ')
    require(len({r['scene_token'] for r in records}) == 256, 'Training scenes differ')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'This job must have no visible GPU')
    spec = importlib.util.spec_from_file_location('native_state_cache_radar_export', a.native)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    import torch
    torch.set_num_threads(2)
    native._prepare_repo(a.repo)
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    write(out/'manifest.json', dict(schema=SCHEMA, source_sha256=sha(__file__),
          original_contract_sha256=EXPECTED, split='train', samples=512, scenes=256,
          shape=[1, 8, 200, 200], layout='B,C,Y,X',
          frame='current LIDAR_TOP', cell_width_m=.512,
          channel_names=['presence', 'clipped_log_count', 'normalized_mean_z',
                         'normalized_mean_RCS', 'normalized_mean_vx_comp_R',
                         'normalized_mean_vy_comp_R', 'normalized_mean_speed_R', 'normalized_mean_lag'],
          inference=False, target_files_opened=False, model_constructed=False,
          environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__)))
    (out/'samples').mkdir()
    result = []
    for i, record in enumerate(records):
        require(time.monotonic()-started < a.max_seconds, 'Time budget exceeded')
        descriptor = record['files']['inputs']
        path = native._sample_directory(root, record)/descriptor['file']
        native._verify_file(path, descriptor)
        inputs = torch.load(str(path), map_location='cpu', weights_only=False)
        native.validate_inputs(inputs)
        radar = inputs['radar_bev']
        require(torch.is_tensor(radar) and radar.dtype == torch.float32 and
                tuple(radar.shape) == (1, 8, 200, 200), 'Radar shape or dtype differs')
        require(native._tensor_digest(radar) == record['native_radar_tensor_sha256'], 'Radar digest differs')
        array = radar.detach().cpu().contiguous().numpy().copy()
        require(np.isfinite(array).all() and np.isin(array[:, 0], [0, 1]).all(), 'Invalid input field')
        absent = array[0, 0] == 0
        require(np.all(array[0, :, absent] == 0), 'Missing radar cell unexpectedly nonzero')
        require(np.max(np.abs(array)) <= 1., 'Normalized radar field outside contract')
        name = 'samples/%04d_%s.npz' % (i, record['sample_token'])
        with (out/name).open('xb') as f:
            np.savez_compressed(f, radar_bev=array,
                                identity_json=np.asarray(json.dumps(identity(record), sort_keys=True)))
        with np.load(out/name, allow_pickle=False) as z:
            require(np.array_equal(z['radar_bev'], array), 'Export roundtrip differs')
        result.append(dict(ordinal=i, identity=identity(record), file=name,
                           sha256=sha(out/name), bytes=(out/name).stat().st_size,
                           input_sha256=descriptor['sha256'],
                           native_radar_tensor_sha256=record['native_radar_tensor_sha256'],
                           occupied_xy_cells=int((~absent).sum())))
        del inputs, radar, array
        require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
        print(json.dumps(dict(event='sample_complete', completed=i+1,
                              seconds=time.monotonic()-started)), flush=True)
    require(sha(a.native) == EXPECTED['native'], 'Native source changed during extraction')
    write(out/'index.json', dict(schema=SCHEMA, records=result))
    write(out/'summary.json', dict(schema=SCHEMA, status='COMPLETE_CPU_RADAR_TRAIN_EXPORT',
          samples=512, scenes=256, seconds=time.monotonic()-started,
          occupied_xy_cells=sum(r['occupied_xy_cells'] for r in result),
          optimizer_updates=0, model_forward=False, CUDA_initialized=False,
          all_input_files_and_radar_digests_verified=True, all_saved_arrays_roundtrip_exact=True))
    write(out/'complete.json', dict(schema=SCHEMA, status='COMPLETE_CPU_RADAR_TRAIN_EXPORT',
          samples=512, source_sha256=sha(__file__),
          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          files_sha256={k: sha(out/k) for k in ('manifest.json', 'index.json', 'summary.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('train-cache', 'selection', 'native', 'runtime', 'repo', 'out'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--max-seconds', type=int, default=900)
    a = parser.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def terminate(signum, frame):
        raise TimeoutError('Signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, terminate)
    signal.alarm(a.max_seconds)
    try:
        require(a.max_seconds == 900, 'Resource budget changed')
        run(a, out, started)
    except BaseException as exc:
        signal.alarm(0)
        write(out/'failed.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
