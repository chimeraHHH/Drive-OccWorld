"""Lossless current-prediction full-grid geometry cache; no Torch or GT support.

Default CLI authenticates assets only. Explicit build constructs frozen train512
and development200; dev0-roundtrip constructs a non-formal one-sample QA cache.
An existing output is never resumed or overwritten. Callers provide an external
complete SHA: manifests are evidence bindings, not signatures or model results.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

SOURCES = {
    'shared_rigid_prediction_geometry_v1.py': 'ef5ea1f0a2e8be5c99ffd366038fd39bc759e4cea1c985824931ce3363b7155a',
    'crn_full_grid_geometry_v1.py': '1c1e7f0799b593b3ad41f282d8506bdc1312a1a8e53994ea8ba595ad12055b9c',
    'object_state_prediction_inputs_v1.py': '55e39715fdf19e5ae1f227a943a843b194eabed08e788ccb2694a040d6f18256',
    'object_state_forecast_development_inputs_v1.py': '7cd498d13d45d5960c6f494ebf7b750718b397d90720ecd745a8c527b0dace06',
    'motion_geometry.py': 'e9d232c3f7aaacd073cfda645868e357afad9e5a2684d07b2f3f2394bd796a31',
}
SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
TRAIN = dict(complete_sha256='82433bc87db02dd68d8672e094a56a38dc988390631d48e4ceab655f9460ce98',
             manifest_sha256='aad73753976920ce72861436ed1d648fe137647b37584c6d7e90a926d48e4695',
             predictions_sha256='b7cab52b8c5c68d6afe1a0dab3728c9878067be826a7eb6e9a0ae7f37e1953ee',
             inference_source_sha256='6a21dbf78da927865f490101ed0d31691f7c244074c7c90d4eb9e3487d5197f8')
DEVELOPMENT = dict(complete_sha256='650f19ae2709bbb8af39ed8c70dcebb7cbb720b7a6bc5fdfa6fc7040a9361b7d',
                   manifest_sha256='8d95afb4ca7e8f86720bdc00b6b1f62084aa76aa081352d4bab08d3c38237d36',
                   predictions_sha256='e1558af2a51ee40e976525d0679135066f93537fc8cfcc873511be44615cf05f')
SHAPE = (200, 200, 16)
EXTENT = (-51.2, -51.2, -5., 51.2, 51.2, 3.)
IDENTITY = ('sample_token', 'scene_token', 'official_index', 'split')
ARRAYS = ('owner', 'owner_original_indices', 'cv_velocity_R', 'c0_R', 'R0_R',
          'wlh', 'velocity_R', 'score', 'classes', 'retained_original_box_indices')
ALIASES = dict(center='c0_R', size='wlh', rotation='R0_R', classes='classes',
               score='score', velocity='velocity_R')
STATUS = 'COMPLETE_SHARED_RIGID_GEOMETRY_CACHE'
QA_STATUS = 'PASS_DEV0_SHARED_RIGID_CACHE_ROUNDTRIP'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def source_sha(): return sha(__file__)


def bind(name):
    p = Path(__file__).with_name(name)
    require(sha(p) == SOURCES[name], 'Frozen source differs: ' + name)
    spec = importlib.util.spec_from_file_location('_cache_' + p.stem, p)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def source_check():
    for name, digest in SOURCES.items():
        require(sha(Path(__file__).with_name(name)) == digest, 'Source differs: ' + name)


def stamp(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def array_receipt(value):
    value = np.asarray(value)
    require(not value.dtype.hasobject, 'Object arrays forbidden')
    return dict(dtype=value.dtype.str, shape=list(value.shape), nbytes=value.nbytes,
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def check_time(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('Geometry cache deadline exceeded; no retry')


def publish(path, writer):
    """Exclusive temp plus atomic no-clobber hard link; no existing file changes."""
    path = Path(path); tmp = path.with_name(path.name + '.partial')
    with tmp.open('xb') as f:
        writer(f); f.flush(); os.fsync(f.fileno())
    os.link(tmp, path); tmp.unlink()
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write_json(path, data):
    content = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    publish(path, lambda f: f.write(content))


def identity(row): return {key: row[key] for key in IDENTITY}


def selection_records(path):
    require(sha(path) == SELECTION_SHA, 'Frozen selection differs')
    rows = read(path)['records']
    train = [identity(r) for r in rows if r['split'] == 'train']
    dev = [identity(r) for r in rows if r['split'] == 'development']
    require(len(train) == 512 and len(dev) == 200, 'Selection coverage differs')
    require(len({r['sample_token'] for r in train + dev}) == 712, 'Duplicate identity')
    return dict(train=train, development=dev)


def authenticate_inputs(selection, train_root, dev_root, raw_root):
    source_check(); chosen = selection_records(selection)
    inputs = bind('object_state_prediction_inputs_v1.py')
    dev_inputs = bind('object_state_forecast_development_inputs_v1.py')
    train = inputs.PredictionInputs(train_root, raw_root, selection, TRAIN, chosen['train'])
    dev = dev_inputs.DevelopmentPredictionInputs(dev_root, raw_root, selection, chosen['development'], DEVELOPMENT)
    return dict(train=train, development=dev), dict(
        train_predictions=train.receipt, development_predictions=dev.receipt,
        raw_manifest_sha256=inputs.RAW_MANIFEST_SHA, raw_complete_sha256=inputs.RAW_COMPLETE_SHA)


def current_pose(raw_root, descriptor, expected, global_ordinal):
    require(descriptor['ordinal'] == global_ordinal and descriptor['identity'] == expected,
            'Raw descriptor identity differs')
    rel = Path(descriptor['file'])
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe carrier path')
    p = Path(raw_root) / rel; before = stamp(p)
    require(sha(p) == descriptor['sha256'], 'Raw carrier SHA differs')
    content = gzip.decompress(p.read_bytes())
    require(hashlib.sha256(content).hexdigest() == descriptor['uncompressed_json_sha256'], 'Raw JSON differs')
    carrier = json.loads(content); frame = carrier['frames'][2]
    require(carrier['identity'] == expected and carrier['ordinal'] == global_ordinal and
            carrier['selection_sha256'] == SELECTION_SHA and frame['relative_frame_index'] == 0 and
            frame['sample_token'] == expected['sample_token'], 'Current pose identity differs')
    G0 = np.asarray(frame['lidar_to_global_column_matrix'], dtype=np.float64)
    require(stamp(p) == before, 'Carrier changed during read')
    # No tracks, annotations, GT support, or future pose crosses this boundary.
    return G0


def validate_arrays(arrays, points):
    require(set(arrays) == set(ARRAYS), 'Cache array keys differ')
    m = len(arrays['score']); q = int(np.prod(SHAPE))
    sizes = dict(owner=(q,), owner_original_indices=(q,), cv_velocity_R=(q, 3),
                 c0_R=(m, 3), R0_R=(m, 3, 3), wlh=(m, 3), velocity_R=(m, 3),
                 score=(m,), classes=(m,), retained_original_box_indices=(m,))
    integers = {'owner', 'owner_original_indices', 'classes', 'retained_original_box_indices'}
    for key, value in arrays.items():
        require(value.shape == sizes[key] and value.dtype == (np.dtype('int64') if key in integers else np.dtype('float64')),
                'Array shape/dtype differs: ' + key)
        require(value.flags.c_contiguous and np.isfinite(value).all(), 'Nonfinite/noncontiguous: ' + key)
    require(points.shape == (q, 3) and points.dtype == np.dtype('float64') and np.isfinite(points).all(), 'Points differ')
    owner = arrays['owner']; covered = owner >= 0; retained = arrays['retained_original_box_indices']
    require(np.all((owner >= -1) & (owner < m)) and np.all(np.diff(retained) > 0) and
            np.all(retained >= 0) and np.all((arrays['classes'] >= 0) & (arrays['classes'] < 8)) and
            np.all(arrays['wlh'] > 0), 'Invalid retained objects/owner')
    require(np.array_equal(retained[owner[covered]], arrays['owner_original_indices'][covered]) and
            np.all(arrays['owner_original_indices'][~covered] == -1) and
            np.all(arrays['cv_velocity_R'][~covered] == 0), 'Owner/CV coverage differs')


def build_sample(task):
    split, ordinal, prediction, descriptor, raw_root, out, points_ledger, deadline = task
    check_time(deadline); tick = time.monotonic(); expected = identity(prediction)
    global_ordinal = ordinal if split == 'train' else 512 + ordinal
    G0 = current_pose(raw_root, descriptor, expected, global_ordinal)
    geometry = bind('shared_rigid_prediction_geometry_v1.py').geometry_from_centered_boxes(
        prediction['boxes'], G0, grid_shape_xyz=SHAPE, extent_xyz=EXTENT)
    require(array_receipt(geometry['points_R']) == points_ledger, 'Per-sample shared points differ')
    arrays = {k: np.ascontiguousarray(geometry[k]) for k in ARRAYS}
    validate_arrays(arrays, geometry['points_R'])
    ledgers = {k: array_receipt(a) for k, a in arrays.items()}
    rel = 'samples/%s/%04d_%s.npz' % (split, ordinal, expected['sample_token'])
    path = Path(out) / rel
    publish(path, lambda f: np.savez_compressed(f, **arrays))
    with np.load(path, allow_pickle=False) as loaded:
        require(set(loaded.files) == set(ARRAYS), 'Written NPZ keys differ')
        for key in ARRAYS:
            require(array_receipt(loaded[key]) == ledgers[key] and
                    loaded[key].tobytes(order='C') == arrays[key].tobytes(order='C'), 'Written bytes differ: ' + key)
    check_time(deadline)
    row = dict(split=split, ordinal=ordinal, raw_global_ordinal=global_ordinal, identity=expected,
               file=rel, sha256=sha(path), arrays=ledgers, raw_carrier=descriptor,
               current_G0_sha256=hashlib.sha256(np.asarray(G0, dtype='<f8').tobytes()).hexdigest(),
               prediction_record_sha256=hashlib.sha256(json.dumps(prediction, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
               geometry_receipt=geometry['receipt'], written_array_bytes_exact=True,
               geometry_and_storage_seconds=time.monotonic()-tick)
    receipt_rel = rel[:-4] + '.complete.json'
    write_json(Path(out)/receipt_rel, row)
    return dict(row, completion_file=receipt_rel, completion_sha256=sha(Path(out)/receipt_rel))


class GeometryCache:
    """No model/import/device work. Full cache is required unless explicit QA flag.

    load('development', i, identity) uses local i, never raw ordinal 512+i.
    Returned points_R is shared immutable storage. Other arrays are immutable too.
    """
    def __init__(self, root, complete_sha256, selection_path, *, allow_dev0_roundtrip=False):
        source_check(); self.root = Path(root); self.chosen = selection_records(selection_path)
        require(not (self.root/'failed.json').exists(), 'Failed cache cannot be consumed')
        require(sha(self.root/'complete.json') == complete_sha256, 'Cache complete SHA differs')
        done = read(self.root/'complete.json'); require(done['schema'] == 'shared-rigid-geometry-cache-complete-v1', 'Cache schema differs')
        qa = done['status'] == QA_STATUS
        require(done['status'] == STATUS or (allow_dev0_roundtrip and qa), 'Cache incomplete or QA-only')
        require(done['selection_sha256'] == SELECTION_SHA and done['source_sha256'] == source_sha(), 'Cache selection/source differs')
        require(set(done['files_sha256']) == {'manifest.json', 'points_R.npy'}, 'Top ledger differs')
        for name, digest in done['files_sha256'].items(): require(sha(self.root/name) == digest, 'Cache file differs: '+name)
        manifest = read(self.root/'manifest.json'); self.manifest = manifest
        require(manifest['schema'] == 'shared-rigid-geometry-cache-v1' and
                manifest['source_sha256'] == source_sha() and manifest['dependencies_sha256'] == SOURCES and
                manifest['selection_sha256'] == SELECTION_SHA and manifest['mode'] == ('dev0-roundtrip' if qa else 'build') and
                manifest['grid_shape_xyz'] == list(SHAPE) and manifest['extent_xyz'] == list(EXTENT), 'Manifest contract differs')
        for key, contract in (('train_predictions', TRAIN), ('development_predictions', DEVELOPMENT)):
            require(all(manifest['provenance'][key][k] == v for k, v in contract.items()), 'Prediction provenance differs: '+key)
        inputs = bind('object_state_prediction_inputs_v1.py')
        require(manifest['provenance']['raw_manifest_sha256'] == inputs.RAW_MANIFEST_SHA and
                manifest['provenance']['raw_complete_sha256'] == inputs.RAW_COMPLETE_SHA, 'Raw provenance differs')
        self.points_path = self.root/'points_R.npy'; self.points_stamp = stamp(self.points_path)
        self.points = np.load(self.points_path, allow_pickle=False); self.points.flags.writeable = False
        require(array_receipt(self.points) == manifest['points_array'] and stamp(self.points_path) == self.points_stamp, 'Points changed')
        require(self.points.shape == (640000, 3) and self.points.dtype == np.dtype('float64'), 'Full grid points differ')
        expected_keys = [('development', 0)] if qa else [(s, i) for s in ('train', 'development') for i in range(len(self.chosen[s]))]
        rows = manifest['records']; require([(r['split'], r['ordinal']) for r in rows] == expected_keys, 'Cache split/order/coverage differs')
        require(done['samples'] == len(rows) and done['split_counts'] == manifest['split_counts'] ==
                dict(train=0 if qa else 512, development=1 if qa else 200), 'Counts differ')
        self.records = {}
        for r in rows:
            split, i = r['split'], r['ordinal']; token = r['identity']['sample_token']
            require(r['identity'] == self.chosen[split][i] and r['raw_global_ordinal'] == (i if split == 'train' else 512+i), 'Cached identity differs')
            require(r['file'] == 'samples/%s/%04d_%s.npz' % (split, i, token) and
                    r['completion_file'] == r['file'][:-4]+'.complete.json', 'Cache file name differs')
            require(r['raw_carrier']['identity'] == r['identity'] and r['raw_carrier']['ordinal'] == r['raw_global_ordinal'] and
                    r['geometry_receipt']['current_G0_sha256'] == r['current_G0_sha256'] and r['written_array_bytes_exact'] is True,
                    'Cached pose/roundtrip differs')
            receipt_path = self.root/r['completion_file']
            require(sha(receipt_path) == r['completion_sha256'] and read(receipt_path) ==
                    {k:v for k,v in r.items() if k not in ('completion_file','completion_sha256')}, 'Sample completion differs')
            self.records[(split, i)] = r
        self.receipt = dict(complete_sha256=complete_sha256, manifest_sha256=done['files_sha256']['manifest.json'],
                            selection_sha256=SELECTION_SHA, source_sha256=source_sha(), dependencies_sha256=dict(SOURCES),
                            points_file_sha256=done['files_sha256']['points_R.npy'], samples=len(rows), split_counts=done['split_counts'],
                            **manifest['provenance'])

    def load(self, split, ordinal, identity):
        require(split in self.chosen and type(ordinal) is int and (split, ordinal) in self.records, 'Invalid split/local ordinal')
        r = self.records[(split, ordinal)]
        require(all(identity[k] == r['identity'][k] for k in IDENTITY), 'Requested cache identity differs')
        require(stamp(self.points_path) == self.points_stamp, 'Shared points file changed after authentication')
        p = self.root/r['file']; before = stamp(p)
        require(sha(p) == r['sha256'], 'NPZ SHA differs')
        with np.load(p, allow_pickle=False) as data:
            require(set(data.files) == set(ARRAYS), 'Loaded NPZ keys differ')
            arrays = {key: data[key] for key in ARRAYS}
        require(stamp(p) == before and {k:array_receipt(v) for k,v in arrays.items()} == r['arrays'], 'NPZ/array bytes changed')
        validate_arrays(arrays, self.points)
        for a in arrays.values(): a.flags.writeable = False
        result = dict(arrays, points_R=self.points, states_numpy={k:arrays[v] for k,v in ALIASES.items()},
                      receipt=dict(r['geometry_receipt']))
        result['receipt']['cache'] = dict(split=split, ordinal=ordinal, raw_global_ordinal=r['raw_global_ordinal'],
            identity=r['identity'], npz_sha256=r['sha256'], current_G0_sha256=r['current_G0_sha256'],
            raw_pose_carrier_sha256=r['raw_carrier']['sha256'], complete_sha256=self.receipt['complete_sha256'])
        return result


def run(args):
    started = time.monotonic(); initial_source_sha = source_sha()
    assets, provenance = authenticate_inputs(args.selection, args.train_predictions, args.development_predictions, args.raw_metadata)
    if args.mode == 'check':
        return dict(status='AUTHENTICATED_INPUTS_ONLY', provenance=provenance, sources=SOURCES, source_sha256=source_sha(), optimizer_updates=0)
    require(args.out is not None and args.max_seconds is not None and 0 < args.max_seconds <= 43200, 'Explicit new out and positive bounded max-seconds required')
    deadline = started + args.max_seconds; check_time(deadline)
    require(1 <= args.workers <= 4, 'Workers must be 1..4')
    if args.workers > 1:
        require(all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')), 'Parallel build requires explicit 2-thread environment before Python')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    for split in ('train','development'): (out/'samples'/split).mkdir(parents=True)
    points = bind('motion_geometry.py').voxel_centers_xyz(SHAPE, EXTENT).reshape(-1, 3)
    point_ledger = array_receipt(points)
    publish(out/'points_R.npy', lambda f: np.save(f, points, allow_pickle=False))
    require(np.load(out/'points_R.npy', allow_pickle=False).tobytes() == points.tobytes(), 'Written shared points differ')
    splits = ('development',) if args.mode == 'dev0-roundtrip' else ('train','development')
    def tasks():
        for split in splits:
            asset = assets[split]; records = asset.records[:1] if args.mode == 'dev0-roundtrip' else asset.records
            for i, record in enumerate(records):
                yield (split, i, record, asset.raw[record['sample_token']], str(asset.raw_root), str(out), point_ledger, deadline)
    rows = []
    try:
        if args.workers == 1:
            results = map(build_sample, tasks())
            for row in results:
                rows.append(row); check_time(deadline)
                print(json.dumps(dict(completed=len(rows),split=row['split'],ordinal=row['ordinal'],seconds=time.monotonic()-started)), flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                for row in pool.map(build_sample, tasks(), chunksize=1):
                    rows.append(row); check_time(deadline)
                    print(json.dumps(dict(completed=len(rows),split=row['split'],ordinal=row['ordinal'],seconds=time.monotonic()-started)), flush=True)
        source_check(); check_time(deadline)
        require(source_sha() == initial_source_sha, 'Builder source changed during run')
        manifest = dict(schema='shared-rigid-geometry-cache-v1', mode=args.mode, source_sha256=source_sha(),
            dependencies_sha256=SOURCES, selection_sha256=SELECTION_SHA, provenance=provenance,
            grid_shape_xyz=list(SHAPE), extent_xyz=list(EXTENT), coordinate_order='XYZ_C_order_Z_fastest',
            points_array=point_ledger, records=rows, split_counts={s:sum(r['split']==s for r in rows) for s in ('train','development')},
            no_GT_or_sparse_support_input=True, lossless_compression_only=True, origin_adaptation_performed=False,
            workers=args.workers, thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            numpy_version=np.__version__, seconds=time.monotonic()-started, max_seconds=args.max_seconds)
        write_json(out/'manifest.json', manifest); check_time(deadline)
        done = dict(schema='shared-rigid-geometry-cache-complete-v1', status=QA_STATUS if args.mode=='dev0-roundtrip' else STATUS,
                    source_sha256=source_sha(), selection_sha256=SELECTION_SHA, samples=len(rows), split_counts=manifest['split_counts'],
                    files_sha256={name:sha(out/name) for name in ('manifest.json','points_R.npy')}, optimizer_updates=0,
                    seconds=time.monotonic()-started)
        check_time(deadline); write_json(out/'complete.json', done)
        if args.mode == 'dev0-roundtrip':
            cache = GeometryCache(out, sha(out/'complete.json'), args.selection, allow_dev0_roundtrip=True)
            actual = cache.load('development', 0, rows[0]['identity'])
            require(array_receipt(actual['points_R']) == point_ledger and
                    {k:array_receipt(actual[k]) for k in ARRAYS} == rows[0]['arrays'] and
                    all(actual['states_numpy'][k] is actual[v] for k,v in ALIASES.items()), 'Actual loader roundtrip differs')
            require(not actual['points_R'].flags.writeable, 'Shared points not readonly')
            check_time(deadline)
            qa = dict(status='PASS_ACTUAL_DEV0_LOADER_ROUNDTRIP', complete_sha256=sha(out/'complete.json'),
                      identity=rows[0]['identity'], all_ten_arrays_and_points_bytes_exact=True, states_aliases_exact=True,
                      actual_cache_load_performed=True, numpy_version=np.__version__, seconds=time.monotonic()-started,
                      GT_or_sparse_support_input=False, torch_imported='torch' in __import__('sys').modules)
            write_json(out/'roundtrip.json', qa)
        return dict(done, complete_sha256=sha(out/'complete.json'))
    except BaseException as exc:
        if not (out/'failed.json').exists():
            write_json(out/'failed.json', dict(status='FAILED_NO_RETRY', exception_type=type(exc).__name__, error=str(exc), seconds=time.monotonic()-started))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('check','build','dev0-roundtrip'), default='check')
    p.add_argument('--selection', required=True); p.add_argument('--train-predictions', required=True)
    p.add_argument('--development-predictions', required=True); p.add_argument('--raw-metadata', required=True)
    p.add_argument('--out'); p.add_argument('--max-seconds', type=float); p.add_argument('--workers', type=int, default=1)
    print(json.dumps(run(p.parse_args()), sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__': main()
