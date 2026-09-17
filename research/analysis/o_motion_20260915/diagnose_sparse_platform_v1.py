"""Read-only CPU comparison of the first full/old-dev intersection.

No output arrays, modified labels, model reads, acceptance tolerance or repair.
All numeric differences are descriptive; production byte-exact gates stay intact.
Run the same file in each environment to compare NumPy/BLAS evidence.
"""
import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

import build_full_sparse_motion_supervision_v1 as full
import build_sparse_motion_supervision_v1 as old

FULL_BUILDER_SHA = '3c5bcf88a70361e0b4649a3c6d2b8869e6c0e8b8cdc9788e3b1d86a191f07e7a'
TOKEN = '296fcfbf2e29489699f1cb5631f38ff5'


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def differences(actual, expected):
    result = dict(actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
                  actual_shape=list(actual.shape), expected_shape=list(expected.shape),
                  actual_bytes_sha256=digest(actual), expected_bytes_sha256=digest(expected))
    compatible = actual.dtype == expected.dtype and actual.shape == expected.shape
    result['dtype_shape_equal'] = bool(compatible)
    result['bytes_exact'] = bool(compatible and actual.tobytes() == expected.tobytes())
    if not compatible:
        return result
    unequal_bits = np.any(actual.reshape(-1).view(np.uint8).reshape(-1, actual.dtype.itemsize)
                          != expected.reshape(-1).view(np.uint8).reshape(-1, expected.dtype.itemsize), axis=1)
    result['elements'] = int(actual.size)
    result['bitwise_different_elements'] = int(unequal_bits.sum())
    result['numerically_different_elements'] = int(np.count_nonzero(actual != expected))
    if actual.dtype.kind == 'f':
        full.require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'Nonfinite diagnostic array')
        error = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        result['max_absolute_difference'] = float(error.max(initial=0.))
        result['actual_absmax'] = float(np.abs(actual).max(initial=0.))
        result['expected_absmax'] = float(np.abs(expected).max(initial=0.))
        bits = 8*actual.dtype.itemsize
        full.require(bits in (32, 64), 'Unsupported float ULP diagnostic')
        unsigned_dtype = np.dtype('uint'+str(bits))
        sign = np.array(1 << (bits-1), dtype=unsigned_dtype)
        def ordered(x):
            u = x.view(unsigned_dtype)
            return np.where((u & sign) != 0, ~u, u | sign)
        ua, ue = ordered(actual), ordered(expected)
        ulp = np.maximum(ua, ue)-np.minimum(ua, ue)
        result['max_ULP_distance_including_signed_zero'] = int(ulp.max(initial=0))
        result['ULP_note'] = 'Monotone IEEE bit rank; near-zero cancellation can have large ULP count despite tiny metre error.'
        result['examples'] = []
        positions = np.flatnonzero(unequal_bits)[:12]
        for position in positions:
            index = np.unravel_index(int(position), actual.shape)
            result['examples'].append(dict(index=list(map(int, index)),
                actual=float(actual[index]), expected=float(expected[index]),
                actual_hex=float(actual[index]).hex(), expected_hex=float(expected[index]).hex(),
                absolute_difference=float(error[index]), ulp_distance=int(ulp[index])))
        if actual.ndim and actual.shape[0] == 4:
            result['per_horizon'] = [dict(bitwise_different_elements=int(unequal_bits.reshape(actual.shape)[h].sum()),
                numerically_different_elements=int(np.count_nonzero(actual[h] != expected[h])),
                max_absolute_difference=float(error[h].max(initial=0.)),
                max_ULP_distance_including_signed_zero=int(ulp[h].max(initial=0))) for h in range(4)]
    return result


def compare_arrays(actual, expected, exclude=()):
    full.require(set(actual) == set(expected), 'Array key sets differ')
    fields = {k: differences(actual[k], expected[k]) for k in sorted(set(actual)-set(exclude))}
    return dict(all_compared_fields_bytes_exact=all(v['bytes_exact'] for v in fields.values()),
                excluded_binding_fields=list(exclude), fields=fields)


def load_raw(root, descriptor):
    payload = (root/descriptor['file']).read_bytes()
    full.require(hashlib.sha256(payload).hexdigest() == descriptor['sha256'], 'Raw payload SHA mismatch')
    unpacked = gzip.decompress(payload)
    full.require(hashlib.sha256(unpacked).hexdigest() == descriptor['uncompressed_json_sha256'], 'Unpacked SHA mismatch')
    return json.loads(unpacked)


def geometry_fingerprint(raw, points):
    g0 = np.asarray(raw['frames'][2]['lidar_to_global_column_matrix'], dtype=np.float64)
    tracks = [t for t in raw['tracks'] if t['valid_mask'][2]]
    transforms = []
    for t in tracks:
        a0 = old.box_to_global(t['global_centers_m'][2], t['global_rotations_wxyz'][2])
        for frame in old.FUTURE:
            if t['valid_mask'][frame]:
                ah = old.box_to_global(t['global_centers_m'][frame], t['global_rotations_wxyz'][frame])
                transform = np.linalg.inv(g0) @ ah @ np.linalg.inv(a0) @ g0
                transforms.append(dict(instance_token=t['instance_token'], frame=frame,
                    transport_sha256=digest(transform), transport_float64=transform.tolist()))
    return dict(grid_centers_sha256=digest(points), G0_sha256=digest(g0),
                inverse_G0_sha256=digest(np.linalg.inv(g0)), transforms=transforms)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-targets', required=True)
    p.add_argument('--old-targets', required=True)
    p.add_argument('--reference-sparse', required=True)
    p.add_argument('--selection', required=True)
    a = p.parse_args(argv)
    started = time.monotonic()
    source = Path(__file__).resolve()
    full.require(full.sha(Path(full.__file__)) == FULL_BUILDER_SHA, 'Changed full builder')
    provenance = full.protected_source_contract(Path(full.__file__))
    full.require(full.sha(Path(old.__file__).with_name('motion_geometry.py')) == full.GEOMETRY_SHA, 'Changed geometry')
    fullroot, oldroot, reference = map(lambda x: Path(x).resolve(),
                                      (a.full_targets, a.old_targets, a.reference_sparse))
    selected, lookup, _ = full.load_bindings(fullroot, reference, Path(a.selection).resolve())
    descriptor = next(r for r in selected if r['identity']['sample_token'] in lookup)
    full.require(descriptor['identity']['sample_token'] == TOKEN and descriptor['ordinal'] == 8, 'Not first failed intersection')
    full.require(full.sha(oldroot/'manifest.json') == old.RAW_MANIFEST_SHA
                 and full.sha(oldroot/'complete.json') == old.RAW_COMPLETE_SHA, 'Changed original raw labels')
    old_descriptor = next(r for r in json.loads((oldroot/'manifest.json').read_text())['records']
                          if r['identity']['sample_token'] == TOKEN)
    old_sparse = lookup[TOKEN]
    full.require(full.sha(reference/old_sparse['file']) == old_sparse['sha256'], 'Changed stored NPZ')
    with np.load(reference/old_sparse['file'], allow_pickle=False) as saved:
        arrays_saved = {k: saved[k].copy() for k in saved.files}
    old_raw, full_raw = load_raw(oldroot, old_descriptor), load_raw(fullroot, descriptor)
    raw_binding = {'selection_sha256', 'ordinal', 'identity'}
    full.require(set(old_raw) == set(full_raw), 'Raw schema keys differ')
    full.require({k:v for k,v in old_raw.items() if k not in raw_binding}
                 == {k:v for k,v in full_raw.items() if k not in raw_binding}, 'Raw physical data differ')
    points = old.voxel_centers_xyz(old.SHAPE, old.EXTENT).reshape(-1, 3)
    arrays_old, counts_old, _ = old.convert(old_raw, old_descriptor, points)
    arrays_full, counts_full, _ = full.convert(full_raw, descriptor, points)
    arrays_repeat, counts_repeat, _ = full.convert(full_raw, descriptor, points)
    config_text = io.StringIO()
    with contextlib.redirect_stdout(config_text):
        np.show_config()
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    result = dict(schema='sparse-platform-comparison-v1', status='DIAGNOSTIC_COMPLETE',
        script_sha256=full.sha(source), source_provenance=provenance,
        full_builder_sha256=FULL_BUILDER_SHA, token=TOKEN,
        full_descriptor=descriptor, original_raw_descriptor=old_descriptor,
        original_sparse_npz_sha256=old_sparse['sha256'],
        environment=dict(platform=platform.platform(), machine=platform.machine(),
            python=sys.version, numpy=np.__version__, blas_config=config_text.getvalue(),
            threadpool_info=pools, thread_environment={k:os.environ.get(k) for k in
                ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}),
        raw_physical_fields_exact=True, raw_metadata_exclusions=sorted(raw_binding),
        stored_vs_old_recomputed=compare_arrays(arrays_old, arrays_saved),
        stored_vs_full_recomputed=compare_arrays(arrays_full, arrays_saved, sorted(full.BINDING_FIELDS)),
        old_vs_full_same_runtime=compare_arrays(arrays_full, arrays_old, sorted(full.BINDING_FIELDS)),
        full_repeat_same_runtime=compare_arrays(arrays_repeat, arrays_full),
        counts_all_exact=(counts_old == counts_full == counts_repeat == old_sparse['counts']),
        geometry_intermediates=geometry_fingerprint(old_raw, points),
        model_or_prediction_read=False, GPU_used=False, optimizer_steps=0,
        label_or_frozen_source_modified=False, seconds=time.monotonic()-started,
        limits='One fixed failed anchor; NumPy/BLAS/ISA attribution requires cross-environment comparison, not inferred from byte mismatch alone.')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
