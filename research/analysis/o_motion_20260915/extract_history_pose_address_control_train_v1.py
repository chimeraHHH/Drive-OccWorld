"""CPU history-GT pose address control on frozen train512 image queries.

This privileged diagnostic is not an input-only or deployable feature. Raw
label files contain seven frames; only slots [-2,-1,0] reach the pose core.
No future pose, occupancy label, training or model forward is used.
"""
import argparse
import datetime
import gzip
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
from PIL import Image

from extract_history_camera_evidence_train_v1 import require, sha, read, write, authenticated_ledger

SCHEMA = 'history-pose-address-control-train-v1'
PARENT_SHA = '0c0d2e640a12e3299e8dacc03189776669cd2e50036a65de823026304119b648'
HISTORY_SHA = '040d62242a541b9685f3f39f1534b3663b0e1142b4cd7d9737cdbfd15d94314d'
MOTION_SHA = '3bd429effb1fbe50ac0aeb77330bc0100ee144bf8406fbd078097fb336301a73'
RAW_MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
RAW_COMPLETE_SHA = 'e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69'
POINTS_SHA = '91bcc80882e41ffd5234cecce4b517ef4081839c89dbb6a048c4226b723ca6fd'
OLD_CORE_SHA = 'c7469179d1628979b30eb256e695fdd6479ffd6e2df047d380d890512f57a5ea'
OLD_EXTRACTOR_SHA = '263103e142b14348b832ace257e23e8cbbecb22073124151f446247f5fa747c0'
TRACK_FIELDS = ('valid_mask', 'annotation_tokens', 'annotation_prev_tokens', 'annotation_next_tokens',
                'global_centers_m', 'global_rotations_wxyz')


def load_module(path, digest):
    require(sha(path) == digest, 'Frozen module changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_npz(root, row):
    p = root / row['file']
    require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Original NPZ changed')
    with np.load(p, allow_pickle=False) as z:
        values = {k: z[k] for k in z.files}
    require(json.loads(values['identity_json'].item()) == row['identity'], 'NPZ sample identity')
    return values


def array_digests(arrays):
    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}


def truncate_history(raw):
    require([f['relative_frame_index'] for f in raw['frames'][:3]] == [-2, -1, 0], 'History slots differ')
    return dict(identity=raw['identity'], frames=raw['frames'][:3],
                tracks=[dict(instance_token=t['instance_token'], **{k: t[k][:3] for k in TRACK_FIELDS})
                        for t in raw['tracks']])


def parent_patch_check(old, cached, pairs):
    """Fixed first anchor, first eight original-valid points per camera only."""
    count = 0; maximum = 0.
    for ci, name in enumerate(old.CAMERA_ORDER):
        ix = np.flatnonzero(cached['valid'] & (cached['camera_index'] == ci))[:8]
        if not len(ix):
            continue
        gray = [old.grayscale(pairs[name][t]['rgb']) for t in ('current', 'past')]
        for hyp in range(2):
            first = old.sample_patches(gray[0], cached['uv'][ix, hyp, 0])
            for broken in (False, True):
                last = old.sample_patches(gray[1], cached['uv'][ix, hyp, 1], gray[1].shape[1] // 2 if broken else 0)
                value = old.regularized_zncc(first, last)
                target = cached['broken_correlation' if broken else 'correlation'][ix, hyp]
                difference = np.abs(value - target)
                # Same server/runtime/functions/recorded UV: require exact stored values.
                require(np.array_equal(value, target), 'First-anchor original patch recomputation differs')
                maximum = max(maximum, float(difference.max(initial=0.))); count += len(ix)
    return dict(anchor_ordinal=0, compared_correlations=count, maximum_absolute_error=maximum,
                first_eight_valid_points_per_camera=True, all512_images_recomputed_for_new_reference=True)


def run(a, out, started):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU job must hide GPUs')
    protocol = read(a.protocol)
    require(protocol['schema'] == SCHEMA and protocol['status'] == 'FROZEN' and
            protocol['resources'] == dict(max_seconds=600, cpu_threads=2), 'Protocol or resources differ')
    require(protocol['extractor_sha256'] == sha(__file__) and
            protocol['parent_core_sha256'] == OLD_CORE_SHA and
            protocol['camera_complete_sha256'] == PARENT_SHA and
            protocol['history_inputs_complete_sha256'] == HISTORY_SHA and
            protocol['motion_complete_sha256'] == MOTION_SHA and
            protocol['raw_manifest_sha256'] == RAW_MANIFEST_SHA and
            protocol['raw_complete_sha256'] == RAW_COMPLETE_SHA and
            protocol['points_file_sha256'] == POINTS_SHA, 'Protocol source/input binding differs')
    here = Path(__file__).resolve().parent
    require(sha(here / 'extract_history_camera_evidence_train_v1.py') == OLD_EXTRACTOR_SHA, 'Parent helper changed')
    old = load_module(here / 'history_camera_evidence_v1.py', OLD_CORE_SHA)
    core = load_module(a.module, protocol['core_sha256'])
    parent, motion, history, rawroot = map(Path, (a.camera_run, a.motion_run, a.history_inputs, a.raw_targets))
    pc = authenticated_ledger(parent, PARENT_SHA); mc = authenticated_ledger(motion, MOTION_SHA)
    hc = authenticated_ledger(history, HISTORY_SHA)
    require(sha(rawroot / 'manifest.json') == RAW_MANIFEST_SHA and sha(rawroot / 'complete.json') == RAW_COMPLETE_SHA,
            'Original raw target contract differs')
    require(sha(a.points_file) == POINTS_SHA, 'Original material grid differs')
    points = np.load(a.points_file, allow_pickle=False)
    require(points.shape == (640000, 3) and points.dtype == np.float64, 'Original grid layout')
    records = read(parent / 'index.json')['records']; mr = read(motion / 'index.json')['records']
    hr = read(history / 'records.json')['records']
    rr = [r for r in read(rawroot / 'manifest.json')['records'] if r['identity']['split'] == 'train']
    require(len(records) == len(mr) == len(hr) == len(rr) == 512, 'Original train support incomplete')
    root = Path(read(history / 'manifest.json')['source_data_root'])
    manifest = dict(schema=SCHEMA, source_sha256=sha(__file__), protocol_sha256=sha(a.protocol),
        core_sha256=sha(a.module), parent_core_sha256=OLD_CORE_SHA,
        camera_complete_sha256=PARENT_SHA, camera_files_sha256=pc['files_sha256'],
        motion_complete_sha256=MOTION_SHA, motion_files_sha256=mc['files_sha256'],
        history_inputs_complete_sha256=HISTORY_SHA, history_inputs_files_sha256=hc['files_sha256'],
        raw_manifest_sha256=RAW_MANIFEST_SHA, raw_complete_sha256=RAW_COMPLETE_SHA, points_file_sha256=POINTS_SHA,
        scope=protocol['scope'], source_data_root=str(root),
        runtime=dict(python=sys.version, numpy=np.__version__, pillow=Image.__version__))
    write(out / 'manifest.json', manifest); (out / 'samples').mkdir()
    output_records = []; total = decoded = 0; reference_counts = {}; pose_counts = {}; audit = None
    for ordinal, (c, m, h, r) in enumerate(zip(records, mr, hr, rr)):
        require(time.monotonic() - started < a.max_seconds, 'CPU time budget')
        require(c['ordinal'] == m['ordinal'] == h['ordinal'] == r['ordinal'] == ordinal and
                c['identity'] == m['identity'] == h['identity'] == r['identity'], 'Original identities differ')
        cached = load_npz(parent, c); frozen_digest = array_digests(cached)
        motion_path = motion / m['file']
        require(motion_path.stat().st_size == m['bytes'] and sha(motion_path) == m['sha256'], 'Motion NPZ changed')
        with np.load(motion_path, allow_pickle=False) as z:
            idx, obj, tokens = z['source_flat_indices'], z['object_index'], z['instance_tokens']
        require(np.array_equal(idx, cached['source_flat_indices']) and len(idx) == c['source_points'], 'Material point support')
        blob = (rawroot / r['file']).read_bytes()
        require(len(blob) == r['bytes'] and hashlib.sha256(blob).hexdigest() == r['sha256'], 'Raw label compressed bytes')
        decoded_raw = gzip.decompress(blob)
        require(hashlib.sha256(decoded_raw).hexdigest() == r['uncompressed_json_sha256'], 'Raw uncompressed label bytes')
        raw = json.loads(decoded_raw)
        require(raw['identity'] == h['identity'] and raw['ordinal'] == ordinal, 'Raw identity')
        past_raw = truncate_history(raw); del raw, decoded_raw, blob
        frames = past_raw['frames']
        require(frames[2]['sample_token'] == h['identity']['sample_token'] and
                frames[1]['sample_token'] == h['previous_sample_token'] and
                frames[2]['lidar_timestamp_us'] == h['t0_lidar_us'] and
                np.allclose(frames[2]['lidar_to_global_column_matrix'], h['lidar_to_global'], rtol=0, atol=1e-9),
                'Historical label reference differs from original camera frame')
        pairs = {}
        for pair in h['cameras']:
            fpair = {}
            for when in ('current', 'past'):
                f = pair[when]; desc = f['image']; path = root / desc['file']
                require(path.stat().st_size == desc['bytes'] and sha(path) == desc['sha256'], 'Raw image changed')
                with Image.open(path) as im:
                    require(im.size == tuple(desc['size_wh']) and im.format == 'JPEG', 'Image header changed')
                    rgb = np.array(im.convert('RGB'), dtype=np.uint8)
                require(f['sample_data']['timestamp'] <= h['input_availability_us'], 'Image outside original availability')
                fpair[when] = dict(timestamp_us=f['sample_data']['timestamp'],
                    camera_to_global=np.asarray(f['camera_to_global']), intrinsic=np.asarray(f['K']), rgb=rgb)
                decoded += 1
            pairs[pair['channel']] = fpair
        if ordinal == 0:
            audit = parent_patch_check(old, cached, pairs)
        result = core.evaluate_pose_address_control(points[idx], np.asarray(h['lidar_to_global']),
                    obj, tokens, past_raw, cached, pairs, chunk_size=4096)
        require(array_digests(cached) == frozen_digest, 'Pose control changed original camera arrays')
        require(np.array_equal(result['camera_index'], cached['camera_index']) and
                np.array_equal(result['original_reason_code'], cached['reason_code']), 'Original view/mask mutated')
        require(np.all(~result['reference_valid'] | cached['valid']), 'Reference expanded original photometric support')
        arrays = dict(identity_json=cached['identity_json'], source_flat_indices=idx, **result)
        name = c['file']
        with (out / name).open('xb') as f:
            np.savez_compressed(f, **arrays)
        with np.load(out / name, allow_pickle=False) as z:
            require(set(z.files) == set(arrays), 'Saved field set differs')
            for key, value in arrays.items():
                require(np.array_equal(value, z[key], equal_nan=True) if value.dtype.kind in 'fc'
                        else np.array_equal(value, z[key]), 'Saved reference array differs: ' + key)
        rc = {str(int(k)): int((result['reference_reason_code'] == k).sum())
              for k in np.unique(result['reference_reason_code'])}
        qc = {str(int(k)): int((result['pose_reason_code'] == k).sum())
              for k in np.unique(result['pose_reason_code'])}
        for key, count in rc.items(): reference_counts[key] = reference_counts.get(key, 0) + count
        for key, count in qc.items(): pose_counts[key] = pose_counts.get(key, 0) + count
        output_records.append(dict(ordinal=ordinal, identity=c['identity'], file=name,
            bytes=(out / name).stat().st_size, sha256=sha(out / name), source_points=len(idx),
            camera_npz_sha256=c['sha256'], motion_npz_sha256=m['sha256'], raw_label_sha256=r['sha256'],
            reference_reason_counts=rc, pose_reason_counts=qc,
            arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()}))
        total += len(idx)
        del pairs, arrays, result, cached, past_raw
        print(json.dumps(dict(event='sample_complete', completed=ordinal + 1, seconds=time.monotonic() - started)), flush=True)
    require(total == 1740053 and decoded == 6144 and sum(reference_counts.values()) == total and
            sum(pose_counts.values()) == 2 * total, 'Incomplete original support')
    require('torch' not in sys.modules, 'CPU pose control imported Torch')
    write(out / 'index.json', dict(schema=SCHEMA, records=output_records, source_points=total,
        reference_reason_counts=reference_counts, pose_reason_counts=pose_counts))
    status = 'COMPLETE_HISTORY_POSE_ADDRESS_CONTROL_TRAIN'
    write(out / 'summary.json', dict(schema=SCHEMA, status=status, samples=512, scenes=256,
        source_points=total, images_decoded=decoded, seconds=time.monotonic() - started,
        reference_reason_counts=reference_counts, pose_reason_counts=pose_counts,
        first_anchor_original_correlation_check=audit, raw_GT_used=True,
        raw_container_includes_future_frames=True, future_slots_passed_to_pose_core=False,
        future_GT_used_in_reference=False, original_cached_arrays_unchanged=True,
        all_saved_arrays_roundtrip_exact=True, fitting=False, threshold_selection=False,
        optimizer_updates=0, model_forward=False, is_deployable=False))
    write(out / 'complete.json', dict(schema=SCHEMA, status=status, samples=512,
        source_sha256=sha(__file__), protocol_sha256=sha(a.protocol), core_sha256=sha(a.module),
        files_sha256={name: sha(out / name) for name in ('manifest.json', 'index.json', 'summary.json')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('protocol', 'module', 'camera-run', 'history-inputs', 'motion-run', 'raw-targets', 'points-file', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--max-seconds', type=int, default=600); a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    def stop(sig, frame): raise TimeoutError('CPU timeout/termination signal ' + str(sig))
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    signal.alarm(a.max_seconds)
    try:
        require(a.max_seconds == 600, 'Fixed resource bound differs'); run(a, out, started)
    except BaseException as exc:
        signal.alarm(0)
        write(out / 'failed.json', dict(error=repr(exc), traceback=traceback.format_exc())); raise
    finally:
        signal.alarm(0)


if __name__ == '__main__': main()
