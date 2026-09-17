"""Build sparse virtual rigid-box point supervision from frozen raw annotations.

CPU/NumPy only. This program never opens model inputs, predictions, occupancy
GT, checkpoints or sensor data. These labels are virtual points inside raw
boxes, not observed points or measured scene flow. Future annotations are
supervision/evaluation labels and must never be supplied as inference inputs.
"""
import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from motion_geometry import box_to_global, rigid_displacement, unique_box_assignment, voxel_centers_xyz

RAW_MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
RAW_COMPLETE_SHA = 'e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69'
SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
GEOMETRY_SHA = 'e9d232c3f7aaacd073cfda645868e357afad9e5a2684d07b2f3f2394bd796a31'
SHAPE = (200, 200, 16)
EXTENT = (-51.2, -51.2, -5., 51.2, 51.2, 3.)
SCHEMA = 'sparse-rigid-box-motion-supervision-v1'
FUTURE = (3, 4, 5, 6)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def conservative_candidates(boxes_R, sizes):
    """Union of conservative box AABBs; exact membership remains the helper's.

    The helper's 1e-9 local-axis boundary tolerance is projected onto all R
    axes; an additional 1e-8 metre pad prevents floating-point prefilter loss.
    This is an acceleration only, checked against full-grid assignment for
    the first two real anchors. No box is dropped by its centre/ROI position.
    """
    mask = np.zeros(SHAPE, dtype=bool)
    axes = [EXTENT[i] + (np.arange(SHAPE[i]) + .5)
            * (EXTENT[i + 3] - EXTENT[i]) / SHAPE[i] for i in range(3)]
    for box, size in zip(boxes_R, sizes):
        half = np.abs(box[:3, :3]) @ (size[[1, 0, 2]] / 2. + 1e-9) + 1e-8
        lo, hi = box[:3, 3] - half, box[:3, 3] + half
        slices = tuple(slice(np.searchsorted(axes[i], lo[i], side='left'),
                             np.searchsorted(axes[i], hi[i], side='right')) for i in range(3))
        mask[slices] = True
    return np.flatnonzero(mask.reshape(-1))


def independent_rotation(q):
    """Separate Rodrigues quaternion identity for real material-point audit."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w = q[0]
    v = q[1:]
    cross = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return (w*w - np.dot(v, v))*np.eye(3) + 2.*np.outer(v, v) + 2.*w*cross


def independent_material_audit(raw, arrays, points, tracks, G0, boxes_R, sizes, candidate_ids, assignment):
    """Check all retained points in two real anchors via explicit local points.

    Does not call rigid_displacement or box_to_global for the independent
    correspondence. The grid assignment additionally has a full-grid exact
    comparison, bypassing the AABB optimization.
    """
    full = unique_box_assignment(points, boxes_R, sizes)
    reconstructed = np.full(len(points), -1, dtype=np.int64)
    reconstructed[candidate_ids] = assignment
    require(np.array_equal(full, reconstructed), 'AABB/full-grid assignment disagreement')
    g_rot, g_trans = G0[:3, :3], G0[:3, 3]
    lookup = {t['instance_token']: t for t in tracks}
    maximum_error = 0.
    checks = 0
    examples = []
    for k, token in enumerate(arrays['instance_tokens'].tolist()):
        t = lookup[token]
        indices = np.flatnonzero(arrays['object_index'] == k)
        p = points[arrays['source_flat_indices'][indices]]
        global_p = p @ g_rot.T + g_trans
        c0 = np.asarray(t['global_centers_m'][2], dtype=np.float64)
        r0 = independent_rotation(t['global_rotations_wxyz'][2])
        local_p = (global_p - c0) @ r0
        require(np.all(np.abs(local_p) <= np.asarray(t['sizes_wlh_m'][2])[[1, 0, 2]]/2. + 1e-8),
                'Retained source not inside its raw rotated box')
        for h, f in enumerate(FUTURE):
            if not t['valid_mask'][f]:
                continue
            ch = np.asarray(t['global_centers_m'][f], dtype=np.float64)
            rh = independent_rotation(t['global_rotations_wxyz'][f])
            expected_global = local_p @ rh.T + ch
            expected_R = (expected_global - g_trans) @ g_rot
            expected_delta = expected_R - p
            actual = arrays['target_displacement_m'][h, indices].astype(np.float64)
            error = float(np.max(np.abs(expected_delta - actual)))
            tolerance = 1e-6 + 2e-7*np.abs(expected_delta)
            require(np.all(np.abs(expected_delta - actual) <= tolerance), 'Raw material-point correspondence failed')
            maximum_error = max(maximum_error, error)
            checks += len(indices)
            if len(examples) < 8:
                examples.append(dict(instance_token=token, horizon_index=h, annotation_token=t['annotation_tokens'][f],
                    source_flat_index=int(arrays['source_flat_indices'][indices[0]]),
                    source_R_m=p[0].tolist(), box_local_material_point_m=local_p[0].tolist(),
                    future_R_independent_m=expected_R[0].tolist(),
                    saved_displacement_m=actual[0].tolist(), max_absolute_error_m=error))
    return dict(identity=raw['identity'], full_grid_assignment_exact=True,
                correspondence='explicit global→box0 local→future box global→fixed t0 LiDAR R',
                checked_valid_point_horizons=checks, maximum_float32_absolute_error_m=maximum_error,
                examples=examples)


def convert(raw, descriptor, points, audit=False):
    require(raw['schema'] == 'raw-nuscenes-motion-target-v1', 'Wrong raw schema')
    require(raw['identity'] == descriptor['identity'] and raw['ordinal'] == descriptor['ordinal'], 'Raw identity mismatch')
    require(raw['selection_sha256'] == SELECTION_SHA, 'Wrong selection')
    frames = raw['frames']
    require(len(frames) == 7 and [f['relative_frame_index'] for f in frames] == list(range(-2, 5)), 'Wrong seven frames')
    token = raw['identity']['sample_token']
    require(frames[2]['sample_token'] == token, 'Wrong t0 frame')
    G0 = np.asarray(frames[2]['lidar_to_global_column_matrix'], dtype=np.float64)
    inverse_G0 = np.linalg.inv(G0)
    dt = np.array([(frames[f]['timestamp_us'] - frames[2]['timestamp_us'])/1e6 for f in FUTURE])
    require(np.isfinite(dt).all() and np.all(np.diff(np.r_[0., dt]) > 0.), 'Nonpositive future dt')
    require(all(np.isclose(dt[h], frames[f]['dt_seconds'], rtol=0., atol=1e-12) for h, f in enumerate(FUTURE)), 'Raw dt mismatch')
    tracks = [t for t in raw['tracks'] if t['valid_mask'][2]]
    require([t['instance_token'] for t in tracks] == sorted({t['instance_token'] for t in tracks}), 'Track identity/order mismatch')
    poses = np.asarray([box_to_global(t['global_centers_m'][2], t['global_rotations_wxyz'][2]) for t in tracks]).reshape(-1, 4, 4)
    boxes_R = inverse_G0[None] @ poses
    sizes = np.asarray([t['sizes_wlh_m'][2] for t in tracks], dtype=np.float64).reshape(-1, 3)
    candidate_ids = conservative_candidates(boxes_R, sizes)
    assignment = unique_box_assignment(points[candidate_ids], boxes_R, sizes)
    keep = assignment >= 0
    source_ids = candidate_ids[keep].astype(np.int64)
    original_object = assignment[keep]
    retained = np.unique(original_object)
    object_index = np.searchsorted(retained, original_object).astype(np.int64)
    K, N = len(retained), len(source_ids)
    selected_tracks = [tracks[i] for i in retained]
    displacement = np.zeros((4, N, 3), dtype=np.float32)
    future_valid = np.zeros((4, K), dtype=bool)
    groups = np.full((4, K), -1, dtype=np.int8)
    center_delta = np.zeros((4, K, 3), dtype=np.float32)
    endpoint_speed = np.full((4, K), np.nan, dtype=np.float64)
    outside = np.zeros(4, dtype=np.int64)
    for k, t in enumerate(selected_tracks):
        selected = np.flatnonzero(object_index == k)
        source_points = points[source_ids[selected]]
        a0 = poses[retained[k]]
        c0 = np.asarray(t['global_centers_m'][2], dtype=np.float64)
        for h, f in enumerate(FUTURE):
            if not t['valid_mask'][f]:
                continue
            future_valid[h, k] = True
            ah = box_to_global(t['global_centers_m'][f], t['global_rotations_wxyz'][f])
            delta = rigid_displacement(source_points, G0, a0, ah)
            displacement[h, selected] = delta.astype(np.float32)
            dc_global = np.asarray(t['global_centers_m'][f], dtype=np.float64) - c0
            center_delta[h, k] = (inverse_G0[:3, :3] @ dc_global).astype(np.float32)
            speed = float(np.linalg.norm(dc_global[:2]) / dt[h])
            endpoint_speed[h, k] = speed
            groups[h, k] = 0 if speed <= .1 else 1 if speed <= .5 else 2
            future_points = source_points + delta
            outside[h] += np.count_nonzero(np.any((future_points < EXTENT[:3]) | (future_points >= EXTENT[3:]), axis=1))
    valid = future_valid[:, object_index]
    arrays = dict(schema=np.array(SCHEMA), sample_token=np.array(token),
        scene_token=np.array(raw['identity']['scene_token']), split=np.array(raw['identity']['split']),
        ordinal=np.array(raw['ordinal'], dtype=np.int64), label_source_sha256=np.array(descriptor['sha256']),
        grid_shape_xyz=np.array(SHAPE, dtype=np.int64), extent_xyz_m=np.array(EXTENT, dtype=np.float64),
        source_flat_indices=source_ids, object_index=object_index,
        target_displacement_m=displacement, valid=valid,
        instance_tokens=np.asarray([t['instance_token'] for t in selected_tracks], dtype='<U32'),
        object_speed_group=groups, center_delta_R_m=center_delta,
        object_future_valid=future_valid, dt_future_seconds=dt)
    require(np.isfinite(displacement).all() and np.isfinite(center_delta).all(), 'Nonfinite output')
    require(np.all(displacement[~valid] == 0.) and np.all(center_delta[~future_valid] == 0.), 'Nonzero missing labels')
    require(np.array_equal(groups == -1, ~future_valid), 'Speed validity mismatch')
    require(np.array_equal(np.unique(object_index), np.arange(K)), 'Noncompact object map')
    require(N == 0 or (np.all(np.diff(source_ids) > 0) and source_ids[0] >= 0 and source_ids[-1] < np.prod(SHAPE)), 'Wrong flat indices')
    counts = dict(union_instances=len(raw['tracks']), t0_present_boxes=len(tracks), retained_objects=K,
        t0_boxes_without_unique_grid_point=len(tracks)-K, source_points=N,
        overlapping_grid_points=int(np.count_nonzero(assignment == -2)),
        uncovered_grid_points=int(np.prod(SHAPE)-N-np.count_nonzero(assignment == -2)),
        candidate_grid_points=len(candidate_ids), valid_points_by_horizon=valid.sum(axis=1).tolist(),
        missing_points_by_horizon=(~valid).sum(axis=1).tolist(),
        valid_objects_by_horizon=future_valid.sum(axis=1).tolist(),
        missing_objects_by_horizon=(~future_valid).sum(axis=1).tolist(),
        valid_future_points_outside_half_open_extent_by_horizon=outside.tolist(),
        object_speed_group_counts_by_horizon=[[int(np.count_nonzero(groups[h] == g)) for g in range(3)] for h in range(4)],
        point_speed_group_counts_by_horizon=[[int(np.count_nonzero(groups[h, object_index] == g)) for g in range(3)] for h in range(4)])
    check = independent_material_audit(raw, arrays, points, tracks, G0, boxes_R, sizes, candidate_ids, assignment) if audit else None
    return arrays, counts, check


def summarize(records):
    result = {}
    for split in ('train', 'development'):
        rows = [r for r in records if r['identity']['split'] == split]
        totals = {}
        for key in rows[0]['counts']:
            summed = np.sum([r['counts'][key] for r in rows], axis=0)
            totals[key] = summed.tolist() if summed.ndim else int(summed)
        result[split] = dict(samples=len(rows), scenes=len({r['identity']['scene_token'] for r in rows}),
            totals=totals, empty_source_samples=sum(r['counts']['source_points'] == 0 for r in rows),
            source_points_min=min(r['counts']['source_points'] for r in rows),
            source_points_max=max(r['counts']['source_points'] for r in rows),
            bytes=sum(r['bytes'] for r in rows))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    root, out = Path(args.targets).resolve(), Path(args.out).resolve()
    source = Path(__file__).resolve()
    geometry = source.with_name('motion_geometry.py')
    source_sha = sha(source)
    require(sha(geometry) == GEOMETRY_SHA, 'Changed geometry source')
    require(sha(root/'manifest.json') == RAW_MANIFEST_SHA and sha(root/'complete.json') == RAW_COMPLETE_SHA, 'Wrong frozen raw labels')
    manifest = json.loads((root/'manifest.json').read_text())
    complete = json.loads((root/'complete.json').read_text())
    require(complete['status'] == 'COMPLETE' and complete['manifest_sha256'] == RAW_MANIFEST_SHA, 'Incomplete labels')
    require(manifest['selection']['sha256'] == SELECTION_SHA, 'Wrong fixed selection')
    selected = manifest['records']
    require(len(selected) == 712 and [r['ordinal'] for r in selected] == list(range(712)), 'Expected ordered712')
    require([r['identity']['split'] for r in selected] == ['train']*512+['development']*200, 'Wrong train/development sequence')
    train_scenes = {r['identity']['scene_token'] for r in selected[:512]}
    dev_scenes = {r['identity']['scene_token'] for r in selected[512:]}
    require(len(train_scenes) == 256 and len(dev_scenes) == 100 and train_scenes.isdisjoint(dev_scenes), 'Wrong/disjoint scenes')
    require(manifest['all_official_scene_memberships_checked'] is True, 'Missing official split check')
    require(not out.exists() and out != root and root not in out.parents, 'Refuse old output or raw-label subtree')
    out.mkdir(parents=True)
    for split in ('train', 'development'):
        (out/split).mkdir()
    points = voxel_centers_xyz(SHAPE, EXTENT).reshape(-1, 3)
    records, audits = [], []
    for ordinal, descriptor in enumerate(selected):
        identity = descriptor['identity']
        expected_file = identity['split']+'/'+identity['sample_token']+'.json.gz'
        require(descriptor['file'] == expected_file, 'Unexpected raw-label path')
        payload = (root/expected_file).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == descriptor['sha256'], 'Raw-label SHA mismatch')
        unpacked = gzip.decompress(payload)
        require(hashlib.sha256(unpacked).hexdigest() == descriptor['uncompressed_json_sha256'], 'Raw JSON SHA mismatch')
        arrays, counts, audit = convert(json.loads(unpacked), descriptor, points, audit=ordinal < 2)
        relative = identity['split']+'/'+identity['sample_token']+'.npz'
        path = out/relative
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(path, allow_pickle=False) as loaded:
            require(set(loaded.files) == set(arrays), 'Output keys mismatch')
            for name, value in arrays.items():
                require(loaded[name].dtype == value.dtype and loaded[name].shape == value.shape
                        and loaded[name].tobytes() == value.tobytes(), 'NPZ byte roundtrip failed: '+name)
        records.append(dict(ordinal=ordinal, identity=identity, file=relative, bytes=path.stat().st_size,
            sha256=sha(path), label_file=expected_file, label_source_sha256=descriptor['sha256'],
            shapes={k:list(v.shape) for k,v in arrays.items()}, dtypes={k:str(v.dtype) for k,v in arrays.items()}, counts=counts))
        if audit is not None:
            audits.append(audit)
        if ordinal < 2 or (ordinal+1) % 64 == 0 or ordinal == 711:
            print(json.dumps(dict(completed=ordinal+1, total=712, seconds=time.monotonic()-started,
                                  sample_token=identity['sample_token'], points=counts['source_points'])), flush=True)
    # Verify final small input and output ledgers before publishing COMPLETE.
    require(sha(source) == source_sha and sha(geometry) == GEOMETRY_SHA, 'Source changed during extraction')
    require(sha(root/'manifest.json') == RAW_MANIFEST_SHA and sha(root/'complete.json') == RAW_COMPLETE_SHA, 'Raw ledger changed')
    require(all(sha(out/r['file']) == r['sha256'] and sha(root/r['label_file']) == r['label_source_sha256'] for r in records), 'Final label/output SHA mismatch')
    stats = summarize(records)
    write_json(out/'statistics.json', stats)
    write_json(out/'material_point_audit.json', dict(schema='sparse-rigid-box-material-point-audit-v1',
        status='PASS', audit_selection='First two actual raw manifest records, ordinals0/1; no model use',
        same_scene=audits[0]['identity']['scene_token'] == audits[1]['identity']['scene_token'],
        geometry_sha256=GEOMETRY_SHA, script_sha256=source_sha, samples=audits))
    contract = dict(schema=SCHEMA, source_grid_shape_xyz=list(SHAPE), source_extent_xyz_m=list(EXTENT),
        source_flatten_order='C: flat=((x*200)+y)*16+z', source_grid_location='voxel centres',
        source_point_semantics='Virtual grid material points uniquely covered by one raw GMO t0 box; not measured points or scene flow',
        assignment='motion_geometry.unique_box_assignment; closed box faces with1e-9m local tolerance; overlap excluded',
        displacement_frame='fixed t0 LiDAR R, metres', displacement_formula='inv(G0) @ Ah @ inv(A0) @ G0 @ p0 - p0',
        size_order='raw wlh; local xyz half extents [l,w,h]/2; no size scaling across time',
        missing_future='displacement and center_delta zero, valid false, object_speed_group -1; never train on invalid zero',
        object_selection='t0 present and at least one unique grid point; no future-valid, speed, visibility or point-count selection',
        object_order='retained original instance_token lexicographic order; compact0..K-1',
        speed_group='per-horizon ||global_center_h.xy-global_center_0.xy||/actual_dt: <=0.1→0, (0.1,0.5]→1, >0.5→2, future missing→-1; endpoint-average speed magnitude, not accumulated path speed',
        future_outside_extent='retained valid displacement; future half-open-ROI outside counts descriptive only; transport may drop support',
        future_newborn='no t0 source point; absent from retained objects',
        precision='float64 geometry/group decisions; float32 stored displacement/center_delta; float64 dt',
        training_inference_boundary='These are separate supervised targets; raw current/future boxes and group labels are not inference inputs',
        fine_GT_or_cache_read=False, model_or_prediction_read=False, GPU_used=False, optimizer_steps=0,
        historical_validation_exposure=True, original_selection_unchanged=True)
    index = dict(schema='sparse-rigid-box-motion-manifest-v1', status='COMPLETE',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), script_sha256=source_sha,
        geometry_sha256=GEOMETRY_SHA, label_source=dict(directory=str(root), manifest_sha256=RAW_MANIFEST_SHA,
        complete_sha256=RAW_COMPLETE_SHA, selection_sha256=SELECTION_SHA), contract=contract,
        samples=712, train_samples=512, development_samples=200, train_dev_scenes_disjoint=True,
        official_split_checked_by_frozen_raw_manifest=True, records=records)
    write_json(out/'manifest.json', index)
    finish = dict(schema='sparse-rigid-box-motion-complete-v1', status='COMPLETE', samples=712,
        train_samples=512, development_samples=200, script_sha256=source_sha, geometry_sha256=GEOMETRY_SHA,
        manifest_sha256=sha(out/'manifest.json'), statistics_sha256=sha(out/'statistics.json'),
        material_point_audit_sha256=sha(out/'material_point_audit.json'),
        raw_manifest_sha256=RAW_MANIFEST_SHA, raw_complete_sha256=RAW_COMPLETE_SHA,
        all712_NPZ_byte_roundtrip_verified=True, all_final_file_hashes_verified=True,
        first_two_actual_material_point_audits_pass=True, elapsed_seconds=time.monotonic()-started,
        fine_GT_or_cache_read=False, model_or_prediction_read=False, GPU_used=False, optimizer_steps=0)
    write_json(out/'complete.json', finish)
    print(json.dumps(dict(status='COMPLETE', manifest_sha256=finish['manifest_sha256'],
                         complete_sha256=sha(out/'complete.json'), seconds=finish['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
