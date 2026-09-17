"""Build full5119 validation sparse virtual rigid-box point supervision.

The original conversion/geometry is source-exact. The old development200
intersection must match all scientific NPZ fields byte for byte; only split,
ordinal and label-source hash may differ because the dataset binding changes.

CPU/NumPy only. This program never opens model inputs, predictions, occupancy
GT, checkpoints or sensor data. These labels are virtual points inside raw
boxes, not observed points or measured scene flow. Future annotations are
supervision/evaluation labels and must never be supplied as inference inputs.
"""
import argparse
import ast
from collections import Counter
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from motion_geometry import box_to_global, rigid_displacement, unique_box_assignment, voxel_centers_xyz

RAW_MANIFEST_SHA = '48ac12f1ce24ed1639c17aed0fcdc8ad2eef2cceff10c181b2a0f82f81fc90bb'
RAW_COMPLETE_SHA = 'de10ddd9013d7efd68249bced50448fb083c6d059546f8cfa0560be02b702252'
SELECTION_SHA = '60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391'
ORIGINAL_BUILDER_SHA = 'a63f5083dad4f5092bdc9d4d87b888dd53f54b95986b2f8b3a13769e9e78f023'
REFERENCE_MANIFEST_SHA = 'cdb11231cbc3213d213e1f3f718fe9d3c90665dccea4154ecf3372ad1ce91efe'
REFERENCE_COMPLETE_SHA = '3d14a03cd61761a1ebc26313463dbff06854907c19c49057667a0ffd257e05bd'
BINDING_FIELDS = frozenset(('split', 'ordinal', 'label_source_sha256'))
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
    for split in ('validation',):
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


def protected_source_contract(source):
    """Require exact source bytes for the original four scientific functions."""
    old = source.with_name('build_sparse_motion_supervision_v1.py')
    require(sha(old) == ORIGINAL_BUILDER_SHA, 'Changed original sparse builder')
    def functions(path):
        code = path.read_text()
        return {n.name: ast.get_source_segment(code, n) for n in ast.parse(code).body
                if isinstance(n, ast.FunctionDef)}
    before, after = functions(old), functions(source)
    names = ('conservative_candidates', 'independent_rotation',
             'independent_material_audit', 'convert')
    require(all(before[n] == after[n] for n in names), 'Changed scientific conversion')
    return dict(original_builder_sha256=ORIGINAL_BUILDER_SHA,
                geometry_sha256=GEOMETRY_SHA,
                source_exact_functions={n: hashlib.sha256(before[n].encode()).hexdigest() for n in names})


def load_bindings(root, reference, selection_path):
    require(sha(root/'manifest.json') == RAW_MANIFEST_SHA
            and sha(root/'complete.json') == RAW_COMPLETE_SHA, 'Wrong frozen full raw labels')
    require(sha(selection_path) == SELECTION_SHA, 'Wrong frozen full selection')
    manifest = json.loads((root/'manifest.json').read_text())
    complete = json.loads((root/'complete.json').read_text())
    selection = json.loads(selection_path.read_text())
    require(complete['status'] == 'COMPLETE' and complete['samples'] == 5119
            and complete['scenes'] == 150 and complete['manifest_sha256'] == RAW_MANIFEST_SHA
            and complete['all5119_files_roundtrip_verified'] is True
            and complete['model_or_prediction_read'] is False
            and complete['optimizer_steps'] == 0, 'Incomplete full raw labels')
    require(manifest['selection']['sha256'] == SELECTION_SHA
            and complete['selection_sha256'] == SELECTION_SHA, 'Wrong raw selection')
    selected = manifest['records']
    require(len(selected) == 5119 and [r['ordinal'] for r in selected] == list(range(5119)), 'Expected ordered5119')
    identities = [r['identity'] for r in selected]
    require(identities == selection['records'], 'Raw identity/selection mismatch')
    require([r['split'] for r in identities] == ['validation']*5119, 'Wrong validation split')
    require([r['official_index'] for r in identities] == list(range(5119)), 'Wrong official ordering')
    require(len({r['sample_token'] for r in identities}) == 5119
            and len({r['scene_token'] for r in identities}) == 150, 'Wrong unique samples/scenes')
    require(manifest['all_official_scene_memberships_checked'] is True, 'Missing official split check')
    require(sha(reference/'manifest.json') == REFERENCE_MANIFEST_SHA
            and sha(reference/'complete.json') == REFERENCE_COMPLETE_SHA, 'Wrong frozen sparse reference')
    prior = json.loads((reference/'manifest.json').read_text())
    prior_done = json.loads((reference/'complete.json').read_text())
    require(prior_done['status'] == 'COMPLETE' and prior_done['samples'] == 712
            and prior_done['manifest_sha256'] == REFERENCE_MANIFEST_SHA
            and prior_done['all_final_file_hashes_verified'] is True, 'Incomplete prior sparse labels')
    refrows = [r for r in prior['records'] if r['identity']['split'] == 'development']
    require(len(refrows) == 200 and len({r['identity']['scene_token'] for r in refrows}) == 100, 'Wrong old dev200')
    lookup = {r['identity']['sample_token']: r for r in refrows}
    full = {r['sample_token']: r for r in identities}
    require(len(lookup) == 200 and set(lookup).issubset(full), 'Missing old dev200 intersection')
    for token, row in lookup.items():
        identity = row['identity']
        require(all(full[token][k] == identity[k] for k in ('sample_token', 'scene_token', 'official_index'))
                and full[token]['historical_partition'] == 'development', 'Changed intersection identity')
        require(row['file'] == 'development/'+token+'.npz', 'Unexpected old sparse path')
    return selected, lookup, prior


def compare_intersection(arrays, counts, descriptor, reference, old):
    """Scientific fields include strings such as instance IDs; no numeric tolerance."""
    path = reference/old['file']
    require(sha(path) == old['sha256'], 'Changed old intersection NPZ')
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as loaded:
        require(set(loaded.files) == set(arrays), 'Changed intersection NPZ keys')
        identity = descriptor['identity']
        require(arrays['split'].item() == 'validation' and loaded['split'].item() == 'development', 'Wrong split rebinding')
        require(arrays['ordinal'].item() == descriptor['ordinal']
                and loaded['ordinal'].item() == old['ordinal'], 'Wrong ordinal rebinding')
        require(arrays['label_source_sha256'].item() == descriptor['sha256']
                and loaded['label_source_sha256'].item() == old['label_source_sha256'], 'Wrong label SHA rebinding')
        for name in sorted(set(arrays)-BINDING_FIELDS):
            actual, expected = arrays[name], loaded[name]
            require(actual.dtype == expected.dtype and actual.shape == expected.shape
                    and actual.tobytes() == expected.tobytes(), 'Changed intersection scientific field: '+name)
            h.update(json.dumps([name, actual.dtype.str, list(actual.shape)], separators=(',', ':')).encode())
            h.update(actual.tobytes())
        require(counts == old['counts'], 'Changed intersection support/counts')
    return dict(sample_token=identity['sample_token'], scene_token=identity['scene_token'],
                full_ordinal=descriptor['ordinal'], old_ordinal=old['ordinal'],
                old_npz_sha256=old['sha256'], old_label_sha256=old['label_source_sha256'],
                full_label_sha256=descriptor['sha256'], scientific_fields_sha256=h.hexdigest(),
                fields_checked=sorted(set(arrays)-BINDING_FIELDS),
                dtype_shape_bytes_exact=True, support_counts_exact=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', required=True)
    parser.add_argument('--reference-sparse', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args(argv)
    started = time.monotonic()
    require(np.isfinite(args.max_seconds) and 0 < args.max_seconds <= 600., 'CPU deadline must be in (0,600] seconds')
    def check_deadline():
        if time.monotonic()-started >= args.max_seconds:
            raise TimeoutError('Bounded label generation deadline; no COMPLETE and no retry')
    root, reference = Path(args.targets).resolve(), Path(args.reference_sparse).resolve()
    selection_path, out = Path(args.selection).resolve(), Path(args.out).resolve()
    source = Path(__file__).resolve()
    geometry = source.with_name('motion_geometry.py')
    source_sha = sha(source)
    require(sha(geometry) == GEOMETRY_SHA, 'Changed geometry source')
    provenance = protected_source_contract(source)
    selected, reference_rows, prior = load_bindings(root, reference, selection_path)
    require(not out.exists() and out != root and root not in out.parents
            and out != reference and reference not in out.parents, 'Refuse old output or input subtree')
    out.mkdir(parents=True)
    (out/'validation').mkdir()
    points = voxel_centers_xyz(SHAPE, EXTENT).reshape(-1, 3)
    records, audits, intersections = [], [], []
    for ordinal, descriptor in enumerate(selected):
        check_deadline()
        identity = descriptor['identity']
        expected_file = 'validation/'+identity['sample_token']+'.json.gz'
        require(descriptor['file'] == expected_file, 'Unexpected raw-label path')
        payload = (root/expected_file).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == descriptor['sha256'], 'Raw-label SHA mismatch')
        unpacked = gzip.decompress(payload)
        require(hashlib.sha256(unpacked).hexdigest() == descriptor['uncompressed_json_sha256'], 'Raw JSON SHA mismatch')
        arrays, counts, audit = convert(json.loads(unpacked), descriptor, points, audit=ordinal < 2)
        token = identity['sample_token']
        intersection = None
        if token in reference_rows:
            intersection = compare_intersection(arrays, counts, descriptor, reference, reference_rows[token])
        relative = 'validation/'+token+'.npz'
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
        if intersection is not None:
            intersection['full_npz_sha256'] = records[-1]['sha256']
            intersections.append(intersection)
        if audit is not None:
            audits.append(audit)
        if ordinal < 2 or (ordinal+1) % 64 == 0 or ordinal == 5118:
            print(json.dumps(dict(completed=ordinal+1, total=5119, seconds=time.monotonic()-started,
                                  sample_token=token, points=counts['source_points'],
                                  old_development_intersections_verified=len(intersections))), flush=True)
    check_deadline()
    require(len(intersections) == 200 and len({r['sample_token'] for r in intersections}) == 200,
            'Not all dev200 intersection fields verified')
    require(sha(source) == source_sha and sha(geometry) == GEOMETRY_SHA, 'Source changed during extraction')
    require(protected_source_contract(source) == provenance, 'Original source changed')
    load_bindings(root, reference, selection_path)
    for row in records:
        check_deadline()
        require(sha(out/row['file']) == row['sha256']
                and sha(root/row['label_file']) == row['label_source_sha256'], 'Final label/output SHA mismatch')
    for row in reference_rows.values():
        require(sha(reference/row['file']) == row['sha256'], 'Old NPZ changed during extraction')
    stats = summarize(records)
    write_json(out/'statistics.json', stats)
    write_json(out/'material_point_audit.json', dict(schema='sparse-rigid-box-material-point-audit-v1',
        status='PASS', audit_selection='First two actual full raw manifest records, ordinals0/1; no model use',
        same_scene=audits[0]['identity']['scene_token'] == audits[1]['identity']['scene_token'],
        geometry_sha256=GEOMETRY_SHA, script_sha256=source_sha, samples=audits))
    write_json(out/'development_intersection_audit.json', dict(schema='full-sparse-motion-dev200-intersection-v1',
        status='PASS', samples=200, scenes=100, comparison='dtype, shape and bytes; no numeric tolerance',
        excluded_binding_fields=sorted(BINDING_FIELDS),
        reference_manifest_sha256=REFERENCE_MANIFEST_SHA, reference_complete_sha256=REFERENCE_COMPLETE_SHA,
        records=intersections))
    # The scientific contract is identical to the frozen, hash-verified original.
    index = dict(schema='sparse-rigid-box-motion-manifest-v1', status='COMPLETE',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), script_sha256=source_sha,
        geometry_sha256=GEOMETRY_SHA, source_provenance=provenance,
        label_source=dict(directory=str(root), manifest_sha256=RAW_MANIFEST_SHA,
        complete_sha256=RAW_COMPLETE_SHA, selection_sha256=SELECTION_SHA), contract=prior['contract'],
        samples=5119, validation_samples=5119, scenes=150, scope='full_native_validation',
        development200_intersection_dtype_shape_bytes_exact=True,
        official_split_checked_by_frozen_raw_manifest=True, records=records)
    write_json(out/'manifest.json', index)
    check_deadline()
    finish = dict(schema='sparse-rigid-box-motion-complete-v1', status='COMPLETE', samples=5119,
        validation_samples=5119, scenes=150, script_sha256=source_sha, geometry_sha256=GEOMETRY_SHA,
        original_builder_sha256=ORIGINAL_BUILDER_SHA,
        manifest_sha256=sha(out/'manifest.json'), statistics_sha256=sha(out/'statistics.json'),
        material_point_audit_sha256=sha(out/'material_point_audit.json'),
        development_intersection_audit_sha256=sha(out/'development_intersection_audit.json'),
        raw_manifest_sha256=RAW_MANIFEST_SHA, raw_complete_sha256=RAW_COMPLETE_SHA,
        selection_sha256=SELECTION_SHA, all5119_NPZ_byte_roundtrip_verified=True,
        all_final_file_hashes_verified=True, development200_intersection_dtype_shape_bytes_exact=True,
        first_two_actual_material_point_audits_pass=True, elapsed_seconds=time.monotonic()-started,
        fine_GT_or_cache_read=False, model_or_prediction_read=False, GPU_used=False, optimizer_steps=0,
        checkpoint_promotion_decision=False, no_flow_measurement_claim=True)
    write_json(out/'complete.json', finish)
    print(json.dumps(dict(status='COMPLETE', manifest_sha256=finish['manifest_sha256'],
                         complete_sha256=sha(out/'complete.json'), seconds=finish['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
