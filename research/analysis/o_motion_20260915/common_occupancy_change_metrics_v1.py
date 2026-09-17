"""Common final-output occupancy/occupancy-change metrics, NumPy only.

Inputs are already decoded native predictions, in fixed t0 LiDAR R, XYZ order.
No logits, threshold selection, model, tracker, morphology, or prediction crop
is used. Raw seven-frame boxes annotate GT-positive voxels AFTER inference.
Future attribution uses the FUTURE box transformed to R, not the t0 box.
These scores are not flow EPE, physical stationarity, or identity accuracy.

Geometry is the existing motion_geometry.py: column-vector SE(3), wxyz
quaternions, size width/length/height, closed faces with 1e-9 m tolerance.
Histograms are GT rows / prediction columns. Never average returned ratios
instead of pooling their sufficient statistics according to the protocol.
"""

import hashlib

import numpy as np

from motion_geometry import box_to_global, unique_box_assignment, _rigid_pose


SCHEMA = 'common-occupancy-change-metrics-v1'
SPEED_GROUPS = ('speed_le_0.1', 'speed_gt_0.1_le_0.5',
                'speed_gt_0.5_le_5', 'speed_gt_5')
POSITIVE_GROUPS = SPEED_GROUPS + (
    'annotated_future_only', 't0_missing_with_history',
    'unknown_no_current_box', 'overlap_current_boxes')
TRANSITION_NAMES = ('00', '01', '10', '11')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _ratio(num, den):
    return float(num / den) if den else None


def _array_digest(array):
    """Typed shape-aware digest; uses canonical C-order and changes no input."""
    a = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode('ascii'))
    h.update(str(tuple(a.shape)).encode('ascii'))
    h.update(a.tobytes(order='C'))
    return h.hexdigest()


def speed_group(speed_mps):
    """Frozen endpoint-average global-XY speed bins, inclusive upper edges."""
    speed = float(speed_mps)
    _require(np.isfinite(speed) and speed >= 0., 'Invalid speed')
    if speed <= .1:
        return SPEED_GROUPS[0]
    if speed <= .5:
        return SPEED_GROUPS[1]
    if speed <= 5.:
        return SPEED_GROUPS[2]
    return SPEED_GROUPS[3]


def confusion_statistics(hist, names):
    """No overall accuracy: preserve all classes, including 01 and 10."""
    a = np.asarray(hist)
    _require(a.shape == (len(names), len(names)) and a.dtype.kind in 'iu'
             and np.all(a >= 0), 'Invalid confusion matrix')
    rows = a.sum(axis=1); cols = a.sum(axis=0); total = int(a.sum())
    classes = {}
    for k, name in enumerate(names):
        tp = int(a[k, k]); fn = int(rows[k])-tp; fp = int(cols[k])-tp
        classes[name] = dict(GT=int(rows[k]), predicted=int(cols[k]), TP=tp,
            FN=fn, FP=fp, TN=total-tp-fn-fp,
            IoU=_ratio(tp, tp+fn+fp), precision=_ratio(tp, tp+fp),
            recall=_ratio(tp, tp+fn))
    return dict(confusion=a.tolist(), row_names=list(names),
                column_names=list(names), valid_voxels=total, classes=classes)


def validate_raw_label(raw):
    """Validate raw schema and poses; no model/GT array is read here.

    A missing current box has no assignable current position. It is preserved
    in the instance ledger, never extrapolated to label unknown foreground.
    """
    _require(isinstance(raw, dict) and raw.get('schema') == 'raw-nuscenes-motion-target-v1',
             'Require original raw seven-frame label schema')
    for field in ('missing_fill_applied', 'old_refine_applied',
                  'visibility_or_point_count_filter_applied',
                  'spatial_ROI_filter_applied'):
        _require(raw.get(field) is False, 'Altered raw-label policy: '+field)
    _require(raw.get('instance_union') == 'all GMO instances in seven original sample annotations',
             'Require full raw seven-frame instance union')
    frames = raw.get('frames'); tracks = raw.get('tracks')
    _require(isinstance(frames, list) and len(frames) == 7
             and isinstance(tracks, list), 'Invalid frames/tracks')
    identity = raw.get('identity', {})
    _require(isinstance(identity.get('sample_token'), str)
             and isinstance(identity.get('scene_token'), str), 'Missing sample identity')
    stamps = []
    for k, frame in enumerate(frames):
        _require(frame.get('sequence_index') == k and frame.get('relative_frame_index') == k-2,
                 'Seven-frame order differs')
        _require(frame.get('scene_token') == identity['scene_token'], 'Cross-scene frame')
        stamp = frame.get('timestamp_us')
        _require(isinstance(stamp, (int, np.integer)) and not isinstance(stamp, (bool, np.bool_)),
                 'timestamp_us must be an integer')
        stamps.append(int(stamp))
        _rigid_pose(frame['lidar_to_global_column_matrix'], 'frame pose')
    _require(all(b > a for a, b in zip(stamps, stamps[1:])), 'Timestamps must strictly increase')
    _require(frames[2]['sample_token'] == identity['sample_token'], 't0 anchor mismatch')
    dt = np.asarray([(t-stamps[2])/1e6 for t in stamps], dtype=np.float64)
    for k, frame in enumerate(frames):
        _require(np.isfinite(frame['dt_seconds']) and
                 abs(float(frame['dt_seconds'])-dt[k]) <= 1e-12, 'dt/timestamp mismatch')
    tokens = []
    for track in tracks:
        token = track.get('instance_token')
        _require(isinstance(token, str) and token, 'Missing instance token')
        tokens.append(token)
        valid = track.get('valid_mask')
        _require(isinstance(valid, list) and len(valid) == 7 and
                 all(isinstance(v, (bool, np.bool_)) for v in valid), 'Invalid valid_mask')
        _require(any(valid) and track.get('gmo_class') == 1, 'Empty/non-GMO union track')
        fields = ('global_centers_m', 'global_rotations_wxyz', 'sizes_wlh_m')
        for field in fields:
            _require(isinstance(track.get(field), list) and len(track[field]) == 7,
                     'Invalid track field: '+field)
        for k in range(7):
            if not valid[k]:
                _require(all(track[f][k] is None for f in fields), 'Missing box has filled geometry')
                continue
            box_to_global(track['global_centers_m'][k], track['global_rotations_wxyz'][k])
            size = np.asarray(track['sizes_wlh_m'][k], dtype=np.float64)
            _require(size.shape == (3,) and np.all(np.isfinite(size)) and np.all(size > 0.),
                     'Invalid width/length/height')
    _require(len(set(tokens)) == len(tokens), 'Duplicate instance token')
    return dict(identity=dict(identity), actual_dt_seconds=dt[2:].tolist(),
                G0=_rigid_pose(frames[2]['lidar_to_global_column_matrix'], 'G0'),
                tracks=tracks)


def _motion_attribution(prediction, foreground, raw_info, horizon, extent):
    frame = horizon+2; tracks = raw_info['tracks']
    active = [t for t in tracks if t['valid_mask'][frame]]
    inverse_G0 = np.linalg.inv(raw_info['G0'])
    boxes = np.asarray([inverse_G0 @ box_to_global(t['global_centers_m'][frame],
        t['global_rotations_wxyz'][frame]) for t in active], dtype=np.float64).reshape(-1, 4, 4)
    sizes = np.asarray([t['sizes_wlh_m'][frame] for t in active], dtype=np.float64).reshape(-1, 3)
    indices = np.argwhere(foreground)
    points = extent[:3] + (indices.astype(np.float64)+.5) * (
        (extent[3:]-extent[:3])/np.asarray(foreground.shape, dtype=np.float64))
    owners = unique_box_assignment(points, boxes, sizes)
    # Prediction positions never come from boxes. Only GT-positive score
    # entries are attributed, after the complete-domain confusion is known.
    positive_predictions = prediction[foreground]
    groups = {name: dict(GT=0, TP=0, FN=0, recall=None) for name in POSITIVE_GROUPS}
    def add(name, mask):
        count = int(np.count_nonzero(mask)); tp = int(np.count_nonzero(positive_predictions[mask]))
        group = groups[name]; group['GT'] += count; group['TP'] += tp; group['FN'] += count-tp
        return count, tp
    add('unknown_no_current_box', owners == -1)
    add('overlap_current_boxes', owners == -2)
    active_rows = {}
    dt = raw_info['actual_dt_seconds'][horizon]
    for j, track in enumerate(active):
        speed = None
        if track['valid_mask'][2]:
            displacement_xy = (np.asarray(track['global_centers_m'][frame], dtype=np.float64)
                               -np.asarray(track['global_centers_m'][2], dtype=np.float64))[:2]
            speed = float(np.linalg.norm(displacement_xy)/dt)
            group = speed_group(speed)
        else:
            group = ('t0_missing_with_history' if any(track['valid_mask'][:2])
                     else 'annotated_future_only')
        count, tp = add(group, owners == j)
        active_rows[track['instance_token']] = dict(instance_token=track['instance_token'],
            current_frame_present=True, t0_frame_present=bool(track['valid_mask'][2]),
            status=group, global_xy_endpoint_speed_mps=speed,
            unique_GT=count, TP=tp, FN=count-tp, recall=_ratio(tp, count),
            support_status='unique_GT_available' if count else 'no_unique_GT_support')
    ledger = []
    for track in tracks:
        row = active_rows.get(track['instance_token'])
        if row is None:
            row = dict(instance_token=track['instance_token'], current_frame_present=False,
                t0_frame_present=bool(track['valid_mask'][2]), status='missing_current_frame',
                global_xy_endpoint_speed_mps=None, unique_GT=0, TP=0, FN=0, recall=None,
                support_status='current_geometry_unavailable')
        ledger.append(row)
    for group in groups.values():
        group['recall'] = _ratio(group['TP'], group['GT'])
    _require(sum(g['GT'] for g in groups.values()) == len(indices), 'GT attribution lost or duplicated voxels')
    _require(sum(g['TP'] for g in groups.values()) == int(positive_predictions.sum()), 'TP attribution mismatch')
    return dict(groups=groups, foreground_voxels=len(indices),
        unique_box_voxels=int(np.count_nonzero(owners >= 0)),
        unknown_voxels=int(np.count_nonzero(owners == -1)),
        overlap_voxels=int(np.count_nonzero(owners == -2)),
        instance_ledger=ledger, union_instance_count=len(tracks),
        active_instance_count=len(active), missing_current_instance_count=len(tracks)-len(active),
        active_instances_without_unique_GT_support=sum(r['unique_GT'] == 0 for r in active_rows.values()))


def evaluate_common_occupancy_change(predicted_gmo, gt, extent, rawlabel7frame):
    """Return JSON-safe sufficient statistics for one anchor, with no mutation.

    predicted_gmo: bool [5,X,Y,Z]; gt: integer [5,X,Y,Z], only 0/1/255.
    extent: [xmin,ymin,zmin,xmax,ymax,zmax] outer cell faces in fixed R.
    rawlabel7frame: unmodified raw-nuscenes-motion-target-v1 label dictionary.
    Its frames [2:7] correspond exactly to the five input grids; the caller
    must bind input tensor/sample/GT hashes to this identity externally.
    """
    pred = np.asarray(predicted_gmo); target = np.asarray(gt)
    _require(pred.dtype == np.dtype(bool), 'predicted_gmo must be bool; no thresholding in metric')
    _require(target.dtype.kind in 'iu', 'GT must be integer classes, not soft/floating labels')
    _require(pred.shape == target.shape and pred.ndim == 4 and pred.shape[0] == 5
             and all(n > 0 for n in pred.shape[1:]), 'Require matching [5,X,Y,Z] grids')
    _require(np.all((target == 0) | (target == 1) | (target == 255)), 'GT outside 0/1/255')
    bounds = np.asarray(extent, dtype=np.float64)
    _require(bounds.shape == (6,) and np.all(np.isfinite(bounds))
             and np.all(bounds[3:] > bounds[:3]), 'Invalid XYZ extent')
    info = validate_raw_label(rawlabel7frame)
    nvox = int(np.prod(pred.shape[1:])); valid0 = target[0] != 255
    binary0 = target[0] == 1
    horizons = []; transitions = []
    for h in range(5):
        valid = target[h] != 255; truth = target[h] == 1
        hist = np.bincount(2*target[h][valid].astype(np.int64)+pred[h][valid], minlength=4).reshape(2, 2)
        row = dict(horizon_index=h, nominal_seconds=h*.5,
            actual_dt_seconds=info['actual_dt_seconds'][h],
            occupancy=confusion_statistics(hist, ('0', '1')),
            ignored_voxels=nvox-int(valid.sum()),
            global_negative=dict(GT=int(hist[0].sum()), FP=int(hist[0, 1]), TN=int(hist[0, 0])),
            motion_positive_attribution=None)
        if h:
            row['motion_positive_attribution'] = _motion_attribution(pred[h], truth, info, h, bounds)
            joint = valid0 & valid
            true_state = 2*binary0[joint].astype(np.uint8)+truth[joint]
            pred_state = 2*pred[0][joint].astype(np.uint8)+pred[h][joint]
            transition = np.bincount((4*true_state+pred_state).astype(np.int64), minlength=16).reshape(4, 4)
            transitions.append(dict(horizon_index=h, nominal_seconds=h*.5,
                actual_dt_seconds=info['actual_dt_seconds'][h],
                **confusion_statistics(transition, TRANSITION_NAMES),
                domain=dict(grid_voxels=nvox, both_valid=int(joint.sum()),
                    t0_valid_h_ignored=int((valid0 & ~valid).sum()),
                    t0_ignored_h_valid=int((~valid0 & valid).sum()),
                    both_ignored=int((~valid0 & ~valid).sum()),
                    t0_valid=int(valid0.sum()), h_valid=int(valid.sum()),
                    t0_foreground_excluded=int((binary0 & ~valid).sum()),
                    h_foreground_excluded=int((truth & ~valid0).sum()))))
        horizons.append(row)
    return dict(schema=SCHEMA, identity=info['identity'], shape_hxyz=list(pred.shape),
        extent_xyz_m=bounds.tolist(), horizons=horizons, transitions=transitions,
        t0_boundary=dict(prediction_full_binary_sha256=_array_digest(pred[0]),
            prediction_on_valid_sha256=_array_digest(pred[0][valid0]),
            gt_valid_mask_sha256=_array_digest(valid0),
            valid_voxels=int(valid0.sum()),
            predicted_foreground_on_valid=int(pred[0][valid0].sum()),
            equals_GT_on_valid=bool(np.array_equal(pred[0][valid0], binary0[valid0])),
            cross_model_equality='Caller must compare full binary digests for O/A/Z; no reference model supplied'),
        contract=dict(axis='XYZ; fixed t0 LiDAR R; no future ego rewarp',
            transition_encoding='2*t0+h: 00,01,10,11; rows GT, columns prediction',
            transition_domain='Only GT-valid at both endpoints; original per-horizon domain remains unchanged',
            positive_assignment='Future frame h+2 valid raw boxes mapped by inv(G0) @ A_h; GT positives only',
            speed='norm(global_center_h[:2]-global_center_t0[:2])/actual_dt_seconds',
            speed_groups=list(SPEED_GROUPS),
            negative_policy='All valid GT-negative FP/TN counted once globally; not copied to motion groups',
            missing_policy='No filled geometry/speed; missing-current instances remain a separate ledger',
            denominator_zero='null; never assign empty precision/recall/IoU a perfect score',
            no_flow_EPE_or_identity_claim=True, low_speed_not_physical_stationarity=True,
            prediction_threshold_selection=False, prediction_components_or_crops=False))
