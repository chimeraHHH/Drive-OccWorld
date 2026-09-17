"""Historical GT-pose address control; NumPy, scoring-only, no image/model I/O.

evaluate_pose_address_control(points_R, lidar_to_global, object_index,
    instance_tokens, raw, camera_result, camera_pairs, chunk_size=4096)

raw is the STRICT three-slot historical projection of original raw7 metadata:
frames and required track arrays have only slots [-2,-1,0]. The caller binds
source identities, timestamps, RGB bytes and metadata receipts. No future slot,
box size, GT speed, visibility, future-valid mask or alternative view is used.
camera_pairs uses the frozen photo core's six camera names and frame dictionaries
{timestamp_us, camera_to_global, intrinsic, rgb}. camera_result is its unchanged
cached output. GT controls only reference addresses on the old photo-valid set.

All output arrays retain N points. Time axis is [current,past]. Geometry for an
available individual time is retained even if the other time is unavailable;
scores, reference patch statistics and pixel departures are NaN unless the
COMPLETE reference is valid. Missing brackets use -1. Original invalid points
have pose reason not_evaluated. No new texture/visibility filter is applied.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

PHOTO_SOURCE_SHA256 = 'c7469179d1628979b30eb256e695fdd6479ffd6e2df047d380d890512f57a5ea'
_path = Path(__file__).with_name('history_camera_evidence_v1.py')
if hashlib.sha256(_path.read_bytes()).hexdigest() != PHOTO_SOURCE_SHA256:
    raise ValueError('Frozen photometric core SHA differs')
_spec = importlib.util.spec_from_file_location('_pose_address_frozen_photo', _path)
PHOTO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PHOTO)
CAMERA_ORDER = PHOTO.CAMERA_ORDER
TIME_ORDER = ('current', 'past')
SLERP_NLERP_DOT = 0.9995
REFERENCE_REASON_CODES = {0: 'valid', 1: 'original_photo_invalid',
    2: 'current_pose_unavailable', 3: 'past_pose_unavailable',
    4: 'current_GT_projection_invalid', 5: 'past_GT_projection_invalid'}
POSE_REASON_CODES = {0: 'available', 1: 'time_outside', 2: 'missing_box',
                     3: 'broken_link', 4: 'not_evaluated'}
TRACK_FIELDS = ('valid_mask', 'annotation_tokens', 'annotation_prev_tokens',
                'annotation_next_tokens', 'global_centers_m', 'global_rotations_wxyz')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _finite(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    _require(a.shape == shape and np.isfinite(a).all(), name + ' shape/finite')
    return a


def _pose(value, name):
    a = _finite(value, (4, 4), name)
    _require(np.allclose(a[3], [0., 0., 0., 1.], rtol=0, atol=1e-10) and
             np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), rtol=0, atol=1e-8) and
             abs(np.linalg.det(a[:3, :3]) - 1.) <= 1e-8, name + ' rigid column pose')
    return a


def normalized_quaternion(q):
    """wxyz, scale-safe unit length; first nonzero component is positive.

    Canonicalization makes q and -q identical, including a 180-degree rotation.
    Relative inverse is quaternion conjugation on unit quaternions; the relative
    scalar q0^-1*q1 is dot(q0,q1), whose sign selects the shortest SLERP arc.
    """
    q = _finite(q, (4,), 'quaternion').copy()
    scale = np.max(np.abs(q)); _require(scale > 0, 'Zero quaternion')
    q /= scale; q /= np.linalg.norm(q)
    if q[np.flatnonzero(q)[0]] < 0:
        q = -q
    return q


def quaternion_matrix(q):
    w, x, y, z = normalized_quaternion(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def slerp(q0, q1, alpha):
    """Shortest-arc normalized SLERP; normalized lerp near coincidence."""
    _require(np.isfinite(alpha) and 0. <= alpha <= 1., 'SLERP alpha outside [0,1]')
    a, b = normalized_quaternion(q0), normalized_quaternion(q1)
    dot = float(np.dot(a, b))
    if dot < 0.:
        b, dot = -b, -dot
    dot = min(1., max(0., dot))  # roundoff only, after shortest-arc sign selection
    if dot > SLERP_NLERP_DOT:
        q = (1. - alpha) * a + alpha * b
    else:
        theta = np.arccos(dot)
        q = (np.sin((1. - alpha) * theta) * a + np.sin(alpha * theta) * b) / np.sin(theta)
    return normalized_quaternion(q)


def interpolate_history_pose(track, sample_times, timestamp_us):
    """Return (reason, bracket, alpha, center, rotation).

    At shared t[-1], choose [slot0,slot1] unconditionally, including when its
    boxes/link are missing. Never change the bracket based on GT availability.
    Both endpoints and A0 must exist even at an interval endpoint. No fallback.
    """
    t0, t1, t2 = sample_times
    if timestamp_us < t0 or timestamp_us > t2:
        return 1, (-1, -1), np.nan, None, None
    lo, hi = (0, 1) if timestamp_us <= t1 else (1, 2)
    alpha = (timestamp_us - sample_times[lo]) / (sample_times[hi] - sample_times[lo])
    slots = (lo, hi, 2)
    if any(not track['valid_mask'][s] or track['annotation_tokens'][s] is None for s in slots):
        return 2, (lo, hi), alpha, None, None
    if (track['annotation_next_tokens'][lo] != track['annotation_tokens'][hi] or
            track['annotation_prev_tokens'][hi] != track['annotation_tokens'][lo]):
        return 3, (lo, hi), alpha, None, None
    c0 = _finite(track['global_centers_m'][lo], (3,), 'Historical box center')
    c1 = _finite(track['global_centers_m'][hi], (3,), 'Historical box center')
    q = slerp(track['global_rotations_wxyz'][lo], track['global_rotations_wxyz'][hi], alpha)
    return 0, (lo, hi), alpha, (1.-alpha)*c0 + alpha*c1, quaternion_matrix(q)


def _frame(frame):
    _require(isinstance(frame, dict), 'Selected camera frame missing')
    _require(isinstance(frame['timestamp_us'], (int, np.integer)), 'Integer camera timestamp')
    pose = _pose(frame['camera_to_global'], 'camera_to_global')
    k = _finite(frame['intrinsic'], (3, 3), 'intrinsic')
    _require(k[0, 0] > 0 and k[1, 1] > 0 and np.array_equal(k[2], [0., 0., 1.]), 'Pinhole K')
    rgb = np.asarray(frame['rgb'])
    _require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3 and
             min(rgb.shape[:2]) >= 8, 'Raw RGB uint8 H,W,3')
    return int(frame['timestamp_us']), pose, k, rgb


def _project_global(points, camera_pose, k, shape):
    camera = (points - camera_pose[:3, 3]) @ camera_pose[:3, :3]
    homogeneous = camera @ k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    depth = camera[:, 2]
    h, w = shape[:2]; r = PHOTO.PATCH_RADIUS
    valid = ((depth > 0) & np.isfinite(uv).all(axis=1) &
             (uv[:, 0] >= r) & (uv[:, 0] < w-1-r) &
             (uv[:, 1] >= r) & (uv[:, 1] < h-1-r))
    # Nonprojectable UV is missing, not serialized +/-inf. Negative depth retained.
    uv[~np.isfinite(uv).all(axis=1)] = np.nan
    return uv, depth, valid


def evaluate_pose_address_control(points_R, lidar_to_global, object_index,
                                  instance_tokens, raw, camera_result, camera_pairs,
                                  chunk_size=4096):
    """Return reference arrays; cached hypotheses and every input are immutable.

    Main reasons are assigned by the explicit priority in REFERENCE_REASON_CODES.
    pose_reason_code[N,2] records both times independently. source identities and
    actual image/source SHA are caller responsibilities, not inferred from GT.
    """
    p = np.asarray(points_R, dtype=np.float64)
    _require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all(), 'points_R [N,3]')
    n = len(p); g0 = _pose(lidar_to_global, 'G0')
    obj = np.asarray(object_index); tokens = np.asarray(instance_tokens)
    _require(obj.shape == (n,) and obj.dtype.kind in 'iu' and tokens.ndim == 1 and
             len(set(tokens.tolist())) == len(tokens) and
             (n == 0 or (obj.min() >= 0 and obj.max() < len(tokens))), 'Object identity/index layout')
    _require(isinstance(chunk_size, int) and chunk_size > 0, 'Positive chunk size')
    frames = raw['frames']
    _require(len(frames) == 3 and [f['relative_frame_index'] for f in frames] == [-2, -1, 0],
             'Require exactly three historical frames; future frames forbidden')
    times = [f['timestamp_us'] for f in frames]
    _require(all(isinstance(t, (int, np.integer)) for t in times) and times[0] < times[1] < times[2],
             'Historical sample timestamps must increase')
    tracks = {t['instance_token']: t for t in raw['tracks']}
    _require(len(tracks) == len(raw['tracks']) and all(t in tracks for t in tokens.tolist()), 'Raw instance identity')
    for token in tokens.tolist():
        _require(all(len(tracks[token][k]) == 3 for k in TRACK_FIELDS), 'Track arrays must contain historical slots only')
    _require(set(camera_pairs) == set(CAMERA_ORDER), 'Exactly six named camera pairs')
    old = camera_result
    ci = np.asarray(old['camera_index']); valid = np.asarray(old['valid']); reason = np.asarray(old['reason_code'])
    _require(ci.shape == valid.shape == reason.shape == (n,) and valid.dtype == np.bool_ and
             np.isin(ci, np.arange(-1, 6)).all() and np.isin(reason, np.arange(5)).all() and
             np.array_equal(valid, reason == 0) and np.all(ci[valid] >= 0), 'Cached camera/validity layout')
    for key, shape in [('uv', (n, 2, 2, 2)), ('correlation', (n, 2)),
                       ('broken_correlation', (n, 2)), ('patch_std', (n, 2, 2)), ('timestamps_us', (n, 2))]:
        _require(np.asarray(old[key]).shape == shape and np.isfinite(np.asarray(old[key])[valid]).all(),
                 'Cached finite valid array: ' + key)
    out = dict(camera_index=ci.copy(), original_reason_code=reason.copy(),
        reference_reason_code=np.full(n, 1, dtype=np.uint8), reference_valid=np.zeros(n, dtype=bool),
        pose_reason_code=np.full((n, 2), 4, dtype=np.uint8),
        reference_uv=np.full((n, 2, 2), np.nan), reference_depth_m=np.full((n, 2), np.nan),
        reference_bracket_indices=np.full((n, 2, 2), -1, dtype=np.int8), reference_alpha=np.full((n, 2), np.nan),
        reference_displacement_R_m=np.full((n, 2, 3), np.nan), reference_patch_std=np.full((n, 2), np.nan),
        reference_pixel_departure_from_D=np.full((n, 2), np.nan), reference_pixel_departure_from_zero=np.full((n, 2), np.nan))
    score_keys = ('reference_correlation', 'reference_broken_correlation', 'reference_residual',
        'reference_broken_residual', 'delta_eD_minus_eGT', 'delta_eZero_minus_eGT',
        'broken_delta_eD_minus_eGT', 'broken_delta_eZero_minus_eGT', 'reference_broken_past_patch_std')
    out.update({key: np.full(n, np.nan) for key in score_keys})
    for camera_index, name in enumerate(CAMERA_ORDER):
        ix = np.flatnonzero(valid & (ci == camera_index))
        if not len(ix):
            continue
        camera_frames = [_frame(camera_pairs[name][when]) for when in TIME_ORDER]
        _require(camera_frames[1][0] < camera_frames[0][0], 'Past camera must precede current')
        _require(np.all(old['timestamps_us'][ix] == np.array([f[0] for f in camera_frames])),
                 'Selected frame times differ from cached hypotheses')
        projection_valid = np.zeros((n, 2), dtype=bool)
        for oi in np.unique(obj[ix]):
            jx = ix[obj[ix] == oi]; track = tracks[tokens[int(oi)]]
            # The material coordinate needs only the current box, never its size.
            has_origin = track['valid_mask'][2] and track['annotation_tokens'][2] is not None
            if has_origin:
                c0 = _finite(track['global_centers_m'][2], (3,), 'Current box center')
                r0 = quaternion_matrix(track['global_rotations_wxyz'][2])
                material = (p[jx] @ g0[:3, :3].T + g0[:3, 3] - c0) @ r0
            for ti, (timestamp, camera_pose, k, rgb) in enumerate(camera_frames):
                status, bracket, alpha, center, rotation = interpolate_history_pose(track, times, timestamp)
                out['pose_reason_code'][jx, ti] = status
                out['reference_bracket_indices'][jx, ti] = bracket
                out['reference_alpha'][jx, ti] = alpha
                if status:
                    continue
                global_points = material @ rotation.T + center
                out['reference_displacement_R_m'][jx, ti] = (global_points-g0[:3, 3]) @ g0[:3, :3] - p[jx]
                uv, depth, good = _project_global(global_points, camera_pose, k, rgb.shape)
                out['reference_uv'][jx, ti] = uv; out['reference_depth_m'][jx, ti] = depth
                projection_valid[jx, ti] = good
        # Assign later-priority reasons first, then overwrite in the stated order.
        out['reference_reason_code'][ix] = 0
        out['reference_reason_code'][ix[~projection_valid[ix, 1]]] = 5
        out['reference_reason_code'][ix[~projection_valid[ix, 0]]] = 4
        out['reference_reason_code'][ix[out['pose_reason_code'][ix, 1] != 0]] = 3
        out['reference_reason_code'][ix[out['pose_reason_code'][ix, 0] != 0]] = 2
        use = ix[out['reference_reason_code'][ix] == 0]
        if not len(use):
            continue
        gray0, gray1 = [PHOTO.grayscale(f[3]) for f in camera_frames]
        for start in range(0, len(use), chunk_size):
            jx = use[start:start+chunk_size]
            current_patch = PHOTO.sample_patches(gray0, out['reference_uv'][jx, 0])
            past_patch = PHOTO.sample_patches(gray1, out['reference_uv'][jx, 1])
            broken = PHOTO.sample_patches(gray1, out['reference_uv'][jx, 1], gray1.shape[1]//2)
            out['reference_patch_std'][jx] = np.stack([current_patch.std(1), past_patch.std(1)], axis=1)
            out['reference_broken_past_patch_std'][jx] = broken.std(1)
            out['reference_correlation'][jx] = PHOTO.regularized_zncc(current_patch, past_patch)
            out['reference_broken_correlation'][jx] = PHOTO.regularized_zncc(current_patch, broken)
        out['reference_valid'][use] = True
        for ai, suffix in ((0, 'zero'), (1, 'D')):
            out['reference_pixel_departure_from_'+suffix][use] = np.linalg.norm(
                out['reference_uv'][use] - old['uv'][use, ai], axis=-1)
    v = out['reference_valid']
    _require(np.array_equal(v, out['reference_reason_code'] == 0), 'Internal reference support mismatch')
    for prefix in ('', 'broken_'):
        corr = out['reference_'+prefix+'correlation']
        out['reference_'+prefix+'residual'][v] = 1.-corr[v]
        out[prefix+'delta_eD_minus_eGT'][v] = corr[v]-old[prefix+'correlation'][v, 1]
        out[prefix+'delta_eZero_minus_eGT'][v] = corr[v]-old[prefix+'correlation'][v, 0]
    for key in score_keys:
        _require(np.isfinite(out[key][v]).all() and np.isnan(out[key][~v]).all(), 'Score missing/finite: '+key)
    return out
