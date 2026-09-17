"""Fixed, input-only historical image correspondence diagnostic (NumPy CPU).

R is the current LiDAR frame; transforms are column-vector local-to-global.
No labels, depth map, association IDs, fitted parameters, image search or I/O.
The caller authenticates source records, original availability and same-camera
current/previous-keyframe identity. Projectable does not mean unoccluded.
"""
import numpy as np

CAMERA_ORDER = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
HORIZONS_SECONDS = (.5, 1., 1.5, 2.)
PATCH_RADIUS = 3
PATCH_SIZE = 7
ZNCC_EPS = 1. / 255.
TEXTURE_STD_MIN = 2. / 255.
REASON_CODES = {0: 'valid', 1: 'no_current_zero_view',
                2: 'missing_previous_keyframe',
                3: 'hypothesis_projection_invalid', 4: 'low_current_zero_texture'}
AXES = {'hypothesis': ['zero', 'D'], 'time': ['current', 'past'],
        'uv': ['pixel_x', 'pixel_y']}
_OFFSETS = np.stack(np.meshgrid(np.arange(-3, 4), np.arange(-3, 4)), -1).reshape(49, 2)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    _require(a.shape == shape and np.isfinite(a).all(), name + ' shape/finite contract')
    return a


def _pose(value, name):
    a = _finite_array(value, (4, 4), name)
    _require(np.allclose(a[3], [0., 0., 0., 1.], atol=1e-10, rtol=0.), name + ' homogeneous row')
    _require(np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-8, rtol=0.) and
             abs(np.linalg.det(a[:3, :3]) - 1.) <= 1e-8, name + ' rigid rotation')
    return a


def _frame(frame, availability_timestamp_us):
    _require(isinstance(frame, dict), 'Camera frame must be a dictionary')
    timestamp = frame['timestamp_us']
    _require(isinstance(timestamp, (int, np.integer)) and
             timestamp <= availability_timestamp_us, 'Camera outside authenticated availability')
    pose = _pose(frame['camera_to_global'], 'camera_to_global')
    k = _finite_array(frame['intrinsic'], (3, 3), 'intrinsic')
    _require(k[0, 0] > 0 and k[1, 1] > 0 and
             np.array_equal(k[2], [0., 0., 1.]), 'Pinhole intrinsic contract')
    rgb = np.asarray(frame['rgb'])
    _require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3 and
             min(rgb.shape[:2]) >= PATCH_SIZE + 1, 'Raw RGB uint8 H,W,3 required')
    return dict(timestamp_us=int(timestamp), camera_to_global=pose, intrinsic=k, rgb=rgb)


def nominal_lsq_velocity(displacement_m):
    """Return [N,3] m/s using all four fixed nominal horizons, never future dt."""
    d = np.asarray(displacement_m, dtype=np.float64)
    _require(d.ndim == 3 and d.shape[0] == 4 and d.shape[2] == 3 and
             np.isfinite(d).all(), 'Expected finite displacement [4,N,3]')
    h = np.asarray(HORIZONS_SECONDS)
    return np.einsum('h,hnc->nc', h, d) / np.dot(h, h)


def grayscale(rgb):
    """BT.601 scalar grayscale in [0,1], original pixel resolution."""
    a = np.asarray(rgb)
    _require(a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3, 'Raw RGB required')
    return np.einsum('hwc,c->hw', a, np.asarray([.299, .587, .114])) / 255.


def project(points_R, velocity_R_mps, lidar_to_global, lidar_timestamp_us, frame):
    """Return uv, camera depth, normalized optical-axis distance, patch-valid."""
    dt = (frame['timestamp_us'] - lidar_timestamp_us) * 1e-6
    p = points_R + dt * velocity_R_mps
    relative = np.linalg.inv(frame['camera_to_global']) @ lidar_to_global
    q = p @ relative[:3, :3].T + relative[:3, 3]
    depth = q[:, 2].copy()
    homogeneous = q @ frame['intrinsic'].T
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]
        optical = np.linalg.norm(q[:, :2] / q[:, 2:3], axis=1)
    height, width = frame['rgb'].shape[:2]
    # Strict upper bound leaves the bilinear +1 neighbour in the raw image.
    valid = ((depth > 0) & np.isfinite(uv).all(axis=1) & np.isfinite(optical) &
             (uv[:, 0] >= PATCH_RADIUS) & (uv[:, 0] < width - 1 - PATCH_RADIUS) &
             (uv[:, 1] >= PATCH_RADIUS) & (uv[:, 1] < height - 1 - PATCH_RADIUS))
    return uv, depth, optical, valid


def sample_patches(gray, uv, horizontal_roll_pixels=0):
    """Bilinear 7x7 patches. Positive shift exactly represents np.roll(image)."""
    height, width = gray.shape
    xy = uv[:, None, :] + _OFFSETS[None, :, :]
    x = xy[:, :, 0] - horizontal_roll_pixels
    y = xy[:, :, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    wx, wy = x - x0, y - y0
    _require(np.all((y0 >= 0) & (y0 + 1 < height)), 'Patch y outside image')
    if horizontal_roll_pixels:
        xa, xb = x0 % width, (x0 + 1) % width
    else:
        _require(np.all((x0 >= 0) & (x0 + 1 < width)), 'Patch x outside image')
        xa, xb = x0, x0 + 1
    return ((1 - wy) * ((1 - wx) * gray[y0, xa] + wx * gray[y0, xb]) +
            wy * ((1 - wx) * gray[y0 + 1, xa] + wx * gray[y0 + 1, xb]))


def regularized_zncc(a, b):
    """Centered correlation, fixed variance regularizer; flat patches give zero."""
    _require(a.ndim == 2 and a.shape == b.shape and a.shape[1] == 49, 'Patch layout differs')
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    numerator = np.einsum('np,np->n', ac, bc)
    denominator = np.sqrt((np.einsum('np,np->n', ac, ac) + 49 * ZNCC_EPS**2) *
                          (np.einsum('np,np->n', bc, bc) + 49 * ZNCC_EPS**2))
    return numerator / denominator


def evaluate_history_correspondence(points_R, velocity_R_mps, lidar_to_global,
                                    lidar_timestamp_us, availability_timestamp_us,
                                    camera_pairs, chunk_size=4096):
    """Input-only field queries; invalid scores are NaN, never silent zeros.

    camera_pairs: CAMERA_ORDER keys, each {current: frame|None, past: frame|None}.
    frame: timestamp_us, camera_to_global[4,4], intrinsic[3,3], rgb uint8[H,W,3].
    Arrays uv/depth/patch_std use [point,hypothesis(zero,D),time(current,past),...].
    Source metadata authentication and original sparse support remain caller duties.
    """
    p = np.asarray(points_R, dtype=np.float64)
    _require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all(), 'Points [N,3] required')
    v = _finite_array(velocity_R_mps, p.shape, 'velocity_R_mps')
    g0 = _pose(lidar_to_global, 'lidar_to_global')
    _require(isinstance(lidar_timestamp_us, (int, np.integer)) and
             isinstance(availability_timestamp_us, (int, np.integer)) and
             availability_timestamp_us >= lidar_timestamp_us, 'LiDAR/availability timestamps invalid')
    _require(isinstance(chunk_size, int) and chunk_size > 0, 'Positive chunk size required')
    _require(set(camera_pairs) == set(CAMERA_ORDER), 'Exactly six named camera entries required')
    frames = []
    for name in CAMERA_ORDER:
        pair = camera_pairs[name]
        current = None if pair['current'] is None else _frame(pair['current'], availability_timestamp_us)
        past = None if pair['past'] is None else _frame(pair['past'], availability_timestamp_us)
        if current is not None and past is not None:
            _require(past['timestamp_us'] < current['timestamp_us'], 'Past must precede same-camera current')
            _require(past['timestamp_us'] < lidar_timestamp_us, 'Previous keyframe must precede LiDAR t0')
        frames.append((current, past))

    n = len(p)
    selected = np.full(n, -1, dtype=np.int8)
    best = np.full(n, np.inf)
    zero = np.zeros_like(v)
    # Camera choice does not inspect D, pixels, past availability or labels.
    for ci, (current, _) in enumerate(frames):
        if current is None:
            continue
        _, _, optical, valid = project(p, zero, g0, lidar_timestamp_us, current)
        use = valid & (optical < best)  # strict < preserves fixed order on ties
        selected[use], best[use] = ci, optical[use]

    out = dict(camera_index=selected, reason_code=np.full(n, 1, dtype=np.uint8),
               valid=np.zeros(n, dtype=bool), optical_axis_distance=np.where(selected >= 0, best, np.nan),
               timestamps_us=np.full((n, 2), -1, dtype=np.int64),
               uv=np.full((n, 2, 2, 2), np.nan), depth_m=np.full((n, 2, 2), np.nan),
               patch_std=np.full((n, 2, 2), np.nan), broken_past_patch_std=np.full((n, 2), np.nan),
               correlation=np.full((n, 2), np.nan), broken_correlation=np.full((n, 2), np.nan),
               residual=np.full((n, 2), np.nan), broken_residual=np.full((n, 2), np.nan),
               score=np.full(n, np.nan), broken_score=np.full(n, np.nan),
               hypothesis_pixel_separation=np.full((n, 2), np.nan))
    for ci, (current, past) in enumerate(frames):
        indices = np.flatnonzero(selected == ci)
        if not len(indices):
            continue
        out['timestamps_us'][indices, 0] = current['timestamp_us']
        if past is None:
            out['reason_code'][indices] = 2
            continue
        out['timestamps_us'][indices, 1] = past['timestamp_us']
        gray_current, gray_past = grayscale(current['rgb']), grayscale(past['rgb'])
        for start in range(0, len(indices), chunk_size):
            ix = indices[start:start + chunk_size]
            projected_valid = np.ones(len(ix), dtype=bool)
            for ai, velocity in enumerate((zero[ix], v[ix])):
                for ti, frame in enumerate((current, past)):
                    uv, depth, _, valid = project(p[ix], velocity, g0, lidar_timestamp_us, frame)
                    out['uv'][ix, ai, ti], out['depth_m'][ix, ai, ti] = uv, depth
                    projected_valid &= valid
            out['hypothesis_pixel_separation'][ix] = np.linalg.norm(
                out['uv'][ix, 1] - out['uv'][ix, 0], axis=-1)
            out['reason_code'][ix] = 3
            jx = ix[projected_valid]
            if not len(jx):
                continue
            patches = [[sample_patches(gray_current if ti == 0 else gray_past,
                                       out['uv'][jx, ai, ti]) for ti in range(2)] for ai in range(2)]
            for ai in range(2):
                for ti in range(2):
                    out['patch_std'][jx, ai, ti] = patches[ai][ti].std(axis=1)
            # This ONE mask is reused for real and broken. No past/control-specific mask.
            textured = out['patch_std'][jx, 0, 0] >= TEXTURE_STD_MIN
            out['reason_code'][jx] = 4
            kx = jx[textured]
            if not len(kx):
                continue
            for ai in range(2):
                a, b = patches[ai][0][textured], patches[ai][1][textured]
                broken_b = sample_patches(gray_past, out['uv'][kx, ai, 1], gray_past.shape[1] // 2)
                out['broken_past_patch_std'][kx, ai] = broken_b.std(axis=1)
                out['correlation'][kx, ai] = regularized_zncc(a, b)
                out['broken_correlation'][kx, ai] = regularized_zncc(a, broken_b)
            out['valid'][kx], out['reason_code'][kx] = True, 0
        del gray_current, gray_past
    valid = out['valid']
    out['residual'][valid] = 1. - out['correlation'][valid]
    out['broken_residual'][valid] = 1. - out['broken_correlation'][valid]
    out['score'][valid] = out['residual'][valid, 0] - out['residual'][valid, 1]
    out['broken_score'][valid] = out['broken_residual'][valid, 0] - out['broken_residual'][valid, 1]
    _require(np.isfinite(out['score'][valid]).all() and np.isfinite(out['broken_score'][valid]).all(),
             'Nonfinite score on common valid support')
    return out
