"""Frozen optical-flow observation evidence, NumPy only; no I/O, GT or model.

evaluate_flow_evidence(camera_result, flow_by_camera) returns an N-row dict.
evaluate_camera_flow_evidence(camera_result, camera_index, flows) returns
(row_indices, local_result_dict) for one original camera, including invalid rows.
This lets a collector release each camera's three flow fields after scoring.
initialize_flow_evidence(camera_result) allocates the full result for that merge.

Each flows dict has exactly FLOW_KEYS, float32 [900,1600,2], unpadded, last axis
(dx,dy) in original pixels. camera_result has the frozen original camera_index,
reason_code, uv[N,hypothesis(zero,D),time(current,past),xy], optional valid and
source_flat_indices. Only those input fields are read. They are never modified.

Primary is original reason 0; the separately labelled extension is reason 4.
Audited old core SHA below writes all four valid UVs before assigning reason 4
(low current-zero texture); its NaN ZNCC scores are not consumed here. Reasons
1/2/3 remain missing. No camera, texture, confidence or FB selection is added.

Bilinear interpolation uses original pixel centres on the CLOSED rectangle
[0,1599] x [0,899], including exact final-row/column centres, without padding or
extrapolation. A forward endpoint outside this rectangle still has a primary
residual; it only makes reverse-flow sampling/FB undefined. Broken flow is an
independent inference on (current, rolled-past); candidate past UV is NOT rolled.
"""
import numpy as np

ORIGINAL_CAMERA_CORE_SHA256 = 'c7469179d1628979b30eb256e695fdd6479ffd6e2df047d380d890512f57a5ea'
CAMERA_ORDER = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
HEIGHT, WIDTH = 900, 1600
FLOW_KEYS = ('current_to_past', 'past_to_current', 'current_to_broken_past')
HYPOTHESIS_ORDER = ('zero', 'D')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _camera_arrays(camera_result):
    ci = np.asarray(camera_result['camera_index'])
    reason = np.asarray(camera_result['reason_code'])
    uv = np.asarray(camera_result['uv'])
    n = len(ci)
    _require(ci.shape == reason.shape == (n,) and ci.dtype == np.int8 and reason.dtype == np.uint8 and
             np.isin(ci, np.arange(-1, 6)).all() and np.isin(reason, np.arange(5)).all(), 'Original camera/reason layout')
    _require(np.array_equal(ci == -1, reason == 1), 'Original no-view reason differs')
    _require(uv.shape == (n, 2, 2, 2) and uv.dtype == np.float64, 'Original float64 UV axes differ')
    eligible = (reason == 0) | (reason == 4)
    _require(np.isfinite(uv[eligible]).all(), 'Original primary/lowtexture UV must be finite')
    if 'valid' in camera_result:
        valid = np.asarray(camera_result['valid'])
        _require(valid.dtype == np.bool_ and valid.shape == (n,) and np.array_equal(valid, reason == 0),
                 'Original primary mask differs from reason 0')
    if 'source_flat_indices' in camera_result:
        idx = np.asarray(camera_result['source_flat_indices'])
        _require(idx.dtype == np.int64 and idx.shape == (n,), 'Original source index layout')
    return ci, reason, uv


def _empty_result(camera_result, rows):
    n = len(rows)
    out = {key: np.asarray(camera_result[key])[rows].copy() for key in
           ('camera_index', 'reason_code', 'source_flat_indices') if key in camera_result}
    out.update(flow_valid=np.zeros(n, dtype=bool), primary_valid=np.zeros(n, dtype=bool),
               fb_valid=np.zeros((n, 2), dtype=bool))
    for key, shape in (
        ('residual', (n, 2)), ('broken_residual', (n, 2)), ('score', (n,)), ('broken_score', (n,)),
        ('hypothesis_flow_separation_px', (n,)), ('sampled_flow', (n, 2, 2)),
        ('sampled_broken_flow', (n, 2, 2)), ('predicted_past_uv', (n, 2, 2)), ('fb_residual', (n, 2))):
        out[key] = np.full(shape, np.nan, dtype=np.float64)
    return out


def _flow_arrays(flows):
    _require(isinstance(flows, dict) and set(flows) == set(FLOW_KEYS), 'Exactly three named flow fields required')
    result = {}
    for key in FLOW_KEYS:
        flow = np.asarray(flows[key])
        _require(flow.dtype == np.float32 and flow.shape == (HEIGHT, WIDTH, 2) and np.isfinite(flow).all(),
                 'Finite unpadded float32 H,W,xy flow required: ' + key)
        result[key] = flow
    return result


def initialize_flow_evidence(camera_result):
    """Full N allocation: original identity copies, false bools and NaN floats.

    Collector usage: result=initialize_flow_evidence(old); for each camera call
    rows,part=evaluate_camera_flow_evidence(old,ci,flows), then assign each
    result[key][rows]=part[key]. Original no-camera rows stay missing.
    """
    ci, _, _ = _camera_arrays(camera_result)
    return _empty_result(camera_result, np.arange(len(ci)))


def bilinear_sample_flow(flow, uv):
    """Return float64 samples and bool validity; missing samples remain NaN.

    No endpoint clipping or extrapolation. At an exact final pixel centre only,
    the unused +1 neighbour is that same final pixel (its interpolation weight
    would otherwise be zero). This keeps the closed-domain boundary defined.
    """
    flow = np.asarray(flow); q = np.asarray(uv, dtype=np.float64)
    _require(flow.dtype == np.float32 and flow.shape == (HEIGHT, WIDTH, 2), 'Flow layout')
    _require(q.ndim == 2 and q.shape[1] == 2, 'UV [N,2] required')
    good = (np.isfinite(q).all(axis=1) & (q[:, 0] >= 0.) & (q[:, 0] <= WIDTH-1) &
            (q[:, 1] >= 0.) & (q[:, 1] <= HEIGHT-1))
    out = np.full(q.shape, np.nan, dtype=np.float64)
    p = q[good]
    if len(p):
        x0 = np.floor(p[:, 0]).astype(np.int64); y0 = np.floor(p[:, 1]).astype(np.int64)
        x1 = np.minimum(x0+1, WIDTH-1); y1 = np.minimum(y0+1, HEIGHT-1)
        wx = (p[:, 0]-x0)[:, None]; wy = (p[:, 1]-y0)[:, None]
        a, b = flow[y0, x0].astype(np.float64), flow[y0, x1].astype(np.float64)
        c, d = flow[y1, x0].astype(np.float64), flow[y1, x1].astype(np.float64)
        out[good] = (1.-wy)*((1.-wx)*a+wx*b) + wy*((1.-wx)*c+wx*d)
        _require(np.isfinite(out[good]).all(), 'Nonfinite sampled flow')
    return out, good


def evaluate_camera_flow_evidence(camera_result, camera_index, flows):
    """One-camera result in original local row order; no flow needed if inactive.

    camera_index is the integer in CAMERA_ORDER. No-view rows (-1) are retained
    by the all-camera wrapper but cannot belong to a per-camera result.
    """
    ci, reason, uv = _camera_arrays(camera_result)
    _require(isinstance(camera_index, (int, np.integer)) and 0 <= camera_index < 6, 'Camera index 0..5 required')
    rows = np.flatnonzero(ci == camera_index)
    out = _empty_result(camera_result, rows)
    active = (reason[rows] == 0) | (reason[rows] == 4)
    if not active.any():
        return rows, out
    fields = _flow_arrays(flows)
    original = rows[active]; q = uv[original]
    current, past = q[:, :, 0, :], q[:, :, 1, :]
    sampled, forward_valid = bilinear_sample_flow(fields['current_to_past'], current.reshape(-1, 2))
    sampled_broken, broken_valid = bilinear_sample_flow(fields['current_to_broken_past'], current.reshape(-1, 2))
    # An authenticated reason 0/4 always has legal current projection. A mismatch
    # is a contract error, never a hidden change to the published population.
    _require(forward_valid.all() and broken_valid.all(), 'Original current UV outside forward sampling domain')
    sampled = sampled.reshape(-1, 2, 2); sampled_broken = sampled_broken.reshape(-1, 2, 2)
    endpoint = current + sampled
    residual = np.linalg.norm(endpoint-past, axis=-1)
    broken_residual = np.linalg.norm(current+sampled_broken-past, axis=-1)
    displacement = past-current
    separation = np.linalg.norm(displacement[:, 1]-displacement[:, 0], axis=-1)
    reverse, reverse_valid = bilinear_sample_flow(fields['past_to_current'], endpoint.reshape(-1, 2))
    reverse = reverse.reshape(-1, 2, 2); reverse_valid = reverse_valid.reshape(-1, 2)
    fb = np.full(residual.shape, np.nan, dtype=np.float64)
    fb[reverse_valid] = np.linalg.norm((sampled+reverse)[reverse_valid], axis=-1)
    out['flow_valid'][active] = True
    out['primary_valid'][active] = reason[original] == 0
    values = dict(residual=residual, broken_residual=broken_residual,
        score=residual[:, 0]-residual[:, 1], broken_score=broken_residual[:, 0]-broken_residual[:, 1],
        hypothesis_flow_separation_px=separation, sampled_flow=sampled,
        sampled_broken_flow=sampled_broken, predicted_past_uv=endpoint,
        fb_valid=reverse_valid, fb_residual=fb)
    for key, value in values.items():
        out[key][active] = value
    for key in ('residual', 'broken_residual', 'score', 'broken_score', 'hypothesis_flow_separation_px',
                'sampled_flow', 'sampled_broken_flow', 'predicted_past_uv'):
        _require(np.isfinite(out[key][active]).all() and np.isnan(out[key][~active]).all(), 'Output finite/missing: '+key)
    _require(np.isfinite(out['fb_residual'][out['fb_valid']]).all() and
             np.isnan(out['fb_residual'][~out['fb_valid']]).all(), 'FB missing contract')
    return rows, out


def evaluate_flow_evidence(camera_result, flow_by_camera):
    """Return all original rows. Mapping needs flows only for reason 0/4 cameras."""
    ci, _, _ = _camera_arrays(camera_result)
    _require(isinstance(flow_by_camera, dict) and set(flow_by_camera) <= set(CAMERA_ORDER), 'Unknown camera flow name')
    out = initialize_flow_evidence(camera_result)
    for camera_index, name in enumerate(CAMERA_ORDER):
        rows, local = evaluate_camera_flow_evidence(camera_result, camera_index, flow_by_camera.get(name))
        for key, value in local.items():
            out[key][rows] = value
    return out
