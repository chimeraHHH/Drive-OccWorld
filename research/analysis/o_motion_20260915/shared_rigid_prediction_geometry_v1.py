"""Pure NumPy current CRN geometry for shared Cpl/Fix rigid state consumers.

Only already-centred current predictions, current G0 and an explicit regular
XYZ grid cross this API. No annotation, GT mask, sparse support, future pose,
origin adaptation, Torch import, device transfer or cache construction occurs.

Keep the two frozen numerical domains: dense CV uses the original full point
matrix; object states use current_states_numpy's per-object matrix-vector path.
Do not rebuild the former from the latter or from float32 packed tokens.
"""
import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np

SOURCES = {
    'crn_full_grid_geometry_v1.py': '1c1e7f0799b593b3ad41f282d8506bdc1312a1a8e53994ea8ba595ad12055b9c',
    'object_state_prediction_inputs_v1.py': '55e39715fdf19e5ae1f227a943a843b194eabed08e788ccb2694a040d6f18256',
    'motion_geometry.py': 'e9d232c3f7aaacd073cfda645868e357afad9e5a2684d07b2f3f2394bd796a31',
}


def _bound(name):
    path = Path(__file__).with_name(name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCES[name]:
        raise ValueError('Frozen source differs: ' + name)
    spec = importlib.util.spec_from_file_location('_shared_rigid_' + path.stem, path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def geometry_from_centered_boxes(boxes, G0, *, grid_shape_xyz=(200, 200, 16),
                                 extent_xyz=(-51.2, -51.2, -5., 51.2, 51.2, 3.)):
    """Return full current geometry; every owner indexes the retained GMO list.

    points_R and cv_velocity_R: float64[Q,3]; owner: int64[Q] in {-1,0..M-1}.
    Flattening is XYZ C-order (Z fastest), not native BEV y*X+x token order.
    owner_original_indices: int64[Q] exactly as the original export/reference.
    retained_original_box_indices: int64[M], original list order, all 8 GMO
    classes with no score/point-count/ROI filtering; uncovered points stay -1.

    states_numpy is EXACT current_states_numpy output (center,size,rotation,
    classes,score,velocity). The c0_R/R0_R/wlh/classes/score/velocity_R keys are
    aliases of these same float64 arrays (classes int64). Caller owns device
    conversion and must retain float64 geometry for rigid_object_motion_v1.
    Do not mutate aliases independently. All inputs remain unchanged.

    receipt includes measured CPU geometry cost; it is neither inference cost
    nor a performance result. No values are learned or fitted in this helper.
    """
    started = time.monotonic()
    full = _bound('crn_full_grid_geometry_v1.py')
    inputs = _bound('object_state_prediction_inputs_v1.py')
    geo = _bound('motion_geometry.py')
    if full.GMO != inputs.GMO: raise ValueError('GMO class order differs')
    tick = time.monotonic()
    field = full.build_full_grid_geometry(boxes, G0, grid_shape_xyz=grid_shape_xyz,
                                          extent_xyz=extent_xyz, box_origin='global_geometric_center')
    full_seconds = time.monotonic() - tick
    states, retained = inputs.current_states_numpy(boxes, G0)
    retained = np.asarray(retained, dtype=np.int64)
    expected = np.flatnonzero(field['boxes']['eligible_GMO'])
    if not np.array_equal(retained, expected): raise ValueError('Retained original box order differs')
    original_owner = field['owner'].reshape(-1)
    covered = original_owner >= 0
    if not np.all((original_owner >= -1) & (original_owner < len(boxes))):
        raise ValueError('Invalid original owner index')
    reverse = np.full(len(boxes), -1, dtype=np.int64)
    reverse[retained] = np.arange(len(retained), dtype=np.int64)
    owner = np.full(original_owner.shape, -1, dtype=np.int64)
    owner[covered] = reverse[original_owner[covered]]
    if np.any(owner[covered] < 0): raise ValueError('Non-GMO object owns a grid point')
    if not np.array_equal(retained[owner[covered]], original_owner[covered]):
        raise ValueError('Original-index round trip failed')
    points = geo.voxel_centers_xyz(grid_shape_xyz, extent_xyz).reshape(-1, 3)
    cv_velocity = field['velocity_R'].reshape(-1, 3)
    if points.shape != cv_velocity.shape or len(points) != len(owner):
        raise ValueError('Full grid field shape differs')
    pose = np.asarray(G0, dtype='<f8')
    receipt = dict(schema='shared-rigid-prediction-geometry-v1', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   dependencies_sha256=dict(SOURCES), grid_shape_xyz=list(field['grid_shape_xyz']),
                   extent_xyz=field['extent_xyz'].tolist(), coordinate_order=field['coordinate_order'],
                   points=len(points), original_boxes=len(boxes), retained_GMO_boxes=len(retained),
                   retained_original_box_indices=retained.tolist(), covered_points=int(covered.sum()),
                   current_G0_sha256=hashlib.sha256(pose.tobytes()).hexdigest(),
                   full_grid_seconds=full_seconds, total_seconds=time.monotonic()-started,
                   owner_original_index_roundtrip_exact=True, states_from_original_current_states_numpy=True,
                   dense_CV_from_original_full_grid=True, box_origin='global_geometric_center',
                   origin_adaptation_performed=False, GT_or_sparse_support_input=False)
    return dict(points_R=points, owner=owner, owner_original_indices=original_owner,
                cv_velocity_R=cv_velocity, c0_R=states['center'], R0_R=states['rotation'],
                wlh=states['size'], velocity_R=states['velocity'], score=states['score'],
                classes=states['classes'], states_numpy=states,
                retained_original_box_indices=retained, receipt=receipt)
