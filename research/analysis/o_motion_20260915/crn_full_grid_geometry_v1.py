"""Pure NumPy CRN current-state geometry on an explicit regular XYZ grid.

Input translations MUST already be global geometric centres (the verified
bottom-to-centre adapter has been applied exactly once upstream). This module
never loads predictions, GT, poses from files, learned weights, or Torch.

The ownership/velocity loop preserves the original operations and full point
matrix shape of predicted_object_state_cv_diagnostic_v2.predicted_velocity_field
(source SHA 0de90351e60330cc1a9208e3233f8b1077614ce69695ff044dd01f2f488516b5).
There is deliberately no AABB/top-k/score threshold/height projection. Real
200x200x16 cost has not yet been measured; do not claim accelerated throughput.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

GEOMETRY_SHA = 'e9d232c3f7aaacd073cfda645868e357afad9e5a2684d07b2f3f2394bd796a31'
REFERENCE_SHA = '0de90351e60330cc1a9208e3233f8b1077614ce69695ff044dd01f2f488516b5'
GMO = ('car', 'bus', 'truck', 'trailer', 'construction_vehicle',
       'motorcycle', 'bicycle', 'pedestrian')


def _geometry():
    path = Path(__file__).with_name('motion_geometry.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != GEOMETRY_SHA:
        raise ValueError('Frozen motion_geometry.py differs')
    spec = importlib.util.spec_from_file_location('_crn_full_grid_geometry', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def build_full_grid_geometry(centered_boxes, G0, *, grid_shape_xyz, extent_xyz,
                             box_origin='global_geometric_center'):
    """Return dense field plus metadata for ALL original export boxes.

    velocity_R: float64 [X,Y,Z,3], metres/second; owner: int64 [X,Y,Z].
    XYZ C-order has Z fastest. Owner is the original list index, or -1; no
    remapping to an eligible-only list. Highest score wins with strict >,
    earlier export wins ties, full 3D closed faces include exactly 1e-9 m.
    Uncovered field is zero. No GT-dependent support or occupancy mask exists.

    boxes keeps arrays in original order: export_index, eligible_GMO, c0_R
    [M,3], R0_R [M,3,3] (box-local to R), wlh [M,3], score [M], and original
    detection_name. c0_R/R0_R derive from inv(G0) @ box_to_global; they are
    metadata, NOT a numerically different route for the containment test.
    All original box metadata must be finite, including non-GMO boxes.
    Inputs are not mutated. box_origin is a caller assertion, not detection.
    """
    if box_origin != 'global_geometric_center':
        raise ValueError('Require already-centred global boxes; never adapt twice')
    if not isinstance(centered_boxes, list):
        raise ValueError('Export boxes must retain their list order')
    geo = _geometry();pose = geo._rigid_pose(G0, 'G0')
    grid = geo.voxel_centers_xyz(grid_shape_xyz, extent_xyz)
    shape = grid.shape[:3];points = grid.reshape(-1, 3)
    global_points = points @ pose[:3, :3].T + pose[:3, 3]
    velocity_global = np.zeros((len(points), 3), dtype=np.float64)
    owner = np.full(len(points), -1, dtype=np.int64)
    best = np.full(len(points), -np.inf, dtype=np.float64)
    inverse = np.linalg.inv(pose)
    count = len(centered_boxes)
    metadata = dict(export_index=np.arange(count, dtype=np.int64),
                    eligible_GMO=np.zeros(count, dtype=bool),
                    c0_R=np.empty((count, 3), dtype=np.float64),
                    R0_R=np.empty((count, 3, 3), dtype=np.float64),
                    wlh=np.empty((count, 3), dtype=np.float64),
                    score=np.empty(count, dtype=np.float64), detection_name=[])
    for index, box in enumerate(centered_boxes):
        score = float(box['detection_score'])
        if not math.isfinite(score): raise ValueError('Nonfinite detection score')
        centre = geo._finite_array(box['translation'], (3,), 'predicted centre')
        size = geo._finite_array(box['size'], (3,), 'predicted wlh')
        if not np.all(size > 0): raise ValueError('Nonpositive predicted box size')
        rotation = geo.quaternion_wxyz_to_matrix(box['rotation'])
        velocity = geo._finite_array(box['velocity'], (2,), 'global velocity')
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation;transform[:3, 3] = centre
        in_R = inverse @ transform
        metadata['c0_R'][index] = in_R[:3, 3]
        metadata['R0_R'][index] = in_R[:3, :3]
        metadata['wlh'][index] = size;metadata['score'][index] = score
        metadata['detection_name'].append(box['detection_name'])
        eligible = box['detection_name'] in GMO
        metadata['eligible_GMO'][index] = eligible
        if not eligible: continue
        # Keep exactly the original global -> box-local operation, without
        # transforming centres/axes to R or evaluating a reduced point array.
        local = (global_points - centre) @ rotation
        inside = np.all(np.abs(local) <= size[[1, 0, 2]] / 2. + 1e-9, axis=1)
        choose = inside & (score > best)
        owner[choose] = index;best[choose] = score
        velocity_global[choose, :2] = velocity
    velocity_R = velocity_global @ np.linalg.inv(pose)[:3, :3].T
    if not np.isfinite(velocity_R).all(): raise ValueError('Nonfinite transformed velocity')
    return dict(velocity_R=velocity_R.reshape(*shape, 3), owner=owner.reshape(shape),
                boxes=metadata, grid_shape_xyz=shape,
                extent_xyz=np.asarray(extent_xyz, dtype=np.float64).copy(),
                box_origin=box_origin, coordinate_order='XYZ_C_order_Z_fastest',
                reference_sha256=REFERENCE_SHA, geometry_sha256=GEOMETRY_SHA)
