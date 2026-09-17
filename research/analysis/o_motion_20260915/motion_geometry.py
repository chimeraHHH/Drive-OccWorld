"""NumPy-only, float64 geometry in metres with column-vector SE(3) matrices.

Quaternion order is w,x,y,z. Box size order is width,length,height; the
box-local x axis is length, y is width, z is height (nuScenes convention).
No data loading, model state, label policy, or train/evaluation masks live here.
"""

import numpy as np


# Validate supplied rigid poses; tolerate float32 metadata roundoff, not scale.
_POSE_ATOL = 1e-6
# Closed box faces, with this absolute tolerance in metres for transform noise.
_BOX_BOUNDARY_ATOL_METRES = 1e-9


def _finite_array(value, shape, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {arr.shape}")
    return arr


def _points(value):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.all(np.isfinite(arr)):
        raise ValueError("points_R must be a finite array of shape [N,3]")
    return arr


def _rigid_pose(value, name):
    pose = _finite_array(value, (4, 4), name)
    if not np.allclose(pose[3], [0., 0., 0., 1.], rtol=0., atol=_POSE_ATOL):
        raise ValueError(f"{name} must have homogeneous last row [0,0,0,1]")
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=_POSE_ATOL)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=_POSE_ATOL)):
        raise ValueError(f"{name} must contain a proper rigid rotation")
    return pose


def quaternion_wxyz_to_matrix(q):
    """Return a 3x3 active rotation, normalizing any finite nonzero quaternion.

    Normalization is scale-safe, including very large/small nonunit q. The
    input is never mutated. Zero or nonfinite quaternions are rejected.
    """
    quat = _finite_array(q, (4,), "q")
    scale = np.max(np.abs(quat))
    if scale == 0.:
        raise ValueError("q must be nonzero")
    unit = quat / scale
    unit = unit / np.linalg.norm(unit)
    w, x, y, z = unit
    return np.array([
        [1. - 2.*(y*y + z*z), 2.*(x*y - z*w), 2.*(x*z + y*w)],
        [2.*(x*y + z*w), 1. - 2.*(x*x + z*z), 2.*(y*z - x*w)],
        [2.*(x*z - y*w), 2.*(y*z + x*w), 1. - 2.*(x*x + y*y)],
    ], dtype=np.float64)


def box_to_global(center, wxyz):
    """Return box-local to global column transform (rotation and translation).

    `center` is the global box centre in metres. Size is deliberately absent:
    this is a rigid coordinate transform, not normalized-box scaling.
    """
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quaternion_wxyz_to_matrix(wxyz)
    out[:3, 3] = _finite_array(center, (3,), "center")
    return out


def rigid_displacement(points_R, G0, A0, Ah):
    """Rigid box proxy displacement at source points, in fixed R, in metres.

    G0 maps reference LiDAR R to global. A0/Ah map the same object's local
    coordinates to global at time 0/h. Return p_h^R - p_0^R using
    inv(G0) @ Ah @ inv(A0) @ G0. There is intentionally no future ego pose:
    camera/LiDAR motion alone cannot create object displacement in fixed R.
    This does not assert that points inside a real object move rigidly.
    """
    points = _points(points_R)
    g0 = _rigid_pose(G0, "G0")
    a0 = _rigid_pose(A0, "A0")
    ah = _rigid_pose(Ah, "Ah")
    transport = np.linalg.inv(g0) @ ah @ np.linalg.inv(a0) @ g0
    # Points receive translation; displacement vectors must not receive it
    # again when expressed in a different coordinate basis.
    moved = points @ transport[:3, :3].T + transport[:3, 3]
    return moved - points


def unique_box_assignment(points_R, box_in_R, sizes_wlh):
    """Return zero-based box indices, -1 for background, -2 for overlaps.

    `box_in_R[j]` maps box-local coordinates to R. Boxes have closed faces
    (absolute boundary tolerance 1e-9 metres). A shared face counts as an
    overlap, never a first-box tie break. Empty point/box arrays are allowed.
    O(Npoints) temporary storage is used rather than an Npoints x Nboxes mask.
    """
    points = _points(points_R)
    boxes = np.asarray(box_in_R, dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[1:] != (4, 4):
        raise ValueError("box_in_R must have shape [Nbox,4,4]")
    sizes = _finite_array(sizes_wlh, (len(boxes), 3), "sizes_wlh")
    if np.any(sizes <= 0.):
        raise ValueError("sizes_wlh must be strictly positive")
    labels = np.full(len(points), -1, dtype=np.int64)
    for index, box in enumerate(boxes):
        box = _rigid_pose(box, f"box_in_R[{index}]")
        inverse = np.linalg.inv(box)
        local = points @ inverse[:3, :3].T + inverse[:3, 3]
        half_extent_xyz = sizes[index, [1, 0, 2]] / 2.
        inside = np.all(np.abs(local) <= half_extent_xyz + _BOX_BOUNDARY_ATOL_METRES, axis=1)
        newly_assigned = inside & (labels == -1)
        already_assigned = inside & (labels != -1)
        labels[newly_assigned] = index
        labels[already_assigned] = -2
    return labels


def voxel_centers_xyz(shape, extent):
    """Return [X,Y,Z,3] centres in metres, with last axis ordered x,y,z.

    `shape` is three positive integer counts (X,Y,Z). `extent` is exactly
    [xmin,ymin,zmin,xmax,ymax,zmax]; it describes outer cell boundaries,
    not the first/last centre. No implicit transpose or clipping is applied.
    """
    dims = tuple(shape)
    if (len(dims) != 3 or any(isinstance(n, (bool, np.bool_))
                            or not isinstance(n, (int, np.integer)) or n <= 0 for n in dims)):
        raise ValueError("shape must contain three positive integer counts")
    bounds = _finite_array(extent, (6,), "extent")
    lo, hi = bounds[:3], bounds[3:]
    if np.any(hi <= lo):
        raise ValueError("extent maxima must exceed minima on every axis")
    axes = [lo[i] + (np.arange(dims[i], dtype=np.float64) + 0.5)
            * ((hi[i] - lo[i]) / dims[i]) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
