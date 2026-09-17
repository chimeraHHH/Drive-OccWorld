"""Source-indexed, differentiable trilinear transport; no model or GT policy.

Axes are [B,C,X,Y,Z], not a grid_sample [D,H,W] convention. A source
voxel centre p_i carries displacement d_i in metres, so its destination is
p_i+d_i. We splat to the eight destination centres with trilinear weights.
Outside-neighbour weights are discarded, NOT clamped or renormalized.

This is a feature transport primitive, not an occupancy probability model.
Coverage counts geometric source weight, including sources whose feature is
zero. Normalization is weighted averaging; it is not a probability union.
CUDA scatter_add can use nondeterministic floating-point atomic reduction.
The function neither changes global precision/determinism nor selects devices.
"""

from itertools import product
import math

import torch


DEFAULT_EXTENT = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


def forward_splat_3d(source, displacement_m, extent=DEFAULT_EXTENT,
                     grid_shape=None):
    """Transport source features to a grid with the same shape and extent.

    Args:
        source: float32/float64 [B,C,X,Y,Z], or [B,C,N] with grid_shape.
        displacement_m: same dtype/device, [B,3,X,Y,Z] or [B,3,N].
            Channels are physical dx,dy,dz, indexed by SOURCE, N=X*Y*Z.
        extent: (xmin,ymin,zmin,xmax,ymax,zmax), in metres.
        grid_shape: optional explicit (X,Y,Z); required for flattened source.

    Returns a dict of tensors:
        numerator [B,C,X,Y,Z]: sum of retained weight * source feature.
        coverage [B,1,X,Y,Z]: sum of retained geometric weights.
        normalized [B,C,X,Y,Z]: numerator/coverage where coverage>0, else 0.
        support [B,1,X,Y,Z]: coverage>0, without an arbitrary threshold.
        source_weight_mass, retained_weight_mass, dropped_weight_mass [B].
        source_feature_sum, retained_feature_sum, dropped_feature_sum [B,C].
            These last quantities are signed channel integrals for general
            features, not necessarily nonnegative physical masses.

    Retained + dropped weight equals X*Y*Z per batch, within floating error.
    Source channel sums equal retained + dropped channel sums likewise.
    There is no source mask, threshold, ego transform, probability clipping,
    implicit inverse-flow conversion, or mixing with another model output.
    Caller must supply displacement and the grid in one physical frame.
    """
    if not isinstance(source, torch.Tensor) or not isinstance(displacement_m, torch.Tensor):
        raise TypeError('source and displacement_m must be torch tensors')
    if source.dtype not in (torch.float32, torch.float64):
        raise TypeError('source must be float32 or float64')
    if displacement_m.dtype != source.dtype or displacement_m.device != source.device:
        raise ValueError('displacement must have exactly the source dtype/device')
    if source.ndim not in (3, 5):
        raise ValueError('source layout must be [B,C,X,Y,Z] or [B,C,N]')
    if grid_shape is None:
        if source.ndim != 5:
            raise ValueError('flattened source requires explicit grid_shape=(X,Y,Z)')
        shape = tuple(source.shape[2:])
    else:
        shape = tuple(grid_shape)
        if len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError('grid_shape must contain three positive integers')
        if source.ndim == 5 and tuple(source.shape[2:]) != shape:
            raise ValueError('grid_shape disagrees with dense source')
    batch, channels = source.shape[:2]
    if batch <= 0 or channels <= 0 or any(n <= 0 for n in shape):
        raise ValueError('all source dimensions must be positive')
    npoints = math.prod(shape)
    if math.prod(source.shape[2:]) != npoints:
        raise ValueError('source length disagrees with grid_shape')
    if tuple(displacement_m.shape) not in ((batch, 3) + shape, (batch, 3, npoints)):
        raise ValueError('displacement layout/shape must be [B,3,X,Y,Z] or [B,3,N]')
    if len(extent) != 6:
        raise ValueError('extent must be xmin,ymin,zmin,xmax,ymax,zmax')
    extent = tuple(float(v) for v in extent)
    if not all(math.isfinite(v) for v in extent):
        raise ValueError('extent must be finite')
    lengths = tuple(extent[i + 3] - extent[i] for i in range(3))
    if any(v <= 0 for v in lengths):
        raise ValueError('extent maxima must exceed minima')
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(displacement_m).all()):
        raise ValueError('source and displacement must contain only finite values')

    values = source.reshape(batch, channels, npoints)
    flow = displacement_m.reshape(batch, 3, npoints)
    cell_size = source.new_tensor([lengths[i] / shape[i] for i in range(3)]).view(1, 3, 1)
    axes = [torch.arange(n, dtype=source.dtype, device=source.device) for n in shape]
    source_index = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=0).reshape(1, 3, npoints)
    # Algebraically (p_i+d_i-min)/cell_size-.5; this form gives exact d=0
    # identity without subtracting rounded physical voxel-centre coordinates.
    destination = source_index + flow / cell_size
    base_float = torch.floor(destination)
    base = base_float.to(torch.long)
    fraction = destination - base_float

    numerator = source.new_zeros(batch, channels, npoints)
    coverage = source.new_zeros(batch, 1, npoints)
    dropped_weight = source.new_zeros(batch)
    dropped_feature = source.new_zeros(batch, channels)
    for corner in product((0, 1), repeat=3):
        coordinates = [base[:, axis] + corner[axis] for axis in range(3)]
        valid = torch.ones((batch, npoints), dtype=torch.bool, device=source.device)
        weight = source.new_ones(batch, npoints)
        for axis in range(3):
            valid = valid & (coordinates[axis] >= 0) & (coordinates[axis] < shape[axis])
            f = fraction[:, axis]
            weight = weight * (f if corner[axis] else 1 - f)
        keep = weight * valid.to(source.dtype)
        discard = weight * (~valid).to(source.dtype)
        linear = coordinates[0] * (shape[1] * shape[2]) + coordinates[1] * shape[2] + coordinates[2]
        # Only make invalid scatter addresses safe. Their weights remain ZERO;
        # this does not clamp out-of-grid feature mass onto a border voxel.
        linear = torch.where(valid, linear, torch.zeros_like(linear)).unsqueeze(1)
        numerator.scatter_add_(2, linear.expand(batch, channels, npoints), values * keep.unsqueeze(1))
        coverage.scatter_add_(2, linear, keep.unsqueeze(1))
        dropped_weight = dropped_weight + discard.sum(dim=1)
        dropped_feature = dropped_feature + (values * discard.unsqueeze(1)).sum(dim=2)

    support = coverage > 0
    denominator = torch.where(support, coverage, torch.ones_like(coverage))
    normalized = numerator / denominator
    return {
        'numerator': numerator.reshape((batch, channels) + shape),
        'coverage': coverage.reshape((batch, 1) + shape),
        'normalized': normalized.reshape((batch, channels) + shape),
        'support': support.reshape((batch, 1) + shape),
        'source_weight_mass': source.new_full((batch,), npoints),
        'retained_weight_mass': coverage.sum(dim=(1, 2)),
        'dropped_weight_mass': dropped_weight,
        'source_feature_sum': values.sum(dim=2),
        'retained_feature_sum': numerator.sum(dim=2),
        'dropped_feature_sum': dropped_feature,
    }
