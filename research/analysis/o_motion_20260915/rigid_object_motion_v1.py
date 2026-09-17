"""Differentiable rigid corrections to a fixed, complete CRN-CV source field.

Single-sample API; XYZ metres in t0 LiDAR R. Rotations are active column-vector
rotations. delta_rotvec_R is an axis-angle vector expressed in R and LEFT
multiplies the current box rotation. All static geometry and outputs are
float64; learned residuals are float32 and retain their gradient through casts.
No labels, score filtering, box ownership construction, or model modes/RNG are
handled here. In particular, owner must already index the supplied M objects.
"""
import math

import torch


HORIZONS_SECONDS = (.5, 1., 1.5, 2.)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(value, shape, dtype, name, device=None, *, fixed=False):
    _require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape),
             name + ' shape differs: expected ' + str(tuple(shape)))
    _require(value.dtype == dtype, name + ' dtype differs: expected ' + str(dtype))
    if device is not None:
        _require(value.device == device, name + ' device differs')
    _require(bool(torch.isfinite(value).all()), name + ' contains nonfinite values')
    if fixed:
        _require(not value.requires_grad, name + ' must be fixed geometry')


def so3_exp(rotvec):
    """Return Exp([rotvec]_x), shape [...,3,3], preserving floating dtype.

    Rodrigues coefficients are sinc(theta/pi) and .5*sinc(theta/(2*pi))**2.
    Neither divides by a clamped angle. torch.linalg.vector_norm has a finite
    zero subgradient; torch.sinc implements the removable singularity at zero.
    Thus the first derivative at rotvec=0 is the cross-product matrix, not zero.
    float64 is supported for analytic/gradcheck use; production residuals are
    converted to float64 by the source-field and pose helpers below.
    """
    _require(isinstance(rotvec, torch.Tensor) and rotvec.ndim >= 1
             and rotvec.shape[-1] == 3
             and rotvec.dtype in (torch.float32, torch.float64),
             'rotvec must be float32/float64 [...,3]')
    _require(bool(torch.isfinite(rotvec).all()), 'Nonfinite rotation vector')
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    skew = skew.reshape(rotvec.shape[:-1] + (3, 3))
    theta = torch.linalg.vector_norm(rotvec, dim=-1)
    a = torch.sinc(theta / math.pi)[..., None, None]
    b = (.5 * torch.sinc(theta / (2. * math.pi)).square())[..., None, None]
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + a * skew + b * (skew @ skew)


class _ExactZeroResidual(torch.autograd.Function):
    """Preserve base bytes at zero residual with ordinary additive Jacobian."""
    @staticmethod
    def forward(ctx, base, residual):
        return torch.where(residual == 0, base, base + residual)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


def _objects(c0_R, delta_center_R, delta_rotvec_R):
    _require(isinstance(c0_R, torch.Tensor) and c0_R.ndim == 2
             and c0_R.shape[1] == 3, 'c0_R must be [M,3]')
    count = c0_R.shape[0]
    _tensor(c0_R, (count, 3), torch.float64, 'c0_R', fixed=True)
    for name, value in (('delta_center_R', delta_center_R),
                        ('delta_rotvec_R', delta_rotvec_R)):
        _tensor(value, (4, count, 3), torch.float32, name, c0_R.device)
    return count


def source_displacement_field(points_R, owner, cv_velocity_R, c0_R,
                              delta_center_R, delta_rotvec_R):
    """Return the complete [4,Q,3] forward material displacement in metres.

    points_R [Q,3] and cv_velocity_R [Q,3] are original float64 geometry.
    owner is fixed int64 [Q], either -1 (uncovered) or an index in [0,M).
    cv_velocity_R must be the already verified full CRN-CV field, not a field
    reconstructed from float32 packed states. No labels/support sampling occur.

    For covered point p owned by j: h*v(p) + delta_c[h,j]
      + (Exp(delta_rot[h,j])-I) @ (p-c0[j]).
    Uncovered points always return exact +0, even if their supplied velocity
    is nonzero. Ownership and current geometric centres never move with h.
    The zero-residual path returns original h*v bytes (including signed zero)
    while preserving derivatives of the residual at zero.
    """
    count = _objects(c0_R, delta_center_R, delta_rotvec_R)
    _require(isinstance(points_R, torch.Tensor) and points_R.ndim == 2
             and points_R.shape[1] == 3, 'points_R must be [Q,3]')
    queries, device = points_R.shape[0], c0_R.device
    _tensor(points_R, (queries, 3), torch.float64, 'points_R', device, fixed=True)
    _tensor(cv_velocity_R, (queries, 3), torch.float64, 'cv_velocity_R', device, fixed=True)
    _tensor(owner, (queries,), torch.int64, 'owner', device, fixed=True)
    _require(bool(((owner >= -1) & (owner < count)).all()),
             'owner must be -1 or a mapped object index in [0,M)')
    horizon = points_R.new_tensor(HORIZONS_SECONDS).reshape(4, 1, 1)
    covered = owner >= 0
    base = horizon * cv_velocity_R.unsqueeze(0)
    base = torch.where(covered[None, :, None], base, torch.zeros_like(base))
    indices = covered.nonzero(as_tuple=False).squeeze(-1)
    # Preserve an empty/zero gradient path for empty objects or uncovered fields.
    if indices.numel() == 0:
        empty_residual = (delta_center_R.to(torch.float64).sum()
                          + delta_rotvec_R.to(torch.float64).sum()) * 0.
        return _ExactZeroResidual.apply(base, empty_residual.expand_as(base))
    assigned = owner.index_select(0, indices)
    offset = points_R.index_select(0, indices) - c0_R.index_select(0, assigned)
    rotation = so3_exp(delta_rotvec_R.to(torch.float64))
    delta_c = delta_center_R.to(torch.float64)
    eye = torch.eye(3, dtype=torch.float64, device=device)
    outputs = []
    # One horizon at a time avoids materializing a [4,Q,3,3] rotation tensor.
    for h in range(4):
        r = (rotation[h] - eye).index_select(0, assigned)
        rotational = torch.einsum('qij,qj->qi', r, offset)
        residual = delta_c[h].index_select(0, assigned) + rotational
        corrected = _ExactZeroResidual.apply(base[h].index_select(0, indices), residual)
        outputs.append(base[h].index_copy(0, indices, corrected))
    return torch.stack(outputs)


def future_poses(c0_R, R0, state_velocity_R, delta_center_R, delta_rotvec_R):
    """Return (centres [4,M,3], rotations [4,M,3,3]), both float64.

    c_h = c0 + h*state_velocity_R + delta_center_R;
    R_h = Exp(delta_rotvec_R) @ R0. Column-vector box-local coordinates map
    to R by R_h @ xi + c_h. Size is unchanged and is not part of this helper.
    Coupled/control selection of residual versus zero belongs to the caller;
    this function never detaches the payload or chooses an experimental arm.
    """
    count = _objects(c0_R, delta_center_R, delta_rotvec_R)
    _tensor(R0, (count, 3, 3), torch.float64, 'R0', c0_R.device, fixed=True)
    _tensor(state_velocity_R, (count, 3), torch.float64,
            'state_velocity_R', c0_R.device, fixed=True)
    horizon = c0_R.new_tensor(HORIZONS_SECONDS).reshape(4, 1, 1)
    base = c0_R.unsqueeze(0) + horizon * state_velocity_R.unsqueeze(0)
    centres = _ExactZeroResidual.apply(base, delta_center_R.to(torch.float64))
    rotations = so3_exp(delta_rotvec_R.to(torch.float64)) @ R0.unsqueeze(0)
    return centres, rotations
