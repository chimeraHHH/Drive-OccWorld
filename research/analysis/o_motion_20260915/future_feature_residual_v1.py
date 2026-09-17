"""Prediction-supported feature transport with dense future-context residuals.

All fields use the original fixed t0 LiDAR output grid. O readout tokens are
Y-major; projected channel z*8+c becomes physical [B,8,X,Y,Z]. O/D, GT policy,
training order and source authentication belong to the caller. This module
reads no labels, boxes, metadata, probabilities or displacement shortcuts.

Only displacement at the splat boundary differs between the matched arms.
The projected source is masked by O's current hard foreground prediction.
Transported features are weighted averages, not occupancy probability mass.
The dense future branch can change destinations with W==0. No old support
influence bound applies to this new residual network.
"""
import math

import torch
from torch import nn

from transport_ops import forward_splat_3d


GRID_SHAPE = (200, 200, 16)
EXTENT_XYZ = (-51.2, -51.2, -5., 51.2, 51.2, 3.)
NORMALIZATION_EPS = 1e-6
PARAMETERS = 41137


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact32(left, right):
    """FP32 value representations, including signed zero."""
    return (left.dtype == right.dtype == torch.float32 and left.shape == right.shape
            and torch.equal(left.contiguous().view(torch.int32),
                            right.contiguous().view(torch.int32)))


def projected_tokens_to_xyz(projected, grid_shape=GRID_SHAPE):
    """[5,B,Y*X,Z*8] -> [B,5,8,X,Y,Z], channel=z*8+c."""
    x, y, z = grid_shape
    require(projected.ndim == 4 and projected.shape[0] == 5
            and projected.shape[2:] == (x*y, z*8), 'Wrong projected Y-major tokens')
    b = projected.shape[1]
    return projected.reshape(5, b, y, x, z, 8).permute(1, 0, 5, 3, 2, 4).contiguous()


def transport_features(source, source_foreground, displacement_m, extent_xyz=EXTENT_XYZ,
                       normalization_eps=NORMALIZATION_EPS):
    """Masked feature/support joint splat; same sources define numerator and W.

    source [B,8,X,Y,Z], mask bool[B,1,X,Y,Z], flow [B,4,3,X,Y,Z].
    At 0<W<eps the fixed denominator floor attenuates the weighted average;
    at W==0 the output is exactly zero. The primitive's unmasked coverage is
    deliberately unused. Signed feature sums are not probability masses.
    """
    require(source.ndim == 5 and source.shape[1] == 8, 'Expected eight source channels')
    b, _, x, y, z = source.shape
    require(source_foreground.shape == (b, 1, x, y, z)
            and source_foreground.dtype == torch.bool, 'Source mask must be a same-grid bool prediction')
    require(displacement_m.shape == (b, 4, 3, x, y, z), 'Wrong source-indexed XYZ displacement shape')
    require(source_foreground.device == source.device and displacement_m.device == source.device
            and displacement_m.dtype == source.dtype, 'Source/flow dtype or device differs')
    require(math.isfinite(normalization_eps) and normalization_eps > 0., 'Positive finite normalization epsilon required')
    require(bool(torch.isfinite(source).all()) and bool(torch.isfinite(displacement_m).all()), 'Nonfinite transport input')
    mask = source_foreground.to(source.dtype)
    payload = torch.cat([source * mask, mask], dim=1)
    normalized, weights = [], []
    for h in range(4):
        splat = forward_splat_3d(payload, displacement_m[:, h], extent_xyz)
        numerator, weight = splat['numerator'][:, :8], splat['numerator'][:, 8:9]
        normalized.append(torch.where(weight > 0, numerator / weight.clamp_min(normalization_eps),
                                      torch.zeros_like(numerator)))
        weights.append(weight)
    transported = torch.stack(normalized, dim=1)
    weight = torch.stack(weights, dim=1)
    source_points = mask.sum(dim=(1, 2, 3, 4))
    retained = weight.sum(dim=(2, 3, 4, 5))
    return dict(transported_features=transported, support_weight=weight,
                source_support_points=source_points,
                retained_support_weight=retained,
                outside_support_weight=source_points[:, None]-retained,
                collision_support_excess=(weight-1).clamp_min(0).sum(dim=(2, 3, 4, 5)))


class FutureFeatureResidual(nn.Module):
    """Shared Linear256->Z*8 and local Conv19->16->1; no normalization layer.

    Inputs: frozen O terminal features [5,B,Y*X,256], O's current hard mask,
    frozen D displacement (or zeros), and frozen O future binary log-odds.
    Decoder channels: future8, transported8, Ologodds1, log1p(W)1, time1.
    Final convolution is zero-initialized. Initially only its gradient can be
    nonzero; projection/first-convolution gradients can emerge after an update.
    """
    def __init__(self, grid_shape=GRID_SHAPE, extent_xyz=EXTENT_XYZ,
                 normalization_eps=NORMALIZATION_EPS):
        super().__init__()
        require(len(grid_shape) == 3 and all(isinstance(n, int) and not isinstance(n, bool) and n > 0
                                            for n in grid_shape), 'Positive integer XYZ shape required')
        require(len(extent_xyz) == 6 and all(math.isfinite(float(v)) for v in extent_xyz)
                and all(extent_xyz[i+3] > extent_xyz[i] for i in range(3)), 'Invalid physical extent')
        require(math.isfinite(normalization_eps) and normalization_eps > 0., 'Invalid epsilon')
        self.grid_shape = tuple(grid_shape)
        self.extent_xyz = tuple(float(v) for v in extent_xyz)
        self.normalization_eps = float(normalization_eps)
        self.projection = nn.Linear(256, self.grid_shape[2]*8)
        self.decoder = nn.Sequential(nn.Conv3d(19, 16, 3, padding=1), nn.ReLU(), nn.Conv3d(16, 1, 1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.register_buffer('nominal_seconds', torch.tensor([.5, 1., 1.5, 2.]))

    def forward(self, readout_features, source_foreground, displacement_m, o_future_logodds):
        x, y, z = self.grid_shape
        require(readout_features.ndim == 4 and readout_features.shape[0] == 5
                and readout_features.shape[2:] == (x*y, 256), 'Expected five native Y-major readout feature frames')
        b = readout_features.shape[1]
        require(b > 0 and o_future_logodds.shape == (b, 4, 1, x, y, z), 'Wrong future margin shape')
        require(readout_features.dtype == o_future_logodds.dtype == self.projection.weight.dtype
                and readout_features.device == o_future_logodds.device == self.projection.weight.device,
                'Feature/model/margin dtype or device differs')
        require(bool(torch.isfinite(readout_features).all()) and bool(torch.isfinite(o_future_logodds).all()),
                'Nonfinite O context or log-odds')
        latent = projected_tokens_to_xyz(self.projection(readout_features), self.grid_shape)
        result = transport_features(latent[:, 0], source_foreground, displacement_m, self.extent_xyz,
                                    self.normalization_eps)
        time = self.nominal_seconds.to(o_future_logodds).view(1, 4, 1, 1, 1, 1).expand(b, 4, 1, x, y, z)
        features = torch.cat([latent[:, 1:], result['transported_features'], o_future_logodds,
                              torch.log1p(result['support_weight']), time], dim=2)
        residual = self.decoder(features.reshape(b*4, 19, x, y, z)).reshape(b, 4, 1, x, y, z)
        require(bool(torch.isfinite(residual).all()), 'Nonfinite signed residual')
        return dict(result, residual=residual)


class _BytePreservingAdd(torch.autograd.Function):
    """Ordinary signed addition with an identity for exact zero, including -0.

    Restoring only the forward representation avoids -0 + +0 -> +0. Backward
    remains the exact derivative of addition; a normal torch.where would kill
    residual gradients at zero initialization and prevent learning.
    """
    @staticmethod
    def forward(ctx, original, residual):
        return torch.where(residual == 0, original, original + residual)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


def compose_prediction(original, base_xyz, residual, oracle):
    """Add residual directly to class1, preserving t0/all other-layer bytes.

    Original native layout and conversions are delegated to the frozen oracle
    helper. No margin reconstruction and no W mask: future context can modify
    uncovered cells. The zero residual identity has a live identity gradient.
    """
    require(base_xyz.ndim == 5 and base_xyz.shape[:2] == (5, 2)
            and residual.shape == (1, 4, 1)+tuple(base_xyz.shape[2:]), 'Wrong native residual composition shape')
    require(original.dtype == base_xyz.dtype == residual.dtype == torch.float32, 'Native composition requires FP32')
    require(bool(torch.isfinite(residual).all()), 'Nonfinite residual')
    logits = base_xyz.clone()
    logits[1:, 1] = _BytePreservingAdd.apply(base_xyz[1:, 1], residual[0, :, 0])
    prediction = oracle.replace_last_logits(original, logits)
    require(exact32(prediction[0], original[0]) and exact32(prediction[:, :-1], original[:, :-1]),
            't0 or intermediate decoder values changed')
    require(exact32(prediction[:, -1, ..., 0], original[:, -1, ..., 0]), 'Background logits changed')
    if not bool(torch.count_nonzero(residual)):
        require(exact32(prediction, original), 'Zero residual failed original O byte identity')
    return prediction


def capture_O_terminal(model, sample, native, *, expected_shape=(5, 1, 40000, 256)):
    """Return (original native prediction, cloned last-layer readout input).

    No hook changes any input/output; the handle is always removed. The caller
    authenticates native sources/weights/boundary before this read-only call.
    A clone prevents the captured view retaining the larger three-layer stack.
    """
    head = model.future_pred_head
    require(not head.soft_weight and len(head.bev_pred_head) == 3, 'Expected three separate native readouts')
    require(not any(getattr(model, k, False) for k in ('turn_on_plan', 'turn_on_flow', 'predict_flow',
                'motion_residual', 'doppler_posterior', 'doppler_advection', 'doppler_flow_loss')),
            'Unsupported planning/flow/M3 branch')
    require(not any(getattr(head, k, False) for k in ('turn_on_flow', 'sem_norm', 'obj_motion_norm')),
            'Unsupported future-head branch')
    require(not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters()),
            'O must already be fully frozen and eval')
    branch = head.bev_pred_head[-1]
    require(not branch._forward_pre_hooks and not branch._forward_hooks, 'Readout already has hooks')
    captured = []
    def capture(_module, inputs):
        require(len(inputs) == 1 and tuple(inputs[0].shape) == tuple(expected_shape)
                and inputs[0].dtype == torch.float32 and bool(torch.isfinite(inputs[0]).all()),
                'Unexpected terminal O feature boundary')
        captured.append(inputs[0].detach().clone())
        return None
    handle = branch.register_forward_pre_hook(capture)
    try:
        original = native.replay(model, sample, training=False)[0]
    finally:
        handle.remove()
    require(len(captured) == 1, 'Terminal O readout was not called exactly once')
    require(original.dtype == torch.float32 and bool(torch.isfinite(original).all()), 'Invalid original O logits')
    return original, captured[0]
