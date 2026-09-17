"""Prediction-supported source transport and a shared local occupancy gate.

No labels, boxes or model metadata enter this module. The caller supplies O's
current probabilities/argmax mask, predicted source displacements and O's
future binary log-odds, all in the same fixed t0 LiDAR XYZ grid. This is a new
candidate component, not the dense-source oracle diagnostic.
"""
import torch
from torch import nn

from transport_ops import forward_splat_3d


def supported_transport(source_probability, source_foreground, displacement_m,
                        extent_xyz):
    """Splat foreground probability mass and source support together.

    Inputs: probability/mask [B,1,X,Y,Z], displacement [B,H,3,X,Y,Z].
    Returns [B,H,1,X,Y,Z] probability mass and support weight. Collisions are
    summed; out-of-grid mass is discarded. Neither output is normalized by W.
    The mask must be O's own current argmax, never a GT motion-support mask.
    """
    if source_probability.ndim != 5 or source_probability.shape[1] != 1:
        raise ValueError('Source probability must be [B,1,X,Y,Z]')
    if source_foreground.shape != source_probability.shape or source_foreground.dtype != torch.bool:
        raise ValueError('Source foreground must be a same-shape boolean prediction')
    if source_probability.device != source_foreground.device:
        raise ValueError('Source probability and prediction mask must share a device')
    b, _, x, y, z = source_probability.shape
    if (displacement_m.ndim != 6 or displacement_m.shape[0] != b
            or tuple(displacement_m.shape[2:]) != (3, x, y, z)):
        raise ValueError('Source displacement must be [B,H,3,X,Y,Z]')
    if displacement_m.shape[1] != 4:
        raise ValueError('Four future keyframes are required')
    if not torch.isfinite(source_probability).all() or not torch.isfinite(displacement_m).all():
        raise ValueError('Nonfinite probability or displacement')
    if (source_probability < 0).any() or (source_probability > 1).any():
        raise ValueError('Source probability must be in [0,1]')
    mask = source_foreground.to(source_probability.dtype)
    source = torch.cat([mask * source_probability, mask], dim=1)
    masses, supports = [], []
    for h in range(4):
        result = forward_splat_3d(source, displacement_m[:, h], extent_xyz)
        masses.append(result['numerator'][:, :1])
        supports.append(result['numerator'][:, 1:2])
    return dict(probability_mass=torch.stack(masses, dim=1),
                support_weight=torch.stack(supports, dim=1),
                source_probability_mass=(mask * source_probability).sum(dim=(2, 3, 4)),
                source_foreground_voxels=mask.sum(dim=(2, 3, 4)))


class SupportedMotionFusion(nn.Module):
    """Shared 4->16->1 pointwise gate; no displacement vector as a shortcut.

    Inputs are O future log-odds, transported log-odds, log(1+W), and nominal
    time. Initial gate=.5. At W==0 the output equals O exactly. t0 is not an
    input/output of this module and must be retained verbatim by its caller.
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv3d(4, 16, 1), nn.ReLU(), nn.Conv3d(16, 1, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.register_buffer('nominal_seconds', torch.tensor([.5, 1., 1.5, 2.]))

    def forward(self, o_future_logodds, probability_mass, support_weight,
                *, fixed_gate=None, disabled=False):
        shape = o_future_logodds.shape
        if len(shape) != 6 or shape[1:3] != (4, 1):
            raise ValueError('Future log-odds must be [B,4,1,X,Y,Z]')
        if probability_mass.shape != shape or support_weight.shape != shape:
            raise ValueError('Future fields must use the same XYZ grid')
        if any(not torch.isfinite(t).all() for t in (o_future_logodds, probability_mass, support_weight)):
            raise ValueError('Nonfinite fusion input')
        if (probability_mass < 0).any() or (support_weight < 0).any():
            raise ValueError('Transport mass/support cannot be negative')
        p = probability_mass.clamp(1e-6, 1.-1e-6)
        transported = torch.log(p) - torch.log1p(-p)
        b, h, _, x, y, z = shape
        time = self.nominal_seconds.to(o_future_logodds).view(1, h, 1, 1, 1, 1).expand(b, h, 1, x, y, z)
        if fixed_gate is None:
            features = torch.cat([o_future_logodds, transported, torch.log1p(support_weight), time], dim=2)
            gate = torch.sigmoid(self.gate(features.reshape(b*h, 4, x, y, z))).reshape(shape)
        else:
            if fixed_gate.shape != shape or not torch.isfinite(fixed_gate).all():
                raise ValueError('Fixed gate must be finite and match the future fields')
            if (fixed_gate < 0).any() or (fixed_gate > 1).any():
                raise ValueError('Fixed gate outside [0,1]')
            gate = fixed_gate
        support = support_weight > 0
        if disabled:
            output = o_future_logodds
        else:
            mixed = (1.-gate)*o_future_logodds + gate*transported
            output = torch.where(support, mixed, o_future_logodds)
        return dict(logodds=output, gate=gate, destination_support=support,
                    transported_logodds=transported)
