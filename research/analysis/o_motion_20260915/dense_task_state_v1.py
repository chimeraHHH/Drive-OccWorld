"""Own-state occupancy prototype, with an exactly parameter-matched direct arm.

Only an authenticated observed BEV tensor enters forward(). No O predictions,
boxes, masks, future features, targets, or future ego poses enter this module.
State content and cumulative displacement are indexed by ORIGINAL source
voxels in the fixed t0 LiDAR frame. This is not an Eulerian velocity field.

The dynamics mixes source neighbours, not future spatial neighbours. It can
change content and thus bypass geometric transport internally; intervention
tests and the direct-content competitor are required, not optional evidence.
"""
import torch
from torch import nn
import torch.nn.functional as F

from transport_ops import forward_splat_3d


class ContextBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation,
                      dilation=dilation, groups=channels),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x + self.net(x)


class DenseTaskState(nn.Module):
    """Binary occupancy [B,5,2,X,Y,Z], displacement [B,4,3,X,Y,Z]."""
    def __init__(self, shape=(200, 200, 16), extent=(-51.2, -51.2, -5., 51.2, 51.2, 3.),
                 channels=8, width=64):
        super().__init__()
        self.shape, self.extent, self.channels = tuple(shape), tuple(extent), channels
        packed = shape[2] * channels
        self.encoder = nn.Sequential(nn.Conv2d(256, packed, 1),
            nn.GroupNorm(8, packed), nn.SiLU(), ContextBlock(packed, 1))
        self.dynamics = nn.Sequential(nn.Conv2d(packed * 2 + 1, width, 1),
            nn.GroupNorm(8, width), nn.SiLU(),
            *[ContextBlock(width, d) for d in (1, 2, 4, 8, 16)])
        self.content_increment = nn.Conv2d(width, packed, 1)
        self.velocity = nn.Conv2d(width, 3 * shape[2], 1)
        self.decoder = nn.Sequential(nn.Conv3d(channels + 1, 16, 3, padding=1),
            nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv3d(16, 2, 1))
        # Both arms start at persistence after the shared current-shape fit.
        nn.init.zeros_(self.content_increment.weight)
        nn.init.zeros_(self.content_increment.bias)
        nn.init.zeros_(self.velocity.weight)
        nn.init.zeros_(self.velocity.bias)

    def to_xyz(self, packed, channels):
        b, _, y, x = packed.shape
        return packed.reshape(b, channels, self.shape[2], y, x).permute(0, 1, 4, 3, 2).contiguous()

    def encode(self, tokens):
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (self.shape[0] * self.shape[1], 256):
            raise ValueError('Observed tokens must be B,Y*X,256 with native y-major flattening')
        b = tokens.shape[0]
        return self.encoder(tokens.transpose(1, 2).reshape(b, 256, self.shape[1], self.shape[0]))

    def decode(self, feature_xyz, coverage):
        # Conv3d treats the three axes symmetrically; retain explicit XYZ order.
        return self.decoder(torch.cat((feature_xyz, torch.log1p(coverage)), dim=1))

    def current(self, tokens):
        state = self.encode(tokens)
        xyz = self.to_xyz(state, self.channels)
        return self.decode(xyz, torch.ones_like(xyz[:, :1]))

    def forward(self, tokens, arm='transport', intervention=None):
        if arm not in ('transport', 'direct') or intervention not in (None, 'zero', 'reverse'):
            raise ValueError('Unknown arm or rendering intervention')
        origin = state = self.encode(tokens)
        xyz = self.to_xyz(state, self.channels)
        logits = [self.decode(xyz, torch.ones_like(xyz[:, :1]))]
        displacement = xyz.new_zeros(xyz.shape[0], 3, *self.shape)
        flows, coverages = [], []
        for h in range(4):
            time = state.new_full((state.shape[0], 1, *state.shape[2:]), (h + 1) / 4.)
            hidden = self.dynamics(torch.cat((origin, state, time), dim=1))
            state = state + self.content_increment(hidden)
            displacement = displacement + .5 * self.to_xyz(self.velocity(hidden), 3)
            xyz = self.to_xyz(state, self.channels)
            flows.append(displacement)
            # No motion feedback into content: the unique velocity head has
            # occupancy gradient only through the transport address/coverage.
            if arm == 'transport' and intervention != 'zero':
                address = -displacement if intervention == 'reverse' else displacement
                splat = forward_splat_3d(xyz, address, extent=self.extent)
                feature, coverage = splat['normalized'], splat['coverage']
            else:
                feature, coverage = xyz, torch.ones_like(xyz[:, :1])
            logits.append(self.decode(feature, coverage))
            coverages.append(coverage)
        return dict(logits=torch.stack(logits, dim=1),
                    displacement=torch.stack(flows, dim=1),
                    coverage=torch.stack(coverages, dim=1))


def geometry_checks(device='cpu'):
    """Independent landmark, outflow, gradient, and source-layout assertions."""
    shape, extent = (7, 9, 4), (-3.5, -4.5, -2., 3.5, 4.5, 2.)
    source = torch.zeros(1, 1, *shape, device=device, dtype=torch.float64)
    source[0, 0, 2, 3, 1] = 1
    flow = torch.zeros(1, 3, *shape, device=device, dtype=torch.float64)
    assert torch.equal(forward_splat_3d(source, flow, extent)['normalized'], source)
    for axis in range(3):
        shifted = flow.clone(); shifted[:, axis] = 1
        result = forward_splat_3d(source, shifted, extent)
        expected = source.clone().zero_(); address = [2, 3, 1]; address[axis] += 1
        expected[(0, 0, *address)] = 1
        assert torch.equal(result['numerator'], expected)
        assert torch.allclose(result['source_weight_mass'],
                              result['retained_weight_mass'] + result['dropped_weight_mass'])
    half = flow.clone(); half[:, 0] = .25; half.requires_grad_()
    rendered = forward_splat_3d(source, half, extent)['numerator']
    assert rendered[0, 0, 2, 3, 1] == .75 and rendered[0, 0, 3, 3, 1] == .25
    grad = torch.autograd.grad(rendered[0, 0, 3, 3, 1], half)[0]
    assert abs(float(grad[0, 0, 2, 3, 1]) - 1.) < 1e-12
    model = DenseTaskState(shape=shape, extent=extent).to(device)
    packed = torch.arange(3 * 4 * 9 * 7, device=device).reshape(1, 12, 9, 7)
    physical = model.to_xyz(packed, 3)
    for c, x, y, z in [(2, 5, 7, 3), (0, 1, 2, 1)]:
        assert physical[0, c, x, y, z] == packed[0, c * 4 + z, y, x]
    tokens = torch.randn(1, 7 * 9, 256, device=device)
    a, b = model(tokens, 'transport'), model(tokens, 'direct')
    assert torch.equal(a['displacement'], b['displacement'])
    assert torch.allclose(a['logits'], b['logits'], atol=2e-6, rtol=2e-6)
    ga = torch.autograd.grad(a['logits'][:, 1:, 1].square().mean(),
                             model.velocity.weight, allow_unused=True)[0]
    gb = torch.autograd.grad(b['logits'][:, 1:, 1].square().mean(),
                             model.velocity.weight, allow_unused=True)[0]
    assert ga is not None and torch.isfinite(ga).all() and ga.norm() > 0
    assert gb is None
    return dict(identity=True, xyz_landmarks=True, fractional_weights=True,
                outflow_accounting=True, displacement_gradient=True,
                native_to_xyz_layout=True, initial_pair_parity=True,
                transport_velocity_gradient=float(ga.norm()), direct_velocity_gradient=None)
