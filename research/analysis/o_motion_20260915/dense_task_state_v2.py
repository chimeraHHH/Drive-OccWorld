"""Coverage-neutral revision of the own-state engineering prototype.

Both arms decode content only. Geometric coverage still normalizes splat
features, but is not an additional learned decoder input. This removes the
v1 confound where occupancy-to-velocity gradients could use density alone.
The content-bypass and source-neighbour interaction limitations remain.
"""
import torch
from torch import nn
import torch.nn.functional as F
from dense_task_state_v1 import DenseTaskState as InitialState
from dense_task_state_v1 import geometry_checks as base_geometry_checks


class DenseTaskState(InitialState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = nn.Sequential(nn.Conv3d(self.channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv3d(16, 2, 1))

    def decode(self, feature_xyz, coverage):
        return self.decoder(feature_xyz)


def geometry_checks(device='cpu'):
    result = base_geometry_checks(device)
    model = DenseTaskState(shape=(7, 9, 4), extent=(-3.5, -4.5, -2., 3.5, 4.5, 2.)).to(device)
    tokens = torch.randn(1, 63, 256, device=device)
    a, b = model(tokens, 'transport'), model(tokens, 'direct')
    assert torch.allclose(a['logits'], b['logits'], atol=2e-6, rtol=2e-6)
    ga = torch.autograd.grad(a['logits'][:, 1:, 1].square().mean(), model.velocity.weight,
                             allow_unused=True)[0]
    gb = torch.autograd.grad(b['logits'][:, 1:, 1].square().mean(), model.velocity.weight,
                             allow_unused=True)[0]
    assert ga is not None and torch.isfinite(ga).all() and ga.norm() > 0 and gb is None
    state = torch.randn(1, 8, 7, 9, 4, device=device)
    c1 = torch.ones_like(state[:, :1]); c2 = torch.randn_like(c1).abs() * 10
    assert torch.equal(model.decode(state, c1), model.decode(state, c2))
    # Asymmetric impulse verifies XYZ resampling addresses rather than hiding
    # an XY transpose on a square grid. Labels still use the legacy block mode.
    field = torch.zeros(1, 1, 200, 200, 16, device=device)
    field[0, 0, 37, 113, 5] = 1.
    locations = []
    for shape in [(256, 256, 20), (512, 512, 40)]:
        up = F.interpolate(field, size=shape, mode='trilinear', align_corners=False)
        flat = int(up.reshape(-1).argmax()); z = flat % shape[2]
        y = (flat // shape[2]) % shape[1]; x = flat // (shape[1]*shape[2])
        expected = [(i+.5)*n/m-.5 for i, n, m in zip([37, 113, 5], shape, [200, 200, 16])]
        assert all(abs(v-e) <= 1.01 for v, e in zip([x, y, z], expected))
        locations.append(dict(shape=shape, peak_xyz=[x, y, z], expected_continuous_xyz=expected))
    result.update(coverage_neutral_decode=True, asymmetric_resampling=locations,
        content_transport_velocity_gradient=float(ga.norm()), identity_velocity_gradient=None)
    return result
