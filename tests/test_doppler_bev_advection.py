import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] /
    'projects/mmdet3d_plugin/bevformer/detectors/doppler_bev_advection.py')
SPEC = importlib.util.spec_from_file_location('doppler_bev_advection', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DopplerBEVAdvection = MODULE.DopplerBEVAdvection


def identity_grid(batch_size, height, width):
    ys = (2.0 * (torch.arange(height).float() + 0.5) / height) - 1.0
    xs = (2.0 * (torch.arange(width).float() + 0.5) / width) - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs)
    return torch.stack((grid_x, grid_y), dim=-1).reshape(
        1, height * width, 2).repeat(batch_size, 1, 1)


class DopplerBEVAdvectionTest(unittest.TestCase):

    def make_module(self, gate_init=0.0):
        return DopplerBEVAdvection(
            point_cloud_range=[0, 0, -1, 5, 5, 1],
            bev_h=5,
            bev_w=5,
            embed_dims=1,
            radar_velocity_norm=1.0,
            time_step=1.0,
            min_dynamic_speed=0.1,
            max_dynamic_speed=10.0,
            gate_init=gate_init)

    def test_zero_gate_preserves_baseline(self):
        module = self.make_module(gate_init=0.0)
        features = torch.randn(1, 25, 1)
        radar = torch.zeros(1, 8, 5, 5)
        radar[:, 0, 2, 2] = 1
        radar[:, 4, 2, 2] = 1
        output = module(features, radar, 1, identity_grid(1, 5, 5))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_positive_x_velocity_moves_prior_right(self):
        module = self.make_module(gate_init=10.0)
        features = torch.zeros(1, 25, 1)
        features[0, 2 * 5 + 2, 0] = 1
        radar = torch.zeros(1, 8, 5, 5)
        radar[:, 0, 2, 2] = 1
        radar[:, 4, 2, 2] = 1
        output = module(features, radar, 1, identity_grid(1, 5, 5))
        output_map = output.view(1, 5, 5, 1)[0, :, :, 0]
        peak_y, peak_x = torch.nonzero(
            output_map == output_map.max(), as_tuple=True)
        self.assertEqual((int(peak_y[0]), int(peak_x[0])), (2, 3))
        self.assertGreater(float(output_map[2, 3]), 0.99)

    def test_no_dynamic_returns_produce_zero_prior(self):
        module = self.make_module(gate_init=10.0)
        output = module(
            torch.randn(2, 25, 1),
            torch.zeros(2, 8, 5, 5),
            2,
            identity_grid(2, 5, 5))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_zero_gate_receives_gradient(self):
        module = self.make_module(gate_init=0.0)
        features = torch.ones(1, 25, 1, requires_grad=True)
        radar = torch.zeros(1, 8, 5, 5)
        radar[:, 0, 2, 2] = 1
        radar[:, 4, 2, 2] = 1
        module(features, radar, 1, identity_grid(1, 5, 5)).sum().backward()
        self.assertIsNotNone(module.prior_gate.grad)
        self.assertGreater(float(module.prior_gate.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
