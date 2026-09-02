import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] /
    'projects/mmdet3d_plugin/bevformer/detectors/'
    'doppler_radial_flow_loss.py')
SPEC = importlib.util.spec_from_file_location(
    'doppler_radial_flow_loss', str(MODULE_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DopplerRadialFlowLoss = MODULE.DopplerRadialFlowLoss


class DopplerRadialFlowLossTest(unittest.TestCase):
    def make_loss(self, **overrides):
        kwargs = dict(
            point_cloud_range=(-1, -1, -1, 1, 1, 1),
            bev_h=2,
            bev_w=2,
            radar_velocity_norm=1.0,
            radar_height_norm=1.0,
            time_lag_channel=None,
            flow_cell_size=(1.0, 1.0),
            time_step=1.0,
            backward_flow=True,
            loss_weight=1.0)
        kwargs.update(overrides)
        return DopplerRadialFlowLoss(**kwargs)

    @staticmethod
    def radar(radial_velocity=0.0, direction=(1.0, 0.0), present=True):
        radar = torch.zeros(1, 11, 2, 2)
        radar[:, 0, 0, 0] = float(present)
        radar[:, 2, 0, 0] = -0.5
        radar[:, 8, 0, 0] = radial_velocity
        radar[:, 9, 0, 0] = direction[0]
        radar[:, 10, 0, 0] = direction[1]
        return radar

    @staticmethod
    def flow(x=0.0, y=0.0, requires_grad=False):
        flow = torch.zeros(2, 1, 4, 2, 3)
        # Radar height -0.5 selects depth bin zero at the occupied cell.
        flow[:, 0, 0, 0, 0] = x
        flow[:, 0, 0, 0, 1] = y
        return flow.requires_grad_(requires_grad)

    def test_backward_flow_is_converted_to_forward_velocity(self):
        result = self.make_loss()(
            self.flow(x=-2.0), self.radar(radial_velocity=2.0))
        self.assertEqual(result['loss_doppler_radial'].item(), 0.0)

    def test_tangential_flow_is_not_supervised(self):
        result = self.make_loss()(
            self.flow(x=0.0, y=17.0), self.radar(radial_velocity=0.0))
        self.assertEqual(result['loss_doppler_radial'].item(), 0.0)

    def test_radial_l1_value_and_gradient(self):
        flow = self.flow(x=0.0, requires_grad=True)
        result = self.make_loss()(
            flow, self.radar(radial_velocity=2.0))
        self.assertAlmostEqual(
            result['loss_doppler_radial'].item(), 2.0, places=6)
        result['loss_doppler_radial'].backward()
        self.assertTrue(torch.isfinite(flow.grad).all())
        self.assertNotEqual(flow.grad[:, 0, 0, 0, 0].abs().sum().item(), 0.0)

    def test_empty_radar_returns_connected_zero(self):
        flow = self.flow(x=-2.0, requires_grad=True)
        result = self.make_loss()(
            flow, self.radar(radial_velocity=2.0, present=False))
        self.assertEqual(result['loss_doppler_radial'].item(), 0.0)
        result['loss_doppler_radial'].backward()
        self.assertEqual(flow.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
