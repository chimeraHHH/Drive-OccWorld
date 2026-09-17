"""CPU-only mechanical tests, including exactly ONE synthetic SGD update.

The toy update verifies optimization reachability, not research training or
motion/occupancy performance. No dataset, checkpoint, GPU, or label loader.
Run: CUDA_VISIBLE_DEVICES='' python test_motion_prediction_head.py
"""

import unittest

import torch
from torch import nn

from motion_prediction_head import (
    HORIZONS_SECONDS, MotionPredictionHead,
    native_tokens_to_conv_bev, readout_to_source_xyz,
)


class MotionHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.shape = (5, 3, 16)  # deliberately X != Y and all 16 heights.

    def test_native_token_y_major_x_minor_layout(self):
        nx, ny, nz = self.shape
        tokens = torch.arange(nx * ny, dtype=torch.float64).view(1, nx * ny, 1)
        tokens = tokens + 1000 * torch.arange(256, dtype=torch.float64).view(1, 1, 256)
        before = tokens.clone()
        bev = native_tokens_to_conv_bev(tokens, self.shape)
        self.assertEqual(tuple(bev.shape), (1, 256, ny, nx))
        for y in range(ny):
            for x in range(nx):
                for channel in (0, 1, 17, 255):
                    self.assertEqual(float(bev[0, channel, y, x]), 1000 * channel + y * nx + x)
        self.assertTrue(torch.equal(tokens, before))

    def test_all_horizons_axes_heights_and_spatial_readout(self):
        nx, ny, nz = self.shape
        raw = torch.empty(2, 4 * 3 * nz, ny, nx, dtype=torch.float64)
        for b in range(2):
            for channel in range(4 * 3 * nz):
                for y in range(ny):
                    for x in range(nx):
                        raw[b, channel, y, x] = b * 1e7 + channel * 10000 + y * 100 + x
        before = raw.clone()
        out = readout_to_source_xyz(raw, nz)
        self.assertEqual(tuple(out.shape), (2, 4, 3, nx, ny, nz))
        for b in range(2):
            for h in range(4):
                for axis in range(3):
                    for z in range(nz):
                        for x, y in ((0, 0), (4, 1), (1, 2)):
                            channel = (h * 3 + axis) * nz + z
                            self.assertEqual(float(out[b, h, axis, x, y, z]), b * 1e7 + channel * 10000 + y * 100 + x)
        self.assertTrue(torch.equal(raw, before))

    def test_exact_architecture_and_zero_initial_output(self):
        model = MotionPredictionHead(self.shape)
        self.assertEqual(model.horizons_seconds, HORIZONS_SECONDS)
        self.assertEqual(len(model.trunk), 6)
        for index, in_channels in ((0, 256), (3, 128)):
            conv = model.trunk[index]
            self.assertIsInstance(conv, nn.Conv2d)
            self.assertEqual((conv.in_channels, conv.out_channels, conv.kernel_size, conv.padding), (in_channels, 128, (3, 3), (1, 1)))
            self.assertIsInstance(model.trunk[index + 1], nn.GroupNorm)
            self.assertEqual(model.trunk[index + 1].num_groups, 8)
            self.assertEqual(model.trunk[index + 1].num_channels, 128)
            self.assertIsInstance(model.trunk[index + 2], nn.ReLU)
            self.assertFalse(model.trunk[index + 2].inplace)
        self.assertEqual((model.readout.in_channels, model.readout.out_channels, model.readout.kernel_size), (128, 192, (1, 1)))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 467904)
        self.assertEqual(int(torch.count_nonzero(model.readout.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(model.readout.bias)), 0)
        tokens = torch.randn(2, 15, 256)
        before = tokens.clone()
        for training in (False, True):
            model.train(training)
            out = model(tokens)
            self.assertEqual(tuple(out.shape), (2, 4, 3, 5, 3, 16))
            self.assertEqual(int(torch.count_nonzero(out)), 0)
        self.assertTrue(torch.equal(tokens, before))

    def test_unbounded_independent_horizon_axis_height_outputs(self):
        model = MotionPredictionHead(self.shape)
        # This is a readout packing test, not learned or label-derived motion.
        values = torch.linspace(-150., 210., 192)
        with torch.no_grad():
            model.readout.bias.copy_(values)
        out = model(torch.zeros(1, 15, 256))
        for h in range(4):
            for axis in range(3):
                for z in range(16):
                    expected = values[(h * 3 + axis) * 16 + z]
                    self.assertTrue(torch.equal(out[0, h, axis, :, :, z], expected.expand(5, 3)))
        self.assertLess(float(out.min()), -100.)
        self.assertGreater(float(out.max()), 100.)

    def test_nonzero_last_layer_grad_and_one_toy_sgd_update(self):
        model = MotionPredictionHead(self.shape).train()
        tokens = torch.randn(2, 15, 256)
        original_tokens = tokens.clone()
        # Pure synthetic target; not an observation or a GT from this project.
        target = torch.linspace(-.3, .8, 4 * 3 * 16).reshape(1, 4, 3, 1, 1, 16)
        target = target.expand(2, 4, 3, 5, 3, 16)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = model(tokens)
        loss_before = (before - target).square().mean()
        loss_before.backward()
        self.assertGreater(float(model.readout.weight.grad.norm()), 0.)
        self.assertGreater(float(model.readout.bias.grad.norm()), 0.)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertEqual(float(model.trunk[0].weight.grad.norm()), 0.)
        optimizer.step()  # Exactly one SGD step; weights are never saved.
        optimizer.zero_grad(set_to_none=True)
        after = model(tokens)
        loss_after = (after - target).square().mean()
        self.assertFalse(torch.equal(before, after))
        self.assertLess(float(loss_after), float(loss_before))
        # Backward only (no second update): first update opened trunk gradients.
        loss_after.backward()
        self.assertGreater(float(model.trunk[0].weight.grad.norm()), 0.)
        self.assertTrue(torch.equal(tokens, original_tokens))

    def test_noncontiguous_native_tokens_and_grad_preservation(self):
        tokens = torch.randn(1, 256, 15, dtype=torch.float64).transpose(1, 2).requires_grad_()
        self.assertFalse(tokens.is_contiguous())
        bev = native_tokens_to_conv_bev(tokens, self.shape)
        bev.sum().backward()
        self.assertTrue(torch.equal(tokens.grad, torch.ones_like(tokens)))

    def test_default_shape_metadata_and_rejected_layouts(self):
        model = MotionPredictionHead()
        self.assertEqual(model.grid_shape, (200, 200, 16))
        with torch.no_grad():
            out = model.eval()(torch.zeros(1, 40000, 256))
        self.assertEqual(tuple(out.shape), (1, 4, 3, 200, 200, 16))
        self.assertEqual(int(torch.count_nonzero(out)), 0)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 15, 256))
        with self.assertRaises(ValueError):
            MotionPredictionHead((5, 3, 0))
        with self.assertRaises(ValueError):
            native_tokens_to_conv_bev(torch.zeros(1, 15, 128), self.shape)
        with self.assertRaises(TypeError):
            native_tokens_to_conv_bev(torch.zeros(1, 15, 256, dtype=torch.int64), self.shape)
        with self.assertRaises(ValueError):
            readout_to_source_xyz(torch.zeros(1, 191, 3, 5), 16)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
