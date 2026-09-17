"""CPU analytical contracts, no data, model checkpoints, optimizer or training.

PyTorch is required: missing PyTorch is a failed prerequisite, never a SKIP
or an alleged backward PASS. The gradient test changes only a tiny local
test instance's last-layer weights to demonstrate connectivity after the
intentional zero-initialization barrier; it performs no optimizer update.
"""

import unittest

import torch

from occupancy_motion_readout_v1 import (
    DEFAULT_PARAMETER_COUNT, OccupancyMotionReadout, counts_to_conv_yx,
    readout_to_xyz, to_native_field,
)


class OccupancyReadoutContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_count_normalization_time_height_channels_and_yx(self):
        shape = (9, 13, 3)
        counts = (torch.arange(2 * 5 * 9 * 13 * 3).reshape(2, 5, 9, 13, 3) % 65).to(torch.uint8)
        original = counts.clone()
        result = counts_to_conv_yx(counts, shape)
        self.assertEqual(tuple(result.shape), (2, 15, 13, 9))
        self.assertEqual(result.dtype, torch.float32)
        for b in range(2):
            for t in range(5):
                for z in range(3):
                    expected = counts[b, t, :, :, z].T.float() / 64
                    self.assertTrue(torch.equal(result[b, t * 3 + z], expected))
        self.assertTrue(torch.equal(counts, original))
        self.assertEqual(float(result.max()), 1.)
        self.assertEqual(float(result.min()), 0.)

    def test_output_channel_horizon_vector_axis_and_height(self):
        raw = torch.arange(2 * 4 * 3 * 5 * 3 * 7, dtype=torch.float32).reshape(2, 60, 3, 7)
        result = readout_to_xyz(raw, 5)
        self.assertEqual(tuple(result.shape), (2, 4, 3, 7, 3, 5))
        for h in range(4):
            for axis in range(3):
                for z in range(5):
                    self.assertTrue(torch.equal(result[:, h, axis, :, :, z],
                                                raw[:, (h * 3 + axis) * 5 + z].transpose(1, 2)))

    def test_formal_parameter_count_and_exact_zero_initialization(self):
        torch.manual_seed(11)
        model = OccupancyMotionReadout()
        actual = sum(p.numel() for p in model.parameters())
        self.assertEqual(actual, 923576)
        self.assertEqual(actual, DEFAULT_PARAMETER_COUNT)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(torch.equal(model.readout.weight, torch.zeros_like(model.readout.weight)))
        self.assertTrue(torch.equal(model.readout.bias, torch.zeros_like(model.readout.bias)))
        counts = torch.randint(0, 65, (1, 5, 128, 128, 10), dtype=torch.uint8)
        with torch.no_grad():
            result = model(counts)
        self.assertEqual(tuple(result.shape), (1, 4, 3, 128, 128, 10))
        self.assertEqual(int(torch.count_nonzero(result)), 0)

    def test_small_odd_grid_zero_head_barrier_then_full_gradient_path(self):
        torch.manual_seed(11)
        model = OccupancyMotionReadout((17, 25, 3))
        counts = torch.randint(0, 65, (1, 5, 17, 25, 3), dtype=torch.uint8)
        output = model(counts)
        self.assertEqual(tuple(output.shape), (1, 4, 3, 17, 25, 3))
        output.sum().backward()
        self.assertGreater(float(model.readout.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.readout.bias.grad.abs().sum()), 0.)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            if not name.startswith('readout.'):
                self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0, name)
        # Controlled local perturbation, not an optimizer step or trained model.
        with torch.no_grad():
            values = torch.linspace(-.02, .03, model.readout.weight.numel())
            model.readout.weight.copy_(values.reshape_as(model.readout.weight))
        model.zero_grad(set_to_none=True)
        field = to_native_field(model(counts), target_shape=(21, 29, 5))
        field.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        for block in list(model.encoder) + list(model.decoder):
            for index in (0, 3):
                self.assertGreater(float(block[index].weight.grad.abs().sum()), 0.)

    def test_constant_fields_preserve_metres_and_interpolation_gradient(self):
        constants = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3, 1, 1, 1) - 7.5
        source = constants.expand(2, 4, 3, 4, 7, 3).clone().requires_grad_()
        result = to_native_field(source, target_shape=(9, 5, 8))
        self.assertEqual(tuple(result.shape), (2, 4, 3, 9, 5, 8))
        torch.testing.assert_close(result, constants.expand_as(result), rtol=0, atol=1e-12)
        result.sum().backward()
        self.assertTrue(bool(torch.isfinite(source.grad).all()))
        # Each output's trilinear weights sum to one, including extended edges.
        torch.testing.assert_close(source.grad.sum(dim=(3, 4, 5)),
                                   torch.full((2, 4, 3), 9 * 5 * 8, dtype=torch.float64),
                                   rtol=0, atol=1e-10)

    def test_xyz_ramps_against_independent_align_false_coordinates(self):
        sx, sy, sz = (4, 7, 3)
        tx, ty, tz = (9, 5, 8)
        source = torch.zeros((1, 4, 3, sx, sy, sz), dtype=torch.float64)
        for h in range(4):
            source[0, h, 0] = h + 2 * torch.arange(sx, dtype=torch.float64)[:, None, None]
            source[0, h, 1] = 10 * h - 3 * torch.arange(sy, dtype=torch.float64)[None, :, None]
            source[0, h, 2] = .7 * h + 5 * torch.arange(sz, dtype=torch.float64)[None, None, :]
        result = to_native_field(source, (tx, ty, tz))
        # Independent closed form; no call to grid_sample/interpolate here.
        x = ((torch.arange(tx, dtype=torch.float64) + .5) * sx / tx - .5).clamp(0, sx - 1)
        y = ((torch.arange(ty, dtype=torch.float64) + .5) * sy / ty - .5).clamp(0, sy - 1)
        z = ((torch.arange(tz, dtype=torch.float64) + .5) * sz / tz - .5).clamp(0, sz - 1)
        for h in range(4):
            expected = torch.stack([
                (h + 2 * x[:, None, None]).expand(tx, ty, tz),
                (10 * h - 3 * y[None, :, None]).expand(tx, ty, tz),
                (.7 * h + 5 * z[None, None, :]).expand(tx, ty, tz),
            ])
            torch.testing.assert_close(result[0, h], expected, rtol=0, atol=1e-12)
        # Upsampling extends endpoint values; metre vectors are never resized in magnitude.
        self.assertAlmostEqual(float(result[0, 0, 0, 0, 0, 0]), 0.)
        self.assertAlmostEqual(float(result[0, 0, 0, -1, 0, 0]), 6.)

    def test_native_default_and_same_grid_identity(self):
        source = torch.zeros(1, 4, 3, 2, 3, 4)
        self.assertIs(to_native_field(source, (2, 3, 4)), source)
        result = to_native_field(source)
        self.assertEqual(tuple(result.shape), (1, 4, 3, 200, 200, 16))
        self.assertEqual(int(torch.count_nonzero(result)), 0)

    def test_reject_wrong_count_dtype_range_shape_and_vector_dtype(self):
        valid = torch.zeros(1, 5, 8, 11, 3, dtype=torch.uint8)
        for wrong in (valid.float(), valid.bool(), valid.to(torch.int64)):
            with self.assertRaises(TypeError):
                counts_to_conv_yx(wrong, (8, 11, 3))
        with self.assertRaises(ValueError):
            counts_to_conv_yx(valid + 65, (8, 11, 3))
        with self.assertRaises(ValueError):
            counts_to_conv_yx(valid, (11, 8, 3))
        with self.assertRaises(ValueError):
            counts_to_conv_yx(valid[:, :4], (8, 11, 3))
        with self.assertRaises(TypeError):
            to_native_field(torch.zeros(1, 4, 3, 2, 3, 4, dtype=torch.uint8))
        with self.assertRaises(ValueError):
            to_native_field(torch.zeros(1, 3, 4, 2, 3, 4))
        with self.assertRaises(ValueError):
            OccupancyMotionReadout((7, 8, 3))


if __name__ == '__main__':
    unittest.main()
