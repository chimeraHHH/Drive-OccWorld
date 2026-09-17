"""Small CPU numerical/axis/gradient tests, no model, dataset, GPU or training.

Run: python test_transport_ops.py
The NumPy oracle independently loops source indices and physical centres.
"""
import itertools
import unittest

import numpy as np
import torch

from transport_ops import forward_splat_3d


def numpy_oracle(source, displacement, extent):
    b, c, nx, ny, nz = source.shape
    shape = (nx, ny, nz)
    lower, upper = np.array(extent[:3]), np.array(extent[3:])
    step = (upper - lower) / np.array(shape)
    out = np.zeros_like(source)
    mass = np.zeros((b, 1) + shape, dtype=source.dtype)
    lost = np.zeros(b, dtype=source.dtype)
    lost_feature = np.zeros((b, c), dtype=source.dtype)
    for batch in range(b):
        for index in np.ndindex(shape):
            p = lower + (np.array(index) + .5) * step
            p = p + displacement[(batch, slice(None)) + index]
            f = (p - lower) / step - .5
            base = np.floor(f).astype(int)
            frac = f - base
            for corner in itertools.product((0, 1), repeat=3):
                target = base + np.array(corner)
                w = np.prod([frac[k] if corner[k] else 1 - frac[k] for k in range(3)])
                value = source[(batch, slice(None)) + index]
                if all(0 <= target[k] < shape[k] for k in range(3)):
                    out[(batch, slice(None)) + tuple(target)] += w * value
                    mass[(batch, 0) + tuple(target)] += w
                else:
                    lost[batch] += w
                    lost_feature[batch] += w * value
    return out, mass, lost, lost_feature


class TransportTests(unittest.TestCase):
    def setUp(self):
        # Non-square grid and unequal metric spacings: dx=2,dy=3,dz=4.
        self.shape = (3, 4, 5)
        self.extent = (-2., -6., -10., 4., 6., 10.)
        self.source = torch.arange(120, dtype=torch.float64).reshape(1, 2, *self.shape) / 7 - 4

    def flow(self, xyz):
        return torch.tensor(xyz, dtype=torch.float64).view(1, 3, 1, 1, 1).expand(1, 3, *self.shape).clone()

    def assert_oracle(self, src, flow, extent):
        out = forward_splat_3d(src, flow, extent)
        expected = numpy_oracle(src.detach().numpy(), flow.detach().numpy(), extent)
        for key, value in zip(('numerator', 'coverage', 'dropped_weight_mass', 'dropped_feature_sum'), expected):
            np.testing.assert_allclose(out[key].detach().numpy(), value, rtol=1e-12, atol=1e-12, err_msg=key)
        torch.testing.assert_close(out['retained_weight_mass'] + out['dropped_weight_mass'], out['source_weight_mass'])
        torch.testing.assert_close(out['retained_feature_sum'] + out['dropped_feature_sum'], out['source_feature_sum'])
        expected_normalized = expected[0] / np.where(expected[1] > 0, expected[1], 1)
        np.testing.assert_allclose(out['normalized'].detach().numpy(), expected_normalized, atol=1e-12, rtol=1e-12)
        return out

    def test_zero_identity_default_extent(self):
        for dtype in (torch.float32, torch.float64):
            src = self.source.to(dtype)
            out = forward_splat_3d(src, self.flow((0, 0, 0)).to(dtype))
            self.assertTrue(torch.equal(out['numerator'], src))
            self.assertTrue(torch.equal(out['normalized'], src))
            self.assertTrue(torch.equal(out['coverage'], torch.ones_like(out['coverage'])))
            self.assertTrue(bool(out['support'].all()))
            self.assertEqual(float(out['dropped_weight_mass'].sum()), 0.)

    def test_non_symmetric_integer_xyz_shift(self):
        flow = self.flow((2, -3, 8))  # +1x, -1y, +2z.
        out = self.assert_oracle(self.source, flow, self.extent)
        target = torch.zeros_like(self.source)
        target[:, :, 1:, :3, 2:] = self.source[:, :, :2, 1:, :3]
        self.assertTrue(torch.equal(out['numerator'], target))
        self.assertEqual(float(out['retained_weight_mass']), 18.)

    def test_fractional_translation_all_axes(self):
        self.assert_oracle(self.source, self.flow((.5, 1.2, 2.4)), self.extent)

    def test_heterogeneous_displacements_and_batches(self):
        generator = torch.Generator().manual_seed(11)
        src = torch.cat((self.source, self.source.flip(-1)), dim=0)
        flow = torch.randn((2, 3) + self.shape, generator=generator, dtype=torch.float64) * .7
        self.assert_oracle(src, flow, self.extent)

    def test_collision_holes_weighted_mean(self):
        # Three x centres collapse into x=1. Coverage is 3, not 1.
        flow = self.flow((0, 0, 0))
        for x in range(3):
            flow[:, 0, x] = (1 - x) * 2
        out = self.assert_oracle(self.source, flow, self.extent)
        self.assertTrue(torch.equal(out['coverage'][:, :, 1], torch.full((1, 1, 4, 5), 3., dtype=torch.float64)))
        torch.testing.assert_close(out['normalized'][:, :, 1], self.source.mean(dim=2))
        self.assertFalse(bool(out['support'][:, :, (0, 2)].any()))
        self.assertEqual(float(out['normalized'][:, :, (0, 2)].abs().sum()), 0.)

    def test_boundary_discards_weights_not_clamps_or_renormalizes(self):
        flow = self.flow((0, 0, 0))
        flow[:, 0, -1, -1, -1] = 1  # half x cell escapes from one source.
        out = self.assert_oracle(self.source, flow, self.extent)
        self.assertEqual(float(out['dropped_weight_mass']), .5)
        self.assertEqual(float(out['coverage'][0, 0, -1, -1, -1]), .5)
        torch.testing.assert_close(out['numerator'][0, :, -1, -1, -1], .5 * self.source[0, :, -1, -1, -1])

    def test_all_outside_is_empty(self):
        out = self.assert_oracle(self.source, self.flow((100, 100, 100)), self.extent)
        self.assertFalse(bool(out['support'].any()))
        self.assertEqual(float(out['normalized'].abs().sum()), 0.)
        self.assertEqual(float(out['dropped_weight_mass']), 60.)

    def test_flattened_layout_equivalence(self):
        flow = self.flow((.5, 1.2, 2.4))
        dense = forward_splat_3d(self.source, flow, self.extent)
        flat = forward_splat_3d(self.source.flatten(2), flow.flatten(2), self.extent, grid_shape=self.shape)
        for key in dense:
            self.assertTrue(torch.equal(dense[key], flat[key]), key)

    def test_gradient_analytic_physical_xyz_and_source(self):
        # A single interior signal; weighted destination-coordinate moment
        # equals displacement, and differentiates by exactly [1,2,3].
        src = torch.zeros((1, 1, 3, 4, 5), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            src[0, 0, 1, 1, 2] = 1
        flow = self.flow((.4, .9, 1.6)).requires_grad_()
        result = forward_splat_3d(src, flow, self.extent)
        centres = [torch.arange(n, dtype=torch.float64) * d + lo + d / 2 for n, d, lo in zip(self.shape, (2., 3., 4.), self.extent[:3])]
        xyz = torch.meshgrid(*centres, indexing='ij')
        objective = (result['numerator'] * (xyz[0] + 2 * xyz[1] + 3 * xyz[2])).sum()
        gs, gd = torch.autograd.grad(objective, (src, flow))
        torch.testing.assert_close(gd[0, :, 1, 1, 2], torch.tensor([1., 2., 3.], dtype=torch.float64))
        self.assertTrue(bool(torch.isfinite(gs).all()))
        self.assertGreater(float(gs.abs().sum()), 0.)

    def test_source_and_displacement_gradcheck(self):
        src = torch.linspace(-.2, .7, 12, dtype=torch.float64).reshape(1, 1, 2, 3, 2).requires_grad_()
        flow = torch.tensor([.23, .31, .47], dtype=torch.float64).view(1, 3, 1, 1, 1).expand(1, 3, 2, 3, 2).clone().requires_grad_()
        extent = (0., 0., 0., 2., 3., 2.)
        self.assertTrue(torch.autograd.gradcheck(lambda s, d: forward_splat_3d(s, d, extent)['numerator'], (src, flow), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_invalid_inputs_fail(self):
        flow = self.flow((0, 0, 0))
        with self.assertRaises(ValueError):
            forward_splat_3d(self.source.flatten(2), flow.flatten(2))
        with self.assertRaises(ValueError):
            forward_splat_3d(self.source, flow, (0, 0, 0, 0, 1, 1))
        with self.assertRaises(ValueError):
            forward_splat_3d(self.source, flow.float())
        with self.assertRaises(ValueError):
            forward_splat_3d(self.source, flow.permute(0, 2, 3, 4, 1))
        flow[0, 0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            forward_splat_3d(self.source, flow)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
