"""Minimal CPU-only analytic checks; no data, models, optimizer, or CUDA use.

Run with a Python environment containing Torch:
  python test_rigid_object_motion_v1.py
"""
import math
import unittest

import torch

from rigid_object_motion_v1 import (
    HORIZONS_SECONDS, future_poses, so3_exp, source_displacement_field,
)


def byte_equal(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().view(torch.uint8),
                            b.contiguous().view(torch.uint8)))


class RigidObjectMotionTests(unittest.TestCase):
    def test_so3_zero_jacobian_and_gradcheck(self):
        v = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        r = so3_exp(v)
        self.assertTrue(torch.equal(r, torch.eye(3, dtype=torch.float64)))
        # d(Exp(w)e_x)_y / dw_z = +1 at zero: active column rotation.
        gradient, = torch.autograd.grad(r[1, 0], v)
        self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient, torch.tensor([0., 0., 1.], dtype=torch.float64),
                                   rtol=0., atol=0.)
        self.assertTrue(torch.autograd.gradcheck(so3_exp, (v,), eps=1e-6,
                                                atol=1e-6, rtol=1e-5))
        small = torch.tensor([1e-9, -2e-9, 3e-9], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(so3_exp, (small,), eps=1e-6,
                                                atol=1e-6, rtol=1e-5))

    def test_zero_residual_cv_bytes_and_live_source_gradient(self):
        p = torch.tensor([[1., 0., 0.], [0., 1., 2.], [7., 8., 9.]], dtype=torch.float64)
        owner = torch.tensor([0, 0, -1], dtype=torch.int64)
        velocity = torch.tensor([[-0., .2, .3], [1., -0., -3.], [3., 4., 5.]], dtype=torch.float64)
        centres = torch.zeros((1, 3), dtype=torch.float64)
        dc = torch.zeros((4, 1, 3), dtype=torch.float32, requires_grad=True)
        dr = torch.zeros_like(dc, requires_grad=True)
        result = source_displacement_field(p, owner, velocity, centres, dc, dr)
        expected = torch.tensor(HORIZONS_SECONDS, dtype=torch.float64)[:, None, None] * velocity[None]
        expected[:, 2] = 0.
        self.assertTrue(byte_equal(result, expected))
        gc, gr = torch.autograd.grad(result[0, 0, 1], (dc, dr))
        self.assertTrue(torch.isfinite(gc).all() and torch.isfinite(gr).all())
        self.assertEqual(float(gc[0, 0, 1]), 1.)
        self.assertEqual(float(gr[0, 0, 2]), 1.)

    def test_known_3d_rotations_translation_and_axes(self):
        # Three independent objects rotate about R x/y/z respectively.
        centres = torch.tensor([[2., 3., 4.], [-1., 2., 3.], [4., -2., 1.]], dtype=torch.float64)
        offset = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], dtype=torch.float64)
        p = centres + offset
        owner = torch.arange(3, dtype=torch.int64)
        vel = torch.tensor([[.1, .2, .3], [1., -2., 3.], [-.5, .4, .2]], dtype=torch.float64)
        dc = torch.tensor([.4, -.2, .7], dtype=torch.float32).expand(4, 3, 3).clone()
        dr = (torch.eye(3, dtype=torch.float32) * (math.pi / 2.)).expand(4, 3, 3).clone()
        result = source_displacement_field(p, owner, vel, centres, dc, dr)
        moved_offset = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
        expected = (torch.tensor(HORIZONS_SECONDS, dtype=torch.float64)[:, None, None] * vel[None]
                    + dc.double() + moved_offset[None] - offset[None])
        torch.testing.assert_close(result, expected, rtol=0., atol=1e-7)

    def test_future_pose_left_composition_matches_material_field(self):
        c0 = torch.tensor([[3., -2., 5.]], dtype=torch.float64)
        # Noncommuting initial R0: left-R and local/right-R must differ.
        r0 = so3_exp(torch.tensor([[math.pi / 2., 0., 0.]], dtype=torch.float64))
        xi = torch.tensor([.2, .7, -1.], dtype=torch.float64)
        p = (r0[0] @ xi + c0[0])[None]
        v = torch.tensor([[.3, -1., .6]], dtype=torch.float64)
        dc = torch.tensor([.1, -.2, .3], dtype=torch.float32).expand(4, 1, 3).clone().requires_grad_()
        dr = torch.tensor([0., 0., .4], dtype=torch.float32).expand(4, 1, 3).clone().requires_grad_()
        ch, rh = future_poses(c0, r0, v, dc, dr)
        direct = (rh[:, 0] @ xi) + ch[:, 0] - p[0]
        field = source_displacement_field(p, torch.tensor([0]), v, c0, dc, dr)[:, 0]
        torch.testing.assert_close(field, direct, rtol=1e-12, atol=1e-12)
        grads = torch.autograd.grad(ch.sum() + rh.sum(), (dc, dr))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_empty_and_uncovered_have_zero_output_and_gradients(self):
        p = torch.tensor([[1., 2., 3.], [2., -1., .5]], dtype=torch.float64)
        for count in (0, 1):
            centres = torch.zeros((count, 3), dtype=torch.float64)
            dc = torch.ones((4, count, 3), dtype=torch.float32, requires_grad=True)
            dr = torch.zeros((4, count, 3), dtype=torch.float32, requires_grad=True)
            result = source_displacement_field(p, torch.full((2,), -1, dtype=torch.int64),
                                               torch.ones_like(p), centres, dc, dr)
            self.assertTrue(byte_equal(result, torch.zeros((4, 2, 3), dtype=torch.float64)))
            grads = torch.autograd.grad(result.sum(), (dc, dr))
            self.assertTrue(all(torch.isfinite(g).all() and bool((g == 0).all()) for g in grads))
            r0 = torch.eye(3, dtype=torch.float64).expand(count, 3, 3).clone()
            ch, rh = future_poses(centres, r0, torch.zeros_like(centres), dc, dr)
            self.assertEqual(ch.shape, (4, count, 3))
            self.assertEqual(rh.shape, (4, count, 3, 3))

    def test_source_float32_residual_finite_difference_and_invalid_owner(self):
        p = torch.tensor([[1.2, -.7, 2.]], dtype=torch.float64)
        c0 = torch.tensor([[.2, .1, -.3]], dtype=torch.float64)
        vel = torch.tensor([[.6, -.2, .1]], dtype=torch.float64)
        owner = torch.tensor([0], dtype=torch.int64)
        dc = torch.zeros((4, 1, 3), dtype=torch.float32, requires_grad=True)
        dr = torch.full((4, 1, 3), .1, dtype=torch.float32, requires_grad=True)
        weight = torch.tensor([.3, -.8, 1.1], dtype=torch.float64)
        def scalar(c, r):
            return (source_displacement_field(p, owner, vel, c0, c, r)[2, 0] * weight).sum()
        gradients = torch.autograd.grad(scalar(dc, dr), (dc, dr))
        eps = 1e-3
        for which in (0, 1):
            for axis in range(3):
                plus = [dc.detach().clone(), dr.detach().clone()]
                minus = [dc.detach().clone(), dr.detach().clone()]
                plus[which][2, 0, axis] += eps
                minus[which][2, 0, axis] -= eps
                # Use the actual representable FP32 step in the denominator.
                step = (plus[which][2, 0, axis].double() - minus[which][2, 0, axis].double())
                finite_difference = (scalar(*plus) - scalar(*minus)) / step
                self.assertAlmostEqual(float(gradients[which][2, 0, axis]),
                                       float(finite_difference), places=5)
        with self.assertRaisesRegex(ValueError, 'mapped object index'):
            source_displacement_field(p, torch.tensor([1]), vel, c0, dc, dr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
