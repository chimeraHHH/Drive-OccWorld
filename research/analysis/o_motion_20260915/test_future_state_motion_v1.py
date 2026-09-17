"""Analytic CPU tests; these do not establish native training or performance."""
import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from future_state_motion_v1 import (
    FutureStateMotionReadout, capture_native_terminal,
    native_state_pair_images, readout_images_to_xyz,
)


class FutureStateMotionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def test_nonsquare_native_pair_and_horizon_batch_order(self):
        nx, ny, nz, batch = 3, 2, 2, 2
        features = torch.empty(5, batch, nx * ny, 256)
        for t in range(5):
            for b in range(batch):
                for y in range(ny):
                    for x in range(nx):
                        features[t, b, y * nx + x] = 10000*t + 1000*b + 100*y + 10*x + torch.arange(256)/256
        original = features.clone()
        images = native_state_pair_images(features, (nx, ny, nz))
        self.assertEqual(tuple(images.shape), (4 * batch, 513, ny, nx))
        for h in range(4):
            for b in range(batch):
                for y in range(ny):
                    for x in range(nx):
                        row = images[h * batch + b, :, y, x]
                        self.assertTrue(torch.equal(row[:256], features[0, b, y*nx+x]))
                        self.assertTrue(torch.equal(row[256:512], features[h+1, b, y*nx+x]))
                        self.assertEqual(row[-1].item(), .5*(h+1))
        self.assertTrue(torch.equal(features, original))

    def test_nonsquare_xyz_output_and_height_channels(self):
        nx, ny, nz, batch = 3, 2, 2, 2
        raw = torch.empty(4*batch, 3*nz, ny, nx)
        for h in range(4):
            for b in range(batch):
                for a in range(3):
                    for z in range(nz):
                        for y in range(ny):
                            for x in range(nx):
                                raw[h*batch+b, a*nz+z, y, x] = 100000*h+10000*b+1000*a+100*z+10*y+x
        out = readout_images_to_xyz(raw, batch, (nx, ny, nz))
        self.assertEqual(tuple(out.shape), (batch, 4, 3, nx, ny, nz))
        for h in range(4):
            for b in range(batch):
                for a in range(3):
                    for z in range(nz):
                        for y in range(ny):
                            for x in range(nx):
                                self.assertEqual(out[b,h,a,x,y,z].item(), 100000*h+10000*b+1000*a+100*z+10*y+x)

    def test_zero_identity_live_last_layer_and_no_unexpected_base_gradient(self):
        head = FutureStateMotionReadout((3, 2, 2))
        features = torch.randn(5, 1, 6, 256, requires_grad=True)
        base = torch.randn(1, 4, 3, 3, 2, 2)
        base.reshape(-1)[0] = -0.
        pred = head(features, base)
        self.assertTrue(torch.equal(pred.view(torch.int32), base.view(torch.int32)))
        pred.square().mean().backward()
        self.assertGreater(head.readout.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(head.readout.bias.grad.abs().sum().item(), 0.)
        self.assertEqual(features.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum().item() == 0. for p in head.trunk.parameters()))
        with self.assertRaises(ValueError):
            head(features, base.clone().requires_grad_())

    def test_after_real_update_K_B_have_same_values_and_different_state_gradients(self):
        head = FutureStateMotionReadout((3, 2, 2))
        inputs = torch.randn(5, 1, 6, 256)
        base = torch.randn(1, 4, 3, 3, 2, 2)
        target = torch.randn_like(base)
        optimizer = torch.optim.SGD(head.parameters(), lr=.01)
        (head(inputs, base)-target).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        control = copy.deepcopy(head)
        fk = inputs.clone().requires_grad_()
        fb = inputs.clone().requires_grad_()
        pk = head(fk, base, detach_features=False)
        pb = control(fb, base, detach_features=True)
        self.assertTrue(torch.equal(pk, pb))
        (pk-target).square().mean().backward()
        (pb-target).square().mean().backward()
        self.assertGreater(fk.grad[1:].abs().sum().item(), 0.)
        self.assertIsNone(fb.grad)
        for a, b in zip(head.parameters(), control.parameters()):
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.equal(a.grad, b.grad))
            self.assertTrue(torch.isfinite(a.grad).all())
        self.assertEqual(sum(p.numel() for p in FutureStateMotionReadout().parameters()), 745392)

    def test_capture_keeps_transition_graph_values_and_removes_hook_on_failure(self):
        model = nn.Module()
        model.future_pred_head = nn.Module()
        model.future_pred_head.soft_weight = False
        model.future_pred_head.transition = nn.Linear(256, 256)
        model.future_pred_head.bev_pred_head = nn.ModuleList([nn.Sequential(nn.Linear(256, 2)) for _ in range(3)])
        model.train()
        x = torch.randn(5, 1, 6, 256)

        def replay(m, sample, training=False):
            z = m.future_pred_head.transition(sample)
            return (torch.stack([branch(z) for branch in m.future_pred_head.bev_pred_head]),)

        native = SimpleNamespace(replay=replay)
        expected = replay(model, x, True)[0]
        pred, captured = capture_native_terminal(model, x, native, training=True, expected_shape=(5,1,6,256))
        self.assertTrue(torch.equal(pred, expected))
        captured[1:].square().mean().backward()
        self.assertGreater(model.future_pred_head.transition.weight.grad.abs().sum().item(), 0.)
        branch = model.future_pred_head.bev_pred_head[-1]
        self.assertFalse(branch._forward_pre_hooks)

        def failed_replay(m, sample, training=False):
            replay(m, sample, training)
            raise RuntimeError('intentional analytic failure')

        with self.assertRaises(RuntimeError):
            capture_native_terminal(model, x, SimpleNamespace(replay=failed_replay), training=True, expected_shape=(5,1,6,256))
        self.assertFalse(branch._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main(verbosity=2)
