"""Small CPU contract/gradient tests; no external model, labels or training.

Requires a real PyTorch installation. These tests do not substitute for the
later native four-update GPU preflight. No synthetic performance is written.
"""
from copy import deepcopy
import unittest

import torch
from torch import nn

from object_state_conditioner_v1 import (
    ObjectStateConditioner, condition_future_head, pack_object_states,
)


def states(batch=1, count=3):
    center = torch.arange(batch * count * 3, dtype=torch.float32).reshape(batch, count, 3)
    size = torch.tensor([2., 4., 1.5]).expand(batch, count, -1).clone()
    rotation = torch.eye(3).expand(batch, count, 3, 3).clone()
    classes = torch.arange(count).remainder(8).expand(batch, -1).clone()
    score = torch.full((batch, count), .7)
    velocity = torch.tensor([8., -3., 1.]).expand(batch, count, -1).clone()
    return pack_object_states(center, size, rotation, classes, score, velocity)


class TinyNativeHead(nn.Module):
    """Only the real forward signature/argument boundary, not a model proxy."""
    def __init__(self, fail_frame=None):
        super().__init__()
        self.bev_embedding = nn.Embedding(6, 256)
        self.fail_frame = fail_frame

    def forward(self, prev_feats, img_metas, target_frame_index,
                action_condition_dict, cond_norm_dict, tgt_points, ref_points,
                bev_h, bev_w, rollout_prior=None):
        if target_frame_index == self.fail_frame:
            raise RuntimeError('sentinel native error')
        return prev_feats[:, -1] if rollout_prior is None else prev_feats[:, -1] + rollout_prior


class ConditionerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)  # Test owns its initialization; production module never seeds.
        self.module = ObjectStateConditioner()
        self.tokens, self.valid = states()
        self.embedding = torch.randn(6, 256)
        self.previous = torch.randn(1, 1, 6, 256)

    def forward(self, tokens=None, use_velocity=True, frame=1):
        return self.module(self.tokens if tokens is None else tokens, self.valid,
                           self.embedding, self.previous, frame, use_velocity=use_velocity)

    def activate(self):
        # A fixed nonzero readout to expose the module's pre-existing dependency,
        # not an optimizer step or a learned performance claim.
        with torch.no_grad():
            self.module.output_projection.weight[:64].copy_(torch.eye(64))

    def test_01_packing_axis_units_no_clipping(self):
        c = torch.tensor([[[102.4, -51.2, 5.12]]])
        s = torch.tensor([[[2., 4., 1.5]]])
        r = torch.tensor([[[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]]])
        v = torch.tensor([[[60., -20., 0.]]])
        t, valid = pack_object_states(c, s, r, torch.tensor([[7]]), torch.tensor([[.9]]), v)
        self.assertEqual(t.shape, (1, 1, 27))
        self.assertTrue(valid.item())
        torch.testing.assert_close(t[..., :3], torch.tensor([[[2., -1., .1]]]))
        torch.testing.assert_close(t[..., 3:6], s / 51.2)
        self.assertTrue(torch.equal(t[..., 6:15], r.reshape(1, 1, 9)))
        self.assertEqual(t[..., 15:23].argmax(-1).item(), 7)
        self.assertTrue(torch.equal(t[..., 24:], torch.tensor([[[3., -1., 0.]]])))
        self.assertEqual(sum(p.numel() for p in self.module.parameters()), 95680)

    def test_02_zero_initialization_and_empty_mixed_batch(self):
        for h in range(1, 5):
            delta = self.forward(frame=h)
            self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        empty, ev = states(count=0)
        self.activate()
        with torch.no_grad():
            self.module.output_projection.bias.fill_(.1)
        out = self.module(empty, ev, self.embedding, self.previous, 1, use_velocity=True)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        tok, val = states(batch=2)
        val[1] = False
        tok[1] = float('nan')  # Padded/invalid values cannot contaminate attention.
        out = self.module(tok, val, self.embedding, self.previous.expand(2, -1, -1, -1), 1,
                          use_velocity=True)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(out[1], torch.zeros_like(out[1])))
        self.assertGreater(float(out[0].abs().sum()), 0.)

    def test_03_matched_parameters_and_velocity_dependency(self):
        paired = deepcopy(self.module)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(self.module.parameters(), paired.parameters())))
        changed = self.tokens.clone()
        changed[..., 24:] += torch.tensor([3., 1., -2.])
        self.assertTrue(torch.equal(changed[..., :24], self.tokens[..., :24]))
        self.activate()
        self.assertTrue(torch.equal(self.forward(use_velocity=False),
                                    self.forward(changed, use_velocity=False)))
        self.assertGreater(float((self.forward() - self.forward(changed)).abs().max()), 1e-8)
        differentiable = self.tokens.clone().requires_grad_()
        g = torch.autograd.grad(self.forward(differentiable, False).square().sum(), differentiable)[0]
        self.assertTrue(torch.equal(g[..., 24:], torch.zeros_like(g[..., 24:])))

    def test_04_zero_readout_live_gradient_and_shared_content_connection(self):
        initial = self.forward()
        initial.sum().backward()
        self.assertGreater(float(self.module.output_projection.weight.grad.abs().sum()), 0.)
        self.module.zero_grad(set_to_none=True)
        self.activate()
        tokens = self.tokens.clone().requires_grad_()
        previous = self.previous.clone().requires_grad_()
        out = self.module(tokens, self.valid, self.embedding, previous, 1, use_velocity=True)
        grads = torch.autograd.grad(out.square().sum(), (tokens, previous, self.module.state_mlp[0].weight))
        self.assertGreater(float(grads[0][..., 24:].abs().sum()), 0.)
        self.assertGreater(float(grads[1].abs().sum()), 0.)
        self.assertGreater(float(grads[2].abs().sum()), 0.)

    def test_05_four_future_hook_order_shapes_rng_modes_cleanup(self):
        head = TinyNativeHead().eval()
        self.module.train()
        previous = self.previous.clone()
        kwargs = dict(tgt_points=None, ref_points=None, bev_h=2, bev_w=3)
        rng = torch.random.get_rng_state().clone()
        with condition_future_head(head, self.module, self.tokens, self.valid, use_velocity=True) as audit:
            for frame in range(1, 5):
                # The final step also checks the originally positional None prior.
                if frame == 4:
                    out = head(previous, {}, frame, {}, {}, None, None, 2, 3, None)
                else:
                    out = head(previous, {}, frame, {}, {}, **kwargs)
                self.assertTrue(torch.equal(out, previous[:, -1]))
        self.assertEqual(audit['frames'], [1, 2, 3, 4])
        self.assertTrue(audit['hook_removed'] and audit['completed'])
        self.assertEqual([e['prior_shape'] for e in audit['entries']], [[1, 6, 256]] * 4)
        self.assertEqual(len(head._forward_pre_hooks), 0)
        self.assertFalse(head.training)
        self.assertTrue(self.module.training)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(torch.equal(previous, self.previous))
        self.assertNotIn('rollout_prior', kwargs)

    def test_06_error_paths_clean_hook_without_masking_native_error(self):
        kwargs = dict(tgt_points=None, ref_points=None, bev_h=2, bev_w=3)
        for frame, prior in ((1, torch.zeros_like(self.previous[:, -1])), (0, None), (2, None)):
            head = TinyNativeHead()
            with self.assertRaises(ValueError):
                with condition_future_head(head, self.module, self.tokens, self.valid, use_velocity=True):
                    head(self.previous, {}, frame, {}, {}, rollout_prior=prior, **kwargs)
            self.assertEqual(len(head._forward_pre_hooks), 0)
        head = TinyNativeHead(fail_frame=2)
        with self.assertRaisesRegex(RuntimeError, 'sentinel native error'):
            with condition_future_head(head, self.module, self.tokens, self.valid, use_velocity=True) as audit:
                head(self.previous, {}, 1, {}, {}, **kwargs)
                head(self.previous, {}, 2, {}, {}, **kwargs)
        self.assertTrue(audit['hook_removed'])
        self.assertFalse(audit['completed'])
        self.assertEqual(len(head._forward_pre_hooks), 0)
        with self.assertRaisesRegex(ValueError, 'before four'):
            with condition_future_head(head, self.module, self.tokens, self.valid, use_velocity=False):
                head(self.previous, {}, 1, {}, {}, **kwargs)
        self.assertEqual(len(head._forward_pre_hooks), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
