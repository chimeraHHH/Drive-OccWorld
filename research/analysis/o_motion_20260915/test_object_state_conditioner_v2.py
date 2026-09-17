"""Small CPU contract/gradient tests; no external model, labels or training.

Requires a real PyTorch installation. These tests do not substitute for the
later native four-update GPU preflight. No synthetic performance is written.
"""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

from object_state_conditioner_v2 import (
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

    def encoded(self, *, frame, use_velocity=True, tokens=None):
        """Observe the actual state-MLP boundary, not a copy of its formula."""
        seen = []
        handle = self.module.state_mlp.register_forward_pre_hook(
            lambda _module, args: seen.append(args[0].detach().clone()))
        try:
            output = self.forward(tokens, use_velocity=use_velocity, frame=frame)
        finally:
            handle.remove()
        self.assertEqual(len(seen), 1)
        return seen[0], output

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
        self.assertEqual(sum(p.numel() for p in self.module.parameters()), 95872)

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

    def test_07_physical_center_steps_current_slots_and_G_unchanged(self):
        # Analytic current predictions only. The second object is outside the
        # nominal cube and has zero confidence: neither is a filtering rule.
        center = torch.tensor([[[-5., 10., 2.], [102.4, -80., -12.]]])
        velocity = torch.tensor([[[4., -2., 1.], [60., -20., 5.]]])
        self.tokens, self.valid = pack_object_states(
            center, torch.tensor([2., 4., 1.5]).expand(1, 2, -1).clone(),
            torch.eye(3).expand(1, 2, 3, 3).clone(), torch.tensor([[0, 7]]),
            torch.tensor([[.7, 0.]]), velocity)
        original = self.tokens.clone()
        expected_metres = (
            [[-3., 9., 2.5], [132.4, -90., -9.5]],
            [[-1., 8., 3.], [162.4, -100., -7.]],
            [[1., 7., 3.5], [192.4, -110., -4.5]],
            [[3., 6., 4.], [222.4, -120., -2.]],
        )
        for frame, physical_center in enumerate(expected_metres, 1):
            encoded, delta = self.encoded(frame=frame)
            self.assertEqual(encoded.shape, (1, 2, 30))
            self.assertTrue(torch.equal(encoded[..., :27], original))
            # Expected values are in metres, independently tabulated above.
            torch.testing.assert_close(encoded[..., 27:] * 51.2,
                torch.tensor([physical_center]), rtol=1e-6, atol=1e-6)
            self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
            g_encoded, _ = self.encoded(frame=frame, use_velocity=False)
            self.assertTrue(torch.equal(g_encoded[..., :24], original[..., :24]))
            self.assertTrue(torch.equal(g_encoded[..., 24:27], torch.zeros(1, 2, 3)))
            self.assertTrue(torch.equal(g_encoded[..., 27:], original[..., :3]))
            self.assertTrue(torch.equal(self.tokens, original))
        # Asking again is not cumulative propagation from the last horizon.
        repeated, _ = self.encoded(frame=1)
        torch.testing.assert_close(repeated[..., 27:] * 51.2,
            torch.tensor([expected_metres[0]]), rtol=1e-6, atol=1e-6)

    def test_08_full_R_velocity_and_geometric_center_consumed_once(self):
        # The loader owns the full global-to-R rotation. Supply its analytic
        # roll-90 result, including nonzero R-z, at the module's real boundary.
        global_to_R = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        velocity_R = global_to_R @ torch.tensor([8., -4., 0.])
        self.assertTrue(torch.equal(velocity_R, torch.tensor([8., 0., -4.])))
        geometric_center = torch.tensor([[[10., 20., 30.]]])
        captured = []
        for height in (4., 40.):
            self.tokens, self.valid = pack_object_states(
                geometric_center, torch.tensor([[[2., 8., height]]]),
                global_to_R.reshape(1, 1, 3, 3), torch.tensor([[0]]),
                torch.tensor([[.8]]), velocity_R.reshape(1, 1, 3))
            encoded, _ = self.encoded(frame=4)
            self.assertTrue(torch.equal(encoded[..., :3], geometric_center / 51.2))
            self.assertTrue(torch.equal(encoded[..., 6:15], global_to_R.reshape(1, 1, 9)))
            torch.testing.assert_close(encoded[..., 27:] * 51.2,
                torch.tensor([[[26., 20., 22.]]]), rtol=1e-6, atol=1e-6)
            captured.append(encoded[..., 27:])
        # No h_box/2, global-z adjustment, or second box-origin adaptation.
        self.assertTrue(torch.equal(captured[0], captured[1]))

    def test_09_CV_channels_alone_connect_horizon_and_velocity(self):
        self.activate()
        with torch.no_grad():
            # Isolate appended centres: direct velocity channels and additive
            # time bias are disabled while the fixed native query is retained.
            self.module.state_mlp[0].weight[:, 24:27].zero_()
            for parameter in self.module.time_mlp.parameters():
                parameter.zero_()
        v_early = self.forward(frame=1)
        v_late = self.forward(frame=4)
        g_early = self.forward(frame=1, use_velocity=False)
        g_late = self.forward(frame=4, use_velocity=False)
        self.assertGreater(float((v_late - v_early).abs().max()), 0.)
        self.assertTrue(torch.equal(g_early, g_late))
        differentiable = self.tokens.clone().requires_grad_()
        value = self.forward(differentiable, frame=4).square().sum()
        grad = torch.autograd.grad(value, differentiable)[0]
        self.assertGreater(float(grad[..., 24:].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(grad).all())

    def test_10_no_GT_future_pose_or_per_example_time_interface(self):
        self.assertEqual(tuple(inspect.signature(pack_object_states).parameters),
            ('center_R', 'size_wlh', 'rotation_R', 'class_index', 'score', 'velocity_R', 'valid'))
        self.assertEqual(tuple(inspect.signature(self.module.forward).parameters),
            ('states', 'valid', 'bev_embedding', 'prev_features', 'target_frame_index', 'use_velocity'))
        for forbidden in ('gt_boxes', 'future_pose', 'dt_future_seconds', 'origin_offset'):
            with self.assertRaises(TypeError):
                self.module(self.tokens, self.valid, self.embedding, self.previous, 1,
                    use_velocity=True, **{forbidden: torch.ones(1)})
        for frame in (0, 5, .5, True):
            with self.assertRaises(ValueError):
                self.forward(frame=frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU receipt execution requires CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConditionerTests)
    names = [case.id() for case in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(schema='object-state-conditioner-cpu-tests-v2',
        status='PASS_CPU_INTERFACE_AND_CV_FEATURE_TESTS' if result.wasSuccessful() else 'FAIL_CPU_TESTS',
        tests_run=result.testsRun, test_names=names,
        failures=[dict(test=t.id(), traceback=error) for t, error in result.failures],
        errors=[dict(test=t.id(), traceback=error) for t, error in result.errors],
        source_sha256=hashlib.sha256(Path(inspect.getfile(ObjectStateConditioner)).read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
            torch_threads=torch.get_num_threads(), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            cuda_initialized=torch.cuda.is_initialized()),
        parameters=95872, current_input_features=27, state_encoder_features=30,
        optimizer_updates=0, native_model_loaded=False, real_data_or_GT_read=False,
        native_O_D_parity_tested=False, performance_evaluated=False,
        scope='CPU interface, analytic coordinate and autograd checks only; native preflight still required')
    if args.receipt:
        with Path(args.receipt).open('x') as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write('\n')
    print(json.dumps(receipt, allow_nan=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
