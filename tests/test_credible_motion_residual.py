"""CPU P2 math/isolation tests; synthetic checks are not model-quality evidence.

Run python tests/test_credible_motion_residual.py. Only PyTorch is required;
the module is imported without loading the MMCV detector package.
"""
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

import torch


DIRECTORY = (Path(__file__).resolve().parents[1] /
             'projects/mmdet3d_plugin/bevformer/detectors')
PACKAGE = '_p2_math_test'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(DIRECTORY)]
sys.modules[PACKAGE] = package
SPEC = importlib.util.spec_from_file_location(
    PACKAGE + '.credible_motion_residual', str(DIRECTORY / 'credible_motion_residual.py'))
P2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P2)


def grid(height=4, width=4):
    yy, xx = torch.meshgrid(torch.arange(height).float(), torch.arange(width).float())
    return torch.stack((2 * (xx + .5) / width - 1,
                        2 * (yy + .5) / height - 1), -1).reshape(1, height * width, 2)


def model(**kwargs):
    options = dict(point_cloud_range=(0, 0, -1, 4, 4, 1), bev_h=4, bev_w=4,
                   embed_dims=4, hidden_dims=8, transport_dims=3)
    options.update(kwargs)
    return P2.CredibleMotionResidual(**options)


def observations():
    return torch.tensor([[[1.5, 1.5, 0., 3., 1., 0., 4., 0., 1.],
                          [1.5, 1.5, 0., 3., 0., 1., 2., 0., 1.],
                          [2.5, 2.5, 0., 3., 1., 0., 1., .1, 1.]]])


class ResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def close(self, a, b, atol=1e-6, rtol=1e-6):
        self.assertTrue(torch.allclose(a, b, atol=atol, rtol=rtol),
                        'max difference {}'.format((a - b).abs().max()))

    def test_occupancy_uses_all_returns_in_training_and_eval(self):
        features, obs = torch.randn(1, 16, 4), observations()
        net = model(nll_weight=.05, holdout_fraction=.99)
        support = torch.tensor([[True, False, False]])
        train = net.prepare(features, obs, training=True, nll_support_mask=support)
        evaluate = net.prepare(features, obs, training=False)
        self.assertEqual(train['support_mask'].sum().item(), 3)
        self.assertEqual(train['holdout_mask'].sum().item(), 0)
        self.assertEqual(train['nll_state']['support_mask'].sum().item(), 1)
        self.assertEqual(train['nll_state']['holdout_mask'].sum().item(), 2)
        self.assertIsNone(evaluate['nll_state'])
        for key in ('mean', 'cov', 'geometry', 'prior_mean', 'prior_std'):
            self.close(train[key], evaluate[key], atol=0, rtol=0)

    def test_holdout_target_changes_loss_and_occupancy_but_not_nll_prediction(self):
        net = model(nll_weight=.05)
        features, obs = torch.randn(1, 16, 4), observations()
        support = torch.tensor([[True, False, True]])
        before = net.prepare(features, obs, training=True, nll_support_mask=support)
        changed = obs.clone()
        changed[0, 1, 6] = -40.
        after = net.prepare(features, changed, training=True, nll_support_mask=support)
        for key in ('mean', 'cov', 'geometry', 'prior_mean', 'prior_std'):
            self.close(before['nll_state'][key], after['nll_state'][key], atol=0, rtol=0)
        self.assertGreater((before['mean'] - after['mean']).abs().max().item(), 1)
        self.assertGreater(abs(before['loss_p2_doppler_nll'].item() -
                               after['loss_p2_doppler_nll'].item()), 1)

    def test_nll_prediction_has_no_holdout_radial_gradient(self):
        net = model(nll_weight=.05)
        obs = observations().requires_grad_(True)
        state = net.prepare(torch.randn(1, 16, 4), obs, training=True,
                            nll_support_mask=torch.tensor([[True, False, True]]))
        grad = torch.autograd.grad(state['nll_state']['mean'].sum(), obs,
                                   retain_graph=True)[0]
        self.assertEqual(grad[0, 1, 6].item(), 0.)
        loss_grad = torch.autograd.grad(state['loss_p2_doppler_nll'], obs)[0]
        self.assertNotEqual(loss_grad[0, 1, 6].item(), 0.)

    def test_random_nll_split_preserves_global_rng_and_remains_differentiable(self):
        net = model(nll_weight=.05, holdout_fraction=.99)
        features, obs = torch.randn(1, 16, 4), observations()
        rng_before = torch.get_rng_state().clone()
        state = net.prepare(features, obs, training=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertGreater(state['nll_state']['holdout_mask'].sum().item(), 0)
        state['loss_p2_doppler_nll'].backward()
        gradient = net.prior_head[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.)
        # Calling NLL again at the same global RNG state yields the same
        # partition. No hidden module generator/cache changes its behavior.
        repeated = net.prepare(features, obs, training=True)
        self.assertTrue(torch.equal(state['nll_state']['support_mask'],
                                    repeated['nll_state']['support_mask']))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

    def test_s1_zero_weight_does_not_construct_holdout_state(self):
        net = model(nll_weight=0., holdout_fraction=.99)
        state = net.prepare(torch.randn(1, 16, 4), observations(), training=True,
                            nll_support_mask=torch.zeros(1, 3, dtype=torch.bool))
        self.assertEqual(state['support_mask'].sum().item(), 3)
        self.assertIsNone(state['nll_state'])
        self.assertNotIn('loss_p2_doppler_nll', state)

    def test_zero_gate_is_exact_baseline_identity_both_modes(self):
        features = torch.randn(1, 16, 4)
        for mode in ('mean', 'capacity'):
            net = model(mode=mode)
            result = net.transport(features, net.prepare(features, observations()), 2, grid())
            self.close(result, torch.zeros_like(result), atol=0, rtol=0)
            self.close(features + result, features, atol=0, rtol=0)
            result.sum().backward()
            for name, parameter in net.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_no_support_has_no_residual_both_modes(self):
        features = torch.randn(1, 16, 4)
        for mode in ('mean', 'capacity'):
            net = model(mode=mode, gate_init=.5)
            result = net.transport(features, net.prepare(features, torch.empty(1, 0, 9)), 2, grid())
            self.close(result, torch.zeros_like(result), atol=0, rtol=0)

    def test_zero_motion_cancels_with_nonidentity_ego_alignment(self):
        features = torch.randn(1, 16, 4)
        net = model(gate_init=.5)
        state = net.prepare(features, observations())
        state['mean'] = state['mean'] * 0.
        ego_grid = grid() + torch.tensor([[[.18, -.12]]])
        self.close(net.transport(features, state, 3, ego_grid),
                   torch.zeros_like(features), atol=0, rtol=0)

    def test_capacity_is_active_at_zero_velocity_and_has_same_parameters(self):
        moving, control = model(mode='mean'), model(mode='capacity', gate_init=.5)
        self.assertEqual({n: tuple(p.shape) for n, p in moving.named_parameters()},
                         {n: tuple(p.shape) for n, p in control.named_parameters()})
        # Exercise gradients away from zero-gate/zero-last-layer initialization.
        with torch.no_grad():
            control.prior_head[-1].weight.normal_(0, .02)
        features = torch.randn(1, 16, 4)
        obs = observations()
        obs[..., 6] = 0.
        state = control.prepare(features, obs)
        state['mean'] = state['mean'] * 0.
        output = control.transport(features, state, 2, grid())
        self.assertGreater(output.abs().sum().item(), 0.)
        output.square().sum().backward()
        for name, parameter in control.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(control.transport_reduce.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(control.transport_expand.weight.grad.abs().sum().item(), 0.)

    def test_destination_gate_preserves_low_confidence_at_isolated_return(self):
        features = torch.randn(1, 16, 4)
        net = model(gate_init=.5)
        state = net.prepare(features, observations()[:, :1])
        full_confidence = net._source_confidence(state)
        net._source_confidence = lambda unused: full_confidence
        full = net.transport(features, state, 1, grid())
        net._source_confidence = lambda unused: full_confidence * .01
        small = net.transport(features, state, 1, grid())
        self.assertGreater(full.abs().sum().item(), 0.)
        self.close(small, full * .01, atol=1e-8, rtol=1e-5)

    def test_impulse_moves_in_correct_direction_with_matching_ego_alignment(self):
        net = model(embed_dims=1, transport_dims=1, gate_init=math.atanh(.5),
                    sensor_std=.01, time_step=1.)
        with torch.no_grad():
            net.transport_reduce.weight.fill_(1.)
            net.transport_expand.weight.fill_(1.)
        features = torch.zeros(1, 16, 1)
        features[0, 5] = 1.
        state = net.prepare(features, observations()[:, :1])
        state['mean'] = state['mean'] * 0.
        state['mean'][:, 0, 1, 1] = 1.
        confidence = net._source_confidence(state)[0, 0, 1, 1].item()
        result = net.transport(features, state, 1, grid()).reshape(4, 4)
        self.assertAlmostEqual(result[1, 1].item(), -.5 * confidence, places=5)
        self.assertAlmostEqual(result[1, 2].item(), .5 * confidence, places=5)
        ego_grid = grid() + torch.tensor([[[.5, 0.]]])
        aligned = net.transport(features, state, 1, ego_grid).reshape(4, 4)
        self.assertAlmostEqual(aligned[1, 0].item(), -.5 * confidence, places=5)
        self.assertAlmostEqual(aligned[1, 1].item(), .5 * confidence, places=5)

    def test_displacement_bound(self):
        net = model(max_speed_mps=4., max_displacement_m=3.)
        displacement = net._bounded_displacement(torch.full((1, 2, 4, 4), 1e5), 10)
        self.assertLessEqual(displacement.square().sum(1).sqrt().max().item(), 3.00001)


if __name__ == '__main__':
    unittest.main(verbosity=2)
