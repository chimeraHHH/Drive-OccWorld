"""CPU-only M3 math and information-isolation tests; no MMCV import required.

Run: python tests/test_doppler_posterior_transport.py
These synthetic checks establish implementation properties, not model quality.
"""
import importlib.util
import math
from pathlib import Path
import unittest

import torch

MODULE_PATH = (Path(__file__).resolve().parents[1] / 'projects/mmdet3d_plugin/'
               'bevformer/detectors/doppler_posterior_transport.py')
SPEC = importlib.util.spec_from_file_location('m3_posterior', str(MODULE_PATH))
M3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M3)


def grid(height, width, batch=1):
    yy, xx = torch.meshgrid(torch.arange(height).float(), torch.arange(width).float())
    return torch.stack((2 * (xx + 0.5) / width - 1,
                        2 * (yy + 0.5) / height - 1), dim=-1).reshape(
                            1, height * width, 2).expand(batch, -1, -1)


def module(**kwargs):
    defaults = dict(point_cloud_range=(0, 0, -1, 8, 8, 1), bev_h=4, bev_w=4,
                    embed_dims=4, hidden_dims=8, transport_dims=3)
    defaults.update(kwargs)
    return M3.DopplerPosteriorTransport(**defaults)


def observation(x=1.0, y=1.0, ux=1.0, uy=0.0, vr=10.0, lag=0.0):
    return [x, y, 0.0, 3.0, ux, uy, vr, lag, 1.0]


class PosteriorMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def assertClose(self, actual, expected, atol=1e-5, rtol=1e-5):
        self.assertTrue(torch.allclose(actual, expected, atol=atol, rtol=rtol),
                        '{} != {}'.format(actual, expected))

    def test_rank_one_update_preserves_tangential_prior(self):
        mean = torch.tensor([3., 7.]).reshape(1, 2, 1, 1)
        std = torch.tensor([5.]).reshape(1, 1, 1, 1)
        info = torch.tensor([[4., 0.], [0., 0.]]).reshape(1, 2, 2, 1, 1)
        vector = torch.tensor([80., 0.]).reshape(1, 2, 1, 1)
        post, cov = M3.solve_isotropic_posterior(mean, std, info, vector)
        self.assertClose(post[:, 1], mean[:, 1])
        self.assertClose(cov[:, 1, 1], std[:, 0].square())
        self.assertAlmostEqual(post[0, 0, 0, 0].item(), 3 + (80 - 12) / 4.04, places=4)
        self.assertLess(cov[0, 0, 0, 0, 0], cov[0, 1, 1, 0, 0])

    def test_opposing_rays_do_not_cancel_information(self):
        model = module(sensor_std=0.1)
        obs = torch.tensor([[observation(vr=12), observation(ux=-1, vr=-12)]])
        state = model.prepare(torch.zeros(1, 16, 4), obs, training=False)
        self.assertGreater(state['mean'][0, 0, 0, 0], 11.99)
        self.assertGreater(state['information_eigenvalues'][0, 1, 0, 0], 199)
        self.assertEqual(state['information_eigenvalues'][0, 0, 0, 0], 0)

    def test_orthogonal_rays_recover_both_components(self):
        model = module(sensor_std=0.01)
        obs = torch.tensor([[observation(vr=6), observation(ux=0, uy=1, vr=-9)]])
        state = model.prepare(torch.zeros(1, 16, 4), obs, training=False)
        self.assertClose(state['mean'][0, :, 0, 0], torch.tensor([6., -9.]), atol=1e-4)

    def test_diagonal_rays_keep_individual_constraints(self):
        model = module(sensor_std=0.01)
        diagonal = math.sqrt(0.5)
        obs = torch.tensor([[observation(ux=diagonal, uy=diagonal, vr=10 * diagonal),
                             observation(ux=diagonal, uy=-diagonal, vr=10 * diagonal)]])
        state = model.prepare(torch.zeros(1, 16, 4), obs, training=False)
        # Averaging u and r separately then renormalizing u would return 7.071.
        self.assertClose(state['mean'][0, :, 0, 0], torch.tensor([10., 0.]), atol=1e-4)

    def test_information_update_is_rotation_equivariant(self):
        mean = torch.tensor([2., -3.])
        u = torch.tensor([[0.6, 0.8], [-1., 0.]])
        weights = torch.tensor([2., 0.5])
        radial = torch.tensor([10., -4.])
        info = torch.einsum('n,ni,nj->ij', weights, u, u)
        vector = torch.einsum('n,ni,n->i', weights, u, radial)
        angle = 0.63
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
        def solve(mu, mat, vec):
            return M3.solve_isotropic_posterior(mu.reshape(1, 2, 1, 1),
                torch.tensor([3.]).reshape(1, 1, 1, 1), mat.reshape(1, 2, 2, 1, 1),
                vec.reshape(1, 2, 1, 1))
        post, cov = solve(mean, info, vector)
        rot_post, rot_cov = solve(rotation @ mean, rotation @ info @ rotation.T,
                                  rotation @ vector)
        self.assertClose(rot_post.flatten(), rotation @ post.flatten())
        self.assertClose(rot_cov.reshape(2, 2), rotation @ cov.reshape(2, 2) @ rotation.T)

    def test_physical_radial_speed_is_not_feature_clipped(self):
        model = module()
        state = model.prepare(torch.zeros(1, 16, 4),
                              torch.tensor([[observation(vr=40)]]), training=False)
        expected = 40.0 / (1.0 + model.sensor_std ** 2 / 25.0)
        self.assertAlmostEqual(state['mean'][0, 0, 0, 0].item(), expected, places=4)
        self.assertGreater(state['mean'][0, 0, 0, 0], 30)
        self.assertEqual(state['geometry'][:, 4:7].abs().sum(), 0)

    def test_empty_input_equals_prior_and_loss_is_finite_zero(self):
        model = module()
        state = model.prepare(torch.zeros(2, 16, 4), torch.empty(2, 0, 9))
        self.assertClose(state['mean'], state['prior_mean'], atol=0, rtol=0)
        self.assertClose(state['cov'][:, 0, 0], state['prior_std'][:, 0].square())
        self.assertClose(state['cov'][:, 1, 1], state['prior_std'][:, 0].square())
        self.assertEqual(state['geometry'].abs().sum(), 0)
        self.assertEqual(state['loss_doppler_nll'].item(), 0)
        state['loss_doppler_nll'].backward()
        self.assertTrue(torch.isfinite(model.prior_head[-1].weight.grad).all())

    def test_invalid_noncausal_and_stale_returns_are_excluded(self):
        model = module()
        rows = [observation(lag=0), observation(lag=.1), observation(lag=-.01),
                observation(lag=.2), observation(ux=2), observation(x=8),
                observation(vr=float('nan')), observation()]
        rows[-1][-1] = 0
        state = model.prepare(torch.zeros(1, 16, 4), torch.tensor([rows]), training=False)
        self.assertEqual(state['support_mask'].tolist(), [[True, True, False, False,
                                                         False, False, False, False]])
        self.assertTrue(torch.isfinite(state['mean']).all())
        self.assertTrue(torch.isfinite(state['cov']).all())
        self.assertEqual(state['diagnostics']['valid_count'].item(), 2)

    def test_heldout_target_cannot_change_conditioned_state(self):
        model = module()
        features = torch.randn(1, 16, 4)
        obs = torch.tensor([[observation(vr=5), observation(vr=10, ux=0, uy=1)]])
        support = torch.tensor([[True, False]])
        before = model.prepare(features, obs, training=True, support_mask=support)
        changed = obs.clone()
        changed[0, 1, 6] = -80
        changed[0, 1, 2] = .8
        changed[0, 1, 3] = 30
        after = model.prepare(features, changed, training=True, support_mask=support)
        for name in ('mean', 'cov', 'geometry', 'prior_mean', 'prior_std'):
            self.assertClose(before[name], after[name], atol=0, rtol=0)
        self.assertGreater(abs(before['loss_doppler_nll'].item() -
                               after['loss_doppler_nll'].item()), 1.0)
        self.assertAlmostEqual(before['geometry'][0, 1, 0, 0].item(),
                               math.log(2.0) / math.log(33.0), places=6)

    def test_random_split_keeps_support_and_inference_uses_all(self):
        model = module(holdout_fraction=0.99999)
        features = torch.zeros(1, 16, 4)
        obs = torch.tensor([[observation(), observation(), observation()]])
        training = model.prepare(features, obs, training=True)
        self.assertEqual(training['support_mask'].sum().item(), 1)
        self.assertEqual(training['holdout_mask'].sum().item(), 2)
        inference = model.prepare(features, obs, training=False)
        self.assertEqual(inference['support_mask'].sum().item(), 3)
        self.assertEqual(inference['holdout_mask'].sum().item(), 0)
        # Explicit states are independent; no mutable last-batch cache.
        self.assertEqual(training['holdout_mask'].sum().item(), 2)

    def test_lag_reduces_measurement_information(self):
        model = module()
        features = torch.zeros(1, 16, 4)
        fresh = model.prepare(features, torch.tensor([[observation(lag=0)]]), training=False)
        aged = model.prepare(features, torch.tensor([[observation(lag=.15)]]), training=False)
        self.assertLess(aged['mean'][0, 0, 0, 0], fresh['mean'][0, 0, 0, 0])
        self.assertGreater(aged['cov'][0, 0, 0, 0, 0], fresh['cov'][0, 0, 0, 0, 0])

    def test_prior_only_ablation_ignores_radial_update(self):
        model = module(use_doppler_update=False)
        state = model.prepare(torch.randn(1, 16, 4),
                              torch.tensor([[observation(vr=40)]]), training=False)
        self.assertClose(state['mean'], state['prior_mean'], atol=0, rtol=0)
        self.assertClose(state['cov'][:, 0, 0], state['prior_std'][:, 0].square())
        self.assertGreater(state['geometry'].abs().sum(), 0)

    def test_sigma_points_match_mean_and_covariance(self):
        mean = torch.tensor([4., -7.]).reshape(1, 2, 1, 1)
        covariance = torch.tensor([[9., 3.], [3., 4.]]).reshape(1, 2, 2, 1, 1)
        points, weights = M3.positive_sigma_points(mean, covariance)
        self.assertTrue((weights > 0).all())
        self.assertClose(weights.sum(), torch.tensor(1.))
        recovered_mean = (points * weights.reshape(1, 5, 1, 1, 1)).sum(dim=1)
        delta = points - mean.unsqueeze(1)
        recovered_cov = torch.einsum('bkchw,bkdhw,k->bcdhw', delta, delta, weights)
        self.assertClose(recovered_mean, mean)
        self.assertClose(recovered_cov, covariance)


class TransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def assertClose(self, actual, expected, atol=1e-5, rtol=1e-5):
        self.assertTrue(torch.allclose(actual, expected, atol=atol, rtol=rtol),
                        '{} != {}'.format(actual, expected))

    def test_integer_translation_moves_features_and_leaves_hole(self):
        features = torch.tensor([1., 2., 3., 4.]).reshape(1, 1, 1, 4)
        displacement = torch.zeros(1, 2, 1, 4)
        displacement[:, 0] = 1
        result, mass = M3.forward_bilinear_splat(features, displacement)
        self.assertClose(result.flatten(), torch.tensor([0., 1., 2., 3.]))
        self.assertClose(mass.flatten(), torch.tensor([0., 1., 1., 1.]))

    def test_subpixel_splat_has_correct_weights(self):
        features = torch.tensor([0., 8., 0., 0.]).reshape(1, 1, 1, 4)
        displacement = torch.zeros(1, 2, 1, 4)
        displacement[:, 0] = .25
        source_weight = torch.tensor([0., 1., 0., 0.]).reshape(1, 1, 1, 4)
        numerator, mass = M3.forward_bilinear_splat(features, displacement, source_weight,
                                                   normalize=False)
        self.assertClose(numerator.flatten(), torch.tensor([0., 6., 2., 0.]))
        self.assertClose(mass.flatten(), torch.tensor([0., .75, .25, 0.]))

    def test_collisions_are_normalized_and_not_overwritten(self):
        features = torch.tensor([2., 6.]).reshape(1, 1, 1, 2)
        displacement = torch.tensor([.5, -.5, 0., 0.]).reshape(1, 2, 1, 2)
        result, mass = M3.forward_bilinear_splat(features, displacement)
        self.assertClose(result.flatten(), torch.tensor([4., 4.]))
        self.assertClose(mass.flatten(), torch.ones(2))

    def test_out_of_bounds_sources_do_not_wrap(self):
        features = torch.ones(1, 1, 2, 3)
        displacement = torch.full((1, 2, 2, 3), 100.0)
        result, mass = M3.forward_bilinear_splat(features, displacement)
        self.assertEqual(result.abs().sum(), 0)
        self.assertEqual(mass.abs().sum(), 0)

    def test_zero_gate_preserves_baseline_but_can_open(self):
        model = module()
        tokens = torch.randn(1, 16, 4)
        state = model.prepare(tokens, torch.empty(1, 0, 9), training=False)
        result = model.transport(tokens, state, 1, grid(4, 4))
        self.assertEqual(result.abs().sum(), 0)
        result.sum().backward()
        self.assertTrue(torch.isfinite(model.prior_gate.grad).all())
        self.assertGreater(model.prior_gate.grad.abs().sum(), 0)

    def test_future_feature_objective_reaches_mean_and_uncertainty(self):
        model = module(gate_init=.5)
        tokens = torch.randn(1, 16, 4, requires_grad=True)
        state = model.prepare(tokens, torch.empty(1, 0, 9), training=False)
        result = model.transport(tokens, state, 1, grid(4, 4))
        objective_weights = torch.randn_like(result)
        (result * objective_weights).sum().backward()
        grad = model.prior_head[-1].bias.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad[:2].abs().sum(), 1e-7)
        self.assertGreater(grad[2].abs(), 1e-7)
        self.assertGreater(model.transport_reduce.weight.grad.abs().sum(), 0)
        self.assertGreater(tokens.grad.abs().sum(), 0)

    def test_mean_and_off_ablation_are_executable(self):
        tokens = torch.randn(1, 16, 4)
        sigma_model = module(gate_init=.5)
        state = sigma_model.prepare(tokens, torch.empty(1, 0, 9), training=False)
        sigma = sigma_model.transport(tokens, state, 1, grid(4, 4))
        sigma_model.transport_mode = 'mean'
        mean = sigma_model.transport(tokens, state, 1, grid(4, 4))
        self.assertGreater((sigma - mean).abs().sum(), 1e-4)
        sigma_model.transport_mode = 'off'
        off = sigma_model.transport(tokens, state, 1, grid(4, 4))
        self.assertEqual(off.abs().sum(), 0)
        off.sum().backward()
        self.assertIsNotNone(sigma_model.transport_reduce.weight.grad)
        self.assertIsNotNone(sigma_model.transport_expand.weight.grad)
        self.assertIsNotNone(sigma_model.prior_gate.grad)

    def test_zero_horizon_uses_identity_push_then_ego_sampling(self):
        model = module(gate_init=.5)
        tokens = torch.randn(1, 16, 4)
        state = model.prepare(tokens, torch.empty(1, 0, 9), training=False)
        result = model.transport(tokens, state, 0, grid(4, 4))
        features = tokens.transpose(1, 2).reshape(1, 4, 4, 4)
        expected = model.transport_expand(model.transport_reduce(features)).flatten(2).transpose(1, 2)
        self.assertClose(result, expected * torch.tanh(model.prior_gate))

    def test_covariance_controls_preserve_trace_and_do_not_modify_state(self):
        model = module()
        original = torch.tensor([[9., 3.], [3., 4.]]).reshape(1, 2, 2, 1, 1)
        saved = original.clone()
        model.covariance_mode = 'isotropic'
        iso = model._transport_covariance(original).reshape(2, 2)
        self.assertClose(iso, torch.eye(2) * 6.5)
        model.covariance_mode = 'rotated'
        rotated = model._transport_covariance(original).reshape(2, 2)
        self.assertClose(rotated, torch.tensor([[4., -3.], [-3., 9.]]))
        self.assertClose(torch.linalg.eigvalsh(rotated), torch.linalg.eigvalsh(original.reshape(2, 2)))
        self.assertClose(original, saved, atol=0, rtol=0)

    def test_parameter_dtype_casts_keep_physical_state_float32(self):
        # CPU float64 convolutions exercise explicit parameter/input casting.
        # CUDA fp16 kernels still require a separate real-stack smoke test.
        model = module(gate_init=.5).double()
        tokens = torch.randn(1, 16, 4, dtype=torch.float32)
        state = model.prepare(tokens, torch.tensor([[observation()]]), training=False)
        result = model.transport(tokens, state, 1, grid(4, 4))
        self.assertEqual(state['mean'].dtype, torch.float32)
        self.assertEqual(state['cov'].dtype, torch.float32)
        self.assertEqual(result.dtype, tokens.dtype)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
