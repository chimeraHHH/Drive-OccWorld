"""Tiny analytic CPU Torch contracts; no data/model/checkpoint dependencies."""
import types
import unittest

import torch
from torch import nn

from future_feature_residual_v1 import (FutureFeatureResidual, PARAMETERS,
    capture_O_terminal, compose_prediction, exact32, projected_tokens_to_xyz,
    transport_features)


class TinyNative:
    """The frozen oracle helper's identical axis algebra, with non-square sizes."""
    shape = (3, 5, 2)

    @classmethod
    def predictions_to_xyz(cls, prediction):
        x, y, z = cls.shape
        return prediction[:, -1, 0, 0].reshape(5, y, x, z, 2).permute(0, 4, 2, 1, 3)

    @classmethod
    def replace_last_logits(cls, original, xyz):
        x, y, z = cls.shape
        output = original.clone()
        output[:, -1, 0, 0] = xyz.permute(0, 3, 2, 4, 1).reshape(5, x*y, z, 2)
        return output


class FutureFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)
        self.shape = (3, 5, 2)
        self.extent = (0., 0., 0., 3., 5., 2.)

    def inputs(self):
        x, y, z = self.shape
        return (torch.randn(5, 1, x*y, 256), torch.ones(1, 1, x, y, z, dtype=torch.bool),
                torch.zeros(1, 4, 3, x, y, z), torch.randn(1, 4, 1, x, y, z))

    def test_parameter_count_zero_initialization_and_no_normalization(self):
        native = FutureFeatureResidual()
        self.assertEqual(sum(p.numel() for p in native.parameters()), PARAMETERS)
        self.assertEqual(PARAMETERS, 41137)
        self.assertFalse(any(isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)) for m in native.modules()))
        model = FutureFeatureResidual(self.shape, self.extent)
        args = self.inputs()
        result = model(*args)
        self.assertEqual(tuple(result['residual'].shape), (1, 4, 1, 3, 5, 2))
        self.assertEqual(torch.count_nonzero(result['residual']).item(), 0)
        empty = list(args); empty[1] = torch.zeros_like(args[1])
        output = model(*empty)
        self.assertEqual(torch.count_nonzero(output['support_weight']).item(), 0)
        self.assertTrue(torch.isfinite(output['residual']).all())

    def test_non_square_token_YX_height_channel_axes(self):
        x, y, z = self.shape
        projected = torch.arange(5*2*x*y*z*8, dtype=torch.float32).reshape(5, 2, y*x, z*8)
        xyz = projected_tokens_to_xyz(projected, self.shape)
        self.assertEqual(tuple(xyz.shape), (2, 5, 8, x, y, z))
        for b in range(2):
            for t in range(5):
                for i, j, k, c in [(0, 0, 0, 0), (2, 4, 1, 7), (1, 3, 0, 4)]:
                    self.assertEqual(float(xyz[b, t, c, i, j, k]),
                                     float(projected[t, b, j*x+i, k*8+c]))

    def test_masked_zero_translation_and_dropped_sources(self):
        source = torch.full((1, 8, 4, 3, 2), 99.)
        source[:, :, 0, 1, 1] = 2.; source[:, :, 3, 2, 0] = 6.
        mask = torch.zeros(1, 1, 4, 3, 2, dtype=torch.bool)
        mask[:, :, 0, 1, 1] = True; mask[:, :, 3, 2, 0] = True
        flow = torch.zeros(1, 4, 3, 4, 3, 2)
        extent = (0., 0., 0., 4., 3., 2.)
        identity = transport_features(source, mask, flow, extent)
        expected = (source*mask).unsqueeze(1).expand(-1, 4, -1, -1, -1, -1)
        self.assertTrue(torch.equal(identity['transported_features'], expected))
        flow[:, :, 0] = 1.
        moved = transport_features(source, mask, flow, extent)
        self.assertTrue(torch.equal(moved['transported_features'][:, :, :, 1, 1, 1], torch.full((1, 4, 8), 2.)))
        self.assertEqual(float(moved['support_weight'].sum()), 4.)
        self.assertTrue(torch.equal(moved['outside_support_weight'], torch.ones(1, 4)))
        self.assertEqual(torch.count_nonzero(moved['transported_features']).item(), 4*8)

    def test_collision_average_empty_and_tiny_positive_weight(self):
        source = torch.zeros(1, 8, 3, 1, 1, dtype=torch.float64)
        source[:, :, 0] = 2.; source[:, :, 2] = 6.
        mask = torch.zeros(1, 1, 3, 1, 1, dtype=torch.bool); mask[:, :, [0, 2]] = True
        flow = torch.zeros(1, 4, 3, 3, 1, 1, dtype=torch.float64)
        flow[:, :, 0, 0] = 1.; flow[:, :, 0, 2] = -1.
        extent = (0., 0., 0., 3., 1., 1.)
        out = transport_features(source, mask, flow, extent)
        self.assertTrue(torch.equal(out['support_weight'][:, :, :, 1], torch.full((1, 4, 1, 1, 1), 2., dtype=torch.float64)))
        self.assertTrue(torch.equal(out['transported_features'][:, :, :, 1], torch.full((1, 4, 8, 1, 1), 4., dtype=torch.float64)))
        self.assertTrue(torch.equal(out['collision_support_excess'], torch.ones(1, 4, dtype=torch.float64)))
        empty = transport_features(source, torch.zeros_like(mask), flow, extent)
        self.assertEqual(torch.count_nonzero(empty['transported_features']).item(), 0)
        self.assertTrue(torch.isfinite(empty['transported_features']).all())
        mask[:, :, 2] = False; flow.zero_(); flow[:, :, 0, 0] = 1.-1e-7
        tiny = transport_features(source, mask, flow, extent, normalization_eps=1e-6)
        weight = tiny['support_weight'][0, 0, 0, 0, 0, 0]
        self.assertGreater(float(weight), 0.); self.assertLess(float(weight), 1e-6)
        self.assertAlmostEqual(float(tiny['transported_features'][0, 0, 0, 0, 0, 0]), float(2.*weight/1e-6), places=14)

    def test_zero_signed_byte_identity_and_live_residual_gradient(self):
        x, y, z = self.shape
        original = torch.randn(5, 3, 1, 1, x*y, z, 2)
        original[1, -1, 0, 0, 0, 0, 1] = -0.
        base = TinyNative.predictions_to_xyz(original)
        residual = torch.zeros(1, 4, 1, x, y, z, requires_grad=True)
        prediction = compose_prediction(original, base, residual, TinyNative)
        self.assertTrue(exact32(prediction, original))
        prediction.sum().backward()
        self.assertTrue(torch.equal(residual.grad, torch.ones_like(residual)))
        signed = residual.detach().clone(); signed[:, 0] = .25; signed[:, 1] = -.5
        result = compose_prediction(original, base, signed, TinyNative)
        xyz = TinyNative.predictions_to_xyz(result)
        self.assertTrue(torch.equal(xyz[1:3, 1], base[1:3, 1]+signed[0, :2, 0]))
        self.assertTrue(exact32(result[0], original[0]))
        self.assertTrue(exact32(result[:, :-1], original[:, :-1]))

    def test_gradients_initial_final_readout_then_projection_and_trunk(self):
        model = FutureFeatureResidual(self.shape, self.extent)
        args = self.inputs()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        residual = model(*args)['residual']
        (residual-1.).square().mean().backward()
        self.assertGreater(float(model.decoder[-1].weight.grad.abs().max()), 0.)
        self.assertEqual(float(model.projection.weight.grad.abs().max()), 0.)
        self.assertEqual(float(model.decoder[0].weight.grad.abs().max()), 0.)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        (model(*args)['residual']-1.).square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(model.projection.weight.grad.abs().max()), 0.)
        self.assertGreater(float(model.decoder[0].weight.grad.abs().max()), 0.)
        # Future context remains a real correction path at empty transport.
        empty = list(args); empty[1] = torch.zeros_like(args[1])
        self.assertGreater(float(model(*empty)['residual'].abs().max()), 0.)

    def test_capture_is_clone_once_and_hook_removed_on_failure(self):
        class Head(nn.Module):
            def __init__(self):
                super().__init__(); self.soft_weight = False
                self.bev_pred_head = nn.ModuleList([nn.Identity() for _ in range(3)])
        class O(nn.Module):
            def __init__(self):
                super().__init__(); self.future_pred_head = Head()
        model = O().eval()
        features = torch.randn(5, 1, 15, 256)
        original = torch.randn(5, 3, 1, 1, 15, 2, 2)
        def replay(m, sample, training):
            self.assertFalse(training)
            m.future_pred_head.bev_pred_head[-1](features)
            return original, None
        result, cloned = capture_O_terminal(model, {}, types.SimpleNamespace(replay=replay),
                                            expected_shape=features.shape)
        self.assertIs(result, original)
        self.assertTrue(exact32(cloned, features))
        self.assertNotEqual(cloned.data_ptr(), features.data_ptr())
        self.assertFalse(model.future_pred_head.bev_pred_head[-1]._forward_pre_hooks)
        def fail(m, sample, training):
            m.future_pred_head.bev_pred_head[-1](features)
            raise RuntimeError('deliberate tiny replay failure')
        with self.assertRaises(RuntimeError):
            capture_O_terminal(model, {}, types.SimpleNamespace(replay=fail), expected_shape=features.shape)
        self.assertFalse(model.future_pred_head.bev_pred_head[-1]._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main(verbosity=2)
