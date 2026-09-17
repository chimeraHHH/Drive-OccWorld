"""Small physical/connection contracts; actual CPU torch execution required."""
import unittest
import torch

from supported_motion_fusion import SupportedMotionFusion, supported_transport


class SupportedFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.extent = (0., 0., 0., 3., 2., 1.)
        self.p = torch.zeros(1, 1, 3, 2, 1)
        self.p[0, 0, 0, 0, 0] = .8
        self.p[0, 0, 2, 1, 0] = .4  # model background: must never be a source
        self.mask = self.p > .5
        self.d = torch.zeros(1, 4, 3, 3, 2, 1)
        self.o = torch.full((1, 4, 1, 3, 2, 1), -.7)

    def test_source_prediction_mask_and_physical_translation(self):
        self.d[:, :, 0] = 1.
        fields = supported_transport(self.p, self.mask, self.d, self.extent)
        self.assertTrue(torch.equal(fields['support_weight'][:, :, 0, 1, 0, 0], torch.ones(1, 4)))
        self.assertTrue(torch.allclose(fields['probability_mass'].sum((2,3,4,5)), torch.full((1,4), .8)))
        self.assertTrue(torch.equal(fields['source_foreground_voxels'], torch.ones(1,1)))

    def test_initial_half_gate_and_exact_O_fallback(self):
        fields = supported_transport(self.p, self.mask, self.d, self.extent)
        result = SupportedMotionFusion()(self.o, fields['probability_mass'], fields['support_weight'])
        self.assertTrue(torch.equal(result['gate'], torch.full_like(self.o, .5)))
        selected = result['destination_support']
        self.assertTrue(torch.equal(result['logodds'][~selected], self.o[~selected]))
        expected = .5*(-.7) + .5*torch.log(torch.tensor(.8/.2))
        self.assertTrue(torch.allclose(result['logodds'][selected], expected.expand(4)))
        disabled = SupportedMotionFusion()(self.o, fields['probability_mass'], fields['support_weight'], disabled=True)
        self.assertTrue(torch.equal(disabled['logodds'], self.o))

    def test_empty_and_outside_support_return_O(self):
        for mask, dx in [(torch.zeros_like(self.mask), 0.), (self.mask, 20.)]:
            displacement = self.d.clone(); displacement[:,:,0] = dx
            fields = supported_transport(self.p, mask, displacement, self.extent)
            result = SupportedMotionFusion()(self.o, fields['probability_mass'], fields['support_weight'])
            self.assertEqual(int(result['destination_support'].sum()), 0)
            self.assertTrue(torch.equal(result['logodds'], self.o))

    def test_occupancy_gradient_reaches_motion_and_gate(self):
        d = self.d.clone(); d[:,:,0] = .25; d.requires_grad_(True)
        model = SupportedMotionFusion()
        fields = supported_transport(self.p, self.mask, d, self.extent)
        result = model(self.o, fields['probability_mass'], fields['support_weight'])
        # One destination, not mass-conserving global sum, resolves direction.
        loss = (result['logodds'][:,:,0,1,0,0] - 1.).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(d.grad).all())
        self.assertGreater(float(d.grad.abs().max()), 0.)
        self.assertGreater(float(model.gate[-1].weight.grad.abs().max()), 0.)

    def test_fixed_gate_recomputes_geometry_and_fallback(self):
        model = SupportedMotionFusion(); self.d[:,:,0] = 1.
        factual = supported_transport(self.p, self.mask, self.d, self.extent)
        result = model(self.o, factual['probability_mass'], factual['support_weight'])
        zero = supported_transport(self.p, self.mask, torch.zeros_like(self.d), self.extent)
        changed = model(self.o, zero['probability_mass'], zero['support_weight'], fixed_gate=result['gate'].detach())
        self.assertTrue(torch.equal(changed['logodds'][:,:,0,1,0,0], self.o[:,:,0,1,0,0]))
        self.assertFalse(torch.equal(result['logodds'], changed['logodds']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
