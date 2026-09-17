"""Tiny CPU engineering tests only, not native-cache or model-result fixtures.

Torch is required: absence is an import error, never a silently skipped test.
Run from a staged directory with the trainer and its pure source/fusion helpers.
"""
import copy
import unittest

import numpy as np
import torch

import train_connected_motion_v1 as trainer
import train_source_motion_v1 as helper
from supported_motion_fusion import SupportedMotionFusion, supported_transport


class ConnectedMotionTests(unittest.TestCase):
    def test_registered_schedule_and_single_order(self):
        self.assertEqual(trainer.learning_rates(0, helper), {'motion': 1e-5, 'gate': 1e-4})
        self.assertEqual(trainer.learning_rates(25, helper), {'motion': 1e-4, 'gate': 1e-3})
        self.assertEqual(trainer.learning_rates(511, helper), {'motion': 1e-5, 'gate': 1e-4})
        orders = helper.planned_orders()
        self.assertEqual(helper.json_hash(orders), 'e47327ffc56ed196d1fde628f0b3841031f59553210c93f06636ae87fdbabba2')
        self.assertEqual(len(orders), 4)
        for order in orders:
            self.assertEqual(sorted(order), list(range(512)))
        self.assertEqual(trainer.TRAINING['physical_loss_coefficient'], 1.)
        self.assertEqual(trainer.TRAINING['occupancy_loss_coefficient'], 1.)

    def test_shared_value_connection_and_gradient_additivity(self):
        torch.manual_seed(11)
        shape = (3, 4, 2); extent = (0., 0., 0., 3., 4., 2.)
        initial = torch.tensor([.17, .23, .11]).reshape(1, 1, 3, 1, 1, 1).expand(1, 4, 3, *shape).clone()
        motion = {arm: torch.nn.Parameter(initial.clone()) for arm in trainer.ARMS}
        gate = {'J': SupportedMotionFusion()}; gate['D'] = copy.deepcopy(gate['J'])
        probability = torch.zeros(1, 1, *shape); probability[0, 0, 1, 1, 0] = .8
        foreground = probability > 0
        future = torch.zeros(1, 4, 1, *shape)
        source = supported_transport(probability, foreground, motion['J'], extent)
        fields = {'J': source, 'D': {k: v.detach() for k, v in source.items()}}
        values, vp, vg = {}, {}, {}
        for arm in trainer.ARMS:
            params = [motion[arm]] + list(gate[arm].parameters())
            fused = gate[arm](future, fields[arm]['probability_mass'], fields[arm]['support_weight'])
            values[arm] = fused['logodds'].detach()
            physical = torch.nn.functional.smooth_l1_loss(motion[arm], torch.full_like(initial, .6), beta=.5)
            occupancy = fused['logodds'].square().mean()
            gp = torch.autograd.grad(physical, params, allow_unused=True, retain_graph=True)
            go = torch.autograd.grad(occupancy, params, allow_unused=True, retain_graph=True)
            gt = torch.autograd.grad(physical + occupancy, params, allow_unused=True)
            self.assertTrue(all(g is None for g in gp[1:]))
            if arm == 'D': self.assertIsNone(go[0])
            else: self.assertGreater(float(go[0].norm()), 0.)
            lhs = trainer.gradient_vector(gt, params)
            rhs = trainer.gradient_vector(gp, params) + trainer.gradient_vector(go, params)
            self.assertTrue(torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-7))
            vp[arm] = gp[0]; vg[arm] = trainer.gradient_vector(go[1:], params[1:])
            self.assertTrue(all(p.grad is None for p in params))
        self.assertTrue(torch.equal(values['J'], values['D']))
        self.assertTrue(torch.equal(vp['J'], vp['D']))
        self.assertTrue(torch.equal(vg['J'], vg['D']))

    def test_actual_detach_boundary_preserves_values_and_physical_path(self):
        for arm in trainer.ARMS:
            parameter = torch.nn.Parameter(torch.tensor([.2, .4, .7]))
            field = trainer.transport_displacement(parameter, arm)
            self.assertTrue(torch.equal(field, parameter))
            self.assertEqual(field.requires_grad, arm == 'J')
            if arm == 'J': self.assertIs(field, parameter)
            physical = parameter.square().sum()
            gate = torch.nn.Parameter(torch.tensor(.5))
            occupancy = (gate * field).sum()
            go = torch.autograd.grad(occupancy, parameter, allow_unused=True, retain_graph=True)[0]
            self.assertEqual(go is None, arm == 'D')
            (physical + occupancy).backward()
            expected = 2*parameter.detach() + (.5 if arm == 'J' else 0.)
            self.assertTrue(torch.equal(parameter.grad, expected))
        with self.assertRaises(ValueError): trainer.transport_displacement(parameter, 'Z')

    def test_empty_physical_sample_keeps_zero_gradient_graph(self):
        sparse = torch.empty((4, 0, 3), requires_grad=True)
        label = dict(target_displacement_m=np.empty((4, 0, 3), np.float32), object_index=np.empty(0, np.int64),
                     valid=np.empty((4, 0), bool), object_speed_group=np.empty((4, 0), np.int8))
        loss, audit = helper.object_group_loss(sparse, label)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue(audit['empty_target'])
        self.assertEqual(audit['valid_horizons'], 0)
        loss.backward()
        self.assertIsNotNone(sparse.grad)
        self.assertEqual(tuple(sparse.grad.shape), (4, 0, 3))

    def test_separate_moments_and_clip_groups(self):
        networks = {a: {k: torch.nn.Parameter(torch.tensor([1., -2.])) for k in ('motion', 'gate')} for a in trainer.ARMS}
        optimizers = {a: {k: torch.optim.AdamW([v], lr=trainer.learning_rates(0, helper)[k],
            betas=(.9, .999), eps=1e-8, weight_decay=.01) for k, v in row.items()} for a, row in networks.items()}
        for arm in trainer.ARMS:
            for component, parameter in networks[arm].items():
                parameter.grad = torch.tensor([100., 200.]) if component == 'motion' else torch.tensor([300., 400.])
                cap = trainer.TRAINING[component+'_grad_clip']
                norm = torch.nn.utils.clip_grad_norm_([parameter], cap)
                self.assertGreater(float(norm), cap)
                self.assertLessEqual(float(parameter.grad.norm()), cap)
                optimizers[arm][component].step()
        states = [opt.state[next(iter(opt.state))] for row in optimizers.values() for opt in row.values()]
        self.assertEqual(len({s['exp_avg'].data_ptr() for s in states}), 4)
        self.assertTrue(torch.equal(networks['J']['motion'], networks['D']['motion']))
        self.assertTrue(torch.equal(networks['J']['gate'], networks['D']['gate']))
        for state in states: self.assertEqual(float(state['step']), 1.)

    def test_comparison_fails_nonfinite_and_outside_tolerance(self):
        left = torch.tensor([1., 0.], dtype=torch.float32)
        near = left + torch.tensor([1e-6, 5e-8])
        self.assertTrue(trainer.tensor_comparison(left, near, rtol=1e-5, atol=1e-7)['passed'])
        self.assertFalse(trainer.tensor_comparison(left, left+1e-2, rtol=1e-5, atol=1e-7)['passed'])
        with self.assertRaises(ValueError):
            trainer.tensor_comparison(left, torch.tensor([float('nan'), 0.]), rtol=1e-5, atol=1e-7)
        with self.assertRaises(ValueError):
            trainer.gradient_vector([torch.tensor(float('inf'))], [torch.nn.Parameter(torch.tensor(0.))])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
