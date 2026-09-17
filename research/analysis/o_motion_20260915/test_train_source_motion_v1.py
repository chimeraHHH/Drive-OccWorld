"""Bounded CPU contracts; optional real PyTorch loss/VJP tests, no training run."""
import importlib.util
import unittest

import numpy as np

from train_source_motion_v1 import (TRAINING, json_hash, object_group_loss,
                                    planned_orders, protocol_template, schedule_lr)

HAS_TORCH = importlib.util.find_spec('torch') is not None


class ScheduleContracts(unittest.TestCase):
    def test_schedule_boundaries(self):
        self.assertEqual(schedule_lr(0), .0001)
        self.assertEqual(schedule_lr(25), .001)
        self.assertEqual(schedule_lr(511), .0001)
        sequence = [schedule_lr(i) for i in range(512)]
        self.assertTrue(all(sequence[i] < sequence[i+1] for i in range(25)))
        self.assertTrue(all(sequence[i] > sequence[i+1] for i in range(25, 511)))
        for i in (-1, 512, .5):
            with self.assertRaises(ValueError):
                schedule_lr(i)

    def test_fixed_order_and_no_reseeding_each_pass(self):
        orders = planned_orders()
        self.assertEqual(len(orders), 4)
        self.assertTrue(all(sorted(row) == list(range(512)) for row in orders))
        self.assertTrue(all(orders[i] != orders[i+1] for i in range(3)))
        self.assertEqual(json_hash(orders), 'e47327ffc56ed196d1fde628f0b3841031f59553210c93f06636ae87fdbabba2')

    def test_template_not_a_frozen_receipt(self):
        p = protocol_template()
        self.assertEqual(p['status'], 'REVIEW_REQUIRED')
        self.assertEqual(p['training'], TRAINING)
        self.assertTrue(all(v is None for v in p['sources_sha256'].values()))


@unittest.skipUnless(HAS_TORCH, 'PyTorch is unavailable; this is not a backward PASS')
class LossAndGradientContracts(unittest.TestCase):
    def test_unequal_points_objects_groups_and_horizons(self):
        import torch
        # h0: stationary object losses .75 and 3.75 -> group2.25;
        # moving object loss1.75 -> h0 mean2.0. h1 only moving -> .75.
        # Equal valid-horizon average1.375; point/global-object means differ.
        values = np.zeros((4, 6, 3), np.float32)
        values[0] = np.array([1., 4., 4., 4., 2., 2.])[:, None]
        values[1, 4:] = 1.
        pred = torch.tensor(values, requires_grad=True)
        obj = np.array([0, 1, 1, 1, 2, 2], np.int64)
        groups = np.array([[0, 0, 2], [-1, -1, 2], [-1]*3, [-1]*3], np.int8)
        label = dict(target_displacement_m=np.zeros_like(values), object_index=obj,
                     valid=(groups >= 0)[:, obj], object_speed_group=groups)
        loss, audit = object_group_loss(pred, label)
        self.assertAlmostEqual(float(loss), 1.375, places=7)
        self.assertEqual(audit['valid_horizons'], 2)
        loss.backward()
        expected = np.zeros_like(values)
        expected[0, 0] = 1/24
        expected[0, 1:4] = 1/72
        expected[0, 4:] = 1/24
        expected[1, 4:] = 1/12
        np.testing.assert_allclose(pred.grad.numpy(), expected, rtol=1e-6, atol=1e-8)

    def test_smooth_l1_quadratic_beta_and_absent_groups(self):
        import torch
        pred = torch.full((4, 1, 3), .25, requires_grad=True)
        groups = np.array([[1], [-1], [-1], [-1]], np.int8)
        label = dict(target_displacement_m=np.zeros((4, 1, 3), np.float32),
                     object_index=np.array([0], np.int64), valid=groups >= 0,
                     object_speed_group=groups)
        loss, _ = object_group_loss(pred, label)
        self.assertAlmostEqual(float(loss), .0625)
        loss.backward()
        np.testing.assert_allclose(pred.grad.numpy()[0, 0], np.full(3, 1/6), atol=1e-7)
        self.assertEqual(int(torch.count_nonzero(pred.grad[1:])), 0)

    def test_empty_source_and_missing_future_are_differentiable_zero(self):
        import torch
        for n, k in ((0, 0), (2, 1)):
            pred = torch.ones((4, n, 3), requires_grad=True)
            label = dict(target_displacement_m=np.zeros((4, n, 3), np.float32),
                         object_index=np.zeros(n, np.int64), valid=np.zeros((4, n), bool),
                         object_speed_group=np.full((4, k), -1, np.int8))
            loss, audit = object_group_loss(pred, label)
            self.assertTrue(audit['empty_target'])
            self.assertEqual(float(loss), 0.)
            loss.backward()
            self.assertIsNotNone(pred.grad)
            self.assertEqual(int(torch.count_nonzero(pred.grad)), 0)


if __name__ == '__main__':
    unittest.main()
