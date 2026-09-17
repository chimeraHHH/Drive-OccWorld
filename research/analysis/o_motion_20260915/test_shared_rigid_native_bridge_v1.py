"""CPU layout/hook checks; these do not replace an actual native preflight."""
from types import SimpleNamespace
import unittest

import torch

from shared_rigid_native_bridge_v1 import (
    inject_future_object_features, source_field_to_native_layout,
)


class BridgeTests(unittest.TestCase):
    def test_non_square_xyz_and_gradient(self):
        shape = (2, 3, 4)
        x = torch.arange(4 * 24 * 3, dtype=torch.float64).reshape(4, 24, 3).requires_grad_()
        y = source_field_to_native_layout(x, shape)
        self.assertEqual(y.shape, (1, 4, 3, 2, 3, 4))
        # Source flat index is (x*Y+y)*Z+z; channel is the last source axis.
        self.assertEqual(float(y[0, 2, 1, 1, 2, 3]), float(x[2, 23, 1]))
        y[0, 2, 1, 1, 2, 3].backward()
        self.assertEqual(float(x.grad[2, 23, 1]), 1.)
        self.assertEqual(int(torch.count_nonzero(x.grad)), 1)

    def test_zero_future_features_byte_exact_and_additive_gradients(self):
        head = SimpleNamespace(bev_pred_head=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)]), soft_weight=False)
        features = torch.zeros(5, 1, 6, 256)
        features[1, 0, 0, 0] = -0.
        original = features.clone()
        delta = torch.zeros(4, 1, 6, 256, requires_grad=True)
        with inject_future_object_features(head, delta) as calls:
            out = [branch(features) for branch in head.bev_pred_head]
        self.assertEqual(calls, [1, 1, 1])
        self.assertTrue(all(torch.equal(v.view(torch.uint8), features.view(torch.uint8)) for v in out))
        sum(v[1:].sum() for v in out).backward()
        self.assertTrue(torch.equal(delta.grad, torch.full_like(delta, 3.)))
        self.assertTrue(torch.equal(features.view(torch.uint8), original.view(torch.uint8)))
        self.assertTrue(all(not b._forward_pre_hooks for b in head.bev_pred_head))

    def test_nonzero_future_only_and_exception_cleanup(self):
        head = SimpleNamespace(bev_pred_head=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)]), soft_weight=False)
        features = torch.arange(5 * 2 * 256, dtype=torch.float32).reshape(5, 1, 2, 256)
        delta = torch.ones(4, 1, 2, 256)
        with inject_future_object_features(head, delta):
            for branch in head.bev_pred_head:
                actual = branch(features)
                self.assertTrue(torch.equal(actual[:1], features[:1]))
                self.assertTrue(torch.equal(actual[1:], features[1:] + 1))
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with inject_future_object_features(head, delta):
                head.bev_pred_head[0](features)
                raise RuntimeError('sentinel')
        self.assertTrue(all(not b._forward_pre_hooks for b in head.bev_pred_head))
        with self.assertRaisesRegex(ValueError, 'all three'):
            with inject_future_object_features(head, delta):
                head.bev_pred_head[0](features)
        self.assertTrue(all(not b._forward_pre_hooks for b in head.bev_pred_head))


if __name__ == '__main__':
    unittest.main(verbosity=2)
