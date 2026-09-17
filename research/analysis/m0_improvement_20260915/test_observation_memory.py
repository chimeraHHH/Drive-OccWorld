"""CPU tests, without importing MMCV or compiling any CUDA extension.

python test_observation_memory.py
Set M0_SOURCE_FILE to the frozen detector file when running from a deployment.
The NumPy/AST tests run without torch; torch tests are explicitly skipped then.
Passing these tests is NOT a real M0/CUDA forward-equivalence receipt.
"""
import ast
import copy
import hashlib
import os
import unittest
from pathlib import Path

import numpy as np
import observation_memory as om

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    torch = None


DEFAULT_SOURCE = (Path(__file__).resolve().parents[2] /
                  "code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/"
                  "bevformer/detectors/drive_occworld.py")
SOURCE = Path(os.environ.get("M0_SOURCE_FILE", str(DEFAULT_SOURCE)))


def source_method():
    text = SOURCE.read_text()
    node = next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name == "future_pred")
    return node, ast.get_source_segment(text, node)


class StaticContractTests(unittest.TestCase):
    def test_exact_source_and_only_three_edits(self):
        node, source = source_method()
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), om.EXPECTED_SOURCE_FILE_SHA256)
        self.assertEqual(om._ast_digest(node), om.EXPECTED_FUTURE_PRED_AST_SHA256)
        for mode in om.MODES:
            fn, receipt = om.derive_future_pred(source, {}, mode)
            self.assertEqual(receipt["source_ast_sha256"], om.EXPECTED_FUTURE_PRED_AST_SHA256)
            self.assertEqual(receipt["structural_changes"],
                             dict(input_guard=1, initial_duplication=int(mode != "native1"),
                                  paired_update=int(mode != "native1")))
            # Identical signatures: no action/metadata arguments dropped.
            self.assertEqual(tuple(__import__('inspect').signature(fn).parameters),
                             tuple(a.arg for a in node.args.args))
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            om.derive_future_pred(source.replace("range(1, future_frame_num + 1)",
                                                 "range(2, future_frame_num + 1)"), {}, "persistent2")

    def test_python310_and_python314_canonicalization(self):
        node, _ = source_method()
        digest = om._ast_digest(node)
        self.assertEqual(digest, "2593f6549742ec99c7c0b9877d63361c57f3126012626f51b7e96db7d494cbba")
        text = om._canonical_ast_dump(node)
        self.assertIn("keywords=[]", text)
        self.assertIn("orelse=[]", text)
        self.assertIn("decorator_list=[]", text)
        self.assertNotIn("type_params=", text)
        # Simulate the additional optional field, independently of host Python.
        node.type_params = []
        node._fields = tuple(k for k in node._fields if k != 'type_params') + ('type_params',)
        self.assertEqual(om._ast_digest(node), digest)
        node.type_params = [ast.Name(id='UnapprovedGeneric', ctx=ast.Load())]
        with self.assertRaisesRegex(ValueError, 'Generic type parameters'):
            om._ast_digest(node)

    def test_head_level_point_row_order(self):
        # Unique entries encode head, point, xy and feature-input column.
        for heads, points, components in ((3, 4, 2), (2, 5, 1)):
            old = np.arange(heads * points * components * 7).reshape(-1, 7)
            expanded = old[om.projection_row_indices(heads, points, components)]
            shaped = expanded.reshape(heads, 2, points, components, 7)
            for h in range(heads):
                for level in range(2):
                    np.testing.assert_array_equal(shaped[h, level],
                                                  old.reshape(heads, points, components, 7)[h])
            # This incorrect tempting implementation must be observably different.
            self.assertFalse(np.array_equal(np.tile(old, (2, 1)), expanded))

    def test_memory_contents_by_horizon(self):
        self.assertEqual([om.memory_plan("persistent2", t) for t in range(1, 5)],
                         [(0, 0), (0, 1), (0, 2), (0, 3)])
        self.assertEqual([om.memory_plan("rolling2", t) for t in range(1, 5)],
                         [(0, 0), (0, 1), (1, 2), (2, 3)])
        self.assertEqual([om.memory_plan("native1", t) for t in range(1, 5)],
                         [(0,), (1,), (2,), (3,)])

    def test_independent_row_vector_se3_geometry(self):
        # Absolute column-vector ego poses: source coordinates = inv(Psource) Ptarget q.
        # Native matrices are transposed and multiplied in row-vector order.
        poses = []
        for x, y, yaw in ((0, 0, 0), (2, 0, 0), (3, 1, np.pi/2), (4, -2, -np.pi/4), (6, 1, .2)):
            p = np.eye(4); c, s = np.cos(yaw), np.sin(yaw)
            p[:2, :2] = [[c, -s], [s, c]]; p[:2, 3] = [x, y]; poses.append(p)
        q = np.array([1.3, -2.7, .4, 1.])
        for mode in om.MODES:
            for t in range(1, 5):
                for source in om.memory_plan(mode, t):
                    native = q @ poses[t].T @ np.linalg.inv(poses[source]).T
                    direct = np.linalg.solve(poses[source], poses[t] @ q)
                    np.testing.assert_allclose(native, direct, atol=1e-12)
        # A static world point moves -x when ego moves +x.
        world = np.array([7., 3., 0., 1.])
        np.testing.assert_allclose(np.linalg.solve(poses[1], world), [5, 3, 0, 1])


if torch is not None:
    class ToyAttention(nn.Module):
        """Independent differentiable multi-level bilinear reference sampler."""
        def __init__(self, c=4, heads=2, points=3):
            super().__init__()
            self.num_heads, self.num_points, self.num_levels = heads, points, 1
            self.sampling_offsets = nn.Linear(c, heads*points*2)
            self.attention_weights = nn.Linear(c, heads*points)
            self.value_proj, self.output_proj = nn.Linear(c, c), nn.Linear(c, c)

        def forward(self, query, memory, refs, height, width):
            b, length, c = query.shape; levels = memory.shape[1]
            h, p, d = self.num_heads, self.num_points, c // self.num_heads
            off = self.sampling_offsets(query).reshape(b, length, h, levels, p, 2)
            weight = self.attention_weights(query).reshape(b, length, h, levels*p)
            weight = weight.softmax(-1).reshape(b, length, h, levels, p)
            value = self.value_proj(memory).reshape(b, levels, height, width, h, d)
            result = query.new_zeros(b, length, h, d)
            for level in range(levels):
                value_l = value[:, level].permute(0, 3, 4, 1, 2).reshape(b*h, d, height, width)
                grid = refs[:, :, None, level, None, :] + off[:, :, :, level] / query.new_tensor([width, height])
                grid = (2*grid-1).permute(0, 2, 1, 3, 4).reshape(b*h, length, p, 2)
                sampled = F.grid_sample(value_l, grid, align_corners=False,
                                        mode="bilinear", padding_mode="zeros")
                sampled = sampled.reshape(b, h, d, length, p).permute(0, 3, 1, 4, 2)
                result = result + (sampled * weight[:, :, :, level, :, None]).sum(3)
            return query + self.output_proj(result.reshape(b, length, c))

    class ToyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.operation_order = ("self_attn", "norm", "cross_attn", "norm", "cross_attn_action")
            self.attentions = nn.ModuleList([nn.Identity(), ToyAttention(), nn.Identity()])

    class ToyHead(nn.Module):
        def __init__(self, height=3, width=5):
            super().__init__()
            self.memory_queue_len, self.history_queue_length, self.embed_dims = 1, 2, 4
            self.height, self.width, self.use_plan_traj = height, width, False
            self.prev_render_neck = None
            self.prev_frame_embedding = nn.Parameter(torch.randn(1, 4))
            self.bev_pred_head = nn.ModuleList([nn.Identity()])
            self.query = nn.Parameter(torch.randn(1, height*width, 4))
            self.transformer = nn.Module(); self.transformer.decoder = nn.Module()
            self.transformer.decoder.layers = nn.ModuleList([ToyLayer(), ToyLayer()])
            self.trace = []

        def forward(self, memory, img_metas, step, action, condition, **kwargs):
            refs = kwargs['ref_points']
            self.trace.append(dict(memory=memory.detach().clone(), refs=refs.detach().clone(),
                                   grad_enabled=torch.is_grad_enabled()))
            x = self.query.expand(memory.shape[0], -1, -1) + .05*step
            val = memory + self.prev_frame_embedding[None, :, None, :]
            for layer in self.transformer.decoder.layers:
                x = layer.attentions[1](x, val, refs, self.height, self.width)
            return x.unsqueeze(0), None

        def forward_head(self, features):
            return features

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__(); self.future_pred_head = ToyHead()
            self.memory_queue_len = 1
            self.bev_h, self.bev_w = 3, 5
            self.future_pred_frame_num = self.test_future_frame_num = 4
            self.turn_on_plan = self.predict_flow = False
            self.motion_residual = self.doppler_posterior = self.doppler_advection = None
            self.trace_transforms = []

        def _get_history_ref_to_previous_transform(self, tensor, count, old_meta, ref_meta):
            assert count == 1
            return torch.eye(4, dtype=tensor.dtype).repeat(tensor.shape[0], 1, 1, 1)

        def _align_bev_coordnates(self, step, transforms, img_metas, plan_traj):
            b, levels = transforms.shape[:2]
            self.trace_transforms.append(transforms.detach().clone())
            h, w = self.future_pred_head.height, self.future_pred_head.width
            yy, xx = torch.meshgrid((torch.arange(h)+.5)/h, (torch.arange(w)+.5)/w, indexing='ij')
            grid = torch.stack((xx, yy), -1).reshape(1, h*w, 2).to(transforms).repeat(b, 1, 1)
            # Deliberate nonzero ego translation, row-vector convention.
            f2r = torch.eye(4, dtype=transforms.dtype).repeat(b, 1, 1)
            f2r[:, 3, 0] = .01*step; f2r[:, 3, 1] = -.02*step
            r2f = torch.linalg.inv(f2r)
            f2hist = f2r[:, None] @ transforms
            q = torch.cat((grid, grid.new_zeros(b, h*w, 1), grid.new_ones(b, h*w, 1)), -1)
            refs = (q[:, None] @ f2hist)[..., :2].permute(0, 2, 1, 3).contiguous()
            return grid, refs, r2f, f2hist.transpose(-1, -2), grid

    # Compile the ACTUAL native function with original source locations, without
    # importing the detector module (which would import MMCV/CUDA extensions).
    method_node, _source_text = source_method()
    _namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method_node], type_ignores=[]), str(SOURCE), 'exec'), _namespace)
    ToyModel.future_pred = _namespace['future_pred']


@unittest.skipIf(torch is None, "torch unavailable; real tensor tests still required on server")
class TensorContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.base = ToyModel().double()

    def run_model(self, model, x, valid=(1, 2, 3, 4), condition=None):
        return model.future_pred(x, {}, {"occ_gts": None} if condition is None else condition,
                                 {"ref_pose_pred": None, "gt_traj": torch.zeros(x.shape[0], 4, 2)},
                                 list(valid), [{} for _ in range(x.shape[0])],
                                 [[{}, {}, {}] for _ in range(x.shape[0])], 3)[0]

    def test_no_rng_and_full_native_and_first_two_steps(self):
        x = torch.randn(2, 1, 15, 4, dtype=torch.float64)
        native_output = self.run_model(self.base, x)
        outputs, receipts = {}, {}
        for mode in om.MODES:
            model = copy.deepcopy(self.base)
            rng = torch.get_rng_state().clone()
            receipts[mode] = om.install_observation_memory(model, mode)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(model.memory_queue_len, 1)
            self.assertEqual(model.future_pred_head.history_queue_length, 2)
            self.assertTrue(all(p.requires_grad for p in model.future_pred_head.parameters()))
            outputs[mode] = self.run_model(model, x)
        torch.testing.assert_close(outputs['native1'], native_output, rtol=0, atol=0)
        for mode in ('persistent2', 'rolling2'):
            torch.testing.assert_close(outputs[mode][:2], native_output[:2], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(outputs['persistent2'][:3], outputs['rolling2'][:3], rtol=0, atol=0)
        self.assertGreater((outputs['persistent2'][3:] - outputs['rolling2'][3:]).abs().max().item(), 1e-8)
        self.assertEqual(receipts['persistent2']['head_parameters_after'], receipts['rolling2']['head_parameters_after'])

    def test_native_no_grad_policy_and_gt_rejection(self):
        model = copy.deepcopy(self.base); om.install_observation_memory(model, 'persistent2')
        self.run_model(model, torch.randn(1, 1, 15, 4, dtype=torch.float64), valid=(1, 3))
        self.assertEqual([r['grad_enabled'] for r in model.future_pred_head.trace], [True, False, True, False])
        with self.assertRaisesRegex(ValueError, 'GT occupancy'):
            self.run_model(model, torch.zeros(1, 1, 15, 4), condition={'occ_gts': torch.zeros(1)})
        with self.assertRaisesRegex(ValueError, 'already installed'):
            om.install_observation_memory(model, 'rolling2')

    def test_feature_and_coordinate_pairing_and_gradient(self):
        initial = torch.tensor([[[[2.]]]], dtype=torch.float64, requires_grad=True)
        init_matrix = torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4)
        for mode in ('persistent2', 'rolling2'):
            f, m = om.initialize_memory(initial, init_matrix)
            for step in range(1, 5):
                expected = om.memory_plan(mode, step)
                torch.testing.assert_close(f.flatten(), torch.tensor([2.+i for i in expected], dtype=f.dtype))
                torch.testing.assert_close(m[0, :, 3, 0], torch.tensor([-i for i in expected], dtype=m.dtype))
                new = initial[:, 0] + step
                matrix = torch.eye(4, dtype=m.dtype).unsqueeze(0); matrix[:, 3, 0] = -step
                f, m = om.update_memory(f, m, new, matrix, mode)
            (grad,) = torch.autograd.grad(f.sum(), initial, retain_graph=True)
            torch.testing.assert_close(grad, torch.full_like(initial, 2.))

    def test_end_to_end_gradients_reach_both_slots(self):
        for mode in ('persistent2', 'rolling2'):
            model = copy.deepcopy(self.base); om.install_observation_memory(model, mode)
            x = torch.randn(1, 1, 15, 4, dtype=torch.float64, requires_grad=True)
            output = self.run_model(model, x)
            output[-1].square().mean().backward()
            self.assertGreater(x.grad.abs().sum().item(), 0.)
            self.assertTrue(torch.isfinite(x.grad).all())
            grad = model.future_pred_head.prev_frame_embedding.grad
            self.assertTrue((grad.abs().sum(-1) > 0).all())
            for _, module in om._cross_attention_modules(model.future_pred_head):
                self.assertTrue(torch.isfinite(module.attention_weights.weight.grad).all())
                grouped = module.attention_weights.weight.grad.reshape(module.num_heads, 2, module.num_points, -1)
                self.assertTrue((grouped.abs().sum((0, 2, 3)) > 0).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
