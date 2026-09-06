"""CPU regression tests; load selected AST nodes without importing MMCV.

Run from the repository root:
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tests/test_occ_target_downsample.py

Loss/gradient tests use the production loss methods with only the fixed grid
constant changed from (256, 256, 20) to (4, 4, 2) in their in-memory AST. All
loss functions, label rules, interpolation and decoder-layer loops are real.
The helper's label tests use its unmodified production AST.
"""

import ast
import importlib.util
import itertools
from pathlib import Path
import unittest
from unittest import mock

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
HEAD_PATH = ROOT / 'projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py'
LOSS_PATH = ROOT / 'projects/mmdet3d_plugin/bevformer/losses'
TEST_GRID = (4, 4, 2)


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEM = _load_file('occ_test_semkitti_loss', LOSS_PATH / 'semkitti_loss.py')
LOVASZ = _load_file('occ_test_lovasz_softmax', LOSS_PATH / 'lovasz_softmax.py')


class _SmallLossGrid(ast.NodeTransformer):
    def visit_Tuple(self, node):
        if (len(node.elts) == 3 and
                all(isinstance(item, ast.Constant) for item in node.elts) and
                tuple(item.value for item in node.elts) == (256, 256, 20)):
            return ast.copy_location(
                ast.Tuple(elts=[ast.Constant(value=n) for n in TEST_GRID],
                          ctx=ast.Load()), node)
        return self.generic_visit(node)


def _load_production_nodes(small_loss_grid=False):
    tree = ast.parse(HEAD_PATH.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and
                  node.name == '_downsample_occ_target')
    head = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'WorldHeadV1')
    methods = [node for node in head.body if isinstance(node, ast.FunctionDef)
               and node.name in ('loss_voxel', 'loss_occ')]
    for method in methods:
        method.decorator_list = []
    stub = ast.ClassDef(name='LossHead', bases=[], keywords=[], body=methods,
                        decorator_list=[])
    module = ast.Module(body=[helper, stub], type_ignores=[])
    if small_loss_grid:
        module = _SmallLossGrid().visit(module)
    ast.fix_missing_locations(module)
    namespace = {
        'torch': torch,
        'F': F,
        'CE_ssc_loss': SEM.CE_ssc_loss,
        'sem_scal_loss': SEM.sem_scal_loss,
        'geo_scal_loss': SEM.geo_scal_loss,
        'lovasz_softmax': LOVASZ.lovasz_softmax,
    }
    exec(compile(module, str(HEAD_PATH), 'exec'), namespace)
    return namespace


PRODUCTION = _load_production_nodes()
DOWNSAMPLE = PRODUCTION['_downsample_occ_target']
SMALL_PRODUCTION = _load_production_nodes(small_loss_grid=True)


def _legacy_downsample(target_voxels, output_size):
    """Frozen pre-change downsampling, deliberately independent of the helper."""
    B, tH, tW, tD = target_voxels.shape
    H, W, D = output_size
    ratio = tH // H
    if ratio != 1:
        target_voxels = target_voxels.reshape(
            B, H, ratio, W, ratio, D, ratio).permute(
                0, 1, 3, 5, 2, 4, 6).reshape(B, H, W, D, ratio ** 3)
        empty_idx = 0
        empty_mask = target_voxels.sum(-1) == empty_idx
        target_voxels = target_voxels.to(torch.int64)
        occ_space = target_voxels[~empty_mask]
        occ_space[occ_space == 0] = -torch.arange(
            len(occ_space[occ_space == 0])).to(occ_space.device) - 1
        target_voxels[~empty_mask] = occ_space
        target_voxels = torch.mode(target_voxels, dim=-1)[0]
        target_voxels[target_voxels < 0] = 255
        target_voxels = target_voxels.long()
    return target_voxels


def _legacy_loss_voxel(head, output_voxels, target_voxels, tag):
    """Frozen original loss math, evaluated on the same smaller test grid."""
    H, W, D = TEST_GRID
    if output_voxels.shape[2] != H:
        output_voxels = F.interpolate(
            output_voxels, size=(H, W, D), mode='trilinear', align_corners=False)
    target_voxels = _legacy_downsample(target_voxels, TEST_GRID)
    assert torch.isnan(output_voxels).sum().item() == 0
    assert torch.isnan(target_voxels).sum().item() == 0
    losses = {
        'loss_voxel_ce_' + tag: head.loss_voxel_ce_weight * SEM.CE_ssc_loss(
            output_voxels, target_voxels,
            head.class_weights.type_as(output_voxels), ignore_index=255),
    }
    if head.multi_loss:
        losses['loss_voxel_sem_scal_' + tag] = (
            head.loss_voxel_sem_scal_weight * SEM.sem_scal_loss(
                output_voxels, target_voxels, ignore_index=255))
        losses['loss_voxel_lovasz_' + tag] = (
            head.loss_voxel_lovasz_weight * LOVASZ.lovasz_softmax(
                torch.softmax(output_voxels, dim=1), target_voxels, ignore=255))
        if head.loss_voxel_geo_scal_weight is not None:
            losses['loss_voxel_geo_scal_' + tag] = (
                head.loss_voxel_geo_scal_weight * SEM.geo_scal_loss(
                    output_voxels, target_voxels,
                    ignore_index=255, non_empty_idx=0))
    return losses


def _head(multi_loss=True, classes=2):
    head = SMALL_PRODUCTION['LossHead']()
    head.class_weights = torch.tensor([1.] + [5.] * (classes - 1))
    head.multi_loss = multi_loss
    head.loss_voxel_ce_weight = 1.0
    head.loss_voxel_sem_scal_weight = 1.0
    head.loss_voxel_lovasz_weight = 1.0
    head.loss_voxel_geo_scal_weight = 1.0
    return head


class OccTargetDownsampleTest(unittest.TestCase):
    def assert_same(self, target, output_size):
        expected = _legacy_downsample(target.clone(), output_size)
        actual = DOWNSAMPLE(target.clone(), output_size)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def test_all_binary_blocks_and_single_return_rule(self):
        blocks = torch.tensor(list(itertools.product((0., 1.), repeat=8)))
        target = blocks.reshape(-1, 2, 2, 2)
        self.assert_same(target, (1, 1, 1))
        counts = blocks.sum(dim=1)
        expected = torch.where(counts == 0, 0, torch.where(counts == 1, 255, 1))
        self.assertTrue(torch.equal(
            DOWNSAMPLE(target, (1, 1, 1)).flatten(), expected))

    def test_all_blocks_with_ignore_label(self):
        blocks = torch.tensor(list(itertools.product((0., 1., 255.), repeat=8)))
        self.assert_same(blocks.reshape(-1, 2, 2, 2), (1, 1, 1))

    def test_random_multiclass_and_multiple_ratios(self):
        generator = torch.Generator().manual_seed(17)
        for dtype in (torch.float32, torch.float64, torch.int64):
            for ratio in (1, 2, 3):
                target = torch.randint(
                    0, 7, (2, 2 * ratio, 3 * ratio, ratio), generator=generator)
                target[target == 6] = 255
                self.assert_same(target.to(dtype), (2, 3, 1))

    def test_noncontiguous_target(self):
        generator = torch.Generator().manual_seed(42)
        target = torch.randint(0, 5, (2, 8, 8, 4), generator=generator).float()
        target[target == 4] = 255
        target = target.transpose(1, 2)
        self.assertFalse(target.is_contiguous())
        expected = _legacy_downsample(target, TEST_GRID)
        actual = DOWNSAMPLE(target, TEST_GRID)
        self.assertTrue(torch.equal(actual, expected))

    def test_already_prepared_target_is_returned_unchanged(self):
        target = torch.ones((2, *TEST_GRID), dtype=torch.int64)
        self.assertIs(DOWNSAMPLE(target, TEST_GRID), target)


class OccupancyLossReuseTest(unittest.TestCase):
    def inputs(self, classes=2):
        generator = torch.Generator().manual_seed(93)
        # Three decoder layers, with a smaller prediction grid to cover
        # trilinear interpolation as well as the full auxiliary-loss path.
        pred = torch.randn(3, 2, classes, 3, 3, 2, generator=generator)
        target = torch.randint(0, classes, (2, 8, 8, 4), generator=generator).float()
        target[:, :2, :2, :2] = 0
        target[:, 2:4, 2:4, :2] = 1
        target[:, 4:6, 4:6, :2] = 255
        return pred, target.transpose(1, 2)

    def test_loss_and_gradient_match_all_decoder_layers(self):
        for multi_loss, classes in ((False, 2), (True, 2), (True, 4)):
            with self.subTest(multi_loss=multi_loss, classes=classes):
                head = _head(multi_loss, classes)
                pred, target = self.inputs(classes)
                before = target.clone()
                expected_pred = pred.clone().requires_grad_()
                actual_pred = pred.clone().requires_grad_()
                expected = {}
                for index, output in enumerate(expected_pred):
                    expected.update(_legacy_loss_voxel(
                        head, output, target, 'inter_{}'.format(index)))
                actual = head.loss_occ(actual_pred, target)
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    self.assertTrue(torch.equal(actual[key], expected[key]), key)
                sum(expected.values()).backward()
                sum(actual.values()).backward()
                self.assertTrue(torch.equal(actual_pred.grad, expected_pred.grad))
                self.assertTrue(torch.equal(target, before))

    def test_downsample_called_once_for_three_layers(self):
        head = _head()
        pred, target = self.inputs()
        helper = SMALL_PRODUCTION['_downsample_occ_target']
        with mock.patch.dict(SMALL_PRODUCTION, {
                '_downsample_occ_target': mock.Mock(wraps=helper)}):
            head.loss_occ(pred, target)
            self.assertEqual(
                SMALL_PRODUCTION['_downsample_occ_target'].call_count, 1)

    def test_direct_call_still_accepts_full_resolution_target(self):
        head = _head()
        pred, target = self.inputs()
        expected = _legacy_loss_voxel(head, pred[0], target, 'direct')
        actual = head.loss_voxel(pred[0], target, 'direct')
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_prepared_target_keeps_geometric_loss_working(self):
        head = _head()
        pred, target = self.inputs()
        prepared = SMALL_PRODUCTION['_downsample_occ_target'](target)
        direct = head.loss_voxel(pred[0], prepared, 'prepared')
        reuse = head.loss_voxel(
            pred[0], prepared, 'prepared', target_voxels_prepared=True)
        for key in direct:
            self.assertTrue(torch.equal(direct[key], reuse[key]), key)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
