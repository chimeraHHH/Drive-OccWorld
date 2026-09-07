"""Execute production observation attachment and real MMCV/PyTorch collation.

Only annotation/pipeline services are replaced. The full production dataset
method is executed unchanged, including its actual DataContainer construction;
this catches the two-dimensional tensor/default-padding integration failure.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import numpy as np
    import torch
    from mmcv.parallel import DataContainer, collate
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_template.py'


def production_prepare_method():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
               node.name == 'NuScenesWorldDatasetTemplate')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and
                  node.name == '_prepare_data_info_single')
    scope = dict(copy=copy, torch=torch, DC=DataContainer)
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), scope)
    return scope['_prepare_data_info_single']


@unittest.skipIf(torch is None, 'Real PyTorch, NumPy and MMCV are required')
class M3ObservationCollateTest(unittest.TestCase):
    def example(self, offset=0.0):
        values = np.arange(4096 * 9, dtype=np.float32).reshape(4096, 9) + offset
        dataset = SimpleNamespace(
            get_data_info=lambda index: dict(sample_idx='current_sample'),
            radar_bev_loader=None, radar_observation_loader=lambda token: values,
            get_future_bboxes=lambda index: ([], np.zeros((1, 1), dtype=np.float32)),
            data_infos=[dict(token='current_sample')],
            traj_api=SimpleNamespace(get_sdc_planning_label=lambda token: (None, None, None)),
            get_trajectory_sampling=lambda info, length: None,
            queue_length=0, future_length=0, use_fine_occ=True,
            record_instance=lambda index, instance_map: instance_map,
            pre_pipeline=lambda info: None, pipeline=lambda info: {})
        example = production_prepare_method()(dataset, 0, occ_load_flag=True)
        return example, torch.from_numpy(values)

    def test_physical_batch_one_preserves_all_4096_by_9_values(self):
        example, expected = self.example()
        self.assertIsNone(example['radar_observations'].pad_dims)
        result = collate([example], samples_per_gpu=1)['radar_observations']
        self.assertTrue(result.stack)
        self.assertEqual(len(result.data), 1)
        self.assertEqual(tuple(result.data[0].shape), (1, 4096, 9))
        self.assertEqual(result.data[0].dtype, torch.float32)
        torch.testing.assert_close(result.data[0][0], expected, rtol=0, atol=0)

    def test_batch_two_stacks_without_padding_or_value_changes(self):
        pairs = [self.example(0.0), self.example(100000.0)]
        result = collate([pair[0] for pair in pairs], samples_per_gpu=2)['radar_observations']
        self.assertEqual(len(result.data), 1)
        self.assertEqual(tuple(result.data[0].shape), (2, 4096, 9))
        torch.testing.assert_close(result.data[0], torch.stack([p[1] for p in pairs]),
                                   rtol=0, atol=0)

    def test_per_gpu_groups_keep_single_sample_shapes_and_order(self):
        pairs = [self.example(0.0), self.example(100000.0)]
        result = collate([pair[0] for pair in pairs], samples_per_gpu=1)['radar_observations']
        self.assertEqual(len(result.data), 2)
        for group, (_, expected) in zip(result.data, pairs):
            self.assertEqual(tuple(group.shape), (1, 4096, 9))
            torch.testing.assert_close(group[0], expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
