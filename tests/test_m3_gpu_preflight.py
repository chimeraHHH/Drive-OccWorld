"""CPU-only checks for diagnostic isolation and the unchanged training contract."""
import copy
from functools import partial
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
    from mmcv.parallel import DataContainer, collate
    from mmcv.runner import EpochBasedRunner, HOOKS
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('m3_gpu_preflight', ROOT / 'tools/m3_gpu_preflight.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config_fixture():
    return dict(
        m3_h200_requires_resolution=False, resume_from=None,
        load_from='/project/checkpoints/r101_dcn_fcos3d_pretrain_l40s_reference.pth',
        fp16=None, gpu_ids=range(2), work_dir='/project/work_dirs/formal', seed=0,
        data=dict(samples_per_gpu=1, workers_per_gpu=2,
                  shuffler_sampler=dict(type='VirtualDistributedGroupSampler', reference_world_size=4),
                  train=dict(ann_file='/shared/train.pkl', pipeline=[dict(type='FullCameraPipeline')])),
        optimizer_config=dict(
            type='EquivalentCumulativeOptimizerHook', cumulative_iters=2, reference_world_size=4,
            expected_optimizer_steps_per_epoch=5983, grad_clip=dict(max_norm=35, norm_type=2)),
        lr_config=dict(policy='EquivalentCosineAnnealing', cumulative_iters=2, warmup_iters=500,
                       warmup='linear', min_lr_ratio=0.001),
        runner=dict(type='EpochBasedRunner', max_epochs=24), total_epochs=24,
        workflow=[('train', 1)], model=dict(doppler_posterior=dict(transport_mode='sigma')),
        optimizer=dict(type='AdamW', lr=1e-4), checkpoint_config=dict(interval=1),
        custom_hooks=[dict(type='SetEpochInfoHook')],
        log_config=dict(interval=20, hooks=[dict(type='TextLoggerHook')]),
        tf32_policy=dict(matmul=True, cudnn=True))


class PreflightConfigTest(unittest.TestCase):
    def test_copy_preserves_data_budget_initialization_and_production_hooks(self):
        original = config_fixture()
        snapshot = copy.deepcopy(original)
        result = MODULE.prepare_config(original, '/project/diagnostics/smoke', 5)
        self.assertEqual(original, snapshot)
        for key in ('data', 'runner', 'total_epochs', 'lr_config', 'optimizer',
                    'model', 'load_from', 'resume_from', 'seed', 'custom_hooks'):
            self.assertEqual(result[key], original[key])
        self.assertIsNone(result['checkpoint_config'])
        self.assertEqual(result['optimizer_config']['type'], 'M3GPUPreflightOptimizerHook')
        self.assertEqual(result['optimizer_config']['expected_optimizer_steps_per_epoch'], 5983)
        self.assertEqual(result['optimizer_config']['preflight_updates'], 5)
        self.assertNotEqual(result['work_dir'], original['work_dir'])

    def test_formal_workdir_and_ancestor_descendant_paths_are_rejected(self):
        for path in ('/project/work_dirs/formal', '/project/work_dirs/formal/diagnostics',
                     '/project/work_dirs'):
            with self.assertRaisesRegex(ValueError, 'separate'):
                MODULE.prepare_config(config_fixture(), path, 3)

    def test_resume_and_trained_checkpoint_are_rejected(self):
        config = config_fixture()
        config['resume_from'] = '/model/epoch_15.pth'
        with self.assertRaisesRegex(ValueError, 'never resume'):
            MODULE.prepare_config(config, '/project/diagnostics/smoke', 3)
        config = config_fixture()
        config['load_from'] = '/model/epoch_15.pth'
        with self.assertRaisesRegex(ValueError, 'FCOS3D'):
            MODULE.prepare_config(config, '/project/diagnostics/smoke', 3)

    def test_invalid_training_semantics_and_unbounded_run_are_rejected(self):
        changes = [
            ('m3_h200_requires_resolution', True), ('fp16', dict(loss_scale=512)),
            ('gpu_ids', range(4)), ('runner', dict(type='EpochBasedRunner', max_epochs=1)),
            ('tf32_policy', dict(matmul=False, cudnn=True)),
            ('workflow', [('train', 1), ('val', 1)])]
        for key, value in changes:
            config = config_fixture()
            config[key] = value
            with self.assertRaises(ValueError):
                MODULE.prepare_config(config, '/project/diagnostics/smoke', 3)
        for updates in (0, 101, 5983):
            with self.assertRaisesRegex(ValueError, 'between 1 and 100'):
                MODULE.prepare_config(config_fixture(), '/project/diagnostics/smoke', updates)

    def test_json_and_timing_report_are_finite_and_explicit(self):
        report = MODULE.timing_summary([10.0, 2.0, 4.0])
        self.assertEqual(report['mean_optimizer_update_seconds'], 16.0 / 3)
        self.assertEqual(report['mean_after_first_update_seconds'], 3.0)
        self.assertIsNone(MODULE.timing_summary([10.0])['mean_after_first_update_seconds'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'rank_0.json'
            MODULE.write_json(path, report)
            self.assertEqual(json.loads(path.read_text()), report)
            self.assertFalse(path.with_suffix('.json.tmp').exists())
            with self.assertRaises(ValueError):
                MODULE.write_json(path, dict(loss=float('nan')))
            self.assertEqual(json.loads(path.read_text()), report)

    def test_finite_prefix_preserves_original_order_and_full_length(self):
        original = list(range(100, 130))
        prefix = MODULE.FiniteSamplerPrefix(original, 6)
        self.assertEqual(len(prefix), len(original))
        self.assertEqual(list(prefix), original[:6])
        self.assertEqual(prefix.dispatched, 6)
        self.assertEqual(list(prefix), original[:6])
        for limit in (0, len(original), len(original) + 1):
            with self.assertRaises(ValueError):
                MODULE.FiniteSamplerPrefix(original, limit)

    def test_worker_cleanup_does_not_accept_an_abnormal_exit(self):
        worker = types.SimpleNamespace(pid=1, exitcode=-6, is_alive=lambda: False)
        iterator = types.SimpleNamespace(
            _send_idx=6, _rcvd_idx=6, _num_yielded=6, _tasks_outstanding=0,
            _workers=[worker], _shutdown_workers=lambda: None)
        loader = types.SimpleNamespace(_iterator=iterator)
        prefix = types.SimpleNamespace(dispatched=6, microsteps=6)
        with self.assertRaisesRegex(RuntimeError, 'did not exit cleanly'):
            MODULE.finish_loader_workers(loader, prefix)


class WorkerTensorDataset:
    """Spawn-picklable real tensors large enough to exercise worker IPC queues."""
    flag = None

    def __init__(self):
        self.flag = np.zeros(23930, dtype=np.uint8)

    def __len__(self):
        return 23930

    def __getitem__(self, index):
        return dict(index=index,
                    img=DataContainer(torch.full((6, 3, 128, 128), float(index % 11)), stack=True),
                    radar_observations=DataContainer(torch.zeros(4096, 9), stack=True, pad_dims=None))


@unittest.skipIf(torch is None, 'Real PyTorch/MMCV required for CPU worker integration')
class PreflightWorkerLifecycleTest(unittest.TestCase):
    def test_real_mmcv_runner_drains_spawn_workers_at_complete_update(self):
        directory = ROOT / 'projects/mmdet3d_plugin/datasets/samplers'
        package_name = '_preflight_worker_sampler'
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]
        sys.modules[package_name] = package

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        load(package_name + '.group_sampler', directory / 'group_sampler.py')
        virtual = load(package_name + '.virtual_group_sampler', directory / 'virtual_group_sampler.py')
        if HOOKS.get('EquivalentCumulativeOptimizerHook') is None:
            production = load('_preflight_worker_optimizer', ROOT /
                              'projects/mmdet3d_plugin/core/hooks/equivalent_training.py')
        else:
            production = sys.modules[HOOKS.get('EquivalentCumulativeOptimizerHook').__module__]
        evidence = {}

        class BoundedCPUHook(production.EquivalentCumulativeOptimizerHook):
            def before_train_epoch(self, runner):
                super().before_train_epoch(runner)
                self.prefix = MODULE.arm_finite_loader(runner.data_loader, 20)

            def after_train_epoch(self, runner):
                super().after_train_epoch(runner)
                evidence.update(MODULE.finish_loader_workers(runner.data_loader, self.prefix))
                raise MODULE.PreflightComplete()

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5))
                self.seen = []

            def train_step(self, batch, optimizer, **kwargs):
                self.seen.extend(batch['index'].tolist())
                return dict(loss=(self.weight - batch['img'].data[0].mean()).square(),
                            log_vars={}, num_samples=1)

        dataset = WorkerTensorDataset()
        sampler = virtual.VirtualDistributedGroupSampler(dataset, 1, 2, 0, 0)
        expected = list(sampler)[:20]
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, sampler=sampler, num_workers=2,
            collate_fn=partial(collate, samples_per_gpu=1), persistent_workers=True,
            prefetch_factor=2, multiprocessing_context='spawn')
        model = ToyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        with tempfile.TemporaryDirectory() as work_dir:
            runner = EpochBasedRunner(model=model, optimizer=optimizer, max_epochs=24,
                                      work_dir=work_dir, logger=logging.getLogger(__name__), meta={})
            runner.register_training_hooks(
                dict(policy='EquivalentCosineAnnealing', cumulative_iters=2,
                     warmup='linear', warmup_iters=500, warmup_ratio=1. / 3,
                     min_lr_ratio=0.001),
                BoundedCPUHook(cumulative_iters=2, reference_world_size=4,
                               expected_optimizer_steps_per_epoch=5983,
                               grad_clip=dict(max_norm=35, norm_type=2)),
                None, None)
            with patch.object(production, 'get_dist_info', return_value=(0, 2)), \
                    patch('mmcv.runner.epoch_based_runner.time.sleep', return_value=None):
                runner.call_hook('before_run')
                with self.assertRaises(MODULE.PreflightComplete):
                    runner.train(loader)
            self.assertFalse(list(Path(work_dir).glob('*.pth')))
        self.assertEqual(model.seen, expected)
        self.assertEqual(runner.iter, 20)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.max_iters, 11966 * 24)
        self.assertEqual(len(loader), 11966)
        self.assertIs(loader.sampler, sampler)
        self.assertEqual(runner.meta['equivalent_training']['optimizer_updates_completed'], 10)
        self.assertEqual(float(optimizer.state[model.weight]['step']), 10.0)
        self.assertTrue(evidence['dataloader_workers_exited_cleanly'])
        self.assertEqual(evidence['counters'], dict(
            dispatched=20, sent=20, received=20, yielded=20, outstanding=0))
        self.assertEqual([worker['exitcode'] for worker in evidence['workers']], [0, 0])
        self.assertIsNone(loader._iterator)


if __name__ == '__main__':
    unittest.main()
