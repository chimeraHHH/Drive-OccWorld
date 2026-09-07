"""Real MMCV/PyTorch checks for optimizer-update equivalence across GPU counts.

Load the small production modules directly, without importing the detector's
CUDA extension stack. These tests do not simulate bitwise augmentation replay.
"""
import importlib.util
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
    from mmcv.runner import HOOKS
    from mmcv.runner.hooks.lr_updater import CosineAnnealingLrUpdaterHook
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, 'PyTorch, NumPy and MMCV are required')
class EquivalentTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT / 'projects/mmdet3d_plugin/datasets/samplers'
        package_name = '_equivalence_test_samplers'
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]
        sys.modules[package_name] = package
        original = load_module(package_name + '.group_sampler', directory / 'group_sampler.py')
        virtual = load_module(package_name + '.virtual_group_sampler', directory / 'virtual_group_sampler.py')
        cls.OriginalSampler = original.DistributedGroupSampler
        cls.VirtualSampler = virtual.VirtualDistributedGroupSampler
        if HOOKS.get('EquivalentCumulativeOptimizerHook') is None:
            cls.hooks = load_module(
                '_equivalence_test_hooks',
                ROOT / 'projects/mmdet3d_plugin/core/hooks/equivalent_training.py')
        else:
            cls.hooks = sys.modules[HOOKS.get('EquivalentCumulativeOptimizerHook').__module__]

    def dataset(self, flags):
        return types.SimpleNamespace(flag=np.asarray(flags, dtype=np.uint8))

    def test_every_update_matches_original_four_rank_indices(self):
        dataset = self.dataset([0] * 13 + [1] * 7 + [2] * 3)
        for seed in (0, 9):
            for epoch in (0, 1, 23):
                original = []
                for rank in range(4):
                    sampler = self.OriginalSampler(dataset, 1, 4, rank, seed)
                    sampler.set_epoch(epoch)
                    original.append(list(sampler))
                for physical_world in (1, 2, 4):
                    accumulation = 4 // physical_world
                    actual = []
                    for rank in range(physical_world):
                        sampler = self.VirtualSampler(dataset, 1, physical_world, rank, seed)
                        sampler.set_epoch(epoch)
                        actual.append(list(sampler))
                        self.assertEqual(len(sampler), len(original[0]) * accumulation)
                        self.assertEqual(sampler.optimizer_steps_per_epoch, len(original[0]))
                        self.assertEqual(sampler.total_size, sum(map(len, original)))
                    for update in range(len(original[0])):
                        for physical_rank in range(physical_world):
                            for micro in range(accumulation):
                                virtual_rank = physical_rank + micro * physical_world
                                self.assertEqual(
                                    actual[physical_rank][update * accumulation + micro],
                                    original[virtual_rank][update])

    def test_reference_padding_gives_5983_updates(self):
        dataset = self.dataset([0] * 23929)
        for world in (1, 2, 4):
            sampler = self.VirtualSampler(dataset, 1, world, 0, 0)
            self.assertEqual(sampler.optimizer_steps_per_epoch, 5983)
            self.assertEqual(sampler.total_size, 23932)

    def test_invalid_physical_batch_and_topology_are_rejected(self):
        dataset = self.dataset([0] * 7)
        for kwargs in (dict(samples_per_gpu=2), dict(num_replicas=3),
                       dict(num_replicas=2, rank=2), dict(reference_world_size=8)):
            with self.assertRaises(ValueError):
                self.VirtualSampler(dataset, **kwargs)

    def make_runner(self, accumulation, steps=6, world=None, model=None):
        if model is None:
            model = torch.nn.Linear(2, 1, bias=True).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        sampler = types.SimpleNamespace(reference_world_size=4, cumulative_iters=accumulation)

        class Loader:
            batch_size = 1
            def __len__(self):
                return steps * accumulation

        loader = Loader()
        loader.sampler = sampler
        return types.SimpleNamespace(
            model=model, optimizer=optimizer, meta={}, iter=0, inner_iter=0,
            epoch=0, max_epochs=24, max_iters=steps * accumulation * 24,
            data_loader=loader,
            outputs={}, log_buffer=types.SimpleNamespace(update=lambda *args: None))

    def test_all_warmup_updates_and_epoch_cosine_match_original(self):
        settings = dict(min_lr_ratio=1e-3, warmup='linear', warmup_iters=500,
                        warmup_ratio=1.0 / 3.0)
        for accumulation in (1, 2, 4):
            original_runner = self.make_runner(1, steps=5983)
            actual_runner = self.make_runner(accumulation, steps=5983)
            original_hook = CosineAnnealingLrUpdaterHook(**settings)
            actual_hook = self.hooks.EquivalentCosineAnnealingLrUpdaterHook(
                cumulative_iters=accumulation, **settings)
            original_hook.before_run(original_runner)
            actual_hook.before_run(actual_runner)
            for epoch in (0, 1, 12, 23):
                original_runner.epoch = actual_runner.epoch = epoch
                original_hook.before_train_epoch(original_runner)
                actual_hook.before_train_epoch(actual_runner)
                for update in list(range(502)) + [5982]:
                    original_runner.iter = epoch * 5983 + update
                    original_hook.before_train_iter(original_runner)
                    expected = original_runner.optimizer.param_groups[0]['lr']
                    for micro in range(accumulation):
                        raw = (epoch * 5983 + update) * accumulation + micro
                        actual_runner.iter = raw
                        actual_hook.before_train_iter(actual_runner)
                        self.assertEqual(actual_runner.optimizer.param_groups[0]['lr'], expected)
                        self.assertEqual(actual_runner.iter, raw)

    def test_lr_rejects_nonreference_schedules(self):
        for settings in (dict(cumulative_iters=3), dict(by_epoch=False),
                         dict(warmup_by_epoch=True)):
            with self.assertRaises(ValueError):
                self.hooks.EquivalentCosineAnnealingLrUpdaterHook(min_lr_ratio=0.001, **settings)

    def start_optimizer_hook(self, runner, accumulation, steps=6):
        hook = self.hooks.EquivalentCumulativeOptimizerHook(
            cumulative_iters=accumulation, expected_optimizer_steps_per_epoch=steps,
            grad_clip=dict(max_norm=0.15, norm_type=2))
        hook.before_run(runner)
        with patch.object(self.hooks, 'get_dist_info', return_value=(0, 4 // accumulation)):
            hook.before_train_epoch(runner)
        return hook

    def sample_loss(self, model, features, targets):
        # A nonlinear probability ratio over a sample's voxels, deliberately
        # unlike pooling all samples before computing the ratio.
        probability = model(features).sigmoid().flatten()
        intersection = (probability * targets).sum()
        return (-torch.log((intersection + 0.01) / (probability.sum() + 0.02))
                -torch.log((intersection + 0.01) / (targets.sum() + 0.02)))

    def test_accumulated_adamw_and_clipping_equal_four_sample_mean(self):
        torch.manual_seed(11)
        features = torch.randn(6, 4, 7, 2, dtype=torch.float64)
        targets = (torch.rand(6, 4, 7, dtype=torch.float64) > 0.45).double()
        initial = torch.nn.Linear(2, 1).double().state_dict()
        for accumulation in (1, 2, 4):
            reference = self.make_runner(1)
            actual = self.make_runner(accumulation)
            reference.model.load_state_dict(initial)
            actual.model.load_state_dict(initial)
            hook = self.start_optimizer_hook(actual, accumulation)
            world = 4 // accumulation
            with patch.object(hook, 'clip_grads', wraps=hook.clip_grads) as clip:
                with patch.object(actual.optimizer, 'step', wraps=actual.optimizer.step) as step:
                    for update in range(6):
                        expected_loss = sum(self.sample_loss(reference.model, features[update, r], targets[update, r])
                                            for r in range(4)) / 4
                        reference.optimizer.zero_grad()
                        expected_loss.backward()
                        torch.nn.utils.clip_grad_norm_(reference.model.parameters(), 0.15, 2)
                        reference.optimizer.step()
                        for micro in range(accumulation):
                            # The mean over physical ranks models DDP's linear
                            # gradient all-reduce for this microstep.
                            physical_rank_losses = [self.sample_loss(
                                actual.model, features[update, rank + micro * world],
                                targets[update, rank + micro * world]) for rank in range(world)]
                            actual.outputs = dict(loss=sum(physical_rank_losses) / world, num_samples=1)
                            actual.iter = actual.inner_iter = update * accumulation + micro
                            hook.after_train_iter(actual)
                            self.assertEqual(step.call_count, update + int(micro == accumulation - 1))
                        for expected, observed in zip(reference.model.parameters(), actual.model.parameters()):
                            torch.testing.assert_close(observed, expected, atol=1e-12, rtol=1e-10)
                    self.assertEqual(clip.call_count, 6)
            hook.after_train_epoch(actual)
            self.assertEqual(actual.meta['equivalent_training']['optimizer_updates_completed'], 6)
            self.assertEqual(actual.meta['equivalent_training']['raw_microsteps_completed'], 6 * accumulation)

    def test_nonlinear_pooling_is_not_mean_of_per_sample_losses(self):
        model = torch.nn.Linear(2, 1).double()
        features = torch.tensor([[[1., 0.], [2., 1.]], [[-3., 2.], [-2., 3.]]], dtype=torch.float64)
        targets = torch.tensor([[1., 1.], [1., 0.]], dtype=torch.float64)
        separate = sum(self.sample_loss(model, features[i], targets[i]) for i in range(2)) / 2
        pooled = self.sample_loss(model, features.reshape(-1, 2), targets.reshape(-1))
        self.assertGreater(abs(float(separate - pooled)), 1e-6)

    def test_partial_update_resume_is_rejected(self):
        runner = self.make_runner(2)
        runner.iter = 1
        hook = self.hooks.EquivalentCumulativeOptimizerHook(cumulative_iters=2)
        with self.assertRaisesRegex(ValueError, 'unfinished optimizer update'):
            hook.before_run(runner)

    def test_changed_topology_on_resume_is_rejected(self):
        runner = self.make_runner(2)
        runner.meta['equivalent_training'] = dict(cumulative_iters=4)
        hook = self.hooks.EquivalentCumulativeOptimizerHook(cumulative_iters=2)
        with self.assertRaisesRegex(ValueError, 'preserve physical topology'):
            hook.before_run(runner)

    def test_completed_epoch_resume_keeps_logical_update_count(self):
        runner = self.make_runner(2)
        runner.iter, runner.epoch = 12, 1
        runner.meta['equivalent_training'] = dict(
            cumulative_iters=2, raw_microsteps_completed=12,
            optimizer_updates_completed=6)
        self.start_optimizer_hook(runner, 2)
        self.assertEqual(runner.meta['equivalent_training']['optimizer_updates_completed'], 6)

    def test_physical_batch_or_wrong_dataset_is_rejected(self):
        runner = self.make_runner(2)
        runner.data_loader.batch_size = 2
        with self.assertRaisesRegex(ValueError, 'physical microbatch=1'):
            self.start_optimizer_hook(runner, 2)
        runner = self.make_runner(2, steps=5)
        with self.assertRaisesRegex(ValueError, 'Dataset/sampler differs'):
            self.start_optimizer_hook(runner, 2, steps=6)

    def test_middle_of_epoch_resume_and_partial_epoch_are_rejected(self):
        runner = self.make_runner(2)
        runner.iter = 2
        runner.meta['equivalent_training'] = dict(cumulative_iters=2)
        with self.assertRaisesRegex(ValueError, 'completed epoch'):
            self.start_optimizer_hook(runner, 2)
        runner = self.make_runner(2)
        hook = self.start_optimizer_hook(runner, 2)
        runner.inner_iter = 0
        with self.assertRaisesRegex(RuntimeError, 'uncommitted gradients'):
            hook.after_train_epoch(runner)

    def test_real_epoch_checkpoint_resume_matches_uninterrupted_training(self):
        from mmcv.runner import EpochBasedRunner

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

            def train_step(self, batch, optimizer, **kwargs):
                return dict(loss=(self.weight * batch['x'] - batch['y']).square().mean(),
                            log_vars={}, num_samples=1)

        class Loader:
            batch_size = 1
            sampler = types.SimpleNamespace(reference_world_size=4, cumulative_iters=4)
            def __len__(self):
                return 8
            def __iter__(self):
                for index in range(8):
                    yield dict(x=torch.tensor([index + 1.], dtype=torch.float64),
                               y=torch.tensor([index % 3 + 0.2], dtype=torch.float64))

        def build_runner(directory):
            model = ToyModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            runner = EpochBasedRunner(
                model=model, optimizer=optimizer, work_dir=directory,
                logger=logging.getLogger('equivalence_checkpoint_test'), max_epochs=2, meta={})
            runner.register_training_hooks(
                dict(policy='EquivalentCosineAnnealing', cumulative_iters=4,
                     warmup='linear', warmup_iters=500, warmup_ratio=1. / 3,
                     min_lr_ratio=0.001),
                dict(type='EquivalentCumulativeOptimizerHook', cumulative_iters=4,
                     expected_optimizer_steps_per_epoch=2,
                     grad_clip=dict(max_norm=0.15, norm_type=2)),
                dict(interval=1, create_symlink=False), None)
            return runner

        with tempfile.TemporaryDirectory() as temp:
            reference_dir = str(Path(temp) / 'reference')
            staged_dir = str(Path(temp) / 'staged')
            reference = build_runner(reference_dir)
            staged = build_runner(staged_dir)
            with patch('mmcv.runner.epoch_based_runner.time.sleep', return_value=None):
                reference.call_hook('before_run')
                reference.train(Loader())
                reference.train(Loader())
                staged.call_hook('before_run')
                staged.train(Loader())
                checkpoint_path = str(Path(staged_dir) / 'epoch_1.pth')
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.assertEqual(checkpoint['meta']['epoch'], 1)
                self.assertEqual(checkpoint['meta']['iter'], 8)
                self.assertEqual(checkpoint['meta']['equivalent_training']['optimizer_updates_completed'], 2)
                resumed = build_runner(staged_dir)
                resumed.resume(checkpoint_path, map_location='cpu')
                resumed.call_hook('before_run')
                resumed.train(Loader())
            self.assertEqual(resumed.iter, 16)
            self.assertEqual(resumed.epoch, 2)
            self.assertEqual(resumed.meta['equivalent_training']['optimizer_updates_completed'], 4)
            torch.testing.assert_close(resumed.model.weight, reference.model.weight, rtol=0, atol=0)
            self.assertEqual(resumed.optimizer.param_groups[0]['lr'], reference.optimizer.param_groups[0]['lr'])

    def test_resume_requires_matching_counter_metadata(self):
        runner = self.make_runner(2)
        runner.iter = 12
        hook = self.hooks.EquivalentCumulativeOptimizerHook(cumulative_iters=2)
        with self.assertRaisesRegex(ValueError, 'checkpoint metadata'):
            hook.before_run(runner)
        runner.meta['equivalent_training'] = dict(
            cumulative_iters=2, raw_microsteps_completed=10,
            optimizer_updates_completed=5)
        with self.assertRaisesRegex(ValueError, 'counters disagree'):
            hook.before_run(runner)


if __name__ == '__main__':
    unittest.main()
