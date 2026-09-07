"""Optimizer-update semantics for L40S-equivalent microbatch-one training.

Keep the model's per-sample nonlinear losses unchanged. DDP averages each
microstep over physical ranks, and this hook averages accumulated microsteps
before a single clipping operation and AdamW update. Checkpoints and MMCV's
``iter`` field count raw microsteps; ``meta.equivalent_training`` also records
the optimizer-update count. Resume is supported at completed epoch boundaries.
"""
from mmcv.runner import HOOKS, OptimizerHook, get_dist_info
from mmcv.runner.hooks.lr_updater import CosineAnnealingLrUpdaterHook


class _OptimizerUpdateView:
    """Read-through runner view; never mutate runner counters during LR hooks."""
    def __init__(self, runner, cumulative_iters):
        self.runner = runner
        self.cumulative_iters = cumulative_iters

    @property
    def iter(self):
        return self.runner.iter // self.cumulative_iters

    @property
    def max_iters(self):
        return self.runner.max_iters // self.cumulative_iters

    def __getattr__(self, name):
        return getattr(self.runner, name)


@HOOKS.register_module()
class EquivalentCosineAnnealingLrUpdaterHook(CosineAnnealingLrUpdaterHook):
    """Use the original epoch cosine schedule and 500 *update* warmup steps."""
    def __init__(self, cumulative_iters=1, **kwargs):
        self.cumulative_iters = int(cumulative_iters)
        if self.cumulative_iters not in (1, 2, 4):
            raise ValueError('cumulative_iters must be 1, 2, or 4')
        if not kwargs.get('by_epoch', True):
            raise ValueError('The reference L40S cosine schedule is by epoch')
        if kwargs.get('warmup_by_epoch', False):
            raise ValueError('Warmup must be specified in optimizer updates')
        super().__init__(**kwargs)

    def before_train_iter(self, runner):
        # Both microsteps of an accumulated update use exactly the same LR.
        # Merely multiplying warmup_iters by accumulation would not do that.
        super().before_train_iter(_OptimizerUpdateView(runner, self.cumulative_iters))


@HOOKS.register_module()
class EquivalentCumulativeOptimizerHook(OptimizerHook):
    """Accumulate complete groups; reject partial epochs and physical batch>1."""
    def __init__(self, cumulative_iters=1, reference_world_size=4,
                 expected_optimizer_steps_per_epoch=None, **kwargs):
        super().__init__(**kwargs)
        self.cumulative_iters = int(cumulative_iters)
        self.reference_world_size = int(reference_world_size)
        self.expected_optimizer_steps_per_epoch = expected_optimizer_steps_per_epoch
        if self.cumulative_iters not in (1, 2, 4) or self.reference_world_size != 4:
            raise ValueError('Require four virtual ranks and accumulation 1, 2, or 4')

    def before_run(self, runner):
        if isinstance(runner.optimizer, dict):
            raise ValueError('Equivalent training supports one optimizer')
        if runner.iter % self.cumulative_iters:
            raise ValueError('Cannot resume inside an unfinished optimizer update')
        if runner.meta is None:
            runner.meta = {}
        previous = runner.meta.get('equivalent_training')
        if runner.iter and previous is None:
            raise ValueError('Resume requires equivalent_training checkpoint metadata')
        if previous is not None and previous.get('cumulative_iters') != self.cumulative_iters:
            raise ValueError('Resume must preserve physical topology and accumulation')
        if previous is not None and (
                previous.get('raw_microsteps_completed', runner.iter) != runner.iter or
                previous.get('optimizer_updates_completed', runner.iter // self.cumulative_iters)
                != runner.iter // self.cumulative_iters):
            raise ValueError('Checkpoint raw and optimizer-update counters disagree')
        runner.optimizer.zero_grad()

    def before_train_epoch(self, runner):
        loader = runner.data_loader
        _, world_size = get_dist_info()
        if world_size * self.cumulative_iters != self.reference_world_size:
            raise ValueError('Physical ranks times accumulation must equal four')
        if loader.batch_size != 1:
            raise ValueError('Equivalent training requires physical microbatch=1')
        sampler = loader.sampler
        if (getattr(sampler, 'reference_world_size', None) != self.reference_world_size or
                getattr(sampler, 'cumulative_iters', None) != self.cumulative_iters):
            raise ValueError('Use the matching VirtualDistributedGroupSampler')
        if not len(loader) or len(loader) % self.cumulative_iters:
            raise ValueError('An epoch must contain complete optimizer updates')
        if runner.iter % len(loader):
            raise ValueError('Only completed epoch checkpoints may be resumed')
        updates = len(loader) // self.cumulative_iters
        if (self.expected_optimizer_steps_per_epoch is not None and
                updates != self.expected_optimizer_steps_per_epoch):
            raise ValueError('Dataset/sampler differs from reference optimizer steps per epoch')
        runner.meta['equivalent_training'] = dict(
            reference_world_size=self.reference_world_size,
            physical_world_size=world_size, physical_microbatch=1,
            cumulative_iters=self.cumulative_iters,
            optimizer_steps_per_epoch=updates,
            optimizer_updates_completed=runner.iter // self.cumulative_iters,
            raw_microsteps_completed=runner.iter,
            checkpoint_iter_unit='microstep', lr_warmup_unit='optimizer_update',
            randomness='same augmentation distribution; not bitwise rank RNG replay')

    def after_train_iter(self, runner):
        (runner.outputs['loss'] / self.cumulative_iters).backward()
        if (runner.inner_iter + 1) % self.cumulative_iters == 0:
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    runner.log_buffer.update(
                        {'grad_norm': float(grad_norm)}, runner.outputs['num_samples'])
            runner.optimizer.step()
            runner.optimizer.zero_grad()
            progress = runner.meta['equivalent_training']
            progress['optimizer_updates_completed'] = (runner.iter + 1) // self.cumulative_iters
            progress['raw_microsteps_completed'] = runner.iter + 1

    def after_train_epoch(self, runner):
        if (runner.inner_iter + 1) % self.cumulative_iters:
            raise RuntimeError('Refusing to end an epoch with uncommitted gradients')
