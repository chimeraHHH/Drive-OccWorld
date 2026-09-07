"""Preserve four single-sample DDP streams when fewer GPUs are available."""
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

from .group_sampler import DistributedGroupSampler
from .sampler import SAMPLER


@SAMPLER.register_module()
class VirtualDistributedGroupSampler(Sampler):
    """Interleave original virtual-rank streams within an optimizer update.

    With two physical ranks, physical rank 0 consumes virtual ranks 0 and 2,
    and physical rank 1 consumes virtual ranks 1 and 3. Each microbatch stays
    at one sample. Thus each logical update sees exactly the original four
    dataset indices, including the original group padding and shuffle.

    This does not reproduce worker augmentation RNG streams or CUDA kernels.
    ``len(sampler)`` counts microsteps, not optimizer updates.
    """

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None,
                 rank=None, seed=0, reference_world_size=4):
        actual_rank, actual_world_size = get_dist_info()
        self.num_replicas = (actual_world_size if num_replicas is None
                             else int(num_replicas))
        self.rank = actual_rank if rank is None else int(rank)
        self.reference_world_size = int(reference_world_size)
        if samples_per_gpu != 1:
            raise ValueError('Equivalent training requires samples_per_gpu=1')
        if (self.reference_world_size != 4 or
                self.num_replicas not in (1, 2, 4)):
            raise ValueError('Require four virtual ranks and 1, 2, or 4 physical ranks')
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError('rank is outside physical world size')
        self.dataset = dataset
        self.samples_per_gpu = 1
        self.cumulative_iters = self.reference_world_size // self.num_replicas
        self.virtual_ranks = tuple(range(self.rank, self.reference_world_size,
                                         self.num_replicas))
        self.streams = [DistributedGroupSampler(
            dataset, samples_per_gpu=1, num_replicas=self.reference_world_size,
            rank=virtual_rank, seed=seed) for virtual_rank in self.virtual_ranks]
        lengths = {len(stream) for stream in self.streams}
        if len(lengths) != 1:
            raise ValueError('Virtual rank stream lengths must match')
        self.optimizer_steps_per_epoch = lengths.pop()
        self.num_samples = self.optimizer_steps_per_epoch * self.cumulative_iters
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0

    def __iter__(self):
        return (index for update_indices in zip(*(iter(s) for s in self.streams))
                for index in update_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        for stream in self.streams:
            stream.set_epoch(epoch)
