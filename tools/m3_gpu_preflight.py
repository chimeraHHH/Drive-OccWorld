"""Run a bounded, real-data M3 preflight through the production MMCV train API.

Launch with torch.distributed.run --nproc_per_node=2. The input must be the
resolved H200 configuration. No dataset truncation, schedule shortening,
training checkpoint, or resume is used. Reports go to a new diagnostics path.
"""
import argparse
import copy
from datetime import timedelta
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def prepare_config(config, diagnostics, updates):
    """Validate the contract and copy only diagnostic runtime overrides.

    This helper has no torch/MMCV dependency so its isolation and schedule
    invariants can be checked on a CPU-only development machine.
    """
    if not 1 <= updates <= 100:
        raise ValueError('Preflight updates must be between 1 and 100')
    if config.get('m3_h200_requires_resolution', True):
        raise ValueError('Supply the resolved H200 configuration')
    if config.get('resume_from'):
        raise ValueError('Preflight must initialize from FCOS3D, never resume training')
    initial = config.get('load_from')
    if not initial or not Path(initial).name.startswith('r101_dcn_fcos3d_pretrain'):
        raise ValueError('Expected the resolved common FCOS3D initialization')
    if config.get('fp16') is not None:
        raise ValueError('The reference experiment uses FP32, without an FP16 hook')
    if (config['data']['samples_per_gpu'] != 1 or len(config['gpu_ids']) != 2 or
            config['data']['workers_per_gpu'] != 2):
        raise ValueError('Require two ranks, physical microbatch one and two workers per rank')
    sampler = config['data']['shuffler_sampler']
    if sampler != dict(type='VirtualDistributedGroupSampler', reference_world_size=4):
        raise ValueError('Use the original four-stream virtual sampler')
    optimizer = config['optimizer_config']
    if (optimizer.get('type') != 'EquivalentCumulativeOptimizerHook' or
            optimizer.get('cumulative_iters') != 2 or
            optimizer.get('reference_world_size') != 4 or
            optimizer.get('expected_optimizer_steps_per_epoch') != 5983 or
            optimizer.get('grad_clip') != dict(max_norm=35, norm_type=2)):
        raise ValueError('Optimizer hook differs from the resolved H200 contract')
    lr = config['lr_config']
    if (lr.get('policy') != 'EquivalentCosineAnnealing' or
            lr.get('cumulative_iters') != 2 or lr.get('warmup_iters') != 500):
        raise ValueError('LR must retain the resolved update-based warmup')
    if config['runner'] != dict(type='EpochBasedRunner', max_epochs=24):
        raise ValueError('Keep the full epoch/cosine budget; preflight stops via its hook')
    if config.get('workflow') != [('train', 1)]:
        raise ValueError('Preflight expects the production train-only workflow')
    if not config['model'].get('doppler_posterior'):
        raise ValueError('The complete M3 posterior/transport must be configured')
    if config['model']['doppler_posterior'].get('transport_mode') != 'sigma':
        raise ValueError('This full-M3 preflight requires sigma transport')
    if config.get('tf32_policy') != dict(matmul=True, cudnn=True):
        raise ValueError('Use the resolved L40S TF32 arithmetic policy')
    output = Path(diagnostics).resolve()
    formal = Path(config['work_dir']).resolve()
    if output == formal or formal in output.parents or output in formal.parents:
        raise ValueError('Diagnostics must be separate from the formal work directory')
    result = copy.deepcopy(config)
    result['work_dir'] = str(output)
    result['checkpoint_config'] = None
    result['optimizer_config']['type'] = 'M3GPUPreflightOptimizerHook'
    result['optimizer_config']['preflight_updates'] = updates
    # Avoid Tensorboard and other logger side effects while retaining the
    # production runner and its normal log-buffer / TextLoggerHook path.
    result['log_config'] = dict(interval=2, hooks=[dict(type='TextLoggerHook')])
    result['m3_preflight'] = dict(
        status='DIAGNOSTIC_ONLY_DO_NOT_RESUME', requested_optimizer_updates=updates,
        original_work_dir=str(formal), checkpoint_saving=False,
        full_dataset_and_epoch_schedule_preserved=True)
    return result


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def timing_summary(values):
    steady = values[1:]
    return dict(
        optimizer_update_seconds=values,
        mean_optimizer_update_seconds=statistics.mean(values) if values else None,
        mean_after_first_update_seconds=statistics.mean(steady) if steady else None,
        first_update_includes_startup=True,
        interpretation='Diagnostic instrumentation is included; not a final training ETA benchmark')


class PreflightComplete(Exception):
    """Stop after complete updates, natural iterator exhaustion and clean workers."""


class FiniteSamplerPrefix:
    """Limit dispatch to a prefix while retaining the full epoch length budget."""
    def __init__(self, sampler, microsteps):
        if not 0 < microsteps < len(sampler):
            raise ValueError('Diagnostic prefix must be shorter than the full sampler')
        self.sampler = sampler
        self.microsteps = int(microsteps)
        self.dispatched = 0

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        self.dispatched = 0
        for index in itertools.islice(iter(self.sampler), self.microsteps):
            self.dispatched += 1
            yield index


def arm_finite_loader(loader, microsteps):
    """Cap indices before any workers prefetch; keep the original sampler/len."""
    if (loader.batch_size != 1 or not loader.persistent_workers or
            getattr(loader, '_iterator', None) is not None):
        raise RuntimeError('Arm a fresh persistent-worker microbatch-one loader before iteration')
    if loader.batch_sampler.sampler is not loader.sampler:
        raise RuntimeError('Unexpected batch sampler; refusing to change diagnostic ordering')
    prefix = FiniteSamplerPrefix(loader.sampler, microsteps)
    loader.batch_sampler.sampler = prefix
    return prefix


def finish_loader_workers(loader, prefix):
    """Require natural exhaustion, then explicitly join and verify worker exits.

    The guarded private attributes describe PyTorch 2.1's actual iterator. They
    let this diagnostic fail on unconsumed tasks or abnormal exits instead of
    relying on a late __del__ after writing PASS.
    """
    iterator = getattr(loader, '_iterator', None)
    if iterator is None:
        raise RuntimeError('Missing persistent-worker iterator at diagnostic completion')
    counters = dict(dispatched=prefix.dispatched, sent=iterator._send_idx,
                    received=iterator._rcvd_idx, yielded=iterator._num_yielded,
                    outstanding=iterator._tasks_outstanding)
    if (any(counters[key] != prefix.microsteps
            for key in ('dispatched', 'sent', 'received', 'yielded')) or
            counters['outstanding'] != 0):
        raise RuntimeError('DataLoader did not naturally drain its finite prefix: ' + str(counters))
    workers = list(iterator._workers)
    iterator._shutdown_workers()
    exits = [dict(pid=worker.pid, exitcode=worker.exitcode, alive=worker.is_alive())
             for worker in workers]
    if not exits or any(worker['alive'] or worker['exitcode'] != 0 for worker in exits):
        raise RuntimeError('DataLoader workers did not exit cleanly: ' + str(exits))
    loader._iterator = None
    return dict(dataloader_workers_exited_cleanly=True, counters=counters, workers=exits,
                full_loader_length=len(loader), dispatched_microsteps=prefix.microsteps,
                method='finite sampler prefix, natural exhaustion, explicit verified shutdown')


def register_preflight_hook(torch, report):
    """Subclass the production optimizer hook; do not duplicate its update loop."""
    from mmcv.runner import HOOKS
    from projects.mmdet3d_plugin.core.hooks.equivalent_training import (
        EquivalentCumulativeOptimizerHook)
    dist = torch.distributed

    def require_all(condition, message):
        valid = torch.tensor(int(bool(condition)), device='cuda', dtype=torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid.item():
            raise RuntimeError(message)

    @HOOKS.register_module()
    class M3GPUPreflightOptimizerHook(EquivalentCumulativeOptimizerHook):
        def __init__(self, preflight_updates=5, **kwargs):
            super().__init__(**kwargs)
            self.target = int(preflight_updates)
            self.micro_losses = []
            self.update_times = []
            self.grad_norms = []
            self.hook_handles = []
            self.path_counts = dict(camera_backbone=0, m3_prior=0, m3_transport=0)
            self.started = None

        def before_run(self, runner):
            super().before_run(runner)
            if runner.iter != 0 or runner.epoch != 0:
                raise ValueError('Preflight must start at epoch zero')
            module = runner.model.module
            self.parameters = [(name, param) for name, param in module.named_parameters()
                               if param.requires_grad]
            self.initial = {name: param.detach().cpu().clone()
                            for name, param in self.parameters}
            for name, layer in (
                    ('camera_backbone', module.img_backbone),
                    ('m3_prior', module.doppler_posterior.prior_head),
                    ('m3_transport', module.doppler_posterior.transport_expand)):
                def observe(layer, inputs, output, key=name):
                    self.path_counts[key] += 1
                self.hook_handles.append(layer.register_forward_hook(observe))
            torch.cuda.reset_peak_memory_stats()

        def before_train_iter(self, runner):
            if self.started is None:
                torch.cuda.synchronize()
                self.started = time.perf_counter()

        def before_train_epoch(self, runner):
            super().before_train_epoch(runner)
            self.prefix = arm_finite_loader(
                runner.data_loader, self.target * self.cumulative_iters)

        def clip_grads(self, params):
            # Called by the real accumulation hook after backward and before
            # clipping/step/zero_grad, so actual accumulated gradients are checked.
            params = list(params)
            gradients = [p.grad for p in params if p.requires_grad and p.grad is not None]
            finite = bool(gradients) and bool(torch.stack([
                torch.isfinite(g).all() for g in gradients]).all().item())
            require_all(finite, 'At least one rank has missing or non-finite accumulated gradients')
            norm = super().clip_grads(params)
            require_all(norm is not None and math.isfinite(float(norm)),
                        'At least one rank has a non-finite gradient norm')
            self.grad_norms.append(float(norm))
            return norm

        def after_train_iter(self, runner):
            loss = runner.outputs['loss']
            require_all(bool(torch.isfinite(loss.detach()).all().item()),
                        'At least one rank has a non-finite full-model loss')
            require_all('loss_m3_doppler_nll' in runner.outputs.get('log_vars', {}),
                        'The full-model training path did not report the M3 NLL')
            self.micro_losses.append(float(loss.detach()))
            super().after_train_iter(runner)
            if (runner.inner_iter + 1) % self.cumulative_iters:
                return
            torch.cuda.synchronize()
            self.update_times.append(time.perf_counter() - self.started)
            progress = runner.meta['equivalent_training']
            completed = progress['optimizer_updates_completed']
            if completed > self.target:
                raise RuntimeError('Preflight exceeded its bounded optimizer-update count')
            self.started = time.perf_counter()

        def after_train_epoch(self, runner):
            # The finite index prefix lets the real DataLoader reach its normal
            # StopIteration before we stop MMCV's outer epoch loop. Checkpoints
            # are disabled, and the full len/24-epoch LR budget was never changed.
            super().after_train_epoch(runner)
            progress = runner.meta['equivalent_training']
            require_all(progress['optimizer_updates_completed'] == self.target and
                        runner.iter == self.target * self.cumulative_iters,
                        'Natural loader completion did not commit the requested updates')
            cleanup_error = None
            try:
                report['dataloader_cleanup'] = finish_loader_workers(runner.data_loader, self.prefix)
                report['dataloader_workers_exited_cleanly'] = True
            except Exception as exc:
                cleanup_error = repr(exc)
                report['dataloader_workers_exited_cleanly'] = False
                report['dataloader_cleanup_error'] = cleanup_error
            require_all(cleanup_error is None,
                        'At least one rank failed verified DataLoader cleanup; see rank reports')
            self.finish(runner)
            raise PreflightComplete()

        def finish(self, runner):
            require_all(all(count > 0 for count in self.path_counts.values()),
                        'Camera, M3 prior and sigma transport must all execute on both ranks')
            # Compare every trainable parameter against rank zero in bounded
            # chunks; this is a full synchronization check, not a sampled digest.
            maximum_difference = torch.zeros((), device='cuda')
            finite_parameters = torch.ones((), device='cuda', dtype=torch.bool)
            for _, parameter in self.parameters:
                flat = parameter.detach().reshape(-1)
                finite_parameters &= torch.isfinite(flat).all()
                for chunk in flat.split(1 << 20):
                    reference = chunk.clone()
                    dist.broadcast(reference, src=0)
                    difference = (chunk - reference).abs().max()
                    maximum_difference = torch.maximum(maximum_difference, difference)
            require_all(bool(finite_parameters.item()), 'Optimizer produced non-finite parameters')
            dist.all_reduce(maximum_difference, op=dist.ReduceOp.MAX)
            require_all(float(maximum_difference) == 0.0,
                        'Trainable parameters differ between DDP ranks after the update')
            changed = []
            m3_changed = []
            maximum_delta = 0.0
            for name, parameter in self.parameters:
                delta = float((parameter.detach().cpu() - self.initial[name]).abs().max())
                maximum_delta = max(maximum_delta, delta)
                if delta > 0:
                    changed.append(name)
                    if name.startswith('doppler_posterior.'):
                        m3_changed.append(name)
            require_all(bool(changed) and bool(m3_changed),
                        'Training must update both model and M3 posterior parameters')
            steps = [float(state['step']) for state in runner.optimizer.state.values()
                     if 'step' in state]
            require_all(bool(steps) and max(steps) == self.target,
                        'AdamW state does not match completed optimizer updates')
            for handle in self.hook_handles:
                handle.remove()
            report.update(
                status='PASS', full_camera_m3_forward_backward=True,
                optimizer_updates_completed=self.target,
                raw_microsteps_completed=runner.iter,
                all_losses_finite=True, all_accumulated_gradients_finite=True,
                microstep_losses=self.micro_losses, gradient_norms_before_clip=self.grad_norms,
                final_lr=[group['lr'] for group in runner.optimizer.param_groups],
                adamw_step_range=[min(steps), max(steps)], path_forward_calls=self.path_counts,
                trainable_parameter_tensors=len(self.parameters),
                changed_parameter_tensors=len(changed), changed_m3_parameter_names=m3_changed,
                maximum_parameter_change=maximum_delta,
                maximum_parameter_difference_between_ranks=float(maximum_difference),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                progress=copy.deepcopy(runner.meta['equivalent_training']),
                timing=timing_summary(self.update_times), checkpoints_written=False)
            self.initial.clear()


def dependency_versions(torch):
    versions = {}
    for name in ('torch', 'torchvision', 'mmcv-full', 'mmdet', 'mmsegmentation',
                 'mmdet3d', 'numpy', 'nuscenes-devkit'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'distribution metadata unavailable'
    versions.update(python=platform.python_version(), cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version())
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Resolved formal H200 config, read without modifying it')
    parser.add_argument('--diagnostics-dir', required=True, help='A new, separate directory')
    parser.add_argument('--updates', type=int, default=5)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    if int(os.environ.get('WORLD_SIZE', '1')) != 2:
        parser.error('Launch with torch.distributed.run --nproc_per_node=2')
    sys.path.insert(0, str(ROOT))
    os.chdir(str(ROOT))
    from mmcv import Config
    config_path = Path(args.config).resolve()
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    original = Config.fromfile(str(config_path))._cfg_dict.to_dict()
    config = prepare_config(original, args.diagnostics_dir, args.updates)
    initialization = Path(config['load_from'])
    if not initialization.is_file() or initialization.stat().st_size != 225215819:
        raise ValueError('Common FCOS3D initialization is missing or has an unexpected size')
    import torch
    from mmcv.runner import init_dist
    if not torch.cuda.is_available():
        raise RuntimeError('A real CUDA runtime is required; no CPU fallback')
    init_dist('pytorch', timeout=timedelta(seconds=args.timeout_seconds), **config['dist_params'])
    rank = torch.distributed.get_rank()
    output = Path(config['work_dir'])
    creation = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
            creation[0] = dict(ok=True)
        except Exception as exc:
            creation[0] = dict(ok=False, error=str(exc))
    torch.distributed.broadcast_object_list(creation, src=0)
    if not creation[0]['ok']:
        raise RuntimeError('Cannot create a new diagnostics directory: ' + creation[0]['error'])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    local_rank = int(os.environ['LOCAL_RANK'])
    selected = visible[local_rank] if local_rank < len(visible) else None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = getattr(properties, 'uuid', None)
    if uuid is None and selected and selected.startswith('GPU-'):
        uuid = selected
    report = dict(status='RUNNING', rank=rank, world_size=2,
                  local_rank=int(os.environ['LOCAL_RANK']), config=str(config_path),
                  config_sha256=config_sha256,
                  diagnostics_dir=str(output), initialization=str(initialization),
                  requested_optimizer_updates=args.updates, dependencies=dependency_versions(torch),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  selected_visible_device=selected, gpu_uuid=str(uuid) if uuid else None,
                  cuda_device_ordinal=torch.cuda.current_device(), hostname=platform.node(),
                  gpu_name=torch.cuda.get_device_name(),
                  gpu_capability=list(torch.cuda.get_device_capability()),
                  gpu_total_memory_bytes=properties.total_memory,
                  process_id=os.getpid(), diagnostic_only=True)
    path = output / ('rank_%d.json' % rank)
    try:
        write_json(path, report)
        if config.get('custom_imports'):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**config['custom_imports'])
        importlib.import_module('projects.mmdet3d_plugin')
        from mmdet.apis import set_random_seed
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models import build_model
        from mmdet3d.utils import get_root_logger
        from projects.mmdet3d_plugin.bevformer.apis.train import custom_train_model
        torch.backends.cuda.matmul.allow_tf32 = config['tf32_policy']['matmul']
        torch.backends.cudnn.allow_tf32 = config['tf32_policy']['cudnn']
        torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', False)
        set_random_seed(config['seed'], deterministic=True)
        report['arithmetic'] = dict(
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark)
        register_preflight_hook(torch, report)
        # MMCV 1.4 dump() requires a filename; a nonempty cfg_text avoids
        # trying to read this new, not-yet-created diagnostic config file.
        diagnostic_config = output / 'preflight_config.py'
        cfg = Config(config, cfg_text='M3 GPU preflight diagnostic config copy',
                     filename=str(diagnostic_config))
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        get_root_logger(log_file=str(output / ('rank_%d.log' % rank)),
                        log_level=cfg.log_level, name='mmdet')
        if rank == 0:
            cfg.dump(str(diagnostic_config))
            write_json(output / 'input_protocol.json', original.get('m3_protocol', {}))
            try:
                inventory = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,driver_version',
                     '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
                (output / 'gpu_inventory.txt').write_text(inventory.stdout + inventory.stderr)
            except (OSError, subprocess.TimeoutExpired) as exc:
                (output / 'gpu_inventory.txt').write_text(str(exc) + '\n')
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        dataset = build_dataset(cfg.data.train)
        model.CLASSES = dataset.CLASSES
        report['dataset_length'] = len(dataset)
        report['dataset_flag_counts'] = torch.bincount(
            torch.as_tensor(dataset.flag, dtype=torch.int64)).tolist()
        try:
            custom_train_model(
                model, [dataset], cfg, distributed=True, validate=False, timestamp=timestamp,
                meta=dict(seed=cfg.seed, config=cfg.pretty_text, diagnostic_only=True))
        except PreflightComplete:
            pass
        else:
            raise RuntimeError('Runner exited without the expected complete-update stop')
        checkpoints = list(output.rglob('*.pth'))
        if checkpoints:
            raise RuntimeError('Preflight unexpectedly wrote checkpoint files')
        write_json(path, report)
        reports = [None] * 2
        torch.distributed.all_gather_object(reports, report)
        if rank == 0:
            if any(r.get('status') != 'PASS' or r.get('rank') != index or
                   r.get('config_sha256') != config_sha256 or
                   r.get('optimizer_updates_completed') != args.updates or
                   r.get('dataloader_workers_exited_cleanly') is not True
                   for index, r in enumerate(reports)):
                raise RuntimeError('Rank reports do not agree on a successful preflight')
            write_json(output / 'preflight_summary.json', dict(
                status='PASS', scope='Full real-data camera/M3 forward, backward and two-rank DDP',
                config=str(config_path), config_sha256=config_sha256, world_size=2,
                optimizer_updates_completed=args.updates,
                dataloader_workers_exited_cleanly=True,
                effective_global_batch=4, reports=reports,
                training_checkpoint_saved=False,
                limitations=['Short diagnostic, not a scientific effectiveness result',
                             'Timing includes checks and startup; do not extrapolate a final ETA']))
            print('M3 GPU preflight PASS: ' + str(output / 'preflight_summary.json'), flush=True)
    except Exception as exc:
        report.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        write_json(path, report)
        raise
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
