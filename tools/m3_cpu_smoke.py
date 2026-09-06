"""Real nuScenes input + configured M3 module smoke; no CUDA training.

This is not a full camera backbone forward or a scientific accuracy result.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from mmcv import Config
from nuscenes.nuscenes import NuScenes

from m3_build_observation_cache import load_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--config', default='projects/configs/radarflowocc/action_condition_GMO_radar_m3.py')
    parser.add_argument('--output', required=True)
    parser.add_argument('--build-detector', action='store_true')
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError('Run this diagnostic with CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = Config.fromfile(args.config)
    assert cfg.model.doppler_advection is None
    assert cfg.model.get('doppler_flow_loss') is None
    assert cfg.data.train.radar_cfg is None
    assert cfg.model.scientific_eval
    report = dict(torch=torch.__version__, numpy=np.__version__,
                  status='CPU_MODULE_SMOKE_ONLY', full_camera_forward=False)
    if args.build_detector:
        # Legacy imports JIT-build CUDA extensions even for CPU construction.
        # Keep build products isolated from any running training job's cache.
        os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(
            Path(args.output).resolve().parent / 'torch_extensions'))
        os.environ.setdefault('MAX_JOBS', '1')
        if not os.environ.get('TORCH_CUDA_ARCH_LIST'):
            raise RuntimeError('CPU-only detector construction requires an explicit '
                               'TORCH_CUDA_ARCH_LIST supported by installed nvcc')
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import projects.mmdet3d_plugin  # registry registration
        from mmdet3d.models import build_model
        model = build_model(cfg.model)
        report['detector_class'] = type(model).__name__
        report['model_parameters'] = sum(p.numel() for p in model.parameters())
        report['predict_flow'] = model.predict_flow
        del model
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.data_root, verbose=False)
    options = dict(cfg.data.train.radar_observation_cfg)
    options.update(cache_dir=None, cache_readonly=False)
    loader = load_loader()(nusc=nusc, **options)
    observations = None
    checked = []
    for sample in nusc.sample[:10]:
        packed = loader(sample['token'])
        valid = packed[:, 8] > 0
        checked.append(dict(token=sample['token'], valid_returns=int(valid.sum())))
        if observations is None and valid.sum() >= 2:
            observations = torch.from_numpy(packed).unsqueeze(0)
    if observations is None:
        raise RuntimeError('No sample with at least two causal returns in the probe')
    report['samples'] = checked
    active = observations[0, :, 8] > 0
    report['selected_lag_range'] = [float(x) for x in
                                  (observations[0, active, 7].min(), observations[0, active, 7].max())]
    report['selected_radial_range'] = [float(x) for x in
                                     (observations[0, active, 6].min(), observations[0, active, 6].max())]
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        'm3_motion_smoke', root / 'projects/mmdet3d_plugin/bevformer/detectors/doppler_posterior_transport.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    motion = module.DopplerPosteriorTransport(**cfg.model.doppler_posterior)
    with torch.no_grad():
        motion.prior_gate.fill_(0.1)
    h, w, c = motion.bev_h, motion.bev_w, motion.embed_dims
    camera = torch.randn(1, h*w, c, requires_grad=True)
    ys, xs = torch.meshgrid(2*(torch.arange(h)+0.5)/h-1,
                           2*(torch.arange(w)+0.5)/w-1)
    grid = torch.stack((xs, ys), dim=-1).reshape(1, h*w, 2)
    start = time.monotonic()
    state = motion.prepare(camera, observations, training=True)
    prior = motion.transport(camera, state, 1, grid)
    loss = prior.square().mean() + 0.05*state['loss_doppler_nll']
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prior).all()
    gradient = motion.prior_head[-1].bias.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    report.update(module_parameters=sum(p.numel() for p in motion.parameters()),
                  feature_shape=list(prior.shape),
                  support_returns=int(state['support_mask'].sum()),
                  heldout_returns=int(state['holdout_mask'].sum()),
                  synthetic_feature_loss=float(loss.detach()),
                  prior_output_bias_gradient=gradient.tolist(),
                  cpu_module_seconds=time.monotonic()-start)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
