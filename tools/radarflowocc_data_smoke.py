#!/usr/bin/env python3
"""Smoke-test the RadarFlowOcc dataset path on one real nuScenes sample."""

import argparse
import importlib
import json
import os
import time

import mmcv
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='projects/configs/radarflowocc/action_condition_GMO_radar.py')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument(
        '--cache-dir', help='Override train.radar_cfg.cache_dir')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cache_dir:
        cfg.data.train.radar_cfg.cache_dir = os.path.abspath(args.cache_dir)
    if cfg.get('plugin', False):
        importlib.import_module('projects.mmdet3d_plugin')

    # The dataset constructor otherwise emits one terminal update per info.
    mmcv.track_iter_progress = lambda iterable: iterable
    started = time.time()
    dataset = build_dataset(cfg.data.train)
    sample = dataset[args.index]
    elapsed = time.time() - started

    radar = sample['radar_bev'].data.numpy()
    presence = radar[0] > 0
    summary = {
        'dataset': type(dataset).__name__,
        'dataset_length': len(dataset),
        'sample_index': args.index,
        'elapsed_seconds': round(elapsed, 3),
        'radar_shape': list(radar.shape),
        'radar_dtype': str(radar.dtype),
        'occupied_cells': int(presence.sum()),
        'channel_nonzero': [
            int(np.count_nonzero(channel)) for channel in radar],
        'channel_min': [float(channel.min()) for channel in radar],
        'channel_max': [float(channel.max()) for channel in radar],
    }

    radar_loader = dataset.radar_bev_loader
    expected_channels = getattr(
        radar_loader, 'num_features', radar_loader.NUM_FEATURES)
    assert radar.shape == (expected_channels, 200, 200)
    assert radar.dtype == np.float32
    assert presence.any()
    assert np.isfinite(radar).all()
    assert np.abs(radar[4:6]).sum() > 0
    if getattr(radar_loader, 'include_radial_features', False):
        radial_direction_norm = np.sqrt(radar[9] ** 2 + radar[10] ** 2)
        assert np.count_nonzero(radial_direction_norm[presence]) > 0
        assert np.isfinite(radar[8:11]).all()
        summary['radial_direction_norm_min'] = float(
            radial_direction_norm[presence].min())
        summary['radial_direction_norm_max'] = float(
            radial_direction_norm[presence].max())
        summary['radial_velocity_mps_min'] = float(
            radar[8][presence].min() *
            radar_loader.velocity_norm)
        summary['radial_velocity_mps_max'] = float(
            radar[8][presence].max() *
            radar_loader.velocity_norm)
    summary['status'] = 'PASS'
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
