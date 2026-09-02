#!/usr/bin/env python3
"""Precompute float32 RadarFlowOcc BEV rasters for nuScenes samples."""

import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
import multiprocessing as mp
import os
from os import path as osp
import tempfile
import time

import mmcv
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset


_RADAR_LOADER = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='projects/configs/radarflowocc/action_condition_GMO_radar.py')
    parser.add_argument(
        '--splits', nargs='+', choices=('train', 'val', 'test'),
        default=('train',))
    parser.add_argument('--cache-dir')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--verify', type=int, default=3)
    return parser.parse_args()


def cache_one(sample_token):
    started = time.time()
    cache_path = _RADAR_LOADER.cache_path(sample_token)
    already_cached = osp.isfile(cache_path)
    bev = _RADAR_LOADER(sample_token)
    return (
        sample_token,
        'cached' if already_cached else 'written',
        time.time() - started,
        int(np.count_nonzero(bev[0])),
    )


def unique_sample_tokens(dataset):
    indices = getattr(dataset, 'usable_index', range(len(dataset.data_infos)))
    return list(dict.fromkeys(dataset.data_infos[index]['token']
                              for index in indices))


def write_manifest(cache_dir, manifest):
    manifest_path = osp.join(cache_dir, 'manifest.json')
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=cache_dir,
                prefix='.manifest.', suffix='.json',
                delete=False) as temp_file:
            temp_path = temp_file.name
            json.dump(manifest, temp_file, indent=2, sort_keys=True)
            temp_file.write('\n')
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path is not None and osp.exists(temp_path):
            os.unlink(temp_path)


def main():
    global _RADAR_LOADER

    args = parse_args()
    if args.workers <= 0:
        raise ValueError('--workers must be positive')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')

    cfg = Config.fromfile(args.config)
    if cfg.get('plugin', False):
        importlib.import_module('projects.mmdet3d_plugin')
    mmcv.track_iter_progress = lambda iterable: iterable

    configured_cache = cfg.data.train.radar_cfg.get('cache_dir', None)
    selected_cache = args.cache_dir or configured_cache
    if selected_cache is None:
        raise ValueError(
            'A cache directory is required via --cache-dir or radar_cfg')
    cache_dir = osp.abspath(selected_cache)
    os.makedirs(cache_dir, exist_ok=True)

    manifest = {
        'cache_format': 'npy-float32-v1',
        'cache_dir': cache_dir,
        'config': osp.abspath(args.config),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'splits': {},
    }
    total_started = time.time()
    fork_context = mp.get_context('fork')

    for split in args.splits:
        split_cfg = copy.deepcopy(cfg.data[split])
        # ``samples_per_gpu`` is a DataLoader option that some validation/test
        # configs keep inside the split dict.  It must not be forwarded to the
        # dataset constructor when precomputing caches directly.
        split_cfg.pop('samples_per_gpu', None)
        split_cfg.radar_cfg.cache_dir = cache_dir
        split_cfg.radar_cfg.cache_readonly = False
        dataset = build_dataset(split_cfg)
        _RADAR_LOADER = dataset.radar_bev_loader
        # Older M0-only loaders predate the explicit feature-count metadata.
        # They always emit the original eight channels, so keep the cache tool
        # compatible without changing their rasterization behavior.
        manifest['num_features'] = getattr(
            _RADAR_LOADER, 'num_features',
            getattr(_RADAR_LOADER, 'NUM_FEATURES', 8))
        manifest['include_radial_features'] = getattr(
            _RADAR_LOADER, 'include_radial_features', False)
        tokens = unique_sample_tokens(dataset)
        if args.limit is not None:
            tokens = tokens[:args.limit]

        print('split={} samples={} workers={} cache={}'.format(
            split, len(tokens), args.workers, cache_dir), flush=True)
        split_started = time.time()
        written = 0
        cached = 0
        occupied_total = 0
        elapsed_total = 0.0

        with fork_context.Pool(processes=args.workers) as pool:
            iterator = pool.imap_unordered(cache_one, tokens, chunksize=1)
            for completed, (_, status, elapsed, occupied) in enumerate(
                    iterator, start=1):
                written += int(status == 'written')
                cached += int(status == 'cached')
                occupied_total += occupied
                elapsed_total += elapsed
                if completed == 1 or completed % 100 == 0 \
                        or completed == len(tokens):
                    wall = time.time() - split_started
                    rate = completed / max(wall, 1e-9)
                    remaining = (len(tokens) - completed) / max(rate, 1e-9)
                    print(
                        'split={} completed={}/{} written={} cached={} '
                        'rate={:.2f} samples/s eta={:.1f} min'.format(
                            split, completed, len(tokens), written, cached,
                            rate, remaining / 60.0),
                        flush=True)

        verify_count = min(args.verify, len(tokens))
        if verify_count:
            positions = np.linspace(
                0, len(tokens) - 1, verify_count, dtype=np.int64)
            for position in positions:
                token = tokens[int(position)]
                cached_bev = _RADAR_LOADER(token)
                online_bev = _RADAR_LOADER.compute(token)
                if not np.array_equal(cached_bev, online_bev):
                    raise AssertionError(
                        'Cache verification failed for {}'.format(token))

        split_wall = time.time() - split_started
        manifest['splits'][split] = {
            'samples': len(tokens),
            'written': written,
            'already_cached': cached,
            'wall_seconds': round(split_wall, 3),
            'worker_seconds': round(elapsed_total, 3),
            'mean_occupied_cells': (
                occupied_total / len(tokens) if tokens else 0.0),
            'verified_samples': verify_count,
        }
        del dataset
        _RADAR_LOADER = None

    manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['wall_seconds'] = round(time.time() - total_started, 3)
    write_manifest(cache_dir, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
