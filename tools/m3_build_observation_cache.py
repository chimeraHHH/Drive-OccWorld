"""Explicit, resumable CPU cache build from current sample tokens only.

Run only after reviewing storage capacity and the resolved config. No SHA256
data pass is performed; NPZ reads enforce the schema and configuration identity.
"""
import argparse
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import types


_WORKER_LOADER = None


def write_status(cache_dir, record):
    """Publish state atomically, including when replacing a previous COMPLETE."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=str(cache),
                                         prefix='.build_status.', suffix='.json',
                                         delete=False) as handle:
            temporary = handle.name
            json.dump(record, handle, indent=2)
        os.replace(temporary, str(cache / 'build_status.json'))
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def process_token(token):
    # Fork inherits the already loaded nuScenes tables and the loader. Only
    # the token is transferred; never serialize point arrays or the database.
    _WORKER_LOADER(token)
    return token


def build_cache(loader, all_tokens, cache_dir, splits, workers=1, limit=None,
                progress_interval=1000):
    """Validate/reuse existing NPZs and build absent entries with the same loader."""
    global _WORKER_LOADER
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('workers must be an integer from 1 to 8')
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('limit must be a positive integer')
    if progress_interval <= 0:
        raise ValueError('progress_interval must be positive')
    all_tokens = sorted(set(all_tokens))
    selected = all_tokens if limit is None else all_tokens[:limit]
    record = dict(status='BUILDING', tokens_processed=0,
                  tokens_required=len(all_tokens), tokens_selected=len(selected),
                  splits=list(splits), workers=workers, limit=limit,
                  manifest=loader.manifest)
    write_status(cache_dir, record)
    try:
        if not all_tokens:
            raise ValueError('The requested splits contain no sample tokens')
        _WORKER_LOADER = loader

        def consume(completions):
            for _ in completions:
                record['tokens_processed'] += 1
                if (record['tokens_processed'] % progress_interval == 0 or
                        record['tokens_processed'] == len(selected)):
                    write_status(cache_dir, record)
                    print('{}/{}'.format(record['tokens_processed'], len(selected)), flush=True)

        if workers == 1:
            consume(map(process_token, selected))
        else:
            if 'fork' not in multiprocessing.get_all_start_methods():
                raise RuntimeError('Parallel cache building requires fork; spawn would copy the nuScenes database')
            context = multiprocessing.get_context('fork')
            with context.Pool(processes=workers) as pool:
                consume(pool.imap_unordered(process_token, selected, chunksize=8))
        # An explicitly limited invocation is always a smoke, even if its
        # limit happens to exceed the number of currently available tokens.
        record['status'] = 'COMPLETE' if limit is None else 'LIMITED_SMOKE_ONLY'
        write_status(cache_dir, record)
        return record
    except BaseException as error:
        record.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        write_status(cache_dir, record)
        raise
    finally:
        _WORKER_LOADER = None


def load_loader():
    # Avoid initializing the detector registry or CUDA extensions for a CPU
    # data preparation job. The production relative radar_bev import is kept.
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType('m3_cache_datasets')
    package.__path__ = [str(root / 'projects/mmdet3d_plugin/datasets')]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        'm3_cache_datasets.radar_observations',
        Path(package.__path__[0]) / 'radar_observations.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NuScenesRadarObservations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--limit', type=int, help='Only for smoke checks; never mark a limited cache complete')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=1,
                        help='Fork workers sharing the already loaded nuScenes database (default: 1)')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be a positive integer')
    # Clear any stale COMPLETE before loading annotations/the database. Even
    # failures during preparation must leave a truthful failure state.
    preparing = dict(status='BUILDING', phase='loading_metadata', tokens_processed=0,
                     tokens_required=None, splits=args.splits, workers=args.workers,
                     limit=args.limit)
    write_status(args.cache_dir, preparing)
    try:
        from mmcv import Config
        import mmcv
        from nuscenes.nuscenes import NuScenes
        cfg = Config.fromfile(args.config)
        tokens = set()
        for split in args.splits:
            ds = cfg.data[split]
            payload = mmcv.load(ds.ann_file)
            infos = payload['infos'] if isinstance(payload, dict) else payload
            tokens.update(info['token'] for info in infos)
        ds = cfg.data[args.splits[0]]
        options = dict(ds.radar_observation_cfg)
        options.update(cache_dir=args.cache_dir, cache_readonly=False)
        nusc = NuScenes(version=ds.get('version', 'v1.0-trainval'), dataroot=ds.data_root, verbose=False)
        loader = load_loader()(nusc=nusc, **options)
    except BaseException as error:
        preparing.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        write_status(args.cache_dir, preparing)
        raise
    record = build_cache(loader, tokens, args.cache_dir, args.splits,
                         workers=args.workers, limit=args.limit)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
