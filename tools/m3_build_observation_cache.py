"""Explicit, resumable CPU cache build from current sample tokens only.

Run only after reviewing storage capacity and the resolved config. No SHA256
data pass is performed; NPZ reads enforce the schema and configuration identity.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

from mmcv import Config
import mmcv
from nuscenes.nuscenes import NuScenes


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
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    tokens = set()
    for split in args.splits:
        ds = cfg.data[split]
        payload = mmcv.load(ds.ann_file)
        infos = payload['infos'] if isinstance(payload, dict) else payload
        tokens.update(info['token'] for info in infos)
    all_tokens = sorted(tokens)
    selected = all_tokens if args.limit is None else all_tokens[:args.limit]
    ds = cfg.data[args.splits[0]]
    options = dict(ds.radar_observation_cfg)
    options.update(cache_dir=args.cache_dir, cache_readonly=False)
    nusc = NuScenes(version=ds.get('version', 'v1.0-trainval'), dataroot=ds.data_root, verbose=False)
    loader = load_loader()(nusc=nusc, **options)
    count = 0
    for token in selected:
        loader(token)
        count += 1
        if count % 1000 == 0:
            print('{}/{}'.format(count, len(selected)), flush=True)
    record = dict(status='COMPLETE' if len(selected) == len(all_tokens) else 'LIMITED_SMOKE_ONLY',
                  tokens_processed=count, tokens_required=len(all_tokens),
                  splits=args.splits, manifest=loader.manifest)
    Path(args.cache_dir, 'build_status.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
