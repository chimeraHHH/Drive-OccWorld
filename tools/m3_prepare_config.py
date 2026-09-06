"""Resolve a common, reviewable experiment protocol; this does not start training."""
import argparse
from pathlib import Path

from mmcv import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', default='projects/configs/radarflowocc/action_condition_GMO_radar_m3.py')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--occ-root', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--initialization', required=True,
                        help='scratch or an explicitly chosen common checkpoint path')
    parser.add_argument('--output', required=True)
    parser.add_argument('--radar-cache', help='Existing M3 per-return cache; read-only during training')
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    cfg = Config.fromfile(args.base_config)
    root = str(Path(args.data_root).resolve()) + '/'
    if not Path(root).is_dir() or not Path(args.occ_root).is_dir():
        raise FileNotFoundError('Both data-root and occ-root must be existing directories')
    if args.initialization != 'scratch' and not Path(args.initialization).is_file():
        raise FileNotFoundError('The explicitly selected initialization checkpoint is missing')
    for split in ('train', 'val', 'test'):
        ds = cfg.data[split]
        ds.data_root = root
        ds.ann_file = root + Path(ds.ann_file).name
        if not Path(ds.ann_file).is_file():
            raise FileNotFoundError(ds.ann_file)
        for step in ds.pipeline:
            if step['type'] == 'LoadOccupancy':
                step['occ_path'] = str(Path(args.occ_root).resolve())
            if step['type'] == 'CropResizeFlipImage':
                step['data_aug_conf']['reisze'] = [args.height]
        if args.radar_cache:
            if not ds.get('radar_observation_cfg'):
                raise ValueError('--radar-cache here accepts only the M3 observation format')
            ds.radar_observation_cfg.cache_dir = str(Path(args.radar_cache).resolve())
            ds.radar_observation_cfg.cache_readonly = True
    cfg.train_pipeline = cfg.data.train.pipeline
    cfg.test_pipeline = cfg.data.test.pipeline
    cfg.evaluation.pipeline = cfg.data.test.pipeline
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = args.workers
    cfg.model.scientific_eval = True
    cfg.optimizer.lr = args.lr
    cfg.total_epochs = args.epochs
    cfg.runner.max_epochs = args.epochs
    cfg.work_dir = str(Path(args.work_dir).resolve())
    cfg.load_from = None if args.initialization == 'scratch' else str(Path(args.initialization).resolve())
    cfg.resume_from = None
    cfg.seed = args.seed
    cfg.m3_protocol = dict(
        status='REQUIRES_TRAINING_AND_EVALUATION',
        ego_condition='given_future_ground_truth_ego_transform_and_action',
        labels='LiDAR-derived GMO/non-GMO occupancy',
        image_resize_train=[args.height], batch_per_gpu=1,
        epochs=args.epochs, lr=args.lr, seed=args.seed,
        initialization=args.initialization,
        backbone_frozen_stages=cfg.model.img_backbone.get('frozen_stages', -1),
        backbone_norm_eval=cfg.model.img_backbone.get('norm_eval', False),
        backbone_norm_requires_grad=cfg.model.img_backbone.get('norm_cfg', {}).get('requires_grad', True),
        note='GPU count/global batch and software revision must be logged by launcher')
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an already resolved experiment config')
    destination.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(destination))
    print(str(destination.resolve()))


if __name__ == '__main__':
    main()
