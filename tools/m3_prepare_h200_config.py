"""Resolve the H200 M3 protocol against the actual L40S configuration.

Run in the selected H200 environment after exposing datasets and checkpoint.
This command only prepares and validates configuration; it never trains.
Path arguments can also be supplied using the documented M3_* environment
variables. The code location is derived from this script, not a host path.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import types


REFERENCE_CHECKPOINT_BYTES = 225215819
OPTIMIZER_UPDATES_PER_EPOCH = 5983
EXPECTED_TRAIN_ANN_BYTES = 518842050
HOST_DEFAULTS = dict(
    data_root='/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/nuscenes_driveocc',
    occ_root='/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/openoccupancy/nuScenes-Occupancy-v0.1',
    pretrained='/storage/data/metaiot_data/huayiming/RadarFlowOcc/checkpoints/r101_dcn_fcos3d_pretrain_l40s_reference.pth',
    work_dir='/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/work_dirs/gmo_radar_m3_h200720_w2_2g_b1_acc2_seed0_fcos3d')
METHOD_MODEL_FIELDS = {
    'doppler_advection', 'doppler_flow_loss', 'doppler_posterior',
    'doppler_nll_weight', 'scientific_eval'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def observation_manifest(dataset):
    """Use the production loader's manifest without loading nuScenes tables.

    The loader keeps devkit imports lazy. Its constructor needs only a small
    object exposing version/dataroot and NumPy; cache_dir=None avoids writes.
    """
    root = Path(__file__).resolve().parents[1]
    path = root / 'projects/mmdet3d_plugin/datasets/radar_observations.py'
    spec = importlib.util.spec_from_file_location('_m3_h200_manifest_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = copy.deepcopy(dataset.get('radar_observation_cfg'))
    _require(isinstance(options, dict) and dataset.get('radar_cfg') is None,
             'Only the M3 per-return observation cache may be bound')
    options.update(cache_dir=None, cache_readonly=False)
    nusc = types.SimpleNamespace(version=dataset.get('version', 'v1.0-trainval'),
                                 dataroot=dataset['data_root'])
    return module.NuScenesRadarObservations(nusc=nusc, **options).manifest


def bind_radar_cache(config, cache_dir):
    """Bind only an explicitly completed, configuration-identical M3 cache.

    Read a single build_status.json; do not scan images, PCDs, or NPZ files.
    During training the readonly loader validates each requested NPZ's own
    schema, sample token and values, and fails if an entry is absent/corrupt.
    """
    cache = Path(cache_dir).resolve()
    status_path = cache / 'build_status.json'
    _require(cache.is_dir() and status_path.is_file(),
             'M3 radar cache requires an existing build_status.json')
    try:
        status = json.loads(status_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError('Cannot read M3 radar cache build_status.json') from error
    _require(isinstance(status, dict) and status.get('status') == 'COMPLETE',
             'M3 radar cache must be COMPLETE, not a limited smoke or unfinished build')
    processed, required = status.get('tokens_processed'), status.get('tokens_required')
    _require(type(processed) is int and type(required) is int and
             processed == required and required > 0,
             'M3 radar cache requires tokens_processed == tokens_required > 0')
    splits = status.get('splits')
    _require(isinstance(splits, list) and all(isinstance(split, str) for split in splits)
             and {'train', 'val'}.issubset(set(splits)),
             'M3 radar cache must cover both requested splits: train and val')
    manifests = {split: observation_manifest(config['data'][split])
                 for split in ('train', 'val', 'test')}
    for split, expected in manifests.items():
        _require(json.dumps(status.get('manifest'), sort_keys=True, separators=(',', ':'))
                 == json.dumps(expected, sort_keys=True, separators=(',', ':')),
                 'M3 radar cache manifest does not match current ' + split + ' configuration')
    # Validate all three before modifying any split, so rejected caches never
    # leave behind a partially bound configuration.
    for split in ('train', 'val', 'test'):
        config['data'][split]['radar_observation_cfg'].update(
            cache_dir=str(cache), cache_readonly=True)
    config['radar_observation_cfg'] = copy.deepcopy(
        config['data']['train']['radar_observation_cfg'])
    config['m3_protocol']['radar_cache'] = dict(
        directory=str(cache), status_file=str(status_path),
        status='COMPLETE', tokens_processed=processed, tokens_required=required,
        covered_splits=splits, train_val_test_readonly=True,
        validation='build record and production manifest; per-entry validation at read time')


def verify_reference(reference):
    """Fail if an accidental historical or differently budgeted run is supplied."""
    _require(reference['data']['samples_per_gpu'] == 1, 'Reference physical batch is not one')
    _require(len(reference['gpu_ids']) == 4, 'Reference must have four physical ranks')
    _require(reference['data']['workers_per_gpu'] == 2, 'Reference workers differ')
    _require(reference['runner'] == dict(type='EpochBasedRunner', max_epochs=24),
             'Reference runner differs from the verified 24-epoch experiment')
    _require(reference['optimizer'] == dict(
        type='AdamW', lr=0.0001, weight_decay=0.01,
        paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1)))),
        'Reference optimizer differs')
    _require(reference['optimizer_config'] == dict(grad_clip=dict(max_norm=35, norm_type=2)),
             'Reference gradient clipping differs')
    expected_lr = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=500,
                       warmup_ratio=1. / 3., min_lr_ratio=0.001)
    _require(reference['lr_config'] == expected_lr, 'Reference LR schedule differs')
    _require(reference.get('fp16') is None, 'Reference is not the verified FP32 run')
    _require(Path(reference['load_from']).name == 'r101_dcn_fcos3d_pretrain.pth',
             'Reference initialization is not the FCOS3D pretrained checkpoint')
    backbone = reference['model']['img_backbone']
    _require(backbone['frozen_stages'] == 1 and backbone['norm_eval'] and
             backbone['norm_cfg']['requires_grad'] is False,
             'Reference backbone freezing/normalization differs')
    augmentations = [s for s in reference['data']['train']['pipeline']
                     if s['type'] == 'CropResizeFlipImage']
    _require(len(augmentations) == 1 and
             augmentations[0]['data_aug_conf']['reisze'] == [720],
             'Use the active train pipeline, which must resize only to 720')


def make_config(candidate, reference, data_root, occ_root, pretrained, work_dir,
                reference_path=None):
    """Build a plain dictionary without importing the heavy training stack."""
    verify_reference(reference)
    config = copy.deepcopy(candidate)
    _require(config['data']['train']['radar_observation_cfg']['nsweeps'] == 1,
             'Do not silently change the M3 one-sweep method')
    _require(config['model']['doppler_posterior']['max_time_lag'] == 0.15,
             'M3 retained-return age differs')
    # The complete shared model and data protocol come from the actual L40S
    # run. Only explicitly named M3 mechanism fields differ.
    model = copy.deepcopy(reference['model'])
    for field in METHOD_MODEL_FIELDS:
        if field in config['model']:
            model[field] = copy.deepcopy(config['model'][field])
        else:
            model.pop(field, None)
    config['model'] = model
    _require(not model['turn_on_flow'] and not model['turn_on_plan'],
             'The comparison requires flow and planning branches disabled')
    config['data'] = copy.deepcopy(reference['data'])
    root = str(Path(data_root).resolve()) + '/'
    occupancy = str(Path(occ_root).resolve())
    for split in ('train', 'val', 'test'):
        dataset = config['data'][split]
        dataset['data_root'] = root
        dataset['ann_file'] = root + Path(dataset['ann_file']).name
        dataset['radar_cfg'] = None
        dataset['radar_observation_cfg'] = copy.deepcopy(
            candidate['data'][split]['radar_observation_cfg'])
        # No M1 clipped raster cache can be reused as per-return M3 input.
        dataset['radar_observation_cfg']['cache_dir'] = None
        dataset['radar_observation_cfg']['cache_readonly'] = False
        for transform in dataset['pipeline']:
            if transform['type'] == 'LoadOccupancy':
                transform['occ_path'] = occupancy
    config['data']['train']['future_metadata_only'] = True
    config['data']['samples_per_gpu'] = 1
    config['data']['workers_per_gpu'] = 2
    config['data']['shuffler_sampler'] = dict(
        type='VirtualDistributedGroupSampler', reference_world_size=4)
    config['data_root'] = root
    config['occ_path'] = occupancy
    config['train_pipeline'] = copy.deepcopy(config['data']['train']['pipeline'])
    config['test_pipeline'] = copy.deepcopy(config['data']['test']['pipeline'])
    config['ida_aug_conf'] = copy.deepcopy(next(
        step['data_aug_conf'] for step in config['train_pipeline']
        if step['type'] == 'CropResizeFlipImage'))
    config['evaluation'] = copy.deepcopy(reference['evaluation'])
    config['evaluation']['pipeline'] = copy.deepcopy(config['test_pipeline'])
    config['optimizer'] = copy.deepcopy(reference['optimizer'])
    config['optimizer_config'] = dict(
        type='EquivalentCumulativeOptimizerHook', cumulative_iters=2,
        reference_world_size=4,
        expected_optimizer_steps_per_epoch=OPTIMIZER_UPDATES_PER_EPOCH,
        grad_clip=copy.deepcopy(reference['optimizer_config']['grad_clip']))
    config['lr_config'] = copy.deepcopy(reference['lr_config'])
    config['lr_config'].update(policy='EquivalentCosineAnnealing',
                               cumulative_iters=2, by_epoch=True)
    for key in ('runner', 'total_epochs', 'checkpoint_config', 'custom_hooks',
                'workflow', 'dist_params', 'log_config'):
        config[key] = copy.deepcopy(reference[key])
    config['log_config']['interval'] = reference['log_config']['interval'] * 2
    config.update(seed=0, gpu_ids=range(2), fp16=None, close_tf32=False,
                  tf32_policy=dict(matmul=True, cudnn=True), cudnn_benchmark=False,
                  load_from=str(Path(pretrained).resolve()), resume_from=None,
                  work_dir=str(Path(work_dir).resolve()), m3_h200_requires_resolution=False)
    config.pop('radar_cfg', None)
    config['radar_observation_cfg'] = copy.deepcopy(config['data']['train']['radar_observation_cfg'])
    config['m3_protocol'] = dict(
        status='HOST_RUNTIME_GPU_SMOKE',
        reference_config=str(reference_path) if reference_path else None,
        initialization='same FCOS3D pretrained artifact as L40S; start epoch 1',
        reference_checkpoint_bytes=REFERENCE_CHECKPOINT_BYTES,
        physical_world_size=2, physical_microbatch=1, cumulative_iters=2,
        effective_global_batch=4, optimizer_updates_per_epoch=OPTIMIZER_UPDATES_PER_EPOCH,
        total_optimizer_updates=OPTIMIZER_UPDATES_PER_EPOCH * 24,
        raw_microsteps_per_epoch=OPTIMIZER_UPDATES_PER_EPOCH * 2,
        warmup_optimizer_updates=500, epochs=24, seed=0,
        train_image_resize=[720], validation_during_training=False,
        precision='FP32 with TF32 matmul and cuDNN enabled, as observed on L40S',
        ego_condition='given future ground-truth ego transforms and actions',
        method_deltas=['M3 one sweep versus M1 five sweeps',
                       'per-return posterior, held-out NLL and sigma-point transport'],
        equivalence='shared optimization/data protocol and effective batch; not bitwise reproduction',
        checkpoint_iter_unit='raw_microstep', lr_warmup_unit='optimizer_update')
    return config


def verify_inputs(config, code_root, create_data_links=False):
    """Check mounted paths; optionally create only absent relative data links."""
    data = Path(config['data_root'])
    occupancy = Path(config['occ_path'])
    checkpoint = Path(config['load_from'])
    _require(data.is_dir() and occupancy.is_dir(), 'Dataset and occupancy mounts must exist')
    for split in ('train', 'val', 'test'):
        _require(Path(config['data'][split]['ann_file']).is_file(),
                 'Missing annotation: ' + config['data'][split]['ann_file'])
    train_annotation = Path(config['data']['train']['ann_file'])
    _require(train_annotation.stat().st_size == EXPECTED_TRAIN_ANN_BYTES,
             'Train annotation byte size differs from the verified L40S/H200 artifact')
    _require(checkpoint.is_file() and checkpoint.stat().st_size == REFERENCE_CHECKPOINT_BYTES,
             'FCOS3D initialization checkpoint missing or unexpected byte size')
    for relative in ('v1.0-trainval', 'samples', 'sweeps', 'can_bus'):
        _require((data / relative).is_dir(), 'Missing nuScenes component: ' + relative)
    root = Path(code_root).resolve()
    for name, destination in (('nuscenes', data), ('nuScenes-Occupancy', occupancy)):
        link = root / 'data' / name
        if link.exists() or link.is_symlink():
            _require(link.resolve() == destination.resolve(),
                     str(link) + ' already exists and resolves to a different dataset')
        elif create_data_links:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(destination, target_is_directory=True)
        else:
            raise ValueError('Relative annotation path requires ' + str(link) +
                             '; supply --create-data-links to create absent links')
    work_dir = Path(config['work_dir'])
    _require(not work_dir.exists() or not any(work_dir.iterdir()),
             'Use a new empty work directory; the formal run must start at epoch 1')


def write_resolved_config(config, output):
    """Support MMCV 1.4's filename-based dump without reading a missing file."""
    from mmcv import Config
    output = Path(output)
    _require(output.suffix == '.py', 'Resolved MMCV config must use a .py filename')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an already resolved config')
    output.parent.mkdir(parents=True, exist_ok=True)
    # MMCV 1.4 dump calls self.filename.endswith; passing only cfg_dict sets
    # filename=None. Supplying filename alone also tries to read that absent
    # file in __init__, so provide nonempty cfg_text as well.
    Config(config, cfg_text='# Resolved H200 M3 equivalent training config\n',
           filename=str(output)).dump(str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--reference-config', default=os.environ.get(
        'M3_L40S_REFERENCE_CONFIG', str(root / 'DOCS/m3_h200_20260907/l40s_resolved_reference.py')))
    parser.add_argument('--base-config', default=str(root / 'projects/configs/radarflowocc/action_condition_GMO_radar_m3_h200_equivalent.py'))
    for option in ('data-root', 'occ-root', 'pretrained', 'work-dir', 'output'):
        default = HOST_DEFAULTS.get(option.replace('-', '_'))
        if option == 'output':
            default = str(root / 'configs_runtime/m3_h200_equivalent.py')
        parser.add_argument('--' + option, default=os.environ.get(
            'M3_' + option.upper().replace('-', '_'), default))
    parser.add_argument('--create-data-links', action='store_true',
                        help='Create absent data links only; never replace existing paths')
    parser.add_argument('--radar-cache', default=os.environ.get('M3_RADAR_CACHE'),
                        help='Completed M3 per-return cache covering train and val; bound readonly')
    args = parser.parse_args()
    for name in ('reference_config', 'data_root', 'occ_root', 'pretrained', 'work_dir', 'output'):
        if not getattr(args, name):
            parser.error('--' + name.replace('_', '-') + ' or its M3_* environment variable is required')
    output = Path(args.output).resolve()
    _require(output.suffix == '.py', 'Resolved MMCV config must use a .py filename')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an already resolved config')
    from mmcv import Config
    reference = Config.fromfile(args.reference_config)._cfg_dict.to_dict()
    candidate = Config.fromfile(args.base_config)._cfg_dict.to_dict()
    config = make_config(candidate, reference, args.data_root, args.occ_root,
                         args.pretrained, args.work_dir, args.reference_config)
    if args.radar_cache:
        bind_radar_cache(config, args.radar_cache)
    verify_inputs(config, root, create_data_links=args.create_data_links)
    write_resolved_config(config, output)
    print(json.dumps(dict(config=str(output), protocol=config['m3_protocol']), indent=2))


if __name__ == '__main__':
    main()
