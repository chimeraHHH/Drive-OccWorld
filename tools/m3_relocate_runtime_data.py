"""Prepare copied H200 data and a new config without touching the running job.

Run only after copy_manifest.json is COPY_VERIFIED. The command rewrites the
local cache's manifest.data_root, preserving the other NPY members byte for
byte, and publishes a separate config. It never changes data symlinks, chooses
a checkpoint, resumes training, or writes the NAS source. Camera annotation
paths remain relative: the caller must separately verify data/nuscenes points
to the copied tree before using the new config. Do not rsync the old radar
cache over the rebased copy afterwards.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import types
import zipfile

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path('/home/wangning/data_cache/RadarFlowOcc_m3_h200')
CACHE_NAME = 'radar_m3_obs_trainval_1sweep_200x200_v1'
CAMERAS = {'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK',
           'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}
REBASE_SCHEMA = 'm3_cache_path_rebase_v1'


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPORT = _load_module('_m3_relocation_config', CODE_ROOT / 'tools/m3_prepare_h200_config.py')
OBSERVATIONS = _load_module('_m3_relocation_observations', CODE_ROOT /
                           'projects/mmdet3d_plugin/datasets/radar_observations.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def destinations():
    return dict(data=LOCAL_ROOT / 'datasets/nuscenes_driveocc',
                occupancy=LOCAL_ROOT / 'datasets/nuScenes-Occupancy-v0.1',
                cache=LOCAL_ROOT / 'cache' / CACHE_NAME)


def filesystem(path):
    return subprocess.check_output(
        ['findmnt', '-no', 'SOURCE,FSTYPE,TARGET', '-T', str(path)], text=True).split()


def local_directory(path):
    path = Path(path)
    require(path.is_absolute() and path.is_dir() and path.resolve() == path,
            'Destination must be a real local directory without symlink ancestors: ' + str(path))
    require(path.stat().st_uid == os.getuid(), 'Destination is not owned by this user: ' + str(path))
    require(filesystem(path) == ['/dev/nvme0n1p2', 'ext4', '/'],
            'Destination must be on the verified H200 system NVMe filesystem: ' + str(path))


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def plain_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Expected a regular non-symlink file: ' + str(path))
    return path


def sync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, contents, overwrite=True):
    """Use a sibling temporary file; exclusive publication never replaces a config."""
    path = Path(path)
    require(not path.is_symlink(), 'Refusing a symlink output: ' + str(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(str(temporary), str(path))
        else:
            os.link(str(temporary), str(path))  # EEXIST rather than overwriting an active config.
            temporary.unlink()
        sync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2) + '\n').encode())


def source_cache(config):
    caches = set()
    for split in ('train', 'val', 'test'):
        dataset = config['data'][split]
        options = dataset['radar_observation_cfg']
        require(dataset.get('radar_cfg') is None and options.get('cache_readonly') is True,
                'Source must use the readonly M3 per-return cache in every split')
        require(isinstance(options.get('cache_dir'), str), 'Source cache_dir must be explicit')
        caches.add(str(Path(options['cache_dir']).resolve()))
    require(len(caches) == 1, 'All splits must share the same source radar cache')
    return Path(caches.pop())


def manifests(config):
    values = [SUPPORT.observation_manifest(config['data'][split])
              for split in ('train', 'val', 'test')]
    require(all(canonical(item) == canonical(values[0]) for item in values),
            'Source train/val/test observation manifests must agree')
    return values[0]


def relocate_config(source):
    """Copy only known data path fields; optimizer, sampler, model and work_dir stay intact."""
    config = copy.deepcopy(source)
    paths = destinations()
    root = str(paths['data']) + '/'
    occupancy, cache = str(paths['occupancy']), str(paths['cache'])
    config['data_root'], config['occ_path'] = root, occupancy
    for split in ('train', 'val', 'test'):
        dataset = config['data'][split]
        require(Path(dataset['ann_file']).parent.resolve() == Path(dataset['data_root']).resolve(),
                'Annotation must be directly under its source data_root')
        dataset['data_root'] = root
        dataset['ann_file'] = root + Path(dataset['ann_file']).name
        dataset['radar_observation_cfg'].update(cache_dir=cache, cache_readonly=True)
        for step in dataset['pipeline']:
            if step.get('type') == 'LoadOccupancy':
                step['occ_path'] = occupancy
    for pipeline in (config.get('train_pipeline', []), config.get('test_pipeline', []),
                     config.get('evaluation', {}).get('pipeline', [])):
        for step in pipeline:
            if step.get('type') == 'LoadOccupancy':
                step['occ_path'] = occupancy
    if 'radar_observation_cfg' in config:
        config['radar_observation_cfg']['cache_dir'] = cache
        config['radar_observation_cfg']['cache_readonly'] = True
    if 'radar_cache' in config.get('m3_protocol', {}):
        config['m3_protocol']['radar_cache'].update(
            directory=cache, status_file=str(paths['cache'] / 'build_status.json'))
    return config


def verify_receipt(source, record_path):
    paths = destinations()
    for path in (LOCAL_ROOT, paths['data'], paths['occupancy'], paths['cache']):
        local_directory(path)
    record_path = plain_file(record_path)
    require(record_path.resolve() == LOCAL_ROOT / 'copy_manifest.json',
            'Use copy_manifest.json from the verified local destination root')
    record = read_json(record_path)
    require(record.get('schema') == 'm3_local_data_copy_v1' and
            record.get('status') == 'COPY_VERIFIED' and
            record.get('destination_root') == str(LOCAL_ROOT),
            'Copy receipt must be COPY_VERIFIED for this exact destination root')
    old_data = Path(source['data']['train']['data_root']).resolve()
    old_occ = {str(Path(step['occ_path']).resolve()) for split in ('train', 'val', 'test')
               for step in source['data'][split]['pipeline'] if step.get('type') == 'LoadOccupancy'}
    require(len(old_occ) == 1, 'Source splits must share one occupancy root')
    expected = dict(images=((old_data / 'samples').resolve(), paths['data'] / 'samples'),
                    metadata=(old_data, paths['data']),
                    occupancy=(Path(old_occ.pop()), paths['occupancy']),
                    radar_cache=(source_cache(source), paths['cache']))
    for name, (old, new) in expected.items():
        component = record.get('components', {}).get(name, {})
        require(component.get('verified') is True and
                type(component.get('files')) is int and component['files'] > 0 and
                type(component.get('bytes')) is int and component['bytes'] > 0,
                'Copy component is not verified with positive counts: ' + name)
        require(Path(component.get('source_root', '')).resolve() == old and
                component.get('destination_root') == str(new),
                'Copy component paths do not match source config and destination: ' + name)
    require(set(record['components']['images'].get('jobs', [])) == CAMERAS,
            'Copy receipt must cover exactly the six camera streams')
    require(source_cache(source) != paths['cache'], 'Source and destination cache must differ')
    return record


def validate_complete(record, manifest):
    require(isinstance(record, dict) and record.get('status') == 'COMPLETE',
            'Source cache build status must be COMPLETE')
    count = record.get('tokens_required')
    require(type(count) is int and count > 0 and
            type(record.get('tokens_processed')) is int and record['tokens_processed'] == count and
            record.get('limit') is None,
            'Source cache must have equal positive processed/required counts and no limit')
    require(isinstance(record.get('splits'), list) and
            {'train', 'val'}.issubset(record['splits']), 'Source cache must cover train and val')
    require(canonical(record.get('manifest')) == canonical(manifest),
            'Source cache manifest differs from source configuration')
    return count


def annotation_inventory(config):
    """Read only the copied annotations, without building a model or nuScenes DB."""
    tokens, images = set(), set()
    annotations = {Path(config['data'][split]['ann_file']) for split in ('train', 'val')}
    for path in sorted(annotations):
        plain_file(path)
        with path.open('rb') as handle:
            payload = pickle.load(handle)
        infos = payload['infos'] if isinstance(payload, dict) else payload
        require(isinstance(infos, list), 'Annotation infos must be a list')
        for info in infos:
            token = info['token']
            OBSERVATIONS.NuScenesRadarObservations._validate_token(token)
            tokens.add(token)
            cameras = info.get('cams')
            require(isinstance(cameras, dict) and set(cameras) == CAMERAS,
                    'Annotation must contain all six camera views')
            for name, camera in cameras.items():
                filename = Path(camera['data_path'])
                require(not filename.is_absolute() and len(filename.parts) == 5 and
                        filename.parts[:3] == ('data', 'nuscenes', 'samples') and
                        filename.parts[3] == name,
                        'Camera filename does not use the copied relative data/nuscenes entry')
                images.add(str(filename.relative_to('data/nuscenes')))
        del payload, infos
    require(tokens, 'Copied annotations have no tokens')
    return tokens, images


def required_tokens(config):
    return annotation_inventory(config)[0]


def validator(dataset):
    options = copy.deepcopy(dataset['radar_observation_cfg'])
    options.update(cache_dir=None, cache_readonly=False)
    return OBSERVATIONS.NuScenesRadarObservations(
        nusc=types.SimpleNamespace(version=dataset.get('version', 'v1.0-trainval'),
                                  dataroot=dataset['data_root']), **options)


def read_entry(path, token, allowed_manifests, checker):
    """Validate with the production rules and retain the original NPY bytes."""
    plain_file(path)
    with zipfile.ZipFile(str(path)) as archive:
        names = archive.namelist()
        require(len(names) == 3 and set(names) ==
                {'observations.npy', 'manifest.npy', 'sample_token.npy'},
                'Invalid or duplicate M3 NPZ members: ' + str(path))
        members = {name: archive.read(name) for name in names}
    arrays = {name: np.load(io.BytesIO(data), allow_pickle=False) for name, data in members.items()}
    manifest = arrays['manifest.npy']
    sample_token = arrays['sample_token.npy']
    require(manifest.shape == () and str(manifest.item()) in allowed_manifests,
            'Unexpected per-entry manifest: ' + str(path))
    require(sample_token.shape == () and str(sample_token.item()) == token,
            'NPZ filename/sample token mismatch: ' + str(path))
    checker._validate_observations(arrays['observations.npy'])
    return str(manifest.item()), members


def rebase_entry(path, token, old_manifest, new_manifest, checker):
    old_json, new_json = canonical(old_manifest), canonical(new_manifest)
    current, members = read_entry(path, token, {old_json, new_json}, checker)
    if current == new_json:
        return False  # Re-validated, already atomically rebased before an interruption.
    output = io.BytesIO()
    replacement = io.BytesIO()
    np.save(replacement, np.asarray(new_json), allow_pickle=False)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in members.items():
            archive.writestr(name, replacement.getvalue() if name == 'manifest.npy' else contents)
    atomic_bytes(path, output.getvalue())
    _, after = read_entry(path, token, {new_json}, checker)
    for name in ('observations.npy', 'sample_token.npy'):
        require(after[name] == members[name], 'Cache array/token bytes changed: ' + str(path))
    return True


@contextlib.contextmanager
def exclusive_locks(cache):
    import fcntl
    # The copy job uses the same root lock. Open it read-only; never write outside cache.
    copy_lock = plain_file(LOCAL_ROOT / '.migration.lock')
    own_lock = cache / '.m3_relocation.lock'
    require(not own_lock.is_symlink(), 'Refusing symlink relocation lock')
    with copy_lock.open('r') as copy_guard, own_lock.open('a') as guard:
        fcntl.flock(copy_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def serialize_config(config, filename):
    from mmcv import Config
    return Config(config, cfg_text='# M3 local data paths; no training started\n',
                  filename=str(filename)).pretty_text.encode()


def run(source, source_config, copy_record, output, progress_interval=1000):
    source_config, output = Path(source_config).resolve(), Path(output).absolute()
    require(output.suffix == '.py', 'New config must have a .py suffix')
    require(not output.is_symlink() and output != source_config,
            'Refusing to overwrite an active config or symlink')
    local_directory(output.parent)
    require(progress_interval > 0, 'progress_interval must be positive')
    verify_receipt(source, copy_record)
    old_cache = source_cache(source)
    old_manifest = manifests(source)
    old_status = read_json(old_cache / 'build_status.json')
    count = validate_complete(old_status, old_manifest)
    config = relocate_config(source)
    new_manifest = manifests(config)
    expected = copy.deepcopy(old_manifest)
    expected['data_root'] = str(destinations()['data'])
    require(canonical(expected) == canonical(new_manifest),
            'Relocation may change only manifest.data_root')
    identity = dict(schema=REBASE_SCHEMA, source_config=str(source_config),
                    source_config_sha256=hashlib.sha256(source_config.read_bytes()).hexdigest(),
                    source_cache=str(old_cache), destination_cache=str(destinations()['cache']),
                    source_manifest=old_manifest, destination_manifest=new_manifest)
    cache = destinations()['cache']
    with exclusive_locks(cache):
        # The copy receipt may have changed between the first read and lock acquisition.
        verify_receipt(source, copy_record)
        status_path = plain_file(cache / 'build_status.json')
        ready_path = cache / 'relocation_ready.json'
        previous = read_json(status_path)
        if output.exists():
            require(previous.get('relocation') is not None and ready_path.is_file() and
                    not ready_path.is_symlink(), 'Refusing to overwrite an existing config')
            prior_ready = read_json(ready_path)
            require(prior_ready.get('schema') == 'm3_local_runtime_data_ready_v1' and
                    prior_ready.get('status') in {'BUILDING', 'FAILED', 'PREPARED_NOT_ACTIVATED'} and
                    prior_ready.get('source_config_sha256') == identity['source_config_sha256'] and
                    prior_ready.get('source_config') == str(source_config) and
                    prior_ready.get('output_config') == str(output) and
                    canonical(prior_ready.get('source_manifest')) == canonical(old_manifest) and
                    canonical(prior_ready.get('destination_manifest')) == canonical(new_manifest),
                    'Existing config is not from this interrupted relocation')
        if previous.get('relocation') is None:
            validate_complete(previous, old_manifest)
            require(previous['tokens_required'] == count, 'Copied/source cache counts differ')
        else:
            require(canonical(previous['relocation']) == canonical(identity) and
                    previous.get('status') in {'BUILDING', 'FAILED', 'COMPLETE'},
                    'Existing relocation belongs to a different source, target or configuration')
        record = copy.deepcopy(old_status)
        record.update(status='BUILDING', phase='rebasing_manifest', tokens_processed=0,
                      manifest=new_manifest, relocation=identity, entries_rewritten=0,
                      entries_already_rebased=0)
        atomic_json(status_path, record)
        ready = dict(schema='m3_local_runtime_data_ready_v1', status='BUILDING',
                     source_config=str(source_config),
                     source_config_sha256=identity['source_config_sha256'],
                     output_config=str(output), copy_record=str(Path(copy_record)),
                     cache=str(cache), cache_status_file=str(status_path),
                     source_manifest=old_manifest, destination_manifest=new_manifest,
                     camera_link=str(CODE_ROOT / 'data/nuscenes'),
                     camera_link_required_target=str(destinations()['data']))
        atomic_json(ready_path, ready)
        try:
            tokens = required_tokens(config)
            actual = {path.stem for path in cache.glob('*.npz')}
            require(len(tokens) == count and actual == tokens,
                    'Local NPZ token set/count differs from copied annotations/source cache')
            checker = validator(config['data']['train'])
            for token in sorted(tokens):
                changed = rebase_entry(cache / (token + '.npz'), token,
                                       old_manifest, new_manifest, checker)
                record['tokens_processed'] += 1
                record['entries_rewritten' if changed else 'entries_already_rebased'] += 1
                if record['tokens_processed'] % progress_interval == 0:
                    atomic_json(status_path, record)
                    print('{}/{} cache entries validated'.format(record['tokens_processed'], count), flush=True)
            record.update(status='COMPLETE', phase='cache_rebased')
            atomic_json(status_path, record)
            SUPPORT.bind_radar_cache(config, cache)
            config['m3_protocol']['data_relocation'] = dict(
                status='PREPARED_NOT_ACTIVATED', source_config=str(source_config),
                source_config_sha256=identity['source_config_sha256'],
                copy_record=str(Path(copy_record)), rebased_tokens=count,
                relative_camera_entry='data/nuscenes',
                camera_link_required_target=str(destinations()['data']),
                resume_policy='No checkpoint selected; preserve full-epoch checkpoint safety')
            config_bytes = serialize_config(config, output)
            if output.exists():
                require(plain_file(output).read_bytes() == config_bytes,
                        'Refusing to overwrite an existing config with different contents')
            else:
                atomic_bytes(output, config_bytes, overwrite=False)
            ready.update(status='PREPARED_NOT_ACTIVATED', tokens_processed=count,
                         camera_annotation_paths_checked=True,
                         output_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                         entries_rewritten=record['entries_rewritten'],
                         entries_already_rebased=record['entries_already_rebased'],
                         training_restarted=False, active_config_modified=False,
                         camera_link_changed=False)
            atomic_json(ready_path, ready)
        except BaseException as error:
            record.update(status='FAILED', phase='relocation_failed',
                          error='{}: {}'.format(type(error).__name__, str(error)))
            atomic_json(status_path, record)
            ready.update(status='FAILED', error=record['error'])
            atomic_json(ready_path, ready)
            raise
    return dict(ready, ready_file=str(ready_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', default=str(CODE_ROOT / 'configs_runtime/m3_h200_equivalent.py'))
    parser.add_argument('--copy-record', default=str(LOCAL_ROOT / 'copy_manifest.json'))
    parser.add_argument('--output', default=str(CODE_ROOT / 'configs_runtime/m3_h200_local_data.py'))
    args = parser.parse_args()
    from mmcv import Config
    source = Config.fromfile(args.source_config)._cfg_dict.to_dict()
    print(json.dumps(run(source, args.source_config, args.copy_record, args.output), indent=2))


if __name__ == '__main__':
    main()
