"""Read-only check before atomically switching the private camera data link.

Accept a complete copy_manifest.json, or the six camera and six metadata
VERIFIED migration_jobs records. Check every annotation camera filename and
local file name/size, without reading image contents or computing hashes.
No symlink, configuration, cache, process or checkpoint is changed here.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_m3_camera_relocation', ROOT / 'tools/m3_relocate_runtime_data.py')
RELOCATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELOCATION)
require = RELOCATION.require
META_DIRECTORIES = ('v1.0-trainval', 'maps', 'can_bus', 'lidarseg/v1.0-trainval')
META_FILES = ('nuscenes_infos_temporal_train_new.pkl', 'nuscenes_infos_temporal_val_new.pkl')


def verified_evidence(source_data, target):
    """Use the existing copy protocol; partial progress alone never authorizes a switch."""
    base = RELOCATION.LOCAL_ROOT
    receipt = base / 'copy_manifest.json'
    if receipt.is_file():
        value = RELOCATION.read_json(RELOCATION.plain_file(receipt))
        if value.get('status') == 'COPY_VERIFIED':
            require(value.get('schema') == 'm3_local_data_copy_v1' and
                    value.get('destination_root') == str(base), 'Wrong final copy receipt identity')
            for name, old, new in [('images', (source_data / 'samples').resolve(), target / 'samples'),
                                   ('metadata', source_data, target)]:
                component = value.get('components', {}).get(name, {})
                require(component.get('verified') is True and
                        Path(component.get('source_root', '')).resolve() == old and
                        component.get('destination_root') == str(new),
                        'Final copy receipt component differs: ' + name)
            return dict(kind='COPY_VERIFIED', path=str(receipt), components=value['components'])
    expected = {camera: (source_data / 'samples' / camera, target / 'samples' / camera)
                for camera in RELOCATION.CAMERAS}
    expected.update({'metadata_' + name.replace('/', '_'): (source_data / name, target / name)
                     for name in META_DIRECTORIES + META_FILES})
    jobs = {}
    for name, (old, new) in expected.items():
        path = RELOCATION.plain_file(base / 'migration_jobs' / (name + '.json'))
        record = RELOCATION.read_json(path)
        require(record.get('status') == 'VERIFIED' and
                record.get('source') == str(old.resolve()) and
                record.get('destination') == str(new) and
                type(record.get('files')) is int and record['files'] > 0 and
                type(record.get('bytes')) is int and record['bytes'] > 0,
                'Missing, incomplete or mismatched camera/metadata copy record: ' + name)
        jobs[name] = record
    return dict(kind='VERIFIED_COMPONENT_JOBS', path=str(base / 'migration_jobs'), jobs=jobs)


def check(source, code_root):
    source_data = Path(source['data']['train']['data_root']).resolve()
    target = RELOCATION.destinations()['data']
    RELOCATION.local_directory(RELOCATION.LOCAL_ROOT)
    RELOCATION.local_directory(target)
    evidence = verified_evidence(source_data, target)
    for name in META_DIRECTORIES:
        RELOCATION.local_directory(target / name)
    for name in META_FILES:
        RELOCATION.plain_file(target / name)
        require((target / name).stat().st_size > 0, 'Empty annotation: ' + name)
    code_root = Path(code_root).resolve()
    link = code_root / 'data/nuscenes'
    require(link.is_symlink() and link.resolve() in (source_data, target),
            'Private code data/nuscenes must be the known source or verified local symlink')
    # Only substitute paths in a private copy to inspect the copied annotations.
    config = RELOCATION.relocate_config(source)
    tokens, required_images = RELOCATION.annotation_inventory(config)
    available, counts, sizes = set(), {}, {}
    for camera in sorted(RELOCATION.CAMERAS):
        directory = target / 'samples' / camera
        RELOCATION.local_directory(directory)
        count, size = 0, 0
        with os.scandir(directory) as entries:
            for entry in entries:
                require(entry.is_file(follow_symlinks=False) and not entry.is_symlink(),
                        'Camera directory contains a non-regular entry: ' + entry.path)
                file_size = entry.stat(follow_symlinks=False).st_size
                require(file_size > 0, 'Empty local camera file: ' + entry.path)
                available.add('samples/' + camera + '/' + entry.name)
                count += 1
                size += file_size
        counts[camera], sizes[camera] = count, size
        if evidence['kind'] == 'VERIFIED_COMPONENT_JOBS':
            record = evidence['jobs'][camera]
            require(record['files'] == count and record['bytes'] == size,
                    'Local camera counts/sizes differ from verified copy: ' + camera)
    if evidence['kind'] == 'COPY_VERIFIED':
        component = evidence['components']['images']
        require(component.get('files') == sum(counts.values()) and component.get('bytes') == sum(sizes.values()),
                'Local camera counts/sizes differ from final copy receipt')
    missing = sorted(required_images - available)
    require(not missing, 'Local camera files missing from annotations: ' + repr(missing[:8]))
    return dict(status='CAMERA_DATA_VERIFIED', local_data_root=str(target),
                camera_link=str(link), current_link_target=str(link.resolve()),
                link_switch_needed=link.resolve() != target,
                annotation_tokens=len(tokens), required_image_files=len(required_images),
                local_image_files=sum(counts.values()), local_image_bytes=sum(sizes.values()),
                camera_file_counts=counts, metadata_components=list(META_DIRECTORIES + META_FILES),
                copy_evidence=evidence['path'], copy_evidence_kind=evidence['kind'],
                filesystem=RELOCATION.filesystem(target),
                training_process_changed=False, link_changed=False,
                limitation='Only future camera opens follow a changed symlink; already-open NAS reads, '
                           'occupancy, radar cache and CAN bus retain their existing paths')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', default=str(ROOT / 'configs_runtime/m3_h200_equivalent.py'))
    parser.add_argument('--code-root', default=str(ROOT))
    args = parser.parse_args()
    # The formal input is already resolved; no heavyweight model import is required.
    source = {key: value for key, value in runpy.run_path(args.source_config).items()
              if not key.startswith('__')}
    require('_base_' not in source, 'Use the resolved running configuration')
    print(json.dumps(check(source, args.code_root), indent=2))


if __name__ == '__main__':
    main()
