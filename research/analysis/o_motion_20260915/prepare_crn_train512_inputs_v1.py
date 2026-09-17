"""Prepare author CRN train metadata and ONLY selected anchors' history radar.

Python 3.8; CPU only. Original datasets, code, environments and exports stay
read-only. New data view; no resume/retry. No detector or optimizer is created.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
KEY_OFFSETS = (0, -2, -4, -6)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('x') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def selected_records(path):
    require(sha(path) == SELECTION_SHA, 'Frozen selection changed')
    records = read(path)['records']
    train = [r for r in records if r['split'] == 'train']
    dev = [r for r in records if r['split'] == 'development']
    require(len(train) == 512 and len({r['sample_token'] for r in train}) == 512 and
            len({r['scene_token'] for r in train}) == 256, 'Wrong train selection')
    require(not {r['scene_token'] for r in train} & {r['scene_token'] for r in dev}, 'Scene overlap')
    return train


def check_sources(repo, contract_path, expected_sha):
    require(sha(contract_path) == expected_sha, 'Source contract changed')
    contract = read(contract_path)
    require(contract['schema'] == 'crn-train512-source-contract-v1' and
            contract['official_commit'] == '5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d', 'Source contract schema')
    for relative, digest in contract['files_sha256'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe source path')
        require(sha(Path(repo)/path) == digest, 'Source changed: '+relative)
    return contract


def history_plan(infos, selection):
    tokens = [r['sample_token'] for r in infos]
    require(len(tokens) == len(set(tokens)) == 28130, 'Full original train metadata is required')
    lookup = {token: index for index, token in enumerate(tokens)}
    plans = []
    for ordinal, identity in enumerate(selection):
        index = lookup[identity['sample_token']]
        require(infos[index]['scene_token'] == identity['scene_token'], 'Train scene differs')
        history = []
        for offset in KEY_OFFSETS:
            cur = index+offset
            # Mirror the author's same-scene fallback, preserving full indices.
            while infos[cur]['scene_token'] != infos[index]['scene_token']:
                cur += 1
            resolved = cur % len(infos)
            require(0 <= resolved <= index and infos[resolved]['timestamp'] <= infos[index]['timestamp'],
                    'History unexpectedly uses a future index')
            history.append(dict(offset=offset, native_index=resolved,
                sample_token=infos[resolved]['sample_token'], timestamp=infos[resolved]['timestamp']))
        plans.append(dict(ordinal=ordinal, identity=identity, native_index=index, history=history))
    return plans


def author_module(path, nusc=None):
    tree = ast.parse(Path(path).read_text())
    if nusc is not None:
        assignments = [node for node in tree.body if isinstance(node, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'nusc' for t in node.targets)]
        require(len(assignments) == 1, 'Author nusc construction changed')
        tree.body = [node for node in tree.body if node is not assignments[0]]
    namespace = {'__name__': '_read_only_author_preparation', 'nusc': nusc}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class SensorOnlyNuScenes:
    """The radar worker may not request annotations or annotation velocities."""
    def __init__(self, original):
        self.original = original
        self.dataroot = original.dataroot
        self.tables = set()

    def get(self, table, token):
        require(table in ('sample', 'sample_data', 'ego_pose', 'calibrated_sensor'),
                'Radar preparation requested a non-sensor table: '+table)
        self.tables.add(table)
        return self.original.get(table, token)


def run(args, out):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation requires CUDA_VISIBLE_DEVICES empty')
    start = time.monotonic()
    def check():
        require(time.monotonic()-start < args.max_seconds, 'Preparation deadline exceeded')
    repo, source = Path(args.repo).resolve(), Path(args.source_data_root).resolve()
    contract = check_sources(repo, args.source_contract, args.source_contract_sha256)
    selection = selected_records(args.selection)
    import mmcv
    import numpy as np
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(source), verbose=True)
    metadata_sha = {p.name: sha(p) for p in sorted((source/'v1.0-trainval').glob('*.json'))}
    train_scenes = {s['token'] for s in nusc.scene if s['name'] in splits.train}
    expected = {s['token'] for s in nusc.sample if s['scene_token'] in train_scenes}
    require(len(train_scenes) == 700 and len(expected) == 28130, 'Official train split differs')
    generate = author_module(repo/'scripts/gen_info.py')['generate_info']
    infos = generate(nusc, splits.train)
    check()
    require({r['sample_token'] for r in infos} == expected, 'Author full train token set differs')
    plans = history_plan(infos, selection)
    needed = sorted({h['native_index'] for plan in plans for h in plan['history']})
    require(len(needed) <= 2048, 'Unexpected history expansion')
    view = out/'data'; view.mkdir()
    for name in ('samples', 'sweeps', 'v1.0-trainval', 'maps'):
        require((source/name).exists(), 'Missing original data directory: '+name)
        (view/name).symlink_to(source/name, target_is_directory=True)
    for name in ('radar_bev_filter', 'radar_pv_filter'):
        (view/name).mkdir()
    # Verify all required image assets before spending CPU on radar projections.
    images = sorted({info['filename'] for index in needed for info in infos[index]['cam_infos'].values()})
    for filename in images:
        require((view/filename).is_file(), 'Missing selected/history image: '+filename)
    info_path = view/'nuscenes_infos_train.pkl'
    mmcv.dump(infos, str(info_path))
    with info_path.open('rb') as handle:
        os.fsync(handle.fileno())
    write(out/'history_plan.json', dict(selection_sha256=SELECTION_SHA, records=plans,
        full_train_samples=len(infos), needed_history_indices=needed))
    sensor_nusc = SensorOnlyNuScenes(nusc)
    bev = author_module(repo/'scripts/gen_radar_bev.py', sensor_nusc)
    pv = author_module(repo/'scripts/gen_radar_pv.py')
    bev['DATA_PATH'] = str(view); pv['DATA_PATH'] = str(view)
    require(bev['N_SWEEPS'] == 8 and not bev['DISABLE_FILTER'] and not bev['DEBUG'], 'Author radar settings changed')
    def project(index):
        check()
        # Prevent annotation fields from entering either projection function.
        original = infos[index]
        sensor_info = {key: original[key] for key in ('lidar_infos', 'cam_infos')}
        bev['worker'](sensor_info); pv['worker'](sensor_info)
        outputs = [view/'radar_bev_filter'/Path(sensor_info['lidar_infos']['LIDAR_TOP']['filename']).name]
        outputs += [view/'radar_pv_filter'/(Path(c['filename']).name+'.bin')
                    for c in sensor_info['cam_infos'].values()]
        rows = []
        for path in outputs:
            require(path.is_file() and path.stat().st_size % 28 == 0, 'Malformed radar file')
            values = np.fromfile(str(path), dtype=np.float32)
            require(np.isfinite(values).all(), 'Nonfinite radar preparation')
            with path.open('rb') as handle:
                os.fsync(handle.fileno())
            rows.append(dict(file=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=sha(path)))
        return dict(native_index=index, sample_token=original['sample_token'], files=rows)
    projected = []
    pool = ThreadPoolExecutor(max_workers=2)
    futures = [pool.submit(project, index) for index in needed]
    try:
        for future in futures:
            row = future.result()
            projected.append(row)
            check()
            write(out/'progress.json', dict(completed_history_frames=len(projected), planned_history_frames=len(needed),
                elapsed_seconds=time.monotonic()-start, GPU_used=False))
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True)
    check_sources(repo, args.source_contract, args.source_contract_sha256)
    require(metadata_sha == {p.name: sha(p) for p in sorted((source/'v1.0-trainval').glob('*.json'))},
            'Source metadata changed during preparation')
    manifest = dict(schema='crn-train512-assets-v1', status='COMPLETE_CRN_TRAIN512_ASSETS',
        selection_sha256=SELECTION_SHA, source_contract_sha256=args.source_contract_sha256,
        author_sources=contract, preparation_source_sha256=sha(__file__),
        source_data_root=str(source), official_split='train', full_train_samples=28130, selected_samples=512,
        selected_scenes=256, selected_history_frames=len(needed), key_offsets=list(KEY_OFFSETS),
        train_info=dict(file=str(info_path.relative_to(out)), sha256=sha(info_path), bytes=info_path.stat().st_size),
        history_plan_sha256=sha(out/'history_plan.json'), metadata_sha256=metadata_sha,
        radar_files=projected, image_files=images,
        sensor_tables_read=sorted(sensor_nusc.tables), projection_annotation_input=False,
        annotation_policy='Author full infos retain labels including box_velocity; projection input excludes ann_infos; inference discards GT batch fields',
        GPU_used=False, model_constructed=False, optimizer_updates=0,
        elapsed_seconds=time.monotonic()-start)
    write(out/'manifest.json', manifest)
    write(out/'complete.json', dict(schema='crn-train512-assets-complete-v1', status=manifest['status'],
        files_sha256={name: sha(out/name) for name in ('manifest.json', 'history_plan.json')},
        train_info_sha256=sha(info_path), source_sha256=sha(__file__), selection_sha256=SELECTION_SHA))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--source-data-root', required=True)
    p.add_argument('--source-contract', required=True)
    p.add_argument('--source-contract-sha256', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-seconds', type=float, default=1800)
    a = p.parse_args()
    require(math.isfinite(a.max_seconds) and 0 < a.max_seconds <= 1800, 'Bounded preparation budget')
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        manifest = run(a, out)
        print(json.dumps(dict(status=manifest['status'], history_frames=manifest['selected_history_frames'],
            seconds=manifest['elapsed_seconds'])))
    except Exception as error:
        write(out/'failed.json', dict(status='FAILED_NO_RETRY', error=repr(error), source_sha256=sha(__file__)))
        raise


if __name__ == '__main__':
    main()
