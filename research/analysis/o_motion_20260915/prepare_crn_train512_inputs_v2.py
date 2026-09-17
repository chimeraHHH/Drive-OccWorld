"""Explicit CPU continuation of a verified stopped v1 preparation prefix.

New output only; unchanged author sensor workers and two threads. The old job
must be terminal and absent before progress or data are read. No automatic
retry, no model, no GPU, no generation of unselected train radar projections.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import time

V1_SHA = '75b5bba81c59348f9c0f190a34065d84ed465814e2cbd98f84578a823dff6e38'
CONTRACT_SHA = 'e147d9deca1409ee060a1ceb384f89424b55a9c40a866576b638f92f7f414e5a'
HERE = Path(__file__).resolve().parent
_v1_path = HERE/'prepare_crn_train512_inputs_v1.py'
if hashlib.sha256(_v1_path.read_bytes()).hexdigest() != V1_SHA:
    raise ValueError('Frozen v1 preparation source changed')
_spec = importlib.util.spec_from_file_location('_crn_verified_prepare_v1', _v1_path)
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)
_v1.require(_v1.sha(_v1.__file__) == V1_SHA, 'Frozen v1 preparation source changed')
# Public pure helpers retained for the separately versioned inference loader.
sha, read, write, require = _v1.sha, _v1.read, _v1.write, _v1.require
selected_records, check_sources, history_plan = _v1.selected_records, _v1.check_sources, _v1.history_plan
SELECTION_SHA, KEY_OFFSETS = _v1.SELECTION_SHA, _v1.KEY_OFFSETS


def require_processes_absent(runner, child):
    for identity in (runner, child):
        require(identity['uid'] == os.getuid() and identity['pid'] > 0 and identity['startticks'] > 0,
                'Old process ownership/identity is invalid')
        require(not (Path('/proc')/str(identity['pid'])).exists(), 'Old runner/child PID remains present')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            text = (proc/'stat').read_text()
            fields = text[text.rfind(')')+2:].split()
            require(not (int(fields[2]) == child['pid'] and fields[0] != 'Z'), 'Old child process group remains live')
        except (FileNotFoundError, ProcessLookupError):
            pass


def stopped_job(args):
    """Authenticate only control files here; do not inspect progress yet."""
    require(sha(args.old_terminal_receipt) == args.old_terminal_receipt_sha256, 'Terminal receipt hash changed')
    terminal = read(args.old_terminal_receipt)
    require(terminal['schema'] == 'crn-train512-stopped-job-v1' and terminal['status'] == 'VERIFIED_STOPPED',
            'Verified terminal receipt required')
    require(all(terminal[k] is True for k in ('runner_missing', 'child_missing', 'owned_process_group_missing')),
            'Old process absence was not verified')
    require(terminal['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'Host boot differs')
    job = Path(args.old_job).resolve()
    for path, key in ((Path(args.old_launch), 'launch_sha256'), (job/'spec.json', 'spec_sha256'), (job/'state.json', 'state_sha256')):
        require(sha(path) == terminal[key], 'Stopped job control file differs: '+key)
    launch, spec, state = read(args.old_launch), read(job/'spec.json'), read(job/'state.json')
    require(state['status'] in ('FAILED', 'TIMEOUT', 'USER_SIGNAL') and 'finished_utc' in state and
            isinstance(state['returncode'], int), 'Old preparation is not terminal')
    runner = dict(pid=launch['runner_pid'], uid=launch['uid'], startticks=launch['startticks'])
    child = state['child']
    require(terminal['runner'] == runner and terminal['child'] == child and state['runner_pid'] == runner['pid'],
            'Old runner/child receipt identity differs')
    require(Path(launch['job']).resolve() == job and launch['stage'] == 'prepare' and
            launch['outer_seconds'] == spec['seconds'] == 1860, 'Wrong old job/stage/budget')
    command = launch['command']
    require(command == spec['command'] == state['command'] and command[1] == '-B', 'Old command differs')
    require(Path(command[2]).name == 'prepare_crn_train512_inputs_v1.py' and sha(command[2]) == V1_SHA,
            'Old job did not execute frozen v1 source')
    require(len(command[3:]) == 14 and len(set(command[3::2])) == 7, 'Unexpected old argv')
    flags = dict(zip(command[3::2], command[4::2]))
    require(set(flags) == {'--repo', '--source-data-root', '--source-contract', '--source-contract-sha256',
        '--selection', '--out', '--max-seconds'} and flags['--max-seconds'] == '1800', 'Unexpected old arguments')
    for key, current in (('--repo', args.repo), ('--source-data-root', args.source_data_root), ('--out', args.old_assets)):
        require(Path(flags[key]).resolve() == Path(current).resolve(), 'Old/new source path differs: '+key)
    require(flags['--source-contract-sha256'] == args.source_contract_sha256 == CONTRACT_SHA,
            'Old/new code contract differs')
    require(sha(flags['--source-contract']) == CONTRACT_SHA and sha(flags['--selection']) == SELECTION_SHA,
            'Old contract or selection file changed')
    require(spec['env']['CUDA_VISIBLE_DEVICES'] == '', 'Old job was not CPU-only')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
        require(os.environ.get(key) == spec['env'][key] == '1', 'Original CPU thread environment differs: '+key)
    package = Path(command[2]).parent
    for name, digest in launch['source_sha256'].items():
        require(Path(name).name == name, 'Unsafe old package filename')
        require(sha(package/name) == digest, 'Old launch source changed: '+name)
    require(launch['source_sha256']['prepare_crn_train512_inputs_v1.py'] == V1_SHA and
            launch['source_sha256']['crn_train512_source_contract_v1.json'] == CONTRACT_SHA and
            launch['source_sha256']['selection_v1.json'] == SELECTION_SHA, 'Old launch fixed sources differ')
    require_processes_absent(runner, child)
    return dict(terminal=terminal, launch=launch, old_job=str(job), runner=runner, child=child,
        terminal_receipt_sha256=sha(args.old_terminal_receipt), launch_sha256=sha(args.old_launch),
        spec_sha256=sha(job/'spec.json'), state_sha256=sha(job/'state.json'))


def copy_exact(source, destination):
    require(source.is_file() and not source.is_symlink(), 'Reusable file must be a regular file')
    before = source.stat()
    digest = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), 'No overwrite during prefix copy')
    shutil.copyfile(str(source), str(destination))
    with destination.open('rb') as handle:
        os.fsync(handle.fileno())
    after = source.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Old source changed during copy')
    require(sha(source) == digest == sha(destination), 'Prefix copy is not byte-exact')
    return dict(bytes=before.st_size, sha256=digest)


def verify_infos(infos, nusc, train_names, needed):
    """Verify the complete token/scene/time order and all consumed sensor data."""
    expected = []
    for scene in nusc.scene:
        if scene['name'] not in train_names:
            continue
        sample = nusc.get('sample', scene['first_sample_token'])
        while True:
            expected.append((sample['token'], sample['scene_token'], sample['timestamp']))
            if not sample['next']:
                break
            sample = nusc.get('sample', sample['next'])
    actual = [(r['sample_token'], r['scene_token'], r['timestamp']) for r in infos]
    require(actual == expected and len(actual) == 28130 and len({r[1] for r in actual}) == 700,
            'Reused full train token/scene/time order differs')
    fields = ('sample_token', 'timestamp', 'filename')
    for index in needed:
        info = infos[index]
        sample = nusc.get('sample', info['sample_token'])
        require(set(info['cam_infos']) == {'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'}, 'Camera set differs')
        require(set(info['lidar_infos']) == {'LIDAR_TOP'}, 'LiDAR set differs')
        for group in ('cam_infos', 'lidar_infos'):
            for channel, sensor in info[group].items():
                current = nusc.get('sample_data', sample['data'][channel])
                require(all(sensor[k] == current[k] for k in fields), 'Reused sensor sample/time/file differs')
                require(sensor['ego_pose'] == nusc.get('ego_pose', current['ego_pose_token']) and
                        sensor['calibrated_sensor'] == nusc.get('calibrated_sensor', current['calibrated_sensor_token']),
                        'Reused pose/calibration differs from current metadata')
                if group == 'cam_infos':
                    require(all(sensor[k] == current[k] for k in ('height', 'width', 'is_key_frame')),
                            'Reused camera geometry differs')


def output_paths(view, info):
    paths = [view/'radar_bev_filter'/Path(info['lidar_infos']['LIDAR_TOP']['filename']).name]
    paths += [view/'radar_pv_filter'/(Path(c['filename']).name+'.bin') for c in info['cam_infos'].values()]
    require(len(paths) == len(set(paths)) == 7, 'Expected exactly one BEV and six PV files')
    return paths


def run(args, out):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU continuation requires CUDA_VISIBLE_DEVICES empty')
    start = time.monotonic()
    def check():
        require(time.monotonic()-start < args.max_seconds, 'Continuation deadline exceeded')
    provenance = stopped_job(args)
    old, repo, source = Path(args.old_assets).resolve(), Path(args.repo).resolve(), Path(args.source_data_root).resolve()
    require(out != old and not (old/'complete.json').exists(), 'Reuse only an incomplete, stopped old output')
    contract = check_sources(repo, args.source_contract, args.source_contract_sha256)
    selection = selected_records(args.selection)
    # Only after verified termination may the old committed prefix be inspected.
    progress_hash, history_hash = sha(old/'progress.json'), sha(old/'history_plan.json')
    progress, history = read(old/'progress.json'), read(old/'history_plan.json')
    count = progress['completed_history_frames']
    require(type(count) is int and 0 <= count <= len(history['needed_history_indices']), 'Invalid committed prefix length')
    require(progress['planned_history_frames'] == len(history['needed_history_indices']) and progress['GPU_used'] is False,
            'Old progress plan differs')
    require(history['selection_sha256'] == SELECTION_SHA and history['full_train_samples'] == 28130,
            'Old history contract differs')
    import mmcv
    import numpy as np
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    infos = mmcv.load(str(old/'data/nuscenes_infos_train.pkl'))
    plans = history_plan(infos, selection)
    needed = sorted({h['native_index'] for p in plans for h in p['history']})
    require(history['records'] == plans and history['needed_history_indices'] == needed and len(needed) <= 2048,
            'Reused metadata/history index plan differs')
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(source), verbose=True)
    metadata_sha = {p.name: sha(p) for p in sorted((source/'v1.0-trainval').glob('*.json'))}
    verify_infos(infos, nusc, set(splits.train), needed)
    view = out/'data'; view.mkdir()
    for name in ('samples', 'sweeps', 'v1.0-trainval', 'maps'):
        require((source/name).exists(), 'Missing source data directory: '+name)
        (view/name).symlink_to(source/name, target_is_directory=True)
    for name in ('radar_bev_filter', 'radar_pv_filter'):
        (view/name).mkdir()
    info_copy = copy_exact(old/'data/nuscenes_infos_train.pkl', view/'nuscenes_infos_train.pkl')
    copy_exact(old/'history_plan.json', out/'history_plan.json')
    images = sorted({c['filename'] for index in needed for c in infos[index]['cam_infos'].values()})
    for name in images:
        require((view/name).is_file(), 'Missing required image: '+name)
    projected = []
    for index in needed[:count]:
        check()
        files = []
        for source_path, destination in zip(output_paths(old/'data', infos[index]), output_paths(view, infos[index])):
            require(source_path.stat().st_size % 28 == 0, 'Malformed reused radar file')
            require(np.isfinite(np.fromfile(str(source_path), dtype=np.float32)).all(), 'Nonfinite reused radar file')
            files.append(dict(file=str(destination.relative_to(out)), **copy_exact(source_path, destination)))
        projected.append(dict(native_index=index, sample_token=infos[index]['sample_token'], files=files))
    write(out/'reused_prefix.json', dict(old_progress_sha256=progress_hash, old_history_plan_sha256=history_hash,
        old_train_info_sha256=info_copy['sha256'], committed_prefix_frames=count, needed_history_indices=needed[:count],
        files=projected, old_prefix_had_persisted_per_file_hash_ledger=False,
        verification='v1 completion-order/fsync contract plus new source/destination SHA, float32 finite and exactly seven files'))
    write(out/'progress.json', dict(completed_history_frames=count, planned_history_frames=len(needed),
        reused_prefix_frames=count, newly_completed_history_frames=0,
        elapsed_seconds=time.monotonic()-start, GPU_used=False))
    sensor_nusc = _v1.SensorOnlyNuScenes(nusc)
    bev = _v1.author_module(repo/'scripts/gen_radar_bev.py', sensor_nusc)
    pv = _v1.author_module(repo/'scripts/gen_radar_pv.py')
    bev['DATA_PATH'] = str(view); pv['DATA_PATH'] = str(view)
    require(bev['N_SWEEPS'] == 8 and not bev['DISABLE_FILTER'] and not bev['DEBUG'], 'Author radar settings differ')
    def project(index):
        check()
        original = infos[index]
        sensor_info = {key: original[key] for key in ('lidar_infos', 'cam_infos')}
        bev['worker'](sensor_info); pv['worker'](sensor_info)
        files = []
        for path in output_paths(view, sensor_info):
            require(path.is_file() and path.stat().st_size % 28 == 0, 'Malformed newly generated radar file')
            require(np.isfinite(np.fromfile(str(path), dtype=np.float32)).all(), 'Nonfinite new radar preparation')
            with path.open('rb') as handle:
                os.fsync(handle.fileno())
            files.append(dict(file=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=sha(path)))
        return dict(native_index=index, sample_token=original['sample_token'], files=files)
    pool = ThreadPoolExecutor(max_workers=2)
    futures = [pool.submit(project, index) for index in needed[count:]]
    try:
        for future in futures:
            projected.append(future.result()); check()
            write(out/'progress.json', dict(completed_history_frames=len(projected), planned_history_frames=len(needed),
                reused_prefix_frames=count, newly_completed_history_frames=len(projected)-count,
                elapsed_seconds=time.monotonic()-start, GPU_used=False))
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True)
    require(len(projected) == len(needed) and [r['native_index'] for r in projected] == needed, 'Full history coverage differs')
    require(sha(old/'progress.json') == progress_hash and sha(old/'history_plan.json') == history_hash and
            sha(old/'data/nuscenes_infos_train.pkl') == info_copy['sha256'], 'Stopped source changed during continuation')
    require_processes_absent(provenance['runner'], provenance['child'])
    check_sources(repo, args.source_contract, args.source_contract_sha256)
    require(metadata_sha == {p.name: sha(p) for p in sorted((source/'v1.0-trainval').glob('*.json'))}, 'Metadata changed')
    check()
    provenance.update(old_assets=str(old), committed_prefix_frames=count, newly_generated_frames=len(needed)-count,
        reused_prefix_sha256=sha(out/'reused_prefix.json'), old_progress_sha256=progress_hash,
        original_prepare_source_sha256=V1_SHA, ignored_uncommitted_old_files=True)
    manifest = dict(schema='crn-train512-assets-v1', status='COMPLETE_CRN_TRAIN512_ASSETS', preparation_revision=2,
        selection_sha256=SELECTION_SHA, source_contract_sha256=args.source_contract_sha256,
        author_sources=contract, preparation_source_sha256=sha(__file__),
        source_data_root=str(source), official_split='train', full_train_samples=28130, selected_samples=512,
        selected_scenes=256, selected_history_frames=len(needed), key_offsets=list(KEY_OFFSETS),
        train_info=dict(file='data/nuscenes_infos_train.pkl', **info_copy),
        history_plan_sha256=sha(out/'history_plan.json'), metadata_sha256=metadata_sha,
        radar_files=projected, image_files=images, reuse=provenance,
        sensor_tables_read=sorted(sensor_nusc.tables), projection_annotation_input=False,
        annotation_policy='Original author full infos retained byte-exact; only sensor metadata enters radar workers; inference discards GT',
        GPU_used=False, model_constructed=False, optimizer_updates=0, elapsed_seconds=time.monotonic()-start)
    write(out/'manifest.json', manifest)
    write(out/'complete.json', dict(schema='crn-train512-assets-complete-v1', status=manifest['status'], preparation_revision=2,
        files_sha256={name: sha(out/name) for name in ('manifest.json', 'history_plan.json', 'reused_prefix.json')},
        train_info_sha256=info_copy['sha256'], source_sha256=sha(__file__), selection_sha256=SELECTION_SHA))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'source-data-root', 'source-contract', 'source-contract-sha256', 'selection',
                 'old-assets', 'old-launch', 'old-job', 'old-terminal-receipt', 'old-terminal-receipt-sha256', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--max-seconds', type=float, default=7200)
    args = p.parse_args()
    require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 7200, 'Bounded continuation budget')
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args, out)
        print(__import__('json').dumps(dict(status=result['status'], reused=result['reuse']['committed_prefix_frames'],
            newly_generated=result['reuse']['newly_generated_frames'], seconds=result['elapsed_seconds'])))
    except Exception as error:
        write(out/'failed.json', dict(status='FAILED_NO_RETRY', error=repr(error), source_sha256=sha(__file__)))
        raise


if __name__ == '__main__':
    main()
