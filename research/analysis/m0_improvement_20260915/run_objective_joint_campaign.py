"""Two bounded native evaluation shards followed by a strict CPU merge.

Server-side, independent of the SSH client; no training, retry or resume.
Children inherit the outer job runner's process group for scoped cleanup.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from objective_joint_evaluation import load_contracts, full_gate, read, require, resolve, sha
from run_objective_supervision_campaign import check_assigned_gpus

MERGE_MAX_SECONDS = 900


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'evaluation-protocol', 'objective-protocol', 'protocol', 'repo', 'config', 'checkpoint'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--mode', choices=('pilot', 'full'), required=True)
    parser.add_argument('--parity-cache')
    parser.add_argument('--full-authorization')
    args = parser.parse_args(argv)
    package = Path(__file__).resolve().parent
    root = Path(args.root).resolve()
    require(not root.exists(), 'New campaign directory required; no overwrite or retry')
    _, _, protocol, _, names, _, _ = load_contracts(args.evaluation_protocol, args.objective_protocol, args.protocol)
    require(protocol['source_sha256'].get(Path(__file__).name) == sha(__file__), 'Wrapper source not frozen')
    if args.mode == 'pilot':
        require(args.parity_cache is not None and args.full_authorization is None, 'Pilot cache/auth contract')
        resources = protocol['resources']['pilot']
        authorization_sha = None
    else:
        require(args.parity_cache is None, 'Full must read native inputs')
        full_gate(args.full_authorization, args.evaluation_protocol, protocol, names)
        resources = read(args.full_authorization)['resources']
        authorization_sha = sha(args.full_authorization)
    cap = resources['max_seconds']
    gpu_before = check_assigned_gpus(['0', '1'])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = time.time()+cap

    def record(filename, value):
        temp = root/(filename+'.tmp')
        with temp.open('w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temp.replace(root/filename)

    manifest = dict(schema='objective-joint-two-gpu-campaign-v1', mode=args.mode,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        evaluation_protocol_sha256=sha(args.evaluation_protocol), wrapper_sha256=sha(__file__),
        full_authorization_sha256=authorization_sha, models=names, resources=resources,
        merge_max_seconds=MERGE_MAX_SECONDS, deadline_unix=deadline, gpu_before=gpu_before,
        automatic_retry=False, optimizer_steps=0)
    record('manifest.json', manifest)
    children = {}; logs = {}; statuses = {}

    def interrupted(*_):
        raise InterruptedError('Evaluation campaign interrupted; no automatic retry')

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for index in (0, 1):
            command = [sys.executable, '-B', str(package/'objective_joint_evaluation.py'),
                '--evaluation-protocol', args.evaluation_protocol, '--objective-protocol', args.objective_protocol,
                '--protocol', args.protocol, '--repo', args.repo, '--config', args.config,
                '--checkpoint', args.checkpoint, '--out', str(root/'evaluation'), '--mode', args.mode,
                '--shard-index', str(index), '--device', 'cuda:0', '--max-seconds', str(cap),
                '--max-allocated-gib', str(resources['max_allocated_gib']), '--deadline-unix', str(deadline)]
            if args.mode == 'pilot':
                command += ['--parity-cache', args.parity_cache]
            else:
                command += ['--full-authorization', args.full_authorization]
            logs[index] = (root/('shard_%02d.log'%index)).open('xb')
            gpu_uuid = gpu_before['assigned_indices_to_uuid'][str(index)]
            child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid),
                stdout=logs[index], stderr=subprocess.STDOUT)
            children[index] = child
            stat = (Path('/proc')/str(child.pid)/'stat').read_text()
            fields = stat[stat.rfind(')')+2:].split()
            statuses[index] = dict(status='RUNNING', pid=child.pid, uid=os.getuid(),
                startticks=int(fields[19]), pgid=int(fields[2]), physical_gpu=index,
                gpu_uuid=gpu_uuid, command=command)
        while any(child.poll() is None for child in children.values()):
            for index, child in children.items():
                code = child.poll()
                if code is not None:
                    statuses[index].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                    require(code == 0, 'Evaluation shard failed; stop this campaign without retry')
            require(time.time() <= deadline+120, 'Evaluation exceeded its deadline and cleanup grace')
            record('state.json', dict(status='EVALUATING', mode=args.mode, shards=statuses,
                seconds=time.monotonic()-started))
            time.sleep(5)
        for index, child in children.items():
            require(child.returncode == 0, 'Incomplete evaluation shard')
            statuses[index].update(status='COMPLETE', exit_code=0)
            complete = read(root/'evaluation'/('shard_%02d_of_02'%index)/'complete.json')
            require(complete['status'] == 'COMPLETE_JOINT_NATIVE_EVALUATION_SHARD' and
                    complete['mode'] == args.mode, 'Missing final shard receipt')
        command = [sys.executable, '-B', str(package/'objective_joint_merge.py'),
            '--evaluation-root', str(root/'evaluation'), '--evaluation-protocol', args.evaluation_protocol,
            '--objective-protocol', args.objective_protocol, '--protocol', args.protocol,
            '--selection', str(resolve(protocol['selections'][args.mode]['file'], args.evaluation_protocol)),
            '--out', str(root/'summary_v1'), '--mode', args.mode]
        if args.mode == 'full':
            command += ['--full-authorization', args.full_authorization]
        record('state.json', dict(status='MERGING', shards=statuses, command=command))
        with (root/'merge.log').open('xb') as stream:
            subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=stream,
                stderr=subprocess.STDOUT, check=True, timeout=MERGE_MAX_SECONDS)
        record('complete.json', dict(status='COMPLETE_OBJECTIVE_JOINT_CAMPAIGN', mode=args.mode,
            manifest_sha256=sha(root/'manifest.json'), summary_complete_sha256=sha(root/'summary_v1'/'complete.json'),
            seconds=time.monotonic()-started, automatic_retry=False, optimizer_steps=0))
        record('state.json', dict(status='COMPLETE', shards=statuses))
    except BaseException as exc:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        cleanup_deadline = time.monotonic()+30
        for child in children.values():
            if child.poll() is None:
                try:
                    child.wait(timeout=max(.1, cleanup_deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
        record('failed.json', dict(status='FAILED_NO_RETRY', error=repr(exc), shards=statuses,
            seconds=time.monotonic()-started, optimizer_steps=0))
        raise
    finally:
        for stream in logs.values():
            stream.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
