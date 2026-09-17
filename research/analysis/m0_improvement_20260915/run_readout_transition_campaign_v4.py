"""Wait for full evaluation, then run one bounded readout/feature diagnostic.

Prepared interface: freeze and audit with the producer before dispatch.
No training, retry, partial-result selection, or dependency process mutation.
"""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from objective_full_dependency import read, require, sha, wait_for_completed_full


def idle_gpu_zero():
    query = lambda field: subprocess.run(['nvidia-smi', field, '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=15).stdout
    inventory = {}
    for row in csv.reader(query('--query-gpu=index,uuid').splitlines()):
        require(len(row) == 2, 'Unexpected GPU inventory')
        index, uuid = [x.strip() for x in row]
        require(index not in inventory, 'Repeated GPU index')
        inventory[index] = uuid
    uuid = inventory.get('0')
    require(uuid == 'GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe', 'Physical GPU0 identity differs')
    for row in csv.reader(query('--query-compute-apps=gpu_uuid,pid').splitlines()):
        require(len(row) == 2, 'Unexpected GPU process query')
        current, pid = [x.strip() for x in row]
        require(pid.isdigit(), 'Unexpected GPU process identity')
        require(current != uuid, 'GPU0 occupied after full completion; refuse to interfere')
    return dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                physical_gpu=0, gpu_uuid=uuid, active_compute_processes=[], snapshot_only=True)


def stage_receipt(directory, mode, protocol_sha):
    directory = Path(directory)
    require(not (directory/'failed.json').exists(), 'Diagnostic stage has failure receipt')
    done = read(directory/'complete.json')
    status = 'PASS_READOUT_TRANSITION_ENGINEERING' if mode == 'pilot' else 'COMPLETE_READOUT_TRANSITION_DIAGNOSTIC'
    require(done['schema'] == 'readout-transition-complete-v1' and done['status'] == status and
            done['mode'] == mode, 'Wrong diagnostic stage completion')
    files = done['files_sha256']
    require(set(files) == {'manifest.json', 'records.jsonl', 'summary.json', 'report.md'}, 'Incomplete stage artifacts')
    for filename, expected in files.items():
        require(sha(directory/filename) == expected, 'Stage artifact changed: '+filename)
    summary = read(directory/'summary.json', files['summary.json'])
    samples, scenes = (2, 2) if mode == 'pilot' else (200, 100)
    require(summary['schema'] == 'readout-transition-summary-v1' and summary['mode'] == mode and
            summary['sources']['diagnostic_protocol_sha256'] == protocol_sha and
            summary['diagnostic_protocol_sha256'] == protocol_sha and
            done['diagnostic_protocol_sha256'] == protocol_sha,
            'Stage protocol differs')
    require(summary['samples'] == done['samples'] == samples and summary['scenes'] == done['scenes'] == scenes and
            summary['all_engineering_gates_passed'] is True and
            summary['optimizer_updates'] == done['optimizer_updates'] == 0,
            'Stage coverage or engineering gate differs')
    return done, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'full-campaign', 'full-job', 'repo', 'config', 'checkpoint', 'protocol',
                 'objective-protocol', 'diagnostic-protocol', 'cache', 'native-runs-root',
                 'runs-root', 'development-summary'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--wait-deadline-unix', type=float, required=True)
    args = parser.parse_args(argv)
    require(os.getuid() == 1009, 'Expected wangning server account')
    package = Path(__file__).resolve().parent
    protocol = read(args.diagnostic_protocol)
    require(protocol['schema'] == 'readout-transition-diagnostic-protocol-v1' and
            protocol['status'] == 'FROZEN', 'Diagnostic protocol not frozen')
    protocol_sha = sha(args.diagnostic_protocol)
    for filename, expected in protocol['source_sha256'].items():
        require(Path(filename).name == filename and sha(package/filename) == expected, 'Staged diagnostic source differs')
    require(protocol['source_sha256'].get(Path(__file__).name) == sha(__file__) and
            protocol['source_sha256'].get('objective_full_dependency.py') == sha(package/'objective_full_dependency.py'),
            'Wrapper/dependency gate must be frozen')
    policy = protocol['resources']
    require(policy['total_max_seconds'] == 3600 and policy['max_allocated_gib'] == 32 and
            policy['pilot'] == {'max_seconds': 300, 'max_allocated_gib': 32} and
            policy['development_ceiling'] == {'max_seconds': 3000, 'max_allocated_gib': 32},
            'Unexpected diagnostic budget')
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    launched = time.monotonic()
    child = None

    def record(filename, value):
        temp = root/(filename+'.tmp')
        with temp.open('w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        temp.replace(root/filename)

    def interrupted(*_):
        raise InterruptedError('Diagnostic campaign interrupted; no retry')

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    record('manifest.json', dict(schema='readout-transition-campaign-v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        diagnostic_protocol_sha256=protocol_sha, wrapper_sha256=sha(__file__),
        dependency_gate_sha256=sha(package/'objective_full_dependency.py'),
        full_campaign=args.full_campaign, full_job=args.full_job,
        wait_deadline_unix=args.wait_deadline_unix, resources=policy,
        optimizer_steps=0, automatic_retry=False, model_selection=False))
    try:
        dependency = wait_for_completed_full(args.full_campaign, args.full_job, args.wait_deadline_unix,
                                             lambda value: record('state.json', value))
        record('dependency_complete.json', dependency)
        gpu = idle_gpu_zero()
        record('gpu_before.json', gpu)
        diagnostic_started = time.monotonic()
        deadline = diagnostic_started+policy['total_max_seconds']
        common = []
        for name in ('repo', 'config', 'checkpoint', 'protocol', 'objective-protocol',
                     'diagnostic-protocol', 'cache', 'native-runs-root', 'runs-root', 'development-summary'):
            common += ['--'+name, getattr(args, name.replace('-', '_'))]

        def run_stage(mode, cap, extra=None):
            nonlocal child
            require(time.monotonic()+cap <= deadline, 'Stage budget exceeds total diagnostic deadline')
            require(sha(args.diagnostic_protocol) == protocol_sha, 'Diagnostic protocol changed')
            for filename, expected in protocol['source_sha256'].items():
                require(sha(package/filename) == expected, 'Staged source changed before '+mode+': '+filename)
            command = [sys.executable, '-B', str(package/'readout_transition_diagnostic_v4.py')]+common+[
                '--mode', mode, '--out', str(root/mode), '--device', 'cuda:0',
                '--max-seconds', str(cap), '--max-allocated-gib', '32']+(extra or [])
            with (root/(mode+'.log')).open('xb') as stream:
                child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['gpu_uuid']),
                                         stdout=stream, stderr=subprocess.STDOUT)
                stat = (Path('/proc')/str(child.pid)/'stat').read_text()
                fields = stat[stat.rfind(')')+2:].split()
                record('state.json', dict(status='RUNNING_'+mode.upper(), pid=child.pid,
                    uid=os.getuid(), startticks=int(fields[19]), pgid=int(fields[2]),
                    command=command, max_seconds=cap, gpu_uuid=gpu['gpu_uuid']))
                child.wait(timeout=min(cap+15, max(.1, deadline-time.monotonic())))
                require(child.returncode == 0, 'Diagnostic stage failed; no retry: '+mode)
            return stage_receipt(root/mode, mode, protocol_sha)

        _, pilot = run_stage('pilot', policy['pilot']['max_seconds'])
        resources = pilot['resources']
        # Resource-derived budget only; no metric or candidate selection is read.
        measured_init = resources['pre_sample_initialization_seconds']
        measured_sample = resources['max_sample_seconds']
        require(math.isfinite(measured_init) and measured_init >= 0 and
                math.isfinite(measured_sample) and measured_sample > 0, 'Invalid pilot resource evidence')
        estimate = (measured_init+200*measured_sample)*1.75+60
        cap = max(300, math.ceil(estimate))
        require(cap <= policy['development_ceiling']['max_seconds'] and
                time.monotonic()+cap <= deadline, 'Measured development budget exceeds frozen ceiling; stop')
        authorization = dict(schema='readout-transition-development-authorization-v1', status='FROZEN',
            diagnostic_protocol_sha256=protocol_sha, pilot_complete_sha256=sha(root/'pilot/complete.json'),
            pilot_summary_sha256=sha(root/'pilot/summary.json'), resources=dict(max_seconds=cap,max_allocated_gib=32),
            resource_derivation=dict(pre_sample_initialization_seconds=measured_init,max_sample_seconds=measured_sample,
                anchors=200,multiplier=1.75,additional_seconds=60,minimum_seconds=300,raw_budget_seconds=estimate),
            elapsed_since_diagnostic_start=time.monotonic()-diagnostic_started,
            model_selection=False, optimizer_steps=0)
        record('development_authorization.json', authorization)
        # Pilot exited; recheck that no other compute job arrived on the chosen GPU.
        require(idle_gpu_zero()['gpu_uuid'] == gpu['gpu_uuid'], 'GPU identity changed')
        run_stage('development', cap, ['--pilot', str(root/'pilot'), '--authorization', str(root/'development_authorization.json')])
        record('complete.json', dict(status='COMPLETE_READOUT_TRANSITION_CAMPAIGN',
            diagnostic_protocol_sha256=protocol_sha, manifest_sha256=sha(root/'manifest.json'),
            dependency_receipt_sha256=sha(root/'dependency_complete.json'),
            pilot_complete_sha256=sha(root/'pilot/complete.json'),
            development_complete_sha256=sha(root/'development/complete.json'),
            development_authorization_sha256=sha(root/'development_authorization.json'),
            diagnostic_seconds=time.monotonic()-diagnostic_started, total_seconds=time.monotonic()-launched,
            optimizer_steps=0, automatic_retry=False, model_selection=False))
        record('state.json', dict(status='COMPLETE', optimizer_steps=0))
    except BaseException as exc:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        record('failed.json', dict(status='FAILED_NO_RETRY', error=repr(exc),
            seconds=time.monotonic()-launched, optimizer_steps=0))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
