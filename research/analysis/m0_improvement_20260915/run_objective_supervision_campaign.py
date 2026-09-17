"""Run independent fixed F/O arms on two assigned GPUs, then a CPU paired summary.

Each arm owns a new directory and its fixed per-arm deadline. No restart or
checkpoint selection is provided. A failed arm does not change its peer's
training; the paired summary is refused unless both finish successfully.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_gpu_state(inventory_csv, compute_csv, gpus):
    """Reject aliases/ambiguous devices and any existing compute PID on either card."""
    if sorted(gpus) != ['0','1']:
        raise ValueError('This campaign accepts only the two physical indices 0 and 1')
    inventory = {}
    for row in csv.reader(inventory_csv.splitlines()):
        if not row:
            continue
        if len(row) != 2:
            raise ValueError('Unexpected nvidia-smi GPU inventory schema')
        index, uuid = [x.strip() for x in row]
        if index in inventory or not index.isdigit() or not uuid.startswith('GPU-'):
            raise ValueError('Ambiguous GPU inventory')
        inventory[index] = uuid
    if any(index not in inventory for index in gpus) or len({inventory[i] for i in gpus}) != 2:
        raise ValueError('Assigned physical GPU indices do not resolve to two distinct UUIDs')
    chosen = {inventory[i] for i in gpus}; processes = []
    for row in csv.reader(compute_csv.splitlines()):
        if not row:
            continue
        if len(row) != 2:
            raise ValueError('Unexpected nvidia-smi compute-process schema')
        uuid, pid = [x.strip() for x in row]
        if not uuid.startswith('GPU-') or not pid.isdigit() or int(pid) <= 0:
            raise ValueError('Invalid compute-process identity')
        if uuid in chosen:
            processes.append(dict(gpu_uuid=uuid,pid=int(pid)))
    if processes:
        raise RuntimeError('Assigned GPU already has compute processes; nothing started: '+json.dumps(processes))
    return dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        assigned_indices_to_uuid={i:inventory[i] for i in gpus}, compute_processes=[],
        both_assigned_gpus_idle=True, scope='read-only pre-spawn snapshot; no process was stopped')


def check_assigned_gpus(gpus):
    inventory = subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],
        text=True,capture_output=True,check=True,timeout=15).stdout
    processes = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        text=True,capture_output=True,check=True,timeout=15).stdout
    return parse_gpu_state(inventory,processes,gpus)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'objective-protocol', 'protocol', 'repo', 'config', 'checkpoint',
                 'train-cache', 'dev-cache', 'control-run', 'gpus'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(argv)
    root = Path(a.root).resolve(); pkg = Path(__file__).resolve().parent
    if root.exists():
        raise ValueError('Campaign output must be new')
    gpus = a.gpus.split(',')
    if sorted(gpus) != ['0','1']:
        raise ValueError('Use physical GPU indices 0,1 or 1,0 in F,O order')
    frozen = json.loads(Path(a.objective_protocol).read_text())
    for name in ('objective_supervision_train.py', Path(__file__).name):
        if sha(pkg/name) != frozen['source_sha256'][name]:
            raise ValueError('Frozen campaign source differs: '+name)
    from objective_supervision_train import load_contracts, validate_engineering
    parent, frozen, _ = load_contracts(a.objective_protocol, a.protocol)
    evidence = validate_engineering(frozen, a.objective_protocol, parent)
    caps = {arm: frozen['resource_policy']['arms'][arm]['max_seconds'] for arm in ('F','O')}
    aggregate_cap = frozen['resource_policy']['aggregate_max_seconds']
    if not all(type(x) is int and x > 0 for x in list(caps.values())+[aggregate_cap]):
        raise ValueError('Every stage needs a positive frozen deadline')
    gpu_before = check_assigned_gpus(gpus)
    root.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    def record(name, value):
        temp = root/(name+'.tmp')
        with temp.open('w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        temp.replace(root/name)
    manifest = dict(schema='objective-supervision-two-gpu-campaign-v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        objective_protocol_sha256=sha(a.objective_protocol), parent_protocol_sha256=sha(a.protocol),
        wrapper_sha256=sha(__file__), arms=['F','O'], assigned_visible_devices=dict(zip(('F','O'),gpus)),
        per_arm_max_seconds=caps, aggregate_max_seconds=aggregate_cap, engineering_evidence=evidence,
        native1_control_reused=True, automatic_retry=False, full_validation=False, gpu_before=gpu_before)
    record('manifest.json', manifest)
    processes = {}; logs = {}; statuses = {}; launches = {}
    def stop(*_):
        raise InterruptedError('Campaign terminated; no automatic restart')
    handlers = {sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        for arm, gpu in zip(('F','O'),gpus):
            command = [sys.executable,'-B',str(pkg/'objective_supervision_train.py'),
                '--objective-protocol',a.objective_protocol,'--protocol',a.protocol,'--repo',a.repo,
                '--config',a.config,'--checkpoint',a.checkpoint,'--train-cache',a.train_cache,
                '--dev-cache',a.dev_cache,'--control-run',a.control_run,'--arm',arm,
                '--out',str(root/'runs'/arm),'--max-seconds',str(caps[arm])]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            logs[arm] = (root/(arm+'.log')).open('xb')
            # Inherit the job_runner process group so its final scoped kill also
            # reaches children if this wrapper itself exceeds cleanup grace.
            processes[arm] = subprocess.Popen(command, env=env, stdout=logs[arm], stderr=subprocess.STDOUT)
            launches[arm] = time.monotonic()
            statuses[arm] = dict(status='RUNNING',pid=processes[arm].pid,visible_device=gpu,command=command)
        while any(p.poll() is None for p in processes.values()):
            for arm, process in processes.items():
                code = process.poll()
                if code is None and time.monotonic()-launches[arm] > caps[arm]+120:
                    # The child's own alarm already gave it 120 seconds to write
                    # its interruption receipt. This is a hard safety backstop.
                    process.kill(); process.wait()
                    statuses[arm].update(status='KILLED_AFTER_DEADLINE_GRACE',exit_code=process.returncode)
                elif code is not None:
                    statuses[arm].update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code)
            record('state.json',dict(status='RUNNING_FIXED_ARMS',arms=statuses,seconds=time.monotonic()-started))
            time.sleep(5)
        for arm, process in processes.items():
            statuses[arm].update(status='COMPLETE' if process.returncode==0 else 'FAILED',exit_code=process.returncode)
        if any(process.returncode != 0 for process in processes.values()):
            raise RuntimeError('At least one fixed arm failed; paired summary is forbidden')
        # Final evaluation is part of each unchanged native training block.
        for arm in ('F','O'):
            complete = json.loads((root/'runs'/arm/'complete.json').read_text())
            if complete['status'] != 'TRAINED_AND_DEVELOPMENT_EVALUATED' or complete['updates'] != 512:
                raise ValueError('Incomplete fixed final arm: '+arm)
        command = [sys.executable,'-B',str(pkg/'objective_supervision_aggregate.py'),
            '--objective-protocol',a.objective_protocol,'--protocol',a.protocol,
            '--cache',a.dev_cache,'--control-run',a.control_run,'--runs-root',str(root/'runs'),
            '--out',str(root/'summary_v1')]
        record('state.json',dict(status='RUNNING_PAIRED_SUMMARY',arms=statuses,command=command))
        with (root/'aggregate.log').open('xb') as stream:
            subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,
                stderr=subprocess.STDOUT,check=True,timeout=aggregate_cap)
        record('complete.json',dict(status='COMPLETE_OBJECTIVE_SUPERVISION_CAMPAIGN',
            objective_protocol_sha256=sha(a.objective_protocol),manifest_sha256=sha(root/'manifest.json'),
            arm_complete_sha256={arm:sha(root/'runs'/arm/'complete.json') for arm in ('F','O')},
            summary_complete_sha256=sha(root/'summary_v1'/'complete.json'), seconds=time.monotonic()-started,
            full_validation=False,automatic_expansion=False,automatic_retry=False))
        record('state.json',dict(status='COMPLETE',arms=statuses,stage='paired_development_summary'))
    except BaseException as exc:
        for sig in handlers:
            signal.signal(sig,signal.SIG_IGN)
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic()+30
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=max(.1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        record('failed.json',dict(status='FAILED_NO_RETRY',error=repr(exc),arms=statuses,
            seconds=time.monotonic()-started,automatic_retry=False))
        raise
    finally:
        for stream in logs.values():
            stream.close()
        for sig,handler in handlers.items():
            signal.signal(sig,handler)


if __name__ == '__main__':
    main()
