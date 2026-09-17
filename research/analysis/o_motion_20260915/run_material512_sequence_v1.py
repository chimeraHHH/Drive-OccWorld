"""Server-resident fixed train/evaluate sequence with no automatic retries.

Run underneath the ownership-scoped job_runner.py. Stage children inherit
the controller process group so its runner can clean up only this job.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    planpath = Path(sys.argv[1])
    assert sha(planpath) == sys.argv[2]
    plan = json.loads(planpath.read_text())
    assert plan['schema'] == 'material512-server-sequence-v1'
    assert plan['status'] == 'FROZEN'
    assert plan['automatic_retries'] == 0
    assert [s['name'] for s in plan['stages']] == [f'{a}_{k}' for a in 'TJD' for k in ('train', 'evaluate')]
    assert plan['controller_sha256'] == sha(__file__)
    assert Path(plan['output']).is_absolute() and Path(plan['cwd']).is_absolute()
    for stage in plan['stages']:
        assert Path(stage['output']).is_absolute()
        assert all(Path(name).is_absolute() for name in stage['bindings'])
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    write(out/'plan.json', plan)
    started = time.monotonic()
    state = dict(status='RUNNING', completed_stages=[], active_stage=None,
                 plan_sha256=sys.argv[2], controller_pid=os.getpid())
    write(out/'state.json', state)
    try:
        for stage in plan['stages']:
            # Output reuse requires a separately reviewed continuation plan.
            assert not Path(stage['output']).exists(), 'Stage output already exists'
            for name, digest in stage['bindings'].items():
                assert sha(name) == digest, 'Stage source/protocol changed: '+name
            command = list(stage['command'])
            if stage['name'].endswith('_evaluate'):
                training_stage = state['completed_stages'][-1]
                assert training_stage['name'] == stage['arm']+'_train'
                command += ['--fit-complete-sha256', training_stage['complete_sha256']]
            with (out/(stage['name']+'.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=plan['cwd'],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                proc = Path('/proc')/str(child.pid)
                text = (proc/'stat').read_text()
                fields = text[text.rfind(')')+2:].split()
                state['active_stage'] = dict(name=stage['name'], pid=child.pid,
                    uid=proc.stat().st_uid, startticks=int(fields[19]),
                    started_elapsed_seconds=time.monotonic()-started)
                write(out/'state.json', state)
                code = child.wait(timeout=stage['max_seconds'])
            assert code == 0, 'Stage failed: '+stage['name']
            donepath = Path(stage['output'])/'complete.json'
            done = json.loads(donepath.read_text())
            assert done['status'] == stage['complete_status']
            assert done['protocol_sha256'] == stage['protocol_sha256']
            assert done['arm'] == stage['arm']
            if stage['name'].endswith('_train'):
                assert done['updates'] == 2048
            else:
                assert done['anchors'] == 200 and done['optimizer_updates'] == 0
            for name, digest in done['files_sha256'].items():
                assert Path(name).name == name and sha(Path(stage['output'])/name) == digest
            state['completed_stages'].append(dict(name=stage['name'],
                complete_sha256=sha(donepath), elapsed_seconds=time.monotonic()-started))
            state['active_stage'] = None
            write(out/'state.json', state)
        state.update(status='COMPLETE', elapsed_seconds=time.monotonic()-started)
        write(out/'state.json', state)
        write(out/'complete.json', state)
    except BaseException as error:
        # The outer runner owns cleanup of this group, including a timed-out
        # stage; no unrelated server task is inspected or signalled here.
        state.update(status='FAILED', error=repr(error), elapsed_seconds=time.monotonic()-started)
        write(out/'state.json', state)
        raise


if __name__ == '__main__':
    main()
