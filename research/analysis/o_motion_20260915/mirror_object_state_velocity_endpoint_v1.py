"""Mirror only the one completed N/P/Z diagnostic; default check is local only.

No launch, retry, remote write, checkpoint load, or partial-score retrieval.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
DISPATCH_SHA = '6a972e7c3e821f2a4d6d70cc229fd4872bb658c095f9edb19477cf0d7a6f078c'
DISPATCHER_SHA = '3cf0a9efafa88cf0819b18c238f59edfd47a8545e0f790da549597d68623d16b'
SOURCE_SHA = 'b7a3e51a115fa3d604a2b1f6413a8ac506ed3770471ae688e0e747b78ec0bbea'
NAME = 'object_state_velocity_assignment_v1'
SCHEMA = 'object-state-velocity-assignment-diagnostic-v1'
RESULT_FILES = {'manifest.json', 'loaded_models.json', 'records.jsonl', 'physical_objects.jsonl', 'summary.json', 'report.md'}
JOB_FILES = {'state.json', 'launch.json', 'spec.json', 'output.log'}


def digest(raw): return hashlib.sha256(raw).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)


def local_binding():
    raw = (HERE / 'receipts' / (NAME + '_dispatch.json')).read_bytes()
    require(digest(raw) == DISPATCH_SHA, 'Actual dispatch receipt changed')
    dispatch = json.loads(raw); require(dispatch['returncode'] == 0, 'Dispatch failed')
    launch = json.loads(dispatch['stdout'])
    require(launch['name'] == NAME and launch['diagnostic_sha256'] == SOURCE_SHA and
            launch['dispatch_source_sha256'] == DISPATCHER_SHA, 'Wrong dispatch/source')
    require(launch['runner'] == dict(pid=70988, uid=1009, startticks=5215033, pgid=70988) and
            launch['child'] == dict(pid=70989, uid=1009, startticks=5215035), 'Wrong actual process identity')
    require(launch['outer_seconds'] == 1260 and launch['inner_seconds'] == 1200 and launch['max_allocated_gib'] == 32,
            'Wrong fixed resource contract')
    require(len(launch['transfer_files']) == 37, 'Source package differs')
    for name, item in launch['transfer_files'].items():
        rel = PurePosixPath(name)
        require(len(rel.parts) == 2 and rel.parts[0] in ('package', 'provenance') and '..' not in rel.parts, 'Unsafe source path')
        candidates = [HERE / rel.name, HERE.with_name('m0_improvement_20260915') / rel.name]
        match = next((p for p in candidates if p.is_file() and digest(p.read_bytes()) == item['sha256']), None)
        require(match is not None and match.stat().st_size == item['bytes'], 'Local source changed: ' + name)
    require(digest((HERE / 'dispatch_object_state_velocity_assignment_v1.py').read_bytes()) == DISPATCHER_SHA,
            'Reviewed dispatcher changed')
    return dict(launch=launch, dispatch_sha256=DISPATCH_SHA, helper_sha256=digest(Path(__file__).read_bytes()),
                result_files=sorted(RESULT_FILES), job_files=sorted(JOB_FILES), schema=SCHEMA)


REMOTE = r'''
import datetime,hashlib,io,json,os,pathlib,sys,tarfile
bound=json.load(sys.stdin);expected=bound['launch'];job=pathlib.Path(expected['job']);out=pathlib.Path(expected['out'])
def sha(raw):return hashlib.sha256(raw).hexdigest()
def regular(p):
 assert p.is_file() and not p.is_symlink(),str(p)
 return p.read_bytes()
assert os.getuid()==1009
launch_bytes=regular(job/'launch.json');launch=json.loads(launch_bytes);assert launch==expected
spec_bytes=regular(job/'spec.json');spec=json.loads(spec_bytes)
state_bytes=regular(job/'state.json');state=json.loads(state_bytes)
assert state['runner_pid']==expected['runner']['pid'] and state['child']==expected['child']
assert state['command']==spec['command']==expected['command'] and spec['seconds']==1260
assert spec['env']['CUDA_VISIBLE_DEVICES']=='0'
args=dict(zip(expected['command'][3::2],expected['command'][4::2]))
assert args['--mode']=='run' and args['--out']==str(out) and args['--max-seconds']=='1200' and args['--max-allocated-gib']=='32'
for name,item in expected['transfer_files'].items():
 raw=regular(job/name);assert len(raw)==item['bytes'] and sha(raw)==item['sha256'],name
live={}
for label in ('runner','child'):
 identity=expected[label];p=pathlib.Path('/proc')/str(identity['pid'])
 try:
  fields=(p/'stat').read_text();fields=fields[fields.rfind(')')+2:].split()
  actual=dict(pid=identity['pid'],uid=p.stat().st_uid,startticks=int(fields[19]),pgid=int(fields[2]))
 except FileNotFoundError:continue
 assert all(actual[k]==identity[k] for k in ('pid','uid','startticks')) and actual['pgid']==identity['pid'],'PID reused/unexpected identity'
 live[label]=actual
if live:
 print(json.dumps(dict(status='LIVE_OR_FINALIZING_NO_RESULTS_FETCHED',job=str(job),job_status=state['status'],live=live,partial_scores_read=False)))
 sys.exit(0)
if state['status']!='EXITED_ZERO' or state.get('returncode')!=0 or not state.get('finished_utc'):
 print(json.dumps(dict(status='NOT_SUCCESSFUL_TERMINAL_NO_RESULTS_FETCHED',job=str(job),job_status=state['status'],returncode=state.get('returncode'),runner_and_child_missing=True,partial_scores_read=False)))
 sys.exit(0)
assert not (out/'failed.json').exists()
done_bytes=regular(out/'complete.json');done=json.loads(done_bytes)
assert done['schema']==bound['schema'] and done['status']=='COMPLETE_SAME_WEIGHT_VELOCITY_DIAGNOSTIC'
assert done['samples']==200 and done['scenes']==100 and done['physical_rows_per_condition']==16074 and done['optimizer_updates']==0
assert done['training_complete_sha256']==args['--complete-sha256'] and done['cpu_audit_complete_sha256']==args['--cpu-audit-complete-sha256']
assert done['source_sha256']==expected['diagnostic_sha256'] and set(done['files_sha256'])==set(bound['result_files'])
files={'complete.json':done_bytes,'_job/state.json':state_bytes,'_job/launch.json':launch_bytes,'_job/spec.json':spec_bytes}
for name in bound['result_files']:
 raw=regular(out/name);assert sha(raw)==done['files_sha256'][name],name
 files[name]=raw
files['_job/output.log']=regular(job/'output.log')
manifest=json.loads(files['manifest.json']);summary=json.loads(files['summary.json'])
assert manifest['schema']==done['schema']==summary['schema'] and summary['status']==done['status'] and summary['optimizer_updates']==0
assert manifest['source_sha256']==done['source_sha256'] and manifest['training_protocol_sha256']==expected['protocol_sha256']
for key in ('training_complete_sha256','cpu_audit_complete_sha256'):assert manifest[key]==done[key]
assert manifest['resources']==dict(max_seconds=1200.0,max_allocated_gib=32.0)
assert len(manifest['dependencies_sha256'])==28
for name,value in manifest['dependencies_sha256'].items():assert expected['transfer_files']['package/'+name]['sha256']==value
# Final immutable terminal snapshot before transmitting any results.
assert regular(job/'state.json')==state_bytes and regular(job/'launch.json')==launch_bytes and regular(job/'spec.json')==spec_bytes
assert regular(out/'complete.json')==done_bytes
assert all(not (pathlib.Path('/proc')/str(expected[k]['pid'])).exists() for k in ('runner','child'))
receipt=dict(schema='object-state-velocity-endpoint-mirror-v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source=str(out),job=str(job),dispatch_sha256=bound['dispatch_sha256'],helper_sha256=bound['helper_sha256'],
 diagnostic_sha256=expected['diagnostic_sha256'],runner=expected['runner'],child=expected['child'],
 runner_and_child_missing=True,source_files_verified=len(expected['transfer_files']),remote_writes=False,
 files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in files.items()})
files['_mirror_receipt.json']=(json.dumps(receipt,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,raw in sorted(files.items()):
  info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;archive.addfile(info,io.BytesIO(raw))
'''


def unpack(raw, bound):
    contents = {}
    expected = RESULT_FILES | {'complete.json', '_mirror_receipt.json'} | {'_job/' + n for n in JOB_FILES}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for member in archive:
            require(member.isfile() and member.name in expected and member.name not in contents, 'Unexpected/duplicate archive member')
            contents[member.name] = archive.extractfile(member).read()
            require(len(contents[member.name]) == member.size, 'Truncated archive member')
    require(set(contents) == expected, 'Missing endpoint artifacts')
    receipt = json.loads(contents['_mirror_receipt.json'])
    require(receipt['dispatch_sha256'] == DISPATCH_SHA and receipt['helper_sha256'] == bound['helper_sha256'] and
            receipt['diagnostic_sha256'] == SOURCE_SHA and receipt['runner_and_child_missing'] is True and
            set(receipt['files']) == expected - {'_mirror_receipt.json'}, 'Mirror provenance differs')
    for name, item in receipt['files'].items():
        require(len(contents[name]) == item['bytes'] and digest(contents[name]) == item['sha256'], 'Local SHA/size differs: ' + name)
    done = json.loads(contents['complete.json'])
    require(set(done['files_sha256']) == RESULT_FILES, 'Local complete ledger differs')
    for name, value in done['files_sha256'].items(): require(digest(contents[name]) == value, 'Local complete SHA differs: ' + name)
    require(json.loads(contents['_job/launch.json']) == bound['launch'], 'Local launch differs')
    return contents, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'mirror'), default='check')
    parser.add_argument('--out', help='New local endpoint directory; required only for mirror')
    args = parser.parse_args();bound = local_binding()
    if args.mode == 'check':
        print(json.dumps(dict(status='LOCAL_BINDINGS_VERIFIED_NO_SSH',dispatch_sha256=DISPATCH_SHA,
                              source_files=len(bound['launch']['transfer_files']),helper_sha256=bound['helper_sha256'])));return
    require(args.out, 'New output required');dest = Path(args.out).resolve()
    require(not dest.exists(), 'Existing destination; no overwrite/retry')
    ssh = HERE.with_name('m0_improvement_20260915') / 'ssh_config'
    command = ['ssh', '-F', str(ssh), '-o', 'StrictHostKeyChecking=yes', '-o', 'BatchMode=yes',
               '-o', 'ConnectTimeout=12', '10.254.30.73', shlex.join([bound['launch']['command'][0], '-B', '-c', REMOTE])]
    result = subprocess.run(command, input=json.dumps(bound).encode(), capture_output=True, timeout=180)
    require(result.returncode == 0, result.stderr.decode(errors='replace')[-6000:])
    if not result.stdout.startswith(b'\x1f\x8b'):
        status = json.loads(result.stdout)
        require(status['status'] in ('LIVE_OR_FINALIZING_NO_RESULTS_FETCHED', 'NOT_SUCCESSFUL_TERMINAL_NO_RESULTS_FETCHED'), 'Unexpected remote status')
        print(json.dumps(status));return
    contents, receipt = unpack(result.stdout, bound)
    dest.parent.mkdir(parents=True, exist_ok=True);dest.mkdir(exist_ok=False)
    # Receipt is written last. An interrupted local write is never called a complete mirror.
    for name in sorted(contents, key=lambda n: n == '_mirror_receipt.json'):
        path = dest / name;path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:stream.write(contents[name])
        require(digest(path.read_bytes()) == digest(contents[name]), 'Persisted local bytes differ: ' + name)
    print(json.dumps(dict(status='MIRRORED_REAL_COMPLETE_ENDPOINT',out=str(dest),files=len(contents),
                          complete_sha256=receipt['files']['complete.json']['sha256'],
                          state_sha256=receipt['files']['_job/state.json']['sha256'],remote_writes=False)))


if __name__ == '__main__': main()
