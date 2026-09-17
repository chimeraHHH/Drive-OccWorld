"""Read-only mirror of the successful, immutable fixed512/dev200 endpoint.

Never launches remote work. Checkpoint tensors remain on the server for the
separate actual CPU audit. Refuses incomplete jobs and existing destinations.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile


REMOTE = r'''
import hashlib,io,json,pathlib,sys,tarfile,datetime
root=pathlib.Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/o_motion_20260915')
job=root/'jobs/object_state_forecast_train_v1';out=root/'object_state_forecast_train_v1'
def read(p):return json.loads(p.read_bytes())
state_bytes=(job/'state.json').read_bytes();state=json.loads(state_bytes);launch=read(job/'launch.json')
assert state['status']=='EXITED_ZERO' and state['returncode']==0 and state['finished_utc']
assert state['runner_pid']==launch['runner_pid']==69091 and launch['uid']==1009 and launch['startticks']==4640525
assert state['child']==dict(pid=69092,uid=1009,startticks=4640529)
assert state['command']==launch['command'] and launch['protocol_sha256']=='93bd80fa938739f40d46a13339a888a4bf1e377385e5cf269ce644f60982ccc9'
assert dict(zip(launch['command'][3::2],launch['command'][4::2]))['--out']==str(out)
assert all(not (pathlib.Path('/proc')/str(pid)).exists() for pid in (69091,69092))
assert not (out/'failed.json').exists()
done_bytes=(out/'complete.json').read_bytes();done=json.loads(done_bytes)
assert done['schema']=='object-state-forecast-training-v1' and done['status']=='COMPLETE_OBJECT_STATE_FORECAST_TRAINING'
assert done['mode']=='train' and done['updates']==512 and done['examples']==2048 and done['evaluated_samples']==200
top={'manifest.json','loaded_models.json','summary.json','physical_gradient_diagnostics.jsonl','development_records.jsonl','development_objects.jsonl'}
assert set(done['files_sha256'])==top and set(done['arm_complete_sha256'])==set(done['final_checkpoints'])=={'V','G'}
files={'complete.json':done_bytes,'_job/state.json':state_bytes}
def add(name,p,digest=None):
 raw=p.read_bytes()
 if digest is not None:assert hashlib.sha256(raw).hexdigest()==digest,name
 assert name not in files and '..' not in pathlib.PurePosixPath(name).parts and not name.startswith('/')
 files[name]=raw
for name in sorted(top):add(name,out/name,done['files_sha256'][name])
manifest=json.loads(files['manifest.json'])
assert manifest['schema']==done['schema'] and manifest['sources']['protocol_sha256']==launch['protocol_sha256']
for arm in ('V','G'):
 prefix='runs/'+arm+'/';folder=out/'runs'/arm
 add(prefix+'complete.json',folder/'complete.json',done['arm_complete_sha256'][arm]);ac=json.loads(files[prefix+'complete.json'])
 assert ac['status']==done['status'] and ac['arm']==arm and ac['updates']==512 and ac['examples']==2048 and ac['evaluated_samples']==200
 assert set(ac['files_sha256'])=={'manifest.json','training.jsonl','final.pth','development_records.jsonl'}
 assert done['final_checkpoints'][arm]['file']==prefix+'final.pth' and done['final_checkpoints'][arm]['sha256']==ac['files_sha256']['final.pth']
 for name in ('manifest.json','training.jsonl','development_records.jsonl'):add(prefix+name,folder/name,ac['files_sha256'][name])
 am=json.loads(files[prefix+'manifest.json'])
 assert am['arm']==arm and am['common_manifest_sha256']==done['files_sha256']['manifest.json'] and am['sources']==manifest['sources']
for name in ('launch.json','spec.json'):add('_job/'+name,job/name)
assert read(job/'spec.json')['command']==launch['command'] and read(job/'spec.json')['seconds']==4860
assert (job/'state.json').read_bytes()==state_bytes and (out/'complete.json').read_bytes()==done_bytes
receipt=dict(schema='object-state-endpoint-mirror-v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source=str(out),job=str(job),runner_and_child_missing=True,checkpoint_bytes_copied=False,
 files={n:dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)) for n,b in files.items()})
files['_mirror_receipt.json']=(json.dumps(receipt,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,raw in sorted(files.items()):
  info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600
  archive.addfile(info,io.BytesIO(raw))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();dest=Path(args.out).resolve()
    if dest.exists():raise FileExistsError(dest)
    here=Path(__file__).resolve().parent
    ssh=here.with_name('m0_improvement_20260915')/'ssh_config'
    result=subprocess.run(['ssh','-F',str(ssh),'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes',
        '-o','ConnectTimeout=12','10.254.30.73','python3 -c '+shlex.quote(REMOTE)],capture_output=True,timeout=120)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-6000:])
    contents={}
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as archive:
        for member in archive:
            name=member.name;rel=Path(name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts or name in contents:
                raise ValueError('Unexpected archive member: '+name)
            contents[name]=archive.extractfile(member).read()
            if len(contents[name])!=member.size:raise ValueError('Truncated member: '+name)
    receipt=json.loads(contents['_mirror_receipt.json'])
    if set(contents)!=set(receipt['files'])|{'_mirror_receipt.json'}:raise ValueError('Mirror ledger mismatch')
    for name,row in receipt['files'].items():
        raw=contents[name]
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('Local bytes mismatch: '+name)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.object-state-mirror-',dir=dest.parent) as temporary:
        stage=Path(temporary)/'endpoint';stage.mkdir()
        for name,raw in contents.items():
            path=stage/name;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as handle:handle.write(raw)
        if dest.exists():raise FileExistsError(dest)
        stage.rename(dest)
    print(json.dumps(dict(status='MIRRORED_REAL_COMPLETE_ENDPOINT',out=str(dest),files=len(contents),
        total_bytes=sum(map(len,contents.values())),complete_sha256=receipt['files']['complete.json']['sha256'],
        formal_state_sha256=receipt['files']['_job/state.json']['sha256'],checkpoint_bytes_copied=False)))


if __name__=='__main__':main()
