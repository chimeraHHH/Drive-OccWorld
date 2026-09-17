"""Read-only observation of the one named server campaign and owned identities.

No launch, stop, restart, model load or partial score calculation. Saves the
observed facts with UTC timestamp; transient SSH failure is not job completion.
"""
import datetime
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import datetime,json,pathlib,subprocess
root=pathlib.Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/o_motion_20260915')
job=root/'jobs/shared_rigid_formal_campaign_v1';campaign=root/'shared_rigid_formal_campaign_v1'
train=root/'shared_rigid_state_train_v1';evaluation=root/'shared_rigid_fixed_dev200_v1'
def read(p):return json.loads(p.read_text()) if p.is_file() else None
launch=read(job/'launch.json');state=read(job/'state.json');progress=read(campaign/'progress.json');live={}
identities={'runner':dict(pid=launch['runner_pid'],uid=launch['uid'],startticks=launch['startticks']),
 'controller':state.get('child')}
if progress and progress.get('child'):identities[progress['phase']]=progress['child']
for role,identity in identities.items():
 if not identity:continue
 p=pathlib.Path('/proc')/str(identity['pid'])
 try:
  f=(p/'stat').read_text().rsplit(')',1)[1].split()
  actual=dict(pid=identity['pid'],uid=p.stat().st_uid,startticks=int(f[19]),state=f[0],pgid=int(f[2]))
  actual['identity_matches']=all(actual[k]==identity[k] for k in ('pid','uid','startticks'));live[role]=actual
 except FileNotFoundError:live[role]=dict(pid=identity['pid'],missing=True)
latest={}
for arm in ('Cpl','Fix'):
 p=train/'runs'/arm/'training.jsonl'
 if p.exists():
  rows=[json.loads(x) for x in p.read_text().splitlines()[-10:]]
  latest[arm]=[dict(update=r['update'],examples=r['examples'],seconds=r['matched_accum4_update_seconds'],
   total_losses=[s['total_loss'] for s in r['samples']],
   physical_losses=[s['physical_loss'] for s in r['samples']],
   gradient_preclip={k:v['preclip_norm'] for k,v in r['gradients'].items()},
   peak_allocated_bytes=r['peak_allocated_bytes']) for r in rows]
lines=(job/'output.log').read_text().splitlines() if (job/'output.log').exists() else []
failures={stage:read(folder/'failed.json') for stage,folder in [('campaign',campaign),('training',train),('evaluation',evaluation)]}
completions={stage:read(folder/'complete.json') for stage,folder in [('campaign',campaign),('training',train),('evaluation',evaluation)]}
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state=state,live=live,
 campaign_progress=progress,training_progress=read(train/'progress.json'),evaluation_progress=read(evaluation/'progress.json'),
 latest_updates=latest,failures=failures,completions=completions,output_tail=lines[-20:],
 gpu_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)),allow_nan=False))
'''


def main():
    n=Path(__file__).resolve().parent;p=n.with_name('m0_improvement_20260915')
    r=subprocess.run(['ssh','-F',str(p/'ssh_config'),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
        '-o','ConnectTimeout=12','10.254.30.73','python3 -'],input=REMOTE,text=True,capture_output=True,timeout=60)
    r.check_returncode();data=json.loads(r.stdout)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path=n/'receipts'/('inspect_shared_rigid_formal_campaign_v1_'+stamp+'.json')
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    print(json.dumps(dict(receipt=str(path),**data),indent=2))


if __name__=='__main__':main()
