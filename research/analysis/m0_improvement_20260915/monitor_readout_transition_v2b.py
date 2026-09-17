"""Read-only snapshot of the fixed after-full diagnostic; no performance reads."""
import datetime
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
REMOTE = r'''
import datetime,json,os,subprocess
from pathlib import Path
root=Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915')
job=root/'jobs/readout_transition_v2b'; campaign=root/'campaign_readout_transition_v2b'
def read(path):return json.loads(path.read_text()) if path.is_file() else None
def process(identity):
 p=Path('/proc')/str(identity['pid'])
 try:
  stat=(p/'stat').read_text();f=stat[stat.rfind(')')+2:].split()
  actual=dict(pid=identity['pid'],uid=p.stat().st_uid,startticks=int(f[19]),state=f[0],pgid=int(f[2]))
  return dict(expected=identity,actual=actual,identity_matches=all(actual[k]==identity[k] for k in ('pid','uid','startticks')),live=f[0]!='Z')
 except FileNotFoundError:return dict(expected=identity,exists=False,identity_matches=False,live=False)
def progress(path):
 if not path.is_file():return None
 with path.open('rb') as stream:
  stream.seek(max(0,path.stat().st_size-80000));lines=stream.read().decode(errors='replace').splitlines()
 for line in reversed(lines):
  try:row=json.loads(line)
  except ValueError:continue
  if isinstance(row,dict) and row.get('event')=='READOUT_DIAGNOSTIC':
   return {k:row[k] for k in ('event','completed','planned','sample_token','seconds')}
 return None
state=read(job/'state.json');launch=read(job/'launch.json');cs=read(campaign/'state.json')
expected=[]
if launch:expected.append(dict(pid=launch['runner_pid'],uid=launch['uid'],startticks=launch['startticks']))
if state and state.get('child'):expected.append(state['child'])
if cs and cs.get('status') in ('RUNNING_PILOT','RUNNING_DEVELOPMENT'):
 expected.append({k:cs[k] for k in ('pid','uid','startticks')})
gpu={}
if state:
 env=read(job/'spec.json')['env']
 for name,query in [('devices','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total'),('processes','--query-compute-apps=gpu_uuid,pid,used_gpu_memory')]:
  q=subprocess.run(['nvidia-smi',query,'--format=csv,noheader'],env=env,text=True,capture_output=True,timeout=15,check=True);gpu[name]=q.stdout
value=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=os.uname().nodename,
 job='readout_transition_v2b',runner_state=state,campaign_state=cs,processes=[process(x) for x in expected],gpu=gpu,
 stages={mode:dict(progress=progress(campaign/(mode+'.log')),complete=read(campaign/mode/'complete.json'),failed=read(campaign/mode/'failed.json')) for mode in ('pilot','development')},
 complete=read(campaign/'complete.json'),failed=read(campaign/'failed.json'),
 dependency_complete=read(campaign/'dependency_complete.json'),performance_read=False)
print(json.dumps(value))
'''


if __name__ == '__main__':
    result = subprocess.run(['ssh','-F',str(BASE/'ssh_config'),'wangning@10.254.30.73','python3 -'],
                            input=REMOTE,text=True,capture_output=True,timeout=60,check=True)
    value=json.loads(result.stdout)
    output=BASE/'receipts'/('readout_transition_v2b_monitor_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    output.write_text(json.dumps(value,indent=2)+'\n')
    value['receipt']=str(output)
    print(json.dumps(value,indent=2))
