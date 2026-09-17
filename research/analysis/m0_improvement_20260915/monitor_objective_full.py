"""Read-only progress/process snapshot. Never read partial model performance."""
import datetime
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
REMOTE = r'''
import datetime,json,os,subprocess
from pathlib import Path
root=Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915')
job=root/'jobs/objective_joint_full_v2'; campaign=root/'campaign_objective_joint_full_v2'
def read(path):return json.loads(path.read_text()) if path.is_file() else None
def process(identity):
 p=Path('/proc')/str(identity['pid'])
 try:
  stat=(p/'stat').read_text();f=stat[stat.rfind(')')+2:].split()
  actual=dict(pid=identity['pid'],uid=p.stat().st_uid,startticks=int(f[19]),state=f[0],pgid=int(f[2]))
  return dict(expected=identity,actual=actual,identity_matches=all(actual[k]==identity[k] for k in ('pid','uid','startticks')))
 except FileNotFoundError:return dict(expected=identity,exists=False,identity_matches=False)
state=read(job/'state.json'); launch=read(job/'launch.json'); cs=read(campaign/'state.json')
expected=[dict(pid=launch['runner_pid'],uid=launch['uid'],startticks=launch['startticks'])]
if state and state.get('child'):expected.append(state['child'])
shards={}
for i in (0,1):
 p=campaign/'evaluation'/('shard_%02d_of_02'%i)
 item=cs.get('shards',{}).get(str(i)) if cs else None
 if item:expected.append({k:item[k] for k in ('pid','uid','startticks')})
 shards[str(i)]=dict(progress=read(p/'progress.json'),complete=read(p/'complete.json'),failed=read(p/'failed.json'))
env=read(job/'spec.json')['env']; gpu={}
for name,query in [('devices','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total'),('processes','--query-compute-apps=gpu_uuid,pid,used_gpu_memory')]:
 q=subprocess.run(['nvidia-smi',query,'--format=csv,noheader'],env=env,text=True,capture_output=True,timeout=15,check=True)
 gpu[name]=q.stdout
value=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=os.uname().nodename,
 job='objective_joint_full_v2',runner_status=state.get('status') if state else None,
 campaign_status=cs.get('status') if cs else None,processes=[process(p) for p in expected],
 gpu=gpu,shards=shards,complete=read(campaign/'complete.json'),failed=read(campaign/'failed.json'),
 merged_complete=read(campaign/'summary_v1/complete.json'),partial_performance_read=False)
print(json.dumps(value))
'''


if __name__ == '__main__':
    result = subprocess.run(['ssh', '-F', str(BASE/'ssh_config'), 'wangning@10.254.30.73', 'python3 -'],
                            input=REMOTE, text=True, capture_output=True, timeout=60, check=True)
    value = json.loads(result.stdout)
    previous = sorted((BASE/'receipts').glob('objective_full_monitor_*.json'))
    if previous:
        old = json.loads(previous[-1].read_text())
        delta = (datetime.datetime.fromisoformat(value['checked_utc']) -
                 datetime.datetime.fromisoformat(old['checked_utc'])).total_seconds()
        rates = {}
        for index in ('0', '1'):
            a, b = old['shards'][index]['progress'], value['shards'][index]['progress']
            if a and b and delta > 0 and b['completed'] > a['completed']:
                per_sample = delta/(b['completed']-a['completed'])
                rates[index] = dict(interval_seconds=delta, completed_in_interval=b['completed']-a['completed'],
                    seconds_per_sample=per_sample, remaining_seconds_linear=(b['planned']-b['completed'])*per_sample)
        value['rate_estimates'] = rates
        value['rate_note'] = 'Recent progress interval only; assumes unchanged throughput and excludes final merge. Not a guarantee.'
    output = BASE/'receipts'/('objective_full_monitor_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    output.write_text(json.dumps(value, indent=2)+'\n')
    value['receipt'] = str(output)
    print(json.dumps(value, indent=2))
