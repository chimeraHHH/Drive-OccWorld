"""One read-only server snapshot; persist evidence and print compact progress."""
import datetime
import json
from pathlib import Path
import subprocess

B = Path(__file__).resolve().parent
REMOTE = r'''
import datetime,json,os,pathlib,subprocess
root=pathlib.Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915')
campaign=root/'campaign_memory_v1'
cache=pathlib.Path('/home/wangning/data_cache/RadarFlowOcc_m0_improvement_20260915/campaign_cache_v1')
def read(p):return json.loads(p.read_text()) if p.is_file() else None
d=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=os.uname().nodename,uid=os.getuid(),jobs={},cache={},cards={},arms={})
for name in ['native_train_cache_v1','memory_campaign_control_v1','memory_campaign_anchor_v1','memory_campaign_aggregate_v1']:
 state=read(root/'jobs'/name/'state.json')
 d['jobs'][name]=state
 if state and state['status'] in ['FAILED','TIMEOUT','USER_SIGNAL']:
  log=root/'jobs'/name/'output.log'
  d['jobs'][name]['error_tail']=log.read_text(errors='replace').splitlines()[-12:] if log.exists() else []
for split in ['train','development']:
 d['cache'][split]=dict(progress=read(cache/split/'progress.json'),complete=read(cache/split/'complete.json'),failure=read(cache/split/'failed.json'))
for role in ['anchor','control']:
 state=read(campaign/'cards'/role/'state.json')
 d['cards'][role]=state
 if state:
  log=campaign/'cards'/role/'stages'/str(state.get('stage'))/'stdout.log'
  d['cards'][role]['log_tail']=log.read_text(errors='replace').splitlines()[-3:] if log.is_file() else []
for arm in ['native1','persistent2','rolling2']:
 run=campaign/'runs'/arm
 a=dict(progress=read(run/'progress.json'),trained=read(run/'training_complete.json'),complete=read(run/'complete.json'))
 log=run/'training.jsonl'
 if log.is_file():
  rows=[json.loads(x) for x in log.read_text().splitlines(keepends=True) if x.endswith('\n') and x.strip()]
  recent=rows[-20:]
  if recent:
   losses=[x['loss_mean'] for x in recent]
   a['recent_updates']=dict(count=len(recent),mean_loss=sum(losses)/len(losses),min_loss=min(losses),max_loss=max(losses))
  if len(recent)>1:
   seconds_per_update=(recent[-1]['seconds']-recent[0]['seconds'])/(recent[-1]['update']-recent[0]['update'])
   a['recent_updates']['seconds_per_update']=seconds_per_update
   a['recent_updates']['training_only_eta_seconds']=max(0,512-recent[-1]['update'])*seconds_per_update
 d['arms'][arm]=a
d['aggregation_wait']=read(campaign/'aggregation_wait_v1'/'state.json')
d['summary_complete']=read(campaign/'summary_v1'/'complete.json')
spec=read(root/'jobs/memory_campaign_control_v1/spec.json')
for key,args in [('gpus',['--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader']),('compute',['--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'])]:
 q=subprocess.run(['nvidia-smi']+args,env=spec['env'],text=True,capture_output=True,timeout=15,check=True)
 d[key]=q.stdout
print(json.dumps(d))
'''

if __name__ == '__main__':
    result = subprocess.run(['ssh','-F',str(B/'ssh_config'),'wangning@10.254.30.73','python3 -'],
                            input=REMOTE,text=True,capture_output=True,timeout=60,check=True)
    value=json.loads(result.stdout)
    path=B/'receipts'/('campaign_monitor_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
    path.write_text(json.dumps(value,indent=2)+'\n')
    compact={k:value[k] for k in ['checked_utc','host','uid','cache','arms','gpus','compute','summary_complete']}
    compact['jobs']={k:v['status'] if v else None for k,v in value['jobs'].items()}
    compact['cards']={k:{key:v.get(key) for key in ['status','stage','error','log_tail']} if v else None for k,v in value['cards'].items()}
    compact['receipt']=str(path)
    print(json.dumps(compact,indent=2))
