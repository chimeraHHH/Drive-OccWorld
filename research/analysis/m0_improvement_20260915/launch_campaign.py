"""Stage two server-resident cards once under one frozen protocol and deadline."""
import datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent
p=b/'protocol_v2.json'
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
protocol=json.loads(p.read_text())
for name,digest in protocol['source_sha256'].items():assert sha(b/name)==digest
root='/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915'
campaign=root+'/campaign_memory_v1'
cache='/home/wangning/data_cache/RadarFlowOcc_m0_improvement_20260915/campaign_cache_v1'
config='/storage/data/metaiot_data/wangning/RadarFlowOcc_sota_p2_20260911/runtime/configs/S0.py'
checkpoint='/storage/data/metaiot_data/wangning/RadarFlowOcc_sota_p2_20260911/assets/m0_epoch24.pth'
deadline=int(time.time())+10800
plan=dict(campaign=campaign,cache_root=cache,protocol_sha256=sha(p),deadline_unix=deadline,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=[])
for role,gpu in [('control','1'),('anchor','')]:
    name='memory_campaign_'+role+'_v1'
    package=root+'/jobs/'+name+'/package'
    args=['--root',campaign,'--cache-root',cache,'--role',role,'--config',config,'--checkpoint',checkpoint,'--protocol',package+'/'+p.name,'--protocol-sha256',sha(p),'--deadline-unix',str(deadline),'--extract-max-seconds','7800','--extract-max-bytes','150000000000']
    script='run_campaign_card.py'
    if role=='anchor':
        args+=['--reuse-complete-caches']
        args=['--producer-job',root+'/jobs/native_train_cache_v1','--gpu','0','--deadline-unix',str(deadline),'--']+args
        script='wait_for_native_cache.py'
    argv=[sys.executable,str(b/'dispatch.py'),'--name',name,'--gpu',gpu,'--seconds','10830','--script',script]
    for filename in list(protocol['source_sha256'])+[p.name,'selection_v1.json']:
        if filename not in [script,'job_runner.py']:argv+=['--include',filename]
    argv+=['--']+args
    plan['jobs'].append(dict(name=name,role=role,dispatch_argv=argv))
receipt=b/'receipts/campaign_launch_plan_v1.json'
with receipt.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
for job in plan['jobs']:
    subprocess.run(job['dispatch_argv'],check=True)
print(json.dumps(dict(status='BOTH_SERVER_CARDS_DISPATCHED',receipt=str(receipt),deadline_unix=deadline)))
