"""Read-only M0 improvement status and small result artifact mirror."""
import argparse,base64,datetime,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fetch',action='append',default=[]);p.add_argument('--job',action='append',default=[]);a=p.parse_args()
extra=base64.b64encode(json.dumps(a.fetch).encode()).decode()
remote=r"""
import base64,datetime,hashlib,json,pathlib
root=pathlib.Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915')
out=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs={},artifacts={})
for p in sorted((root/'jobs').glob('*')):
 if not p.is_dir():continue
 state=json.loads((p/'state.json').read_text()) if (p/'state.json').exists() else {}
 log=p/'output.log';state['log_tail']=log.read_text(errors='replace').splitlines()[-12:] if log.exists() else []
 rlog=p/'runner.log';state['runner_tail']=rlog.read_text(errors='replace').splitlines()[-5:] if rlog.exists() else []
 out['jobs'][p.name]=state
for name in json.loads(base64.b64decode('__EXTRA__')):
 assert '..' not in pathlib.Path(name).parts and not pathlib.Path(name).is_absolute()
 p=(pathlib.Path('/home/wangning/data_cache/RadarFlowOcc_m0_improvement_20260915')/name[len('cache/'):]) if name.startswith('cache/') else root/name
 if not p.is_file():out['artifacts'][name]={'missing':True};continue
 assert p.stat().st_size<10000000
 b=p.read_bytes();out['artifacts'][name]=dict(data=base64.b64encode(b).decode(),sha256=hashlib.sha256(b).hexdigest())
print(json.dumps(out))
""".replace('__EXTRA__',extra)
r=subprocess.run(['ssh','-F',str(Path(__file__).with_name('ssh_config')),'wangning@10.254.30.73','python3 -'],input=remote,text=True,capture_output=True,timeout=65);r.check_returncode();d=json.loads(r.stdout)
base=Path(__file__).resolve().parent;(base/'receipts').mkdir(exist_ok=True)
for name,v in d['artifacts'].items():
 if v.get('missing'):continue
 b=base64.b64decode(v.pop('data'));assert hashlib.sha256(b).hexdigest()==v['sha256']
 target=base/'server_results'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');(base/'receipts'/(stamp+'.json')).write_text(json.dumps(d,indent=2)+'\n')
compact=dict(checked_utc=d['checked_utc'],jobs={},artifacts=d['artifacts'])
for name,state in d['jobs'].items():
 if a.job and name not in a.job:continue
 compact['jobs'][name]={k:state.get(k) for k in ['status','seconds','returncode','child']}
 if state.get('status')!='EXITED_ZERO':
  compact['jobs'][name]['log_tail']=[line[:700] for line in state.get('log_tail',[])[-4:]]
print(json.dumps(compact,ensure_ascii=False,indent=2))
