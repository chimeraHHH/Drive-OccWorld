"""Read a named durable job, preserve process identity and optional final files.

This is an observer only. It never launches, stops, or retries a server job.
"""
import argparse
import base64
import datetime
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--output', help='Relative output path below this research root')
    parser.add_argument('--full', action='store_true', help='Fetch final JSON/JSONL outputs')
    args = parser.parse_args()
    assert args.name.replace('_', '').isalnum()
    output = args.output or args.name
    assert not Path(output).is_absolute() and '..' not in Path(output).parts
    task = dict(name=args.name, output=output, full=args.full)
    remote = r'''
from pathlib import Path
import base64, datetime, hashlib, json
t=json.loads(base64.b64decode('__TASK__'))
root=Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/o_motion_20260915')
j=root/'jobs'/t['name']; o=root/t['output']
launch=json.loads((j/'launch.json').read_text())
state=json.loads((j/'state.json').read_text())
live={}
for role,identity in [('runner',dict(pid=launch['runner_pid'],uid=launch['uid'],startticks=launch['startticks'])),('child',state.get('child',{}))]:
 if not identity: live[role]={'identity_unavailable':True}; continue
 try:
  p=Path('/proc')/str(identity['pid']); stat=(p/'stat').read_text(); f=stat[stat.rfind(')')+2:].split()
  actual=dict(pid=identity['pid'],uid=p.stat().st_uid,startticks=int(f[19]),state=f[0],cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
  actual['identity_matches']=all(actual[k]==identity[k] for k in ('pid','uid','startticks'))
  live[role]=actual
 except FileNotFoundError:live[role]=dict(pid=identity['pid'],missing=True)
files={}
paths=[('job/'+f,j/f) for f in ('launch.json','state.json','output.log','runner.log')]
if o.is_dir():
 names=('manifest.json','summary.json','complete.json','failed.json','progress.json')
 paths += [('output/'+f,o/f) for f in names]
 if t['full']:
  assert state['status']!='RUNNING', 'Full artifact capture requires a terminal runner'
  paths += [('output/'+p.name,p) for p in o.iterdir() if p.suffix in ('.json','.jsonl','.md') and p.name not in names]
  if (o/'runs').is_dir():
   paths += [('output/'+str(p.relative_to(o)),p) for p in (o/'runs').rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.md')]
for key,p in paths:
 if p.is_file():
  data=p.read_bytes();files[key]=dict(data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest())
latest=[]
logs=[('',o/'training.jsonl')]
if (o/'runs').is_dir():logs += [(p.name,p/'training.jsonl') for p in (o/'runs').iterdir() if p.is_dir()]
for arm,log in logs:
 if not log.is_file():continue
 for line in log.read_text().splitlines()[-6:]:
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  latest.append(dict(arm=arm,**{k:r[k] for k in ('update','examples','loss','lr','preclip_grad_norm','seconds','shared_update_wall_seconds','resources','peak_allocated_bytes') if k in r}))
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),name=t['name'],state=state,live=live,latest_updates=latest,files=files)))
'''.replace('__TASK__', base64.b64encode(json.dumps(task).encode()).decode())
    base = Path(__file__).resolve().parent
    cmd = ['ssh', '-F', str(base.with_name('m0_improvement_20260915') / 'ssh_config'),
           '10.254.30.73', 'python3 -']
    result = subprocess.run(cmd, input=remote, text=True, capture_output=True, timeout=60)
    result.check_returncode()
    observation = json.loads(result.stdout)
    import hashlib
    for key, item in observation.pop('files').items():
        prefix, name = key.split('/', 1)
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        directory = base / 'server_results' / ('jobs' if prefix == 'job' else 'training') / args.name
        directory.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(item['data'])
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (base / 'receipts' / ('inspect_' + args.name + '_' + stamp + '.json')).write_text(
        json.dumps(observation, indent=2) + '\n')
    print(json.dumps(observation, indent=2))
    out = base / 'server_results/training' / args.name
    for name in ('summary.json', 'failed.json'):
        if (out / name).exists():
            value = json.loads((out / name).read_text())
            if name == 'summary.json':
                value = {k:v for k,v in value.items() if k != 'development'}
                if isinstance(value.get('scores'), dict):
                    value['scores'] = {arm: ({k:v for k,v in score.items()
                        if k not in ('horizons', 'transitions')} if isinstance(score, dict) else score)
                        for arm, score in value['scores'].items()}
            print(name, json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
