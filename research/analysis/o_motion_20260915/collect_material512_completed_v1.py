"""Retrieve completed small text artifacts; never copy model weights or arrays."""
import argparse,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path

n=Path(__file__).resolve().parent;workspace=n.parent.parent
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('T_train','T_evaluate','J_train','J_evaluate','D_train','D_evaluate','fit_preflight','fit_full'));a=ap.parse_args()
if a.stage.startswith('fit_'):
 relative='material512_train_fit_'+a.stage[4:]+'_v1';dest=n/'server_results/diagnostics'/relative
else:
 relative='dense_material512_v1/'+a.stage;dest=n/'server_results/training'/relative
remote=r'''
import hashlib,io,json,sys,tarfile
from pathlib import Path
n=Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/o_motion_20260915')
relative=sys.argv[1];root=n/relative
assert n in root.parents
p=root/'complete.json';done=json.loads(p.read_text());assert done['status'].startswith('COMPLETE_')
data={'complete.json':p.read_bytes()}
for name,digest in done['files_sha256'].items():
 if not name.endswith(('.json','.jsonl')):continue
 assert Path(name).name==name
 b=(root/name).read_bytes();assert len(b)<60000000 and hashlib.sha256(b).hexdigest()==digest
 data[name]=b
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
 for name,b in data.items():
  item=tarfile.TarInfo(name);item.size=len(b);tar.addfile(item,io.BytesIO(b))
sys.stdout.buffer.write(buf.getvalue())
'''
r=subprocess.run(['ssh','-F',str(workspace/'analysis/m0_improvement_20260915/ssh_config'),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12','10.254.30.73',shlex.join(['python3','-c',remote,relative])],capture_output=True,timeout=60)
assert r.returncode==0,r.stderr.decode()
assert len(r.stdout)<50000000
with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as tar:
 members=tar.getmembers();assert len(members)<20 and sum(m.size for m in members)<120000000
 data={}
 for m in members:
  assert m.isfile() and Path(m.name).name==m.name and m.name.endswith(('.json','.jsonl'))
  assert m.name not in data;data[m.name]=tar.extractfile(m).read()
done=json.loads(data['complete.json']);expected={k:v for k,v in done['files_sha256'].items() if k.endswith(('.json','.jsonl'))}
assert set(data)==set(expected)|{'complete.json'}
for name,digest in expected.items():assert hashlib.sha256(data[name]).hexdigest()==digest
if a.stage.startswith('fit_'):
 assert done['protocol_sha256']==hashlib.sha256((n/'material512_train_fit_protocol_v1.json').read_bytes()).hexdigest()
else:
 p=n/('dense_material512_'+('training' if a.stage.endswith('_train') else 'evaluation')+'_protocol_v1.json')
 assert done['protocol_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
dest.mkdir(parents=True,exist_ok=True)
for name,b in data.items():
 p=dest/name
 if p.exists():assert p.read_bytes()==b,'Refuse changed collected artifact'
 else:p.write_bytes(b)
print(json.dumps(dict(stage=a.stage,directory=str(dest),files=len(data),complete_sha256=hashlib.sha256(data['complete.json']).hexdigest(),status=done['status'])))
