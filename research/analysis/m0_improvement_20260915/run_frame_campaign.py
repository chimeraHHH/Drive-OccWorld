"""Run one fixed reference-frame candidate, then its CPU paired summary on server."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['root','frame-protocol','protocol','repo','config','checkpoint','train-cache','dev-cache','control-run']:
        p.add_argument('--'+field,required=True)
    a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=False)
    pkg=Path(__file__).resolve().parent;began=time.monotonic()
    def record(name,data):
        temp=root/(name+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(root/name)
    frozen=json.loads(Path(a.frame_protocol).read_text())
    cap=frozen['resource_policy']['max_seconds']
    manifest=dict(schema='single-reference-frame-server-campaign-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        frame_protocol_sha256=sha(a.frame_protocol),parent_protocol_sha256=sha(a.protocol),
        wrapper_sha256=sha(__file__),candidate='reference',new_control_training=False)
    record('manifest.json',manifest)
    run=root/'runs'/'reference'
    train=[sys.executable,'-B',str(pkg/'frame_train.py'),'--frame-protocol',a.frame_protocol,'--protocol',a.protocol,
        '--repo',a.repo,'--config',a.config,'--checkpoint',a.checkpoint,'--train-cache',a.train_cache,
        '--dev-cache',a.dev_cache,'--control-run',a.control_run,'--out',str(run),'--max-seconds',str(cap)]
    summary=[sys.executable,'-B',str(pkg/'frame_aggregate.py'),'--cache',a.dev_cache,'--reference-run',str(run),
        '--native-runs-root',str(Path(a.control_run).parent),'--protocol',a.frame_protocol,
        '--parent-protocol',a.protocol,'--out',str(root/'summary_v1')]
    try:
        for stage,command in [('train_reference',train),('paired_development_summary',summary)]:
            record('state.json',dict(status='RUNNING_STAGE',stage=stage,command=command))
            subprocess.run(command,check=True)
        record('complete.json',dict(status='COMPLETE_REFERENCE_CAMPAIGN',seconds=time.monotonic()-began,
            frame_protocol_sha256=sha(a.frame_protocol),reference_complete_sha256=sha(run/'complete.json'),
            summary_complete_sha256=sha(root/'summary_v1'/'complete.json'),
            full_validation=False,automatic_expansion=False))
        record('state.json',dict(status='COMPLETE',stage='paired_development_summary'))
    except BaseException as exc:
        record('failed.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-began));raise


if __name__=='__main__':main()
