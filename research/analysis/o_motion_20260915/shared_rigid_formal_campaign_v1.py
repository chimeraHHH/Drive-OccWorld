"""Server-resident train -> fixed dev200 -> frozen analysis, one attempt only.

The external durable job_runner owns the dedicated process group and the hard
whole-campaign timeout. Child stages inherit that group. Failure never selects
another checkpoint, repeats a stage, or launches a replacement process.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    with temporary.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    parser.add_argument('--plan-sha256',required=True);a=parser.parse_args();started=time.monotonic()
    require(sha(a.plan)==a.plan_sha256,'Frozen campaign plan changed');p=read(a.plan)
    require(p['schema']=='shared-rigid-formal-campaign-v1' and p['status']=='FROZEN' and
            p['seed']==11 and p['training_updates_per_arm']==512 and p['evaluation_samples']==200 and
            p['no_retry'] is True and p['no_resume'] is True,'Wrong campaign plan')
    require(p['source_sha256']==sha(__file__) and p['outer_seconds']==6900,'Campaign source/resource changed')
    out=Path(p['out']);out.mkdir(parents=True,exist_ok=False);receipts={}
    def check_files():
        for name,digest in p['package_sources_sha256'].items():
            require(Path(name).name==name and sha(Path(__file__).with_name(name))==digest,'Package changed: '+name)
    def execute(stage,additional=()):
        check_files();item=p['stages'][stage];command=[*item['command'],*additional]
        require(command[:2]==[sys.executable,'-B'],'Unexpected Python command')
        if stage!='analysis':require(not Path(item['output']).exists(),'Stage output exists; never overwrite/retry')
        else:require(not Path(item['output']).exists(),'Analysis output exists')
        tick=time.monotonic();child=subprocess.Popen(command,stdin=subprocess.DEVNULL)
        proc=Path('/proc')/str(child.pid);fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        identity=dict(pid=child.pid,uid=proc.stat().st_uid,startticks=int(fields[19]),pgid=os.getpgid(child.pid))
        require(identity['uid']==1009 and identity['pgid']==os.getpgrp(),'Child left owned process group')
        receipt=dict(stage=stage,command=command,identity=identity,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out/'progress.json',dict(phase=stage,status='RUNNING',child=identity,seconds=time.monotonic()-started))
        write(out/(stage+'_launch.json'),receipt)
        # On timeout/failure, controller exits; external job_runner cleans its
        # identity-checked whole child group, including any remaining worker.
        code=child.wait(timeout=item['outer_seconds'])
        receipt.update(returncode=code,seconds=time.monotonic()-tick)
        write(out/(stage+'_exit.json'),receipt);require(code==0,'Stage failed; no follow-up or retry: '+stage)
        receipts[stage]=receipt
    try:
        check_files();write(out/'plan.json',p)
        execute('training')
        train=Path(p['stages']['training']['output']);done=read(train/'complete.json')
        require(done['status']=='TRAINING_COMPLETE_PENDING_FIXED_DEV200' and done['updates']==512 and
                done['examples']==2048 and done['evaluated_samples']==0,'Fixed training endpoint incomplete')
        train_sha=sha(train/'complete.json')
        execute('evaluation',('--training-complete-sha256',train_sha))
        evaluation=Path(p['stages']['evaluation']['output']);done=read(evaluation/'complete.json')
        require(done['status']=='COMPLETE_SHARED_RIGID_FIXED_DEV200' and done['evaluated_samples']==200 and
                done['fixed_training_complete_sha256']==train_sha,'Final evaluation incomplete')
        eval_sha=sha(evaluation/'complete.json')
        execute('analysis',('--evaluation-complete-sha256',eval_sha))
        analysis=Path(p['stages']['analysis']['output']);summary=read(analysis)
        require(summary['schema']=='shared-rigid-joint-summary-v1' and summary['samples']==200 and summary['scenes']==100,
                'Fixed summary incomplete')
        check_files();write(out/'complete.json',dict(schema=p['schema'],status='COMPLETE_FIXED_CAMPAIGN_NOT_GOAL_CLAIM',
            plan_sha256=a.plan_sha256,training_complete_sha256=train_sha,evaluation_complete_sha256=eval_sha,
            analysis_sha256=sha(analysis),stages=receipts,seconds=time.monotonic()-started,goal_achieved_not_asserted=True))
        write(out/'progress.json',dict(phase='complete',status='COMPLETE',seconds=time.monotonic()-started))
    except BaseException as exc:
        write(out/'failed.json',dict(status='FAILED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),
            completed_stages=receipts,seconds=time.monotonic()-started));raise


if __name__=='__main__':main()
