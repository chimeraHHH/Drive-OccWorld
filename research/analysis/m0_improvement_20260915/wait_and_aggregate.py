"""CPU-only, one-shot waiter for the already launched memory campaign v1.

No deployment, CUDA imports, training, protocol edits, scientific retry, or
process termination. Wait for BOTH card receipts AND their job runners to
report EXITED_ZERO. Then execute the protocol-bound aggregate_memory.py in
this job's inherited environment/PGID, under the same global deadline.
The aggregator alone creates ROOT/summary_v1; existing output is never reused.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


PROTOCOL_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
DEADLINE = 1789417106
OWNER_UID = 1009
JOB_NAMES = {r:'memory_campaign_' + r + '_v1' for r in ['control','anchor']}
TERMINAL_FAILURES = {'FAILED','TIMEOUT','USER_SIGNAL','STOPPING','INTERRUPTED','DEADLINE'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, exclusive=False):
    path=Path(path); tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    try:
        with tmp.open('x') as f:
            json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        if exclusive:
            os.link(str(tmp),str(path)); tmp.unlink()
        else:
            os.replace(str(tmp),str(path))
        fd=os.open(str(path.parent),os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if tmp.exists(): tmp.unlink()


def options(argv, flags=()):
    """Strict argv parsing; duplicate/positional/unknown options fail closed."""
    result={}; i=0
    values={'--root','--cache-root','--role','--config','--checkpoint','--protocol',
            '--protocol-sha256','--deadline-unix','--extract-max-seconds',
            '--extract-max-bytes','--producer-job','--gpu'}
    while i<len(argv):
        key=argv[i]
        require(key not in result and key in values.union(flags),'Invalid/duplicate command argument: '+key)
        if key in flags:
            result[key]=True; i+=1
        else:
            require(i+1<len(argv) and not argv[i+1].startswith('--'),'Missing command argument: '+key)
            result[key]=argv[i+1]; i+=2
    return result


def card_command(command, role, job, deadline):
    require(isinstance(command,list) and all(isinstance(x,str) for x in command),'Structured command required')
    require(len(command)>3 and command[1]=='-B','Expected Python -B command')
    require(Path(command[0]).resolve()==Path(sys.executable).resolve(),'Different Python interpreter')
    package=job/'package'
    if role=='control':
        require(Path(command[2]).resolve()==package/'run_campaign_card.py','Wrong control entry point')
        args=command[3:]
    else:
        require(Path(command[2]).resolve()==package/'wait_for_native_cache.py','Wrong anchor entry point')
        require(command[3:].count('--')==1,'Anchor separator missing/duplicated')
        boundary=command.index('--',3)
        wrapper=options(command[3:boundary])
        require(set(wrapper)=={'--producer-job','--gpu','--deadline-unix'},'Unexpected wrapper options')
        require(Path(wrapper['--producer-job']).resolve()==job.parent/'native_train_cache_v1','Wrong native cache producer')
        require(wrapper['--gpu']=='0' and float(wrapper['--deadline-unix'])==deadline,'Anchor GPU/deadline mismatch')
        args=command[boundary+1:]
    parsed=options(args,flags={'--reuse-complete-caches'})
    expected={'--root','--cache-root','--role','--config','--checkpoint','--protocol',
              '--protocol-sha256','--deadline-unix','--extract-max-seconds','--extract-max-bytes'}
    if role=='anchor': expected.add('--reuse-complete-caches')
    require(set(parsed)==expected and parsed['--role']==role,'Card CLI role/options mismatch')
    return parsed


class Waiter:
    def __init__(self, args):
        self.root=Path(args.root).resolve(); self.cache=Path(args.cache_root).resolve()
        self.protocol_path=Path(args.protocol).resolve(); self.package=Path(__file__).resolve().parent
        self.deadline=args.deadline_unix
        require(os.getuid()==OWNER_UID,'Expected campaign owner UID 1009')
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Waiter must be dispatched with no visible GPU')
        require(self.root.name=='campaign_memory_v1' and self.root.is_dir(),'Wrong/missing campaign root')
        require(self.deadline==DEADLINE and math.isfinite(self.deadline),'Frozen global deadline mismatch')
        self.start=time.monotonic(); self.end=self.start+max(0,self.deadline-time.time())
        self.directory=self.root/'aggregation_wait_v1'
        self.directory.mkdir(exist_ok=False)
        self.lock=(self.directory/'owner.lock').open('x')
        fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.summary=self.root/'summary_v1'
        self.state=dict(schema='m0-memory-wait-and-aggregate-v1',status='VALIDATING',
            root=str(self.root),cache_root=str(self.cache),protocol=str(self.protocol_path),
            protocol_sha256=PROTOCOL_SHA,deadline_unix=self.deadline,pid=os.getpid(),
            pgid=os.getpgid(0),uid=os.getuid(),started_utc=utc(),automatic_retry=False)
        self.update()
        self.jobs={}

    def update(self, **fields):
        self.state.update(fields); self.state.update(updated_utc=utc(),seconds=time.monotonic()-self.start)
        write(self.directory/'state.json',self.state)

    def remaining(self):
        return min(self.deadline-time.time(),self.end-time.monotonic())

    def stop_check(self):
        require(self.remaining()>0,'GLOBAL_DEADLINE: incomplete campaign/aggregation is not retried')

    def contract(self):
        self.stop_check()
        require(sha(self.protocol_path)==PROTOCOL_SHA,'Frozen protocol mismatch')
        protocol=read(self.protocol_path)
        require(protocol['status']=='FROZEN_BEFORE_CANDIDATE_TRAINING','Protocol not frozen')
        require(sha(self.protocol_path.with_name('selection_v1.json'))==protocol['selection_sha256'],'Selection hash mismatch')
        for name,digest in protocol['source_sha256'].items():
            require(Path(name).name==name,'Source path must be basename')
            require(sha(self.package/name)==digest,'Waiter package frozen source mismatch: '+name)
        require('aggregate_memory.py' in protocol['source_sha256'],'Unbound aggregator')
        for path,digest in protocol['runtime_source_sha256'].items():
            require(Path(path).is_absolute() and sha(path)==digest,'Runtime source changed: '+path)
        return protocol

    def launch(self, role, protocol):
        job=self.root.parent/'jobs'/JOB_NAMES[role]
        for path in [job,job/'launch.json',job/'spec.json',job/'package']:
            require(path.stat().st_uid==OWNER_UID,'Foreign job/receipt owner')
        launch,spec=read(job/'launch.json'),read(job/'spec.json')
        require(launch['name']==JOB_NAMES[role] and Path(launch['job']).resolve()==job,'Wrong job identity')
        require(launch['uid']==OWNER_UID and launch['runner_pid']>0 and launch['startticks']>0,'Invalid runner identity')
        require(launch['command']==spec['command'],'Launch/spec command mismatch')
        command=card_command(launch['command'],role,job,self.deadline)
        require(Path(command['--root']).resolve()==self.root and Path(command['--cache-root']).resolve()==self.cache,'Root/cache CLI mismatch')
        require(float(command['--deadline-unix'])==self.deadline and command['--protocol-sha256']==PROTOCOL_SHA,'CLI deadline/protocol mismatch')
        require(Path(command['--protocol']).resolve()==job/'package'/'protocol_v2.json','Wrong card protocol path')
        require(sha(command['--protocol'])==PROTOCOL_SHA,'Card protocol bytes differ')
        require(sha(command['--config'])==protocol['config_sha256'],'Card config changed')
        require(int(command['--extract-max-seconds'])==protocol['resource_policy']['extraction_timeout_seconds'],'Extraction time cap mismatch')
        require(int(command['--extract-max-bytes'])==protocol['resource_policy']['max_cache_bytes_per_split'],'Extraction storage cap mismatch')
        expected=dict(protocol['source_sha256'],**{'protocol_v2.json':PROTOCOL_SHA,'selection_v1.json':protocol['selection_sha256']})
        require(launch['script_sha256']==expected,'Job package manifest differs from frozen sources')
        for name,digest in expected.items():
            require(sha(job/'package'/name)==digest,'Launched source bytes changed: '+name)
        require(spec['env']['CUDA_VISIBLE_DEVICES']==('1' if role=='control' else ''),'Initial role GPU binding changed')
        require(spec['env']['PYTHONDONTWRITEBYTECODE']=='1','Frozen sources require no pycache writes')
        require(spec['seconds']==10830,'Runner bound differs from launch')
        require(Path(spec['cwd']).resolve()==Path(spec['env']['M0_SOURCE_FILE']).resolve().parents[4],'Wrong native working directory')
        return dict(job=job,launch=launch,spec=spec,command=command,
                    launch_sha256=sha(job/'launch.json'),spec_sha256=sha(job/'spec.json'))

    def card_status(self, role, protocol):
        info=self.jobs[role]; job=info['job']; launch=info['launch']
        require(sha(job/'launch.json')==info['launch_sha256'] and sha(job/'spec.json')==info['spec_sha256'],'Job launch/spec mutated')
        path=job/'state.json'
        if not path.is_file(): return dict(role=role,status='WAITING_JOB_STATE',ready=False)
        require(path.stat().st_uid==OWNER_UID,'Foreign runner state')
        state=read(path)
        require(state['runner_pid']==launch['runner_pid'] and state['command']==launch['command'],'Runner state identity mismatch')
        status=state['status']
        require(status not in TERMINAL_FAILURES,'Job terminal failure: '+role+': '+status)
        require(status in ('RUNNING','EXITED_ZERO'),'Unexpected runner state: '+status)
        child=state.get('child')
        if child:
            require(child['uid']==OWNER_UID and child['pid']>0 and child['startticks']>0,'Wrong child identity')
        card=self.root/'cards'/role
        require(not (card/'failed.json').exists(),'Card failed: '+role)
        # Stable runner metadata binds finished processes; inspect /proc only
        # while RUNNING, avoiding false claims about a subsequently reused PID.
        proc=Path('/proc')/str(launch['runner_pid'])
        if status=='RUNNING' and proc.exists():
            try:
                s=(proc/'stat').read_text(); fields=s[s.rfind(')')+2:].split()
                require(proc.stat().st_uid==OWNER_UID and int(fields[19])==launch['startticks'],'Live runner identity changed')
                cmdline=(proc/'cmdline').read_bytes().split(b'\0')
                require(str(job/'package'/'job_runner.py').encode() in cmdline and str(job).encode() in cmdline,'Live runner command changed')
            except FileNotFoundError:
                pass  # runner may finish between state and /proc read
        if (card/'state.json').exists():
            require((card/'state.json').stat().st_uid==OWNER_UID,'Foreign card state')
            cs=read(card/'state.json')
            require(cs['status'] not in TERMINAL_FAILURES,'Card terminal failure: '+role)
            require(cs['role']==role and Path(cs['root']).resolve()==self.root and Path(cs['cache_root']).resolve()==self.cache,'Card identity/root/cache mismatch')
            require(cs['deadline_unix']==self.deadline and Path(cs['package']).resolve()==job/'package','Card deadline/package mismatch')
            require(child and cs['pgid']==child['pid'],'Card escaped owned runner child PGID')
            if role=='control': require(cs['pid']==child['pid'],'Direct control PID mismatch')
        if status!='EXITED_ZERO' or not (card/'complete.json').is_file():
            return dict(role=role,status=status,card_complete=(card/'complete.json').is_file(),ready=False)
        require(state['returncode']==0 and child,'Runner not successfully exited')
        complete,manifest=read(card/'complete.json'),read(card/'manifest.json')
        require(complete==read(card/'state.json') and complete['status']=='COMPLETE','Card complete/state differ')
        for key in ['role','pid','pgid','root','cache_root','package','deadline_unix']:
            require(complete[key]==manifest[key],'Card manifest identity mismatch: '+key)
        contract=manifest['contract']
        for key in ['config_sha256','selection_sha256','source_sha256','runtime_source_sha256']:
            require(contract[key]==protocol[key],'Card frozen contract mismatch: '+key)
        require(contract['protocol_sha256']==PROTOCOL_SHA and contract['deadline_unix']==self.deadline,'Card frozen protocol/deadline mismatch')
        require(contract['checkpoint_sha256']==protocol['m0_sha256'],'Card source checkpoint differs')
        stages=['train_persistent2'] if role=='anchor' else ['extract_development','train_native1','train_rolling2']
        require([x['stage'] for x in complete['completed_stages']]==stages,'Missing/unexpected completed stages')
        for row in complete['completed_stages']:
            stage=card/'stages'/row['stage']
            require(not (stage/'failed.json').exists() and sha(stage/'complete.json')==row['receipt_sha256'],'Stage completion hash/failure mismatch')
            receipt=read(stage/'complete.json'); stage_launch=read(stage/'launch.json')
            require(receipt['status']=='PASS' and receipt['returncode']==0 and receipt['automatic_retry'] is False,'Stage did not pass once')
            require(receipt['contract']==contract and stage_launch['contract']==contract,'Stage contract mismatch')
            require(receipt['argv']==stage_launch['argv'] and stage_launch['pgid']==complete['pgid'],'Stage launch binding mismatch')
            require(receipt['stdout_sha256']==sha(stage/'stdout.log'),'Stage log changed')
        for split in ['train','development']:
            meta=complete['caches'][split]
            require(Path(meta['directory']).resolve()==self.cache/split,'Card cache directory mismatch')
            require(meta['index_sha256']==sha(self.cache/split/'index.json') and
                    meta['complete_sha256']==sha(self.cache/split/'complete.json'),'Card cache metadata changed')
        return dict(role=role,status=status,ready=True,job_state_sha256=sha(path),
            launch_sha256=info['launch_sha256'],spec_sha256=info['spec_sha256'],
            card_complete_sha256=sha(card/'complete.json'),card_manifest_sha256=sha(card/'manifest.json'))

    def run(self):
        protocol=self.contract()
        require(not self.summary.exists(),'summary_v1 already exists; no overwrite/reuse')
        self.jobs={role:self.launch(role,protocol) for role in JOB_NAMES}
        require(self.jobs['control']['command']['--config']==self.jobs['anchor']['command']['--config'] and
                self.jobs['control']['command']['--checkpoint']==self.jobs['anchor']['command']['--checkpoint'],'Cards use different model paths')
        write(self.directory/'manifest.json',dict(self.state,script_sha256=sha(__file__),
            source_sha256=protocol['source_sha256'],jobs={r:{k:str(v) if isinstance(v,Path) else v for k,v in x.items() if k!='spec'} for r,x in self.jobs.items()}),exclusive=True)
        while True:
            self.stop_check()
            rows=[self.card_status(role,protocol) for role in JOB_NAMES]
            self.update(status='WAITING_FOR_BOTH_CARDS_AND_RUNNERS',dependencies=rows)
            if all(row['ready'] for row in rows): break
            time.sleep(min(30.,max(.001,self.remaining())))
        self.contract(); require(not self.summary.exists(),'Summary appeared while waiting')
        argv=[sys.executable,'-B',str(self.package/'aggregate_memory.py'),
              '--cache',str(self.cache/'development'),'--runs-root',str(self.root/'runs'),
              '--protocol',str(self.protocol_path),'--out',str(self.summary)]
        write(self.directory/'aggregate_launch.json',dict(argv=argv,pgid=os.getpgid(0),
            uid=os.getuid(),deadline_unix=self.deadline,dependencies=rows,shell=False,
            inherited_environment=True,new_session=False,source_sha256=sha(self.package/'aggregate_memory.py')),exclusive=True)
        self.update(status='AGGREGATING',argv=argv)
        with (self.directory/'aggregate_stdout.log').open('x') as log:
            try:
                result=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                      timeout=max(.001,self.remaining()),check=False)
            finally: log.flush(); os.fsync(log.fileno())
        self.stop_check(); require(result.returncode==0,'Aggregator failed; no retry')
        self.contract()
        receipt=read(self.summary/'complete.json')
        require(receipt['status']=='COMPLETE' and not (self.summary/'failed.json').exists(),'Summary failed/incomplete')
        require(receipt['summary_sha256']==sha(self.summary/'summary.json'),'Summary hash mismatch')
        for name,digest in receipt['files_sha256'].items():
            require(Path(name).name==name and sha(self.summary/name)==digest,'Summary artifact changed')
        summary=read(self.summary/'summary.json')
        require(summary['status']=='COMPLETED_DEVELOPMENT_COMPARISON' and summary['protocol_sha256']==PROTOCOL_SHA,'Summary protocol/state mismatch')
        require(summary['script_sha256']==protocol['source_sha256']['aggregate_memory.py'],'Summary source changed')
        self.update(status='COMPLETE',summary=str(self.summary),summary_complete_sha256=sha(self.summary/'complete.json'),
                    summary_sha256=receipt['summary_sha256'],aggregate_stdout_sha256=sha(self.directory/'aggregate_stdout.log'))
        write(self.directory/'complete.json',self.state,exclusive=True)


def self_test():
    assert options(['--role','anchor','--reuse-complete-caches'],{'--reuse-complete-caches'})=={'--role':'anchor','--reuse-complete-caches':True}
    for bad in [['--role','control','--role','anchor'],['--role'],['--shell','x'],['positional']]:
        try: options(bad)
        except RuntimeError: pass
        else: raise AssertionError('Accepted invalid argv')
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'receipt.json'; write(path,dict(status='PASS'),exclusive=True)
        before=sha(path)
        try: write(path,dict(status='OVERWRITE'),exclusive=True)
        except FileExistsError: pass
        else: raise AssertionError('Overwrote completion')
        assert sha(path)==before
        # Actual local subprocess validates PGID inheritance and structured argv.
        output=subprocess.check_output([sys.executable,'-c','import os; print(os.getpgid(0))'],text=True)
        assert int(output)==os.getpgid(0)
    print(json.dumps(dict(status='PASS_CPU_SELF_TEST',checks=['strict_argv','immutable_receipt','inherited_PGID'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root');p.add_argument('--cache-root');p.add_argument('--protocol')
    p.add_argument('--deadline-unix',type=float);p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test: self_test(); return 0
    require(all(getattr(args,key) is not None for key in ['root','cache_root','protocol','deadline_unix']),'Required: --root --cache-root --protocol --deadline-unix')
    waiter=Waiter(args)
    def interrupted(number,_frame): raise RuntimeError('SIGNAL_'+str(number))
    for sig in [signal.SIGTERM,signal.SIGINT]: signal.signal(sig,interrupted)
    try:
        waiter.run(); return 0
    except BaseException as exc:
        waiter.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        write(waiter.directory/'failed.json',waiter.state,exclusive=True)
        traceback.print_exc(); return 1
    finally: waiter.lock.close()


if __name__=='__main__':
    raise SystemExit(main())
