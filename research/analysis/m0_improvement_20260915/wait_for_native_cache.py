"""Wait for our already dispatched cache job, then launch its dependent card.

This controller does not use CUDA while waiting, retry extraction, stop any
process, or reclaim occupied GPUs. Child stages inherit the job-runner group.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--producer-job',required=True)
    p.add_argument('--gpu',choices=['0','1'],required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('card_args',nargs=argparse.REMAINDER)
    a=p.parse_args();producer=Path(a.producer_job).resolve()
    launch=json.loads((producer/'launch.json').read_text())
    assert Path(launch['job']).resolve()==producer
    assert launch['uid']==os.getuid()==1009
    assert launch['name']=='native_train_cache_v1'
    assert launch['command'][2].endswith('/native_state_cache.py')
    assert '--split' in launch['command'] and launch['command'][launch['command'].index('--split')+1]=='train'
    package=Path(__file__).resolve().parent
    args=a.card_args[1:] if a.card_args[:1]==['--'] else a.card_args
    assert float(args[args.index('--deadline-unix')+1])==a.deadline_unix
    cache_root=Path(args[args.index('--cache-root')+1]).resolve()
    assert Path(launch['command'][launch['command'].index('--out')+1]).resolve()==cache_root/'train'
    protocol=Path(args[args.index('--protocol')+1])
    contract=json.loads(protocol.read_text())
    assert launch['script_sha256']['native_state_cache.py']==contract['source_sha256']['native_state_cache.py']
    assert launch['script_sha256']['selection_v1.json']==contract['selection_sha256']
    uuid=['GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe','GPU-000b6236-3632-a001-9667-1f02cbb61c8b'][int(a.gpu)]
    end=time.monotonic()+max(0,a.deadline_unix-time.time())
    print(json.dumps(dict(event='WAITING_FOR_OWN_CACHE',producer=str(producer),gpu=a.gpu)),flush=True)
    while time.time()<a.deadline_unix and time.monotonic()<end:
        state=json.loads((producer/'state.json').read_text()) if (producer/'state.json').is_file() else {}
        if state:
            assert state['runner_pid']==launch['runner_pid']
            assert state['command']==launch['command']
        status=state.get('status')
        if status in ['FAILED','TIMEOUT','USER_SIGNAL','STOPPING']:
            raise RuntimeError('Producer did not succeed: '+str(state))
        if status=='EXITED_ZERO':
            assert state['returncode']==0
            q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],
                             check=True,text=True,capture_output=True,timeout=15)
            if uuid not in q.stdout:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu)
                print(json.dumps(dict(event='CACHE_COMPLETE_GPU_FREE_START_CARD',gpu=a.gpu)),flush=True)
                subprocess.run([sys.executable,'-B',str(package/'run_campaign_card.py')]+args,
                               env=env,stdin=subprocess.DEVNULL,check=True)
                return
        time.sleep(min(30,max(.01,end-time.monotonic())))
    raise TimeoutError('Deadline while waiting for completed cache and available GPU')


if __name__=='__main__':main()
