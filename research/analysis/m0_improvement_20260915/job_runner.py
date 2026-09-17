"""Bounded server job with durable receipt and ownership-scoped child cleanup."""
import datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
job=Path(sys.argv[1]);spec=json.loads((job/'spec.json').read_text())
start=time.monotonic()
stop_signal=None
def request_stop(signum,frame):
 global stop_signal
 stop_signal=signum
signal.signal(signal.SIGTERM,request_stop)
signal.signal(signal.SIGINT,request_stop)
state=dict(status='RUNNING',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner_pid=os.getpid(),command=spec['command'])
def save():
 p=job/'state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p)
save()
with (job/'output.log').open('x') as log:
 child=subprocess.Popen(spec['command'],env=spec['env'],cwd=spec['cwd'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 s=(Path('/proc')/str(child.pid)/'stat').read_text();fields=s[s.rfind(')')+2:].split()
 identity=dict(pid=child.pid,uid=os.getuid(),startticks=int(fields[19]));state['child']=identity;save()
 while child.poll() is None and stop_signal is None and time.monotonic()-start<spec['seconds']:
  try:child.wait(timeout=min(1,max(.01,spec['seconds']-(time.monotonic()-start))))
  except subprocess.TimeoutExpired:pass
 if child.poll() is None:
  reason='USER_SIGNAL' if stop_signal is not None else 'TIMEOUT'
  state.update(status='STOPPING',stop_reason=reason,stop_signal=stop_signal);save()
  p=Path('/proc')/str(child.pid)
  if p.exists():
   s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
   assert p.stat().st_uid==identity['uid'] and int(f[19])==identity['startticks']
   assert os.getpgid(child.pid)==child.pid
   try:os.killpg(child.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:
    # The unreaped child pins this PID; still verify identity before force stop.
    s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
    assert p.stat().st_uid==identity['uid'] and int(f[19])==identity['startticks']
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait()
  state.update(status=reason,returncode=child.returncode)
 else:state.update(status='EXITED_ZERO' if child.returncode==0 else 'FAILED',returncode=child.returncode)
 # If the controller died unexpectedly, descendants can outlive its PID.
 # Only this dedicated session/group, this UID, and later start times qualify.
 def owned_group_members():
  members=[]
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
    if int(f[2])!=identity['pid']:continue
    assert int(f[3])==identity['pid'] and p.stat().st_uid==identity['uid'] and int(f[19])>=identity['startticks'],'Unexpected group identity; refuse cleanup'
    if f[0]!='Z':members.append(int(p.name))
   except (FileNotFoundError,ProcessLookupError):pass
  return members
 remaining=owned_group_members()
 if remaining:
  state['residual_group_cleanup']=remaining
  try:os.killpg(identity['pid'],signal.SIGTERM)
  except ProcessLookupError:pass
  deadline=time.monotonic()+20
  while owned_group_members() and time.monotonic()<deadline:time.sleep(.2)
  remaining=owned_group_members()
  if remaining:
   try:os.killpg(identity['pid'],signal.SIGKILL)
   except ProcessLookupError:pass
  state['residual_group_after_term']=remaining
state.update(seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
save()
