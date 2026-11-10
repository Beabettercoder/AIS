"""Replay only after two idle observations of one non-service GPU."""
from pathlib import Path
import time,json,subprocess,datetime,os
O=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/independent_audit/single');PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
assert (O/'independent_single_metrics_audit_v2.json').exists() and not (O/'audit_single_replay.py.log').exists()
def rows():
 d=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return {int(x.split(',')[0]):(int(x.split(',')[1]),int(x.split(',')[2])) for x in d.splitlines()}
while True:
 a=rows();time.sleep(5);b=rows();free=[n for n in (3,1,2,0,5,6,7) if a[n][0]<200 and b[n][0]<200 and a[n][1]==0 and b[n][1]==0]
 if free:break
 time.sleep(15)
gpu=free[0];env=os.environ.copy();env['SINGLE_AUDIT_GPU']=str(gpu)
(O/'replay_launch.json').write_text(json.dumps({'selected_gpu':gpu,'two_idle_observations':[a[gpu],b[gpu]],'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
with (O/'audit_single_replay.py.log').open('x') as f:r=subprocess.run([PY,str(O/'audit_single_replay.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
(O/'audit_finished.json').write_text(json.dumps({'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metrics_v2':'passed','replay_exit_code':r.returncode,'gpu':gpu},indent=2));print('GPU',gpu,'REPLAY_EXIT',r.returncode,flush=True)
if r.returncode:raise SystemExit(r.returncode)
