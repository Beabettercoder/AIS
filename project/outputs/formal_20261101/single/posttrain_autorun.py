"""Wait for owned training; freeze, evaluate, summarize in that order; stop on any failure."""
import json,time,subprocess,sys,hashlib
from pathlib import Path
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python';done=P/'autorun_status.json';assert not done.exists();t=time.time()
def save(stage,**kw):
 q={'stage':stage,'elapsed_seconds':round(time.time()-t,2),**kw};tmp=P/'autorun_status.tmp';tmp.write_text(json.dumps(q,indent=2));tmp.replace(done);print(json.dumps(q),flush=True)
def valid_json(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
save('waiting_for_training')
while True:
 a=valid_json(P/'launch_result.json');b=valid_json(P/'cde2_accelerated_result.json');c=valid_json(P/'graph_accelerated_result.json')
 if a is not None and b is not None and c is not None:break
 if time.time()-t>4*3600:save('failed_wait_timeout');sys.exit(1)
 time.sleep(10)
for stage,filename in [('freeze','freeze_checkpoints.py'),('evaluate','launch_evaluation.py'),('summarize','summarize_single.py')]:
 save(stage+'_running')
 with (P/(stage+'_autorun.log')).open('x') as f:r=subprocess.run([PY,str(P/filename)],stdout=f,stderr=subprocess.STDOUT)
 if r.returncode!=0:save(stage+'_failed',exit_code=r.returncode);sys.exit(r.returncode)
 if stage=='evaluate':
  x=valid_json(P/'test_launch_result.json')
  if x is None or len(x)!=24 or any(z['exit']!=0 for z in x):save('evaluate_failed_worker',worker_results=x);sys.exit(1)
 save(stage+'_done')
save('complete')
