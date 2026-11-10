"""Independent audits after completed test; choose an idle, non-service GPU."""
from pathlib import Path
import time,json,subprocess,concurrent.futures,datetime,os
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');O=P.parent/'independent_audit/single';PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
if (O/'audit_finished.json').exists():raise FileExistsError('Audit already finished')
f=O/'audit_final_start.json';assert not f.exists();f.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'choose idle GPU0/1/2/7 after final evaluation, per parent authorization'},indent=2))
while not ((P/'frozen_checkpoint_thresholds.json').is_file() and (P/'single_formal_summary.json').is_file() and (P/'单主体八骨干三种子正式结果.md').is_file()):time.sleep(10)
def rows():
 d=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return {int(x.split(',')[0]):(int(x.split(',')[1]),int(x.split(',')[2])) for x in d.splitlines()}
def run(name):
 env=os.environ.copy();gpu=None
 if 'replay' in name:
  while gpu is None:
   a=rows();time.sleep(5);b=rows()
   for n in (1,2,0,7):
    if a[n][0]<200 and b[n][0]<200 and a[n][1]==0 and b[n][1]==0:gpu=n;break
   if gpu is None:time.sleep(15)
  env['SINGLE_AUDIT_GPU']=str(gpu)
 with (O/(name+'.log')).open('x') as log:r=subprocess.run([PY,str(O/name)],stdout=log,stderr=subprocess.STDOUT,env=env)
 return {'script':name,'exit_code':r.returncode,'gpu':gpu}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(run,['audit_single_metrics.py','audit_single_replay.py']))
(O/'audit_finished.json').write_text(json.dumps({'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':result},indent=2));print(json.dumps(result),flush=True)
if any(x['exit_code'] for x in result):raise SystemExit(1)
