"""Wait until fixed test summary is complete; replay only when GPU7 is idle."""
from pathlib import Path
import time,json,subprocess,concurrent.futures,datetime
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');O=P.parent/'independent_audit/single';PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
if (O/'audit_finished.json').exists():raise FileExistsError('Audit already finished')
f=O/'audit_resumed.json';assert not f.exists();f.write_text(json.dumps({'resumed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'GPU7 became occupied by another workload; wait for idle before replay'},indent=2))
while not ((P/'frozen_checkpoint_thresholds.json').is_file() and (P/'single_formal_summary.json').is_file() and (P/'单主体八骨干三种子正式结果.md').is_file()):time.sleep(10)
def run(name):
 if 'replay' in name:
  while True:
   data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
   row=[x.strip().split(',') for x in data.splitlines() if x.strip().split(',')[0].strip()=='7'][0]
   if int(row[1])<200 and int(row[2])==0:
    time.sleep(5)
    d2=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rr=[x.strip().split(',') for x in d2.splitlines() if x.strip().split(',')[0].strip()=='7'][0]
    if int(rr[1])<200 and int(rr[2])==0:break
   time.sleep(15)
 with (O/(name+'.log')).open('x') as log:r=subprocess.run([PY,str(O/name)],stdout=log,stderr=subprocess.STDOUT)
 return {'script':name,'exit_code':r.returncode}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(run,['audit_single_metrics.py','audit_single_replay.py']))
(O/'audit_finished.json').write_text(json.dumps({'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':result},indent=2));print(json.dumps(result),flush=True)
if any(x['exit_code'] for x in result):raise SystemExit(1)
