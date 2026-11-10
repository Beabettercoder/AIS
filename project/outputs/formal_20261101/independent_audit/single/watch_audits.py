"""Wait until fixed test summary is completely produced, then run independent audits."""
from pathlib import Path
import time,json,subprocess,concurrent.futures,datetime
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');O=P.parent/'independent_audit/single';PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
if (O/'audit_started.json').exists():raise FileExistsError('Audit already launched')
(O/'audit_started.json').write_text(json.dumps({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'waiting for final single task summary'},indent=2))
while not ((P/'frozen_checkpoint_thresholds.json').is_file() and (P/'single_formal_summary.json').is_file() and (P/'单主体八骨干三种子正式结果.md').is_file()):time.sleep(10)
# Test set scoring has already completed and model thresholds were frozen; no new training or optimization.
def run(name):
 with (O/(name+'.log')).open('x') as f:r=subprocess.run([PY,str(O/name)],stdout=f,stderr=subprocess.STDOUT)
 return {'script':name,'exit_code':r.returncode}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(run,['audit_single_metrics.py','audit_single_replay.py']))
(O/'audit_finished.json').write_text(json.dumps({'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':result},indent=2));print(json.dumps(result),flush=True)
if any(x['exit_code'] for x in result):raise SystemExit(1)
