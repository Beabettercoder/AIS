from pathlib import Path
import json,time,subprocess,datetime
S=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');D=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/cases/v6');status=D/'replay_status.json'
def save(phase,detail=None):status.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'detail':detail},indent=2))
assert not status.exists();save('awaiting_all_single_models_and_frozen_thresholds')
while True:
 p=S/'launch_result.json'
 if p.exists():
  z=json.loads(p.read_text())
  if any(q.get('exit_code',1)!=0 for q in z):save('skipped_training_failed',z);raise SystemExit(2)
 frozen=S/'frozen_checkpoint_thresholds.json'
 if frozen.exists():
  try:
   z=json.loads(frozen.read_text());assert len(z['arms'])==96;break
  except (json.JSONDecodeError,OSError,KeyError,AssertionError):pass
 time.sleep(5)
for name,exe in [('score_cases.py','/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'),('render_cases.py','/mnt/sda/WangZijia/AIS/.venv/bin/python')]:
 save('running_'+name)
 with (D/(name+'.log')).open('x') as f:r=subprocess.run([exe,str(D/name)],stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:save('failed',{'script':name,'exit_code':r.returncode});raise SystemExit(r.returncode)
save('complete')
