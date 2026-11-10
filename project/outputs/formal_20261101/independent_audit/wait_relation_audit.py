from pathlib import Path
import json,time,subprocess,sys,datetime
O=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/relation');D=O.parent/'independent_audit';script=D/'audit_relation.py';status=D/'relation_audit_status.json'
def save(phase,detail=None):status.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'detail':detail},indent=2))
assert not status.exists();save('waiting_for_final_summary')
while True:
 p=O/'pipeline_status.json'
 if p.exists():
  z=json.loads(p.read_text())
  if z.get('phase')=='failed':save('skipped_pipeline_failed',z);raise SystemExit(2)
 if (O/'heldout.lock.json').exists() and (O/'summary.json').exists():
  try:json.loads((O/'summary.json').read_text());break
  except (json.JSONDecodeError,OSError):pass
 time.sleep(5)
save('running')
with (D/'relation_audit.log').open('x') as f:r=subprocess.run([sys.executable,str(script)],stdout=f,stderr=subprocess.STDOUT)
save('complete' if r.returncode==0 else 'failed',{'exit_code':r.returncode});raise SystemExit(r.returncode)
