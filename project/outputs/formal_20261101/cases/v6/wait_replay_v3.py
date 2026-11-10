from pathlib import Path
import json,time,subprocess,datetime,hashlib
S=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');D=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/cases/v6');status=D/'replay_v3_status.json'
def save(phase,detail=None):status.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'detail':detail},indent=2))
assert not status.exists() and not (D/'case_scores.json').exists() and not (D/'rolling_scores.json').exists();locks=json.loads((D/'rolling_input_code_lock.json').read_text());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in locks['sha256'].items());add=json.loads((D/'case_figure_addendum_lock.json').read_text());assert hashlib.sha256((D/'render_seedbands.py').read_bytes()).hexdigest()==add['render_seedbands_sha256'];save('awaiting_validated_96_arm_freeze',{'previous_watchers':'conservatively skipped because 4 duplicate-starts correctly rejected; actual results must be validated by single freeze script','nonoverlap_windows':'5 MATSON,16 SARA','rolling_windows':'30 MATSON,59 SARA','prescore_locked':True})
while True:
 frozen=S/'frozen_checkpoint_thresholds.json'
 if frozen.exists():
  try:
   z=json.loads(frozen.read_text());assert len(z['arms'])==96;break
  except (json.JSONDecodeError,OSError,KeyError,AssertionError):pass
 time.sleep(5)
for name,exe in [('score_cases.py','/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'),('render_cases.py','/mnt/sda/WangZijia/AIS/.venv/bin/python'),('score_rolling.py','/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'),('render_rolling.py','/mnt/sda/WangZijia/AIS/.venv/bin/python'),('render_seedbands.py','/mnt/sda/WangZijia/AIS/.venv/bin/python')]:
 save('running_'+name)
 with (D/(name+'.log')).open('x') as f:r=subprocess.run([exe,str(D/name)],stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:save('failed',{'script':name,'exit_code':r.returncode});raise SystemExit(r.returncode)
save('complete')
