from pathlib import Path
import hashlib,json,time,subprocess,datetime
D=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/cases/v6');status=D/'seedband_status.json';assert not status.exists();status.write_text(json.dumps({'phase':'waiting_for_case_scores'}));p=D/'case_figure_addendum_lock.json';a=json.loads(p.read_text());assert hashlib.sha256((D/'render_seedbands.py').read_bytes()).hexdigest()==a['render_seedbands_sha256']
while True:
 z=json.loads((D/'replay_status.json').read_text())
 if z.get('phase')=='failed' or z.get('phase')=='skipped_training_failed':status.write_text(json.dumps({'phase':'skipped','replay':z}));raise SystemExit(2)
 if z.get('phase')=='complete':break
 time.sleep(5)
status.write_text(json.dumps({'phase':'rendering'}))
with (D/'render_seedbands.log').open('x') as f:r=subprocess.run(['/mnt/sda/WangZijia/AIS/.venv/bin/python',str(D/'render_seedbands.py')],stdout=f,stderr=subprocess.STDOUT)
status.write_text(json.dumps({'phase':'complete' if r.returncode==0 else 'failed','exit_code':r.returncode}));raise SystemExit(r.returncode)
