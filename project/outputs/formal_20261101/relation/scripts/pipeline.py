from pathlib import Path
import json,time,subprocess,sys,hashlib,datetime
O=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/relation');PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python';VENV='/mnt/sda/WangZijia/AIS/.venv/bin/python'
progress=O/'pipeline_status.json'
def write(phase,extra=None):progress.write_text(json.dumps({'phase':phase,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'detail':extra},indent=2),encoding='utf8')
def run(name,cmd):
 write(name);print('START',name,flush=True)
 with (O/(name+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:write('failed',{'step':name,'exit_code':r.returncode});raise SystemExit(r.returncode)
 print('DONE',name,flush=True)
try:
 write('await_training')
 while True:
  p=O/'status.json'
  if p.exists():
   status=json.loads(p.read_text());jobs=status.get('jobs',{})
   failed=[k for k,v in jobs.items() if v['status']=='failed']
   if failed:write('failed',{'training_failed':failed});raise SystemExit(1)
   if len(jobs)==24 and all(v['status']=='completed' for v in jobs.values()):break
  time.sleep(10)
 run('freeze_test',[PY,str(O/'scripts/freeze_test.py')])
 run('build_test',[PY,str(O/'scripts/build_test.py'),'--split','test'])
 lock=json.loads((O/'heldout.lock.json').read_text())
 for name,h in lock['postfreeze_scripts_sha256'].items():
  assert hashlib.sha256((O/'scripts'/name).read_bytes()).hexdigest()==h,('post-freeze code changed',name)
 run('eval_launch',[PY,str(O/'scripts/launch_eval.py')])
 run('summarize',[PY,str(O/'scripts/summarize.py')])
 run('render',[PY,str(O/'scripts/render.py')])
 run('render_figure',[VENV,str(O/'scripts/render_figure.py')])
 write('complete')
except Exception as e:
 write('failed',{'error':repr(e)});raise
