import concurrent.futures,subprocess,time,json,threading,os,sys
from pathlib import Path
O=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/relation'); PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python';S=O/'scripts/train.py'
if (O/'launch.json').exists():raise FileExistsError('Existing launch refused')
names=['neuralcde','itransformer','mamba','patchtst','transformer','gru','tcn','graph'];jobs=[(n,s) for n in names for s in (0,1,2)]
import queue
q=queue.Queue()
for x in jobs:q.put(x)
lock=threading.Lock();progress={'jobs':{},'devices':[3,5,6,7],'started':time.time()}
def save():
 tmp=O/'status.tmp';tmp.write_text(json.dumps(progress,indent=2));os.replace(tmp,O/'status.json')
def work(gpu):
 while True:
  try:n,s=q.get_nowait()
  except queue.Empty:return
  key=f'{n}/seed{s}'
  with lock:progress['jobs'][key]={'status':'running','gpu':gpu,'start':time.time()};save()
  log=O/'logs'/f'{n}_seed{s}.log'
  with log.open('x') as f:code=subprocess.run([PY,str(S),'--name',n,'--seed',str(s),'--gpu',str(gpu)],stdout=f,stderr=subprocess.STDOUT).returncode
  with lock:progress['jobs'][key].update({'status':'completed' if code==0 else 'failed','exit_code':code,'end':time.time()});save()
  q.task_done()
(O/'logs').mkdir(exist_ok=False)
(O/'launch.json').write_text(json.dumps({'jobs':jobs,'devices':[3,5,6,7],'service_gpu4':'excluded','user_authorized_unconditional':True},indent=2))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(work,[3,5,6,7]))
progress['completed']=time.time();save()
if any(v['exit_code']!=0 for v in progress['jobs'].values()):sys.exit(1)
