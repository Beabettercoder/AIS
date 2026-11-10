import concurrent.futures,queue,subprocess,sys,json,time,os,hashlib,threading
from pathlib import Path
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
assert (P/'frozen_train_protocol.json').is_file(); assert not (P/'launch_manifest.json').exists()
plan=[]
for n in ('tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'):
 for s in (0,1,2):
  if s==0 and n in ('tcn','gru','transformer','itransformer'):continue
  plan.append((n,s))
assert len(plan)==20
for n,s in plan:assert not (P/'runs'/n/('seed'+str(s))).exists(),(n,s)
(P/'logs').mkdir(exist_ok=False)
long=[('neuralcde',0),('neuralcde',1),('neuralcde',2)]
shared=queue.Queue()
for x in plan:
 if x not in long:shared.put(x)
assigned={0:[long[0],long[2]],1:[long[1]],2:[],3:[]}
gpus=[1,2,1,2]; lock=threading.Lock();results=[]
manifest={'plan':[{'name':n,'seed':s,'arms':4} for n,s in plan],'gpu_workers':gpus,'total_new_model_runs':80,'reused_models':16,'note':'explicit user-authorized; fixed epoch 25; no held-out access'}
(P/'launch_manifest.json').write_text(json.dumps(manifest,indent=2))
def worker(i):
 pending=assigned[i].copy()
 while True:
  if pending:item=pending.pop(0)
  else:
   try:item=shared.get_nowait()
   except queue.Empty:return
  n,s=item;log=P/'logs'/f'{n}_seed{s}.log'
  print('START',n,s,'gpu',gpus[i],flush=True);start=time.time()
  with log.open('x') as f:
   z=subprocess.run([PY,str(P/'train_single.py'),'--name',n,'--seed',str(s),'--gpu',str(gpus[i])],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
  r={'name':n,'seed':s,'gpu':gpus[i],'exit_code':z.returncode,'wall_seconds':round(time.time()-start,3),'log':str(log)}
  with lock:
   results.append(r);(P/'launch_progress.json').write_text(json.dumps(results,indent=2))
  print('END',n,s,z.returncode,'sec',r['wall_seconds'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 list(pool.map(worker,range(4)))
(P/'launch_result.json').write_text(json.dumps(results,indent=2));print('MATRIX_FINISHED',len(results),'failed',sum(r['exit_code']!=0 for r in results),flush=True)
