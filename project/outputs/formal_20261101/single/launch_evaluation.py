import concurrent.futures,queue,subprocess,sys,json,time,threading
from pathlib import Path
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python';assert (P/'frozen_checkpoint_thresholds.json').exists();assert not (P/'test_launch_manifest.json').exists()
q=queue.Queue();plan=[(n,s) for n in ('tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph') for s in (0,1,2)]
for n,s in plan:
 assert not (P/'runs'/n/f'seed{s}'/'test').exists();q.put((n,s))
(P/'test_logs').mkdir(exist_ok=False);(P/'test_launch_manifest.json').write_text(json.dumps({'plan':plan,'gpus':[1,2],'notes':'all 96 models final epoch25 and val-calibrated thresholds frozen before any test predictions'},indent=2));results=[];lock=threading.Lock()
def worker(i):
 gpu=[1,2,1,2][i]
 while True:
  try:n,s=q.get_nowait()
  except queue.Empty:return
  with (P/'test_logs'/f'{n}_seed{s}.log').open('x') as f:
   z=subprocess.run([PY,str(P/'evaluate_fixed.py'),'--name',n,'--seed',str(s),'--gpu',str(gpu)],stdout=f,stderr=subprocess.STDOUT)
  with lock:results.append({'name':n,'seed':s,'gpu':gpu,'exit':z.returncode});(P/'test_progress.json').write_text(json.dumps(results,indent=2))
  print('EVALUATED',n,s,z.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,range(4)))
(P/'test_launch_result.json').write_text(json.dumps(results,indent=2));print('TEST_DONE',len(results),'failed',sum(x['exit']!=0 for x in results),flush=True)
