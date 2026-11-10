import os,subprocess,json,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
python='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
trainer=p/'historical/train_historical.py'; workers=[]
for model,gpu in [('tcn','0'),('gru','1')]:
  commands=[]
  for policy in ['clean','all']:
    cmd=[python,str(trainer),'--data',str(p/'data'),'--out',str(p/'runs'),'--protocol',str(p/'protocol.json'),'--backbone',model,'--policy',policy,'--device','cuda:0']
    if model=='tcn' and policy=='clean':cmd.append('--resume')
    commands.append(cmd)
  worker=p/('worker_'+model+'.py')
  worker.write_text('import subprocess,sys,json,time\ncommands='+repr(commands)+'\nfor cmd in commands:\n print("START",json.dumps(cmd),flush=True)\n r=subprocess.run(cmd)\n print("END",r.returncode,flush=True)\n if r.returncode:sys.exit(r.returncode)\n',encoding='utf-8')
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
  log=(p/('worker_'+model+'.log')).open('x');proc=subprocess.Popen([python,str(worker)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
  workers.append(dict(model=model,gpu=gpu,pid=proc.pid,commands=commands))
(p/'launch.json').write_text(json.dumps({'time_utc':time.time(),'workers':workers},indent=2),encoding='utf-8');print(json.dumps(workers,indent=2))
