from pathlib import Path
import argparse,queue,subprocess,concurrent.futures,json,os,time
R=Path(__file__).resolve().parent;PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['train','test'],required=True);ap.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3,5,6,7]);a=ap.parse_args();assert 4 not in a.gpus
q=queue.Queue()
for arch in ['neuralcde','mamba','patchtst','graph_temporal','transformer','itransformer','tcn','gru']:
 for seed in [0,1,2]:q.put((arch,seed))
def run(gpu):
    events=[]
    while not (R/'STOP').exists():
        try:arch,seed=q.get_nowait()
        except queue.Empty:break
        label=f'field_{a.mode}_{arch}_{seed}';p=R/'logs'/(label+'.log');started=time.time()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with p.open('a') as f:r=subprocess.run([PY,str(R/'train_field_extension.py'),'--mode',a.mode,'--backbone',arch,'--device','cuda:0','--seeds',str(seed)],stdout=f,stderr=subprocess.STDOUT,env=env)
        row={'job':label,'gpu':gpu,'returncode':r.returncode,'seconds':time.time()-started};events.append(row);print(json.dumps(row),flush=True)
    return events
with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:events=[x for f in [pool.submit(run,g) for g in a.gpus] for x in f.result()]
(R/f'field_{a.mode}_launcher_results.json').write_text(json.dumps(events,indent=2));print('DONE',len(events),'FAILED',sum(x['returncode']!=0 for x in events),flush=True)
