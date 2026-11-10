"""Run bounded, resumable study jobs on explicitly assigned idle GPUs."""
from pathlib import Path
import argparse, json, subprocess, time, os, concurrent.futures, queue, sys
ROOT=Path(__file__).resolve().parent
PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
def worker(gpu,jobs,mode):
    events=[]
    while not (ROOT/'STOP').exists():
        try:task,arch,seed=jobs.get_nowait()
        except queue.Empty:break
        label=f'{mode}_{task}_{arch}_seed{seed}';log=ROOT/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
        cmd=[PY,str(ROOT/'train_study.py'),'--mode',mode,'--task',task,'--backbone',arch,'--device','cuda:0','--seeds',str(seed)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        started=time.time()
        with log.open('a',encoding='utf8') as f:
            f.write('\nJOB_START '+json.dumps({'gpu':gpu,'time':started,'cmd':cmd})+'\n');f.flush()
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
        event={'job':label,'gpu':gpu,'returncode':result.returncode,'elapsed_seconds':time.time()-started,'log':str(log)};events.append(event)
        (ROOT/'job_events').mkdir(exist_ok=True)
        (ROOT/'job_events'/(label+'.json')).write_text(json.dumps(event,indent=2),encoding='utf8')
        print(json.dumps(event),flush=True)
        # Failures are recorded; unrelated jobs can complete. Failed arms are never counted complete.
    return events
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['train','test'],required=True);ap.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3,5,6,7]);ap.add_argument('--tasks',nargs='+',default=['individual','relation']);ap.add_argument('--backbones',nargs='+',default=['neuralcde','patchtst','mamba','graph_temporal','transformer','itransformer','tcn','gru']);a=ap.parse_args()
    assert 4 not in a.gpus and len(set(a.gpus))==len(a.gpus)
    pending=queue.Queue()
    for arch in a.backbones:
        for task in a.tasks:
            for seed in (0,1,2):pending.put((task,arch,seed))
    print('LAUNCH',a.mode,'jobs',pending.qsize(),'gpus',a.gpus,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        results=[x for f in [pool.submit(worker,g,pending,a.mode) for g in a.gpus] for x in f.result()]
    out={'mode':a.mode,'events':results,'stopped':(ROOT/'STOP').exists(),'jobs_not_started':pending.qsize()}
    (ROOT/(a.mode+'_launcher_results.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
    print('LAUNCHER_DONE',json.dumps({'jobs':len(results),'failed':sum(x['returncode']!=0 for x in results),'not_started':pending.qsize()}),flush=True)
