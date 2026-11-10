"""Run four forecasting workers with persistent logs and live status files."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def atomic(path, obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--out',type=Path,default=Path('parallel_runs'))
    p.add_argument('--reports',type=Path,default=Path('reports'))
    p.add_argument('--gpus',nargs=4,type=int,default=[2,3,5,6])
    p.add_argument('--detach',action='store_true')
    args=p.parse_args();here=Path(__file__).resolve().parent;os.chdir(here)
    if len(set(args.gpus))!=4:raise ValueError('Provide four different free GPU indices.')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.detach:
        cmd=[sys.executable,str(Path(__file__).resolve()),'--data',str(args.data.resolve()),
             '--out',str(args.out.resolve()),'--reports',str(args.reports.resolve()),
             '--gpus',*map(str,args.gpus)]
        with (args.out/'supervisor.log').open('a') as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(args.out/'supervisor_start.json',dict(pid=proc.pid,command=cmd,started_unix=time.time()))
        print(f'Supervisor PID {proc.pid}; progress: {args.out}/supervisor_status.json',flush=True);return
    status=dict(stage='training',started_unix=time.time(),workers=[]);jobs=[]
    for model,gpu in zip(['tcn','gru','transformer','itransformer'],args.gpus):
        cmd=[sys.executable,'-u',str(here/'train.py'),'--data',str(args.data),'--out',str(args.out/model),
             '--models',model,'--device',f'cuda:{gpu}','--threads','2','--resume']
        log=(args.out/f'{model}.log').open('a');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        item=dict(model=model,gpu=gpu,pid=proc.pid,command=cmd,returncode=None);status['workers'].append(item);jobs.append((proc,log,item))
    while True:
        for proc,log,item in jobs:item['returncode']=proc.poll()
        status['updated_unix']=time.time();atomic(args.out/'supervisor_status.json',status)
        if all(item['returncode'] is not None for _,_,item in jobs):break
        time.sleep(5)
    for _,log,_ in jobs:log.close()
    if any(item['returncode']!=0 for _,_,item in jobs):
        status['stage']='training_error';atomic(args.out/'supervisor_status.json',status);raise SystemExit('A worker failed; inspect its log. Completed runs are preserved.')
    status['stage']='summarizing';atomic(args.out/'supervisor_status.json',status)
    cmd=[sys.executable,'-u',str(here/'summarize.py'),'--runs',str(args.out),'--out',str(args.reports),'--bootstrap','2000']
    code=subprocess.call(cmd)
    if code==0:
        code=subprocess.call([sys.executable,str(here/'independent_checks.py'),'--runs',str(args.out),
            '--out',str(args.reports/'independent_postflight.json')])
    status['stage']='complete' if code==0 else 'evaluation_error';status['finished_unix']=time.time()
    atomic(args.out/'supervisor_status.json',status);raise SystemExit(code)

if __name__=='__main__':main()
