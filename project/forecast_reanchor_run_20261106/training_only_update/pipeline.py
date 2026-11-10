"""User-started transparent pipeline; Ctrl+C stops workers, exact resume is supported."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,default=DEFAULT_PILOT);p.add_argument('--work',type=Path,default=Path('work'))
    p.add_argument('--gpus',nargs='+',type=int,default=[2,3,5,6]);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();a.work=a.work.resolve();a.work.mkdir(parents=True,exist_ok=True)
    if len(a.gpus) not in [1,4] or len(set(a.gpus))!=len(a.gpus):p.error('Provide one GPU for sequential runs or four distinct GPUs.')
    lock=T.acquire_lock(a.work);children=[]
    def stage(name,**kw):save(a.work/'pipeline_status.json',dict(stage=name,updated_utc=T.utc(),**kw));print('\n'+name,flush=True)
    def run(script,*extra):subprocess.run([sys.executable,'-u',str(ROOT/script),*map(str,extra)],check=True)
    try:
        stage('1/7 PREPARE: re-anchored data, 12 Raw reuse audits')
        run('prepare.py','--pilot',a.pilot,'--work',a.work)
        if a.prepare_only:stage('prepared_only');return
        stage('2/7 TRAIN: 36 new runs; 12 Raw models reused')
        queue=list(CONFIG['models']);available=list(a.gpus);active=[];completed=[];printed={}
        for path in (a.work/'runs').rglob('progress.json'):
            rr=read(path)
            if rr.get('arm')!='P0':printed[str(path.parent)]=rr.get('completed_epochs',0)
        logs=a.work/'logs';logs.mkdir(exist_ok=True)
        while queue or active:
            while queue and available:
                model=queue.pop(0);gpu=available.pop(0);f=(logs/f'{model}.log').open('a')
                cmd=[sys.executable,'-u',str(ROOT/'train_worker.py'),'--work',str(a.work),'--model',model,'--device',f'cuda:{gpu}']
                proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);children.append(proc);active.append((proc,f,model,gpu))
            for item in list(active):
                proc,f,model,gpu=item;code=proc.poll()
                if code is not None:
                    f.close();active.remove(item);available.append(gpu)
                    if code:raise RuntimeError(f'{model} failed, inspect {logs/model}.log')
                    completed.append(model)
            current=[];done=0
            for file in (a.work/'runs').rglob('progress.json'):
                r=read(file)
                if r.get('arm')=='P0':continue
                event_file=file.with_name('events.jsonl')
                if event_file.exists():
                    for line in event_file.read_text(encoding='utf-8').splitlines():
                        try:ev=__import__('json').loads(line)
                        except ValueError:continue
                        if ev.get('event')=='epoch_complete' and ev['epoch']>printed.get(str(file.parent),0):
                            printed[str(file.parent)]=ev['epoch']
                            zh={'P1':'运动学门控','P2':'门控＋普通背景场','P3':'门控＋可信背景场'}[r['arm']]
                            print('\r'+' '*180+'\r'+f"{r['model']:12s} 种子 {r['seed']} | {zh} | 第 {ev['epoch']:02d}/{r['total_epochs']} 轮 | "
                                f"训练损失 {ev['train_loss']:.6f} | 验证 ADE {ev['val_ADE_m']:.2f} m / FDE {ev['val_FDE_m']:.2f} m | "
                                f"权重 weights/epoch_{ev['epoch']:03d}.pt",flush=True)
                if r.get('stage')=='complete':done+=1
                else:current.append(f"{r.get('model')} s{r.get('seed')} {r.get('arm')} epoch {r.get('epoch',0)}/{r.get('total_epochs')} batch {r.get('batch',0)}/{r.get('total_batches',0)} loss={r.get('train_loss')}")
            width=24;n=int(width*done/36)
            print('\r'+'['+'#'*n+'-'*(width-n)+f'] {done}/36 | '+'; '.join(current)+' '*12,end='',flush=True)
            save(a.work/'pipeline_status.json',dict(stage='training',completed_new_runs=done,total_new_runs=36,reused_raw=12,active=current,updated_utc=T.utc()))
            if queue or active:time.sleep(1)
        stage('3/7 FREEZE: all 48 checkpoints, normalizer and protocol')
        run('freeze.py','--work',a.work)
        stage('4/7 VALIDATION RESULTS')
        run('report_results.py','--work',a.work,'--split','val')
        stage('5/7 BUILD TEST: only after freeze')
        run('build_test.py','--pilot',a.pilot,'--work',a.work)
        stage('6/7 FROZEN TEST EVALUATION')
        run('evaluate_test.py','--work',a.work,'--device',f'cuda:{a.gpus[0]}')
        stage('7/7 TEST RESULTS AND INDEPENDENT AUDIT')
        run('report_results.py','--work',a.work,'--split','test')
        run('audit_run.py','--work',a.work)
        stage('complete',validation_results=str(a.work/'reports/val'),test_results=str(a.work/'reports/test'),
              checkpoints=str(a.work/'runs'))
    except BaseException as e:
        for proc in children:
            if proc.poll() is None:proc.terminate()
        stage('interrupted' if isinstance(e,KeyboardInterrupt) else 'error',message=str(e));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
