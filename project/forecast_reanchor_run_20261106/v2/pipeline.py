"""User-run v2 training-policy, data-quota and model-capacity comparisons. No PDF."""
from __future__ import annotations
import argparse,json,hashlib,os,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'frozen_pilot'))
import train as legacy
sys.path.insert(0,str(HERE))
import data as D

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj):legacy.atomic_json(p,obj)
def sha(p):return legacy.digest(p)
def code_hashes():
    paths=list(HERE.glob('*.py'))+list((ROOT/'frozen_pilot').glob('*.py'))+[ROOT/'reanchor.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def plans(a):
    def spec(name,hidden,quota,policy):return dict(name=name,hidden=hidden,train_per_region=quota,policy=policy,
        history_steps=a.history_steps,future_steps=a.future_steps,batch_size=a.batch_size,epochs=a.epochs,
        itransformer_epochs=a.itransformer_epochs,learning_rate=a.lr,models=a.models,seeds=a.seeds)
    out=[spec('main',a.hidden_dim,a.train_per_region,'cosine_best')]
    if not a.main_only:
        out += [spec(f'capacity_h{h}',h,a.train_per_region,'cosine_best') for h in [64,128,256] if h!=a.hidden_dim]
        if a.baseline_train_per_region!=a.train_per_region:
            out.append(spec('data_small',a.hidden_dim,a.baseline_train_per_region,'cosine_best'))
        out.append(spec('training_constant',a.hidden_dim,a.baseline_train_per_region,'constant_last'))
    return out

def freeze(exp,settings,norm):
    rows=[]
    for model in settings['models']:
        for seed in settings['seeds']:
            hashes=set();counts=set()
            for arm in ['P0','P1','P2','P3']:
                folder=exp/f'runs/{model}/seed_{seed}/{arm}';r=read(folder/'result.json')
                assert r['status']=='complete'
                hashes.add(r.get('initial_model_hash',r.get('initial_hash')))
                counts.add(r.get('parameter_count'))
                path=Path(r['selected_checkpoint']);assert path.exists()
                assert sha(path)==r['selected_checkpoint_sha256']
                rows.append(dict(model=model,seed=seed,arm=arm,run_dir=str(folder.resolve()),selected_checkpoint=str(path.resolve()),
                    selected_checkpoint_sha256=sha(path),best_val=r['best_val'],last_epoch=r['last_epoch']))
            assert len(hashes)==1 and None not in hashes,'Paired initializations not equal'
            assert len(counts)==1 and None not in counts,'Paired registered parameter count not equal'
    obj=dict(settings=settings,models=rows,norm_sha256=sha(norm),code_hashes=code_hashes(),
        selection='Base validation ADE minimum for cosine_best, predeclared final epoch for constant_last',
        test_status='target-MMSI held out from training; evaluation pool reused from earlier research',frozen_utc=legacy.utc())
    path=exp/'freeze.json'
    if path.exists():
        old=read(path);assert old['models']==obj['models'] and old['code_hashes']==obj['code_hashes'];return old
    save(path,obj);return obj

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=ROOT/'work_v2');p.add_argument('--project',type=Path,default=Path('/mnt/sda/WangZijia/AIS'))
    p.add_argument('--pilot',type=Path,default=Path('/mnt/sda/WangZijia/AIS/forecast_pilot_20261105'))
    p.add_argument('--gpus',nargs='+',type=int,default=[2,3,5,6]);p.add_argument('--hidden-dim',type=int,choices=[64,128,256],default=64)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--history-steps',type=int,default=12);p.add_argument('--future-steps',type=int,default=8)
    p.add_argument('--train-per-region',type=int,default=2000);p.add_argument('--baseline-train-per-region',type=int,default=700)
    p.add_argument('--val-per-region',type=int,default=150);p.add_argument('--test-per-region',type=int,default=150)
    p.add_argument('--epochs',type=int,default=25);p.add_argument('--itransformer-epochs',type=int,default=75);p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--models',nargs='+',choices=['tcn','gru','transformer','itransformer'],default=['tcn','gru','transformer','itransformer'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--main-only',action='store_true');p.add_argument('--inventory-only',action='store_true');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();work=a.work.resolve();work.mkdir(parents=True,exist_ok=True)
    if not a.gpus or len(set(a.gpus))!=len(a.gpus):p.error('Use distinct GPU indices')
    if min(a.batch_size,a.epochs,a.itransformer_epochs,a.train_per_region,a.baseline_train_per_region,a.val_per_region,a.test_per_region)<=0:p.error('Budgets must be positive')
    if a.train_per_region<a.baseline_train_per_region:p.error('Expanded quota must be >= baseline quota')
    if len(set(a.models))!=len(a.models) or len(set(a.seeds))!=len(a.seeds):p.error('Duplicate models/seeds not allowed')
    if a.plan_only:
        print(json.dumps(dict(experiments=plans(a),total_runs=len(plans(a))*len(a.models)*len(a.seeds)*4),indent=2));return
    lock=legacy.acquire_lock(work);children=[]
    try:
        specs=plans(a)
        if a.inventory_only:
            inv=D.inventory(a.project,a.pilot,a.history_steps,a.future_steps);save(work/'inventory.json',inv)
            print('Inventory saved:',work/'inventory.json');return
        norm_source=ROOT/'work/data/normalizer.json'
        if not norm_source.exists():norm_source=a.pilot/'parallel_runs/tcn/tcn/seed_0/P0/normalizer.json'
        plan=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},experiments=specs,code_hashes=code_hashes(),
            normalizer_sha256=sha(norm_source),raw_retrained=True,
            evaluation_policy='All comparisons predeclared before main test; test scores never used to choose later experiments',
            reused_test_sources=True,outputs='weights, logs, validation/test predictions and CSV/JSON/Markdown numeric summaries only')
        marker=work/'plan.json'
        if marker.exists():assert read(marker)==plan,'Plan/code changed: use a new --work directory'
        else:save(marker,plan)
        norm=work/'normalizer.json'
        if norm.exists():assert sha(norm)==plan['normalizer_sha256']
        else:shutil.copy2(norm_source,norm)
        inv=D.inventory(a.project,a.pilot,a.history_steps,a.future_steps);save(work/'inventory.json',inv)
        runs_per=len(a.models)*len(a.seeds)*4;total=runs_per*len(specs)
        print(f'Predeclared {len(specs)} experiments, {total} training runs; every arm including Raw is trained.',flush=True)
        data_cache={};shared_eval=None
        for num,spec in enumerate(specs,1):
            exp=work/spec['name'];exp.mkdir(exist_ok=True)
            data_dir=work/f"data/q{spec['train_per_region']}_h{a.history_steps}_f{a.future_steps}"
            if spec['train_per_region'] not in data_cache:
                D.build(a.project,a.pilot,data_dir,'train',spec['train_per_region'],a.history_steps,a.future_steps)
                if shared_eval is None:
                    D.build(a.project,a.pilot,data_dir,'val',a.val_per_region,a.history_steps,a.future_steps);shared_eval=data_dir
                else:
                    for name in ['val.npz','val_manifest.json','val_provenance.json']:shutil.copy2(shared_eval/name,data_dir/name)
                data_cache[spec['train_per_region']]=data_dir
            spec={**spec,'data_dir':str(data_dir.resolve()),'norm_path':str(norm.resolve()),'norm_sha256':sha(norm)}
            specpath=exp/'experiment.json'
            if specpath.exists():assert read(specpath)==spec
            else:save(specpath,spec)
            queue=list(a.models);available=a.gpus[:];active=[];seen={};logs=exp/'logs';logs.mkdir(exist_ok=True)
            for progress in (exp/'runs').rglob('progress.json'):
                try:seen[str(progress.parent)]=read(progress).get('completed_epochs',0)
                except (OSError,ValueError):pass
            while queue or active:
                while queue and available:
                    model=queue.pop(0);gpu=available.pop(0);f=(logs/f'{model}.log').open('a')
                    cmd=[sys.executable,'-u',str(HERE/'worker.py'),'--experiment',str(exp),'--model',model,'--device',f'cuda:{gpu}']
                    proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);children.append(proc);active.append((proc,f,model,gpu))
                for job in list(active):
                    proc,f,m,g=job;code=proc.poll()
                    if code is not None:
                        f.close();active.remove(job);available.append(g)
                        if code:raise RuntimeError(f'{spec["name"]}/{m} failed; inspect {logs/m}.log')
                done=0;current=[]
                for path in (exp/'runs').rglob('progress.json'):
                    try:r=read(path)
                    except (OSError,ValueError):continue
                    done+=r.get('stage')=='complete'
                    if r.get('stage')!='complete':current.append(r)
                    history=path.with_name('history.csv')
                    if history.exists():
                        import csv
                        try:rows=list(csv.DictReader(history.open()))
                        except (OSError,ValueError):rows=[]
                        for row in rows:
                            if not all(row.get(k) for k in ['epoch','lr','train_loss','val_ADE_m','val_FDE_m']):continue
                            epoch=int(row['epoch'])
                            if epoch>seen.get(str(path.parent),0):
                                seen[str(path.parent)]=epoch
                                print('\r'+' '*180+'\r'+f"[{num}/{len(specs)} {spec['name']}] {path.parent.parent.parent.name} {path.parent.parent.name} {path.parent.name} "
                                    f"epoch {epoch} | lr={float(row['lr']):.7f} loss={float(row['train_loss']):.6f} | "
                                    f"val ADE={float(row['val_ADE_m']):.2f}m FDE={float(row['val_FDE_m']):.2f}m",flush=True)
                state=dict(stage='training',experiment=spec['name'],experiment_index=num,experiment_count=len(specs),completed_in_experiment=done,runs_per_experiment=runs_per,total_planned=total,active=current)
                save(work/'progress.json',state)
                print(f'\r[{num}/{len(specs)} {spec["name"]}] {done}/{runs_per} complete '+' '*20,end='',flush=True)
                if queue or active:time.sleep(1)
            print('',flush=True);frozen=freeze(exp,spec,norm)
            # One shared fixed evaluation cohort. Later settings were predeclared before main test.
            if not (shared_eval/'test.npz').is_file() or not (shared_eval/'test_manifest.json').is_file():
                D.build(a.project,a.pilot,shared_eval,'test',a.test_per_region,a.history_steps,a.future_steps,exp/'freeze.json')
            else:
                assert sha(shared_eval/'test.npz')==read(shared_eval/'test_manifest.json')['npz_sha256'],'Test cache hash mismatch'
            if data_dir!=shared_eval:
                for name in ['test.npz','test_manifest.json','test_provenance.json']:shutil.copy2(shared_eval/name,data_dir/name)
            from engine import evaluate
            for row in frozen['models']:
                assert sha(row['selected_checkpoint'])==row['selected_checkpoint_sha256']
                evaluate(data_dir,exp/f"test_runs/{row['model']}/seed_{row['seed']}/{row['arm']}",Path(row['run_dir']),split='test',device=f'cuda:{a.gpus[0]}',freeze_path=exp/'freeze.json')
            from report import summarize
            for split in ['val','test']:summarize(exp,exp/f'reports/{split}',split=split,data_dir=data_dir)
            save(exp/'complete.json',dict(status='complete',settings=spec,freeze_sha256=sha(exp/'freeze.json')))
        from report import compare
        for split in ['val','test']:compare([work/s['name'] for s in specs],work/f'comparisons/{split}',split=split)
        save(work/'progress.json',dict(stage='complete',experiments=[s['name'] for s in specs],total_planned=total))
        print('COMPLETE:',work/'comparisons',flush=True)
    except BaseException as e:
        for c in children:
            if c.poll() is None:c.terminate()
        save(work/'progress.json',dict(stage='interrupted' if isinstance(e,KeyboardInterrupt) else 'error',message=str(e)));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
