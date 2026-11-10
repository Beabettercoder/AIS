"""Forecasting engine v2: dynamic history, paired capacity, cosine LR, val-only selection.

No test file is accessed by train(). Test evaluation requires a checkpoint freeze.
"""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'frozen_pilot'))
from forecast_models import ForecastModel,state_hash,MODEL_NAMES,ARMS
import train as legacy

VIEWS=('base','target_jump','target_drift','target_speed','neighbor_stress')
ARM_ZH={'P0':'原始输入','P1':'重锚运动学门控','P2':'门控＋普通场','P3':'门控＋可信场'}

def sha(path):return legacy.digest(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,data):legacy.atomic_json(Path(path),data)
def stamp():return legacy.utc()
def source_hashes():
    return {p.name:sha(p) for p in [Path(__file__),ROOT/'frozen_pilot/forecast_models.py',ROOT/'frozen_pilot/vendor_encoders.py',ROOT/'frozen_pilot/train.py']}

def load_data(path,training=False):
    path=Path(path);views=('base',) if training else VIEWS
    common=['future_xy','query_s','time_s','mmsi','region','window_id','maneuver']
    fields=['raw','gated','f0','f1','f0_valid','f1_valid','anchor_raw','anchor_gate','vel_raw','vel_gate']
    keys=common+[v+'__'+k for v in views for k in fields]
    with np.load(path,allow_pickle=False) as z:
        missing=set(keys)-set(z.files)
        if missing:raise ValueError(f'Missing keys in {path}: {sorted(missing)}')
        d={k:z[k] for k in keys}
    n=len(d['window_id']);h=d['time_s'].shape[1];k=d['query_s'].shape[1]
    if n==0 or len(np.unique(d['window_id']))!=n:raise ValueError('Empty/duplicate windows')
    if d['future_xy'].shape!=(n,k,2) or d['future_xy'].dtype!=np.float64:raise ValueError('Future reference must be float64 [N,K,2]')
    if not np.isfinite(d['future_xy']).all() or not np.isfinite(d['query_s']).all():raise ValueError('Invalid reference/query values')
    if np.any(d['query_s']<=0) or np.any(np.diff(d['query_s'],axis=1)<=0) or np.any(np.diff(d['time_s'],axis=1)<=0):raise ValueError('Non-causal time order')
    if h<4 or k<1:raise ValueError('History >=4, future >=1 required')
    for v in views:
        expected={'raw':(n,h,9),'gated':(n,h,9),'f0':(n,h,8),'f1':(n,h,8),'f0_valid':(n,h),'f1_valid':(n,h),
                  'anchor_raw':(n,2),'anchor_gate':(n,2),'vel_raw':(n,2),'vel_gate':(n,2)}
        for name,shape in expected.items():
            a=d[v+'__'+name]
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'Bad {v}/{name}')
            if name.startswith('anchor') and a.dtype!=np.float64:raise ValueError('Anchor must remain float64')
    return d

def prepare(d,view,arm,norm):
    raw=arm=='P0';n,h,_=d[view+'__raw'].shape
    x=(d[view+('__raw' if raw else '__gated')].astype(float)-np.asarray(norm['self_mean']))/np.asarray(norm['self_std'])
    if arm in ('P2','P3'):
        fn='f0' if arm=='P2' else 'f1';orig=d[view+'__'+fn];valid=d[view+'__'+fn+'_valid'].astype(bool)
        field=(orig.astype(float)-np.asarray(norm['field_mean']))/np.asarray(norm['field_std']);field[~valid]=0
        rho=orig[...,6].copy();rho[~valid]=0
    else:field=np.zeros((n,h,8));rho=np.zeros((n,h))
    return dict(x=torch.from_numpy(x.astype('f4')),field=torch.from_numpy(field.astype('f4')),rho=torch.from_numpy(rho.astype('f4')),
        query_s=torch.from_numpy(d['query_s'].astype('f4')),times=torch.from_numpy(d['time_s'].astype('f4')),
        anchor=torch.from_numpy(d[view+('__anchor_raw' if raw else '__anchor_gate')].astype('f8')),
        velocity=torch.from_numpy(d[view+('__vel_raw' if raw else '__vel_gate')].astype('f4')),
        target=torch.from_numpy(d['future_xy'].astype('f8')))

def forward(model,tensors,idx,device,arm):
    b={k:v[idx].to(device) for k,v in tensors.items()}
    pred=model(b['x'],b['field'],b['rho'],b['query_s'],b['anchor'],b['velocity'],arm in ('P2','P3'),times=b['times'])
    return pred,b['target']

@torch.inference_mode()
def infer(model,tensors,device,arm,batch):
    model.eval();pred=[]
    for i in range(0,len(tensors['x']),batch):pred.append(forward(model,tensors,slice(i,i+batch),device,arm)[0].cpu().numpy())
    pred=np.concatenate(pred);err=np.linalg.norm(pred-tensors['target'].numpy(),axis=-1)
    if not np.isfinite(err).all():raise FloatingPointError('Non-finite forecast error')
    return pred,err

def metrics(err,d):
    result={'ADE_m':float(err.mean()),'FDE_m':float(err[:,-1].mean()),'windows':len(err),'target_mmsi':len(np.unique(d['mmsi']))}
    regions=np.unique(d['region']);result['region_macro_ADE_m']=float(np.mean([err[d['region']==r].mean() for r in regions]))
    result['region_macro_FDE_m']=float(np.mean([err[d['region']==r,-1].mean() for r in regions]))
    m=d['maneuver'].astype(bool)
    result['observed_maneuver']={'windows':int(m.sum()),'ADE_m':float(err[m].mean()) if m.any() else None,'FDE_m':float(err[m,-1].mean()) if m.any() else None}
    return result

def persist(folder,view,pred,err,d,settings,epoch,split='val',prefix='predictions'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/f'{prefix}_{view}.npz';tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,pred_xy=pred,future_xy=d['future_xy'],point_error=err,ADE=err.mean(1),FDE=err[:,-1],query_s=d['query_s'],
            mmsi=d['mmsi'],region=d['region'],window_id=d['window_id'],maneuver=d['maneuver'],split=split,view=view,
            model=settings['model'],arm=settings['arm'],seed=settings['seed'],hidden=settings['hidden'],policy=settings['policy'],selected_epoch=epoch)
    tmp.replace(path)

def configure(device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    return device

def checkpoint(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);legacy.save_checkpoint(path,payload)

def best_payload(model_state,best,config_hash,initial_hash,settings):
    return dict(model=model_state,completed_epoch=best['epoch'],best_val=copy.deepcopy(best),
                config_hash=config_hash,initial_model_hash=initial_hash,settings=settings)

def train(data_dir,out,model,seed,arm,hidden=64,batch=128,epochs=25,lr=.002,policy='cosine_best',norm_path=None,device='cuda:0'):
    data_dir=Path(data_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if model not in MODEL_NAMES or arm not in ARMS or hidden not in (64,128,256) or policy not in ('cosine_best','constant_last'):raise ValueError('Unsupported model/arm/hidden/policy')
    if epochs<1 or batch<1 or lr<=0:raise ValueError('Positive epochs/batch/lr required')
    if norm_path is None:norm_path=ROOT/'work/data/normalizer.json'
    norm_path=Path(norm_path);norm=read(norm_path)
    tr=load_data(data_dir/'train.npz',True);va=load_data(data_dir/'val.npz',False)
    if set(tr['mmsi'])&set(va['mmsi']):raise ValueError('Training/validation target identities overlap')
    h=tr['time_s'].shape[1];k=tr['query_s'].shape[1]
    if va['time_s'].shape[1]!=h or va['query_s'].shape[1]!=k:raise ValueError('Train/val horizon mismatch')
    settings=dict(model=model,seed=int(seed),arm=arm,hidden=int(hidden),history_steps=h,future_steps=k,batch_size=int(batch),epochs=int(epochs),lr=lr,
        policy=policy,optimizer='AdamW',weight_decay=1e-4,gradient_clip=1.,
        scheduler='CosineAnnealingLR' if policy=='cosine_best' else 'none',eta_min=lr*.01 if policy=='cosine_best' else lr,
        selection='minimum base-validation ADE, earliest exact tie' if policy=='cosine_best' else 'fixed final epoch',
        norm_sha256=sha(norm_path),normalizer_content_hash=legacy.object_hash(norm),data_sha256={s:sha(data_dir/f'{s}.npz') for s in ['train','val']},source_sha256=source_hashes(),
        torch_version=torch.__version__,numpy_version=np.__version__,device=str(device))
    config_hash=legacy.object_hash(settings)
    old_protocol=out/'protocol.json'
    if old_protocol.exists() and read(old_protocol)['config_hash']!=config_hash:raise RuntimeError(f'Configuration/data/code changed: {out}; use a new run directory')
    if (out/'result.json').exists():
        result=read(out/'result.json')
        if result.get('status')=='complete':
            if sha(result['selected_checkpoint'])!=result['selected_checkpoint_sha256']:raise RuntimeError('Selected checkpoint modified')
            print(f'SKIP completed {model} seed={seed} {arm} h={hidden} {policy}',flush=True);return result
    device=configure(device);lock=legacy.acquire_lock(out);start=time.monotonic()
    try:
        legacy.seed_all(seed);net=ForecastModel(model,history_steps=h,hidden=hidden);init=state_hash(net)
        save(old_protocol,dict(**settings,config_hash=config_hash,initial_model_hash=init,parameter_count=sum(p.numel() for p in net.parameters())))
        save(out/'normalizer.json',norm);net.to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=lr*.01) if policy=='cosine_best' else None
        history=[];completed=0;best={'epoch':0,'ADE_m':None,'FDE_m':None};best_state=None
        lastpath=out/'last_checkpoint.pt';bestpath=out/'best_val_checkpoint.pt'
        if lastpath.exists():
            ck=torch.load(lastpath,map_location=device,weights_only=False)
            if ck['config_hash']!=config_hash or ck['initial_model_hash']!=init:raise RuntimeError('Resume checkpoint version mismatch')
            net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
            if scheduler is not None:scheduler.load_state_dict(ck['scheduler'])
            legacy.restore_rng(ck['rng'],device);completed=ck['completed_epoch'];history=ck['history'];best=ck['best_val']
            best_state={k:v.detach().cpu().clone() for k,v in ck['best_model'].items()}
            # The complete last checkpoint is authoritative after interruption between file writes.
            checkpoint(bestpath,best_payload(best_state,best,config_hash,init,settings))
            print(f'RESUME {model} s{seed} {arm}: {completed}/{epochs}',flush=True)
        progress=dict(model=model,seed=seed,arm=arm,hidden=hidden,policy=policy,stage='training',epoch=completed,completed_epochs=completed,total_epochs=epochs,
                      total_batches=math.ceil(len(tr['window_id'])/batch),batch=0)
        def status(**kw):
            progress.update(kw);progress['updated_utc']=stamp();progress['elapsed_seconds']=time.monotonic()-start;save(out/'progress.json',progress)
        trt=prepare(tr,'base',arm,norm);vat=prepare(va,'base',arm,norm)
        status();legacy.event(out,'start',config_hash=config_hash,resume_epoch=completed)
        for ep in tqdm(range(completed,epochs),initial=completed,total=epochs,desc=f'{model} s{seed} {arm} h{hidden}',unit='epoch'):
            net.train();order=np.random.default_rng(300000+seed*10000+ep).permutation(len(tr['window_id']));losses=[];counts=[]
            current_lr=float(opt.param_groups[0]['lr']);last_write=0.
            for bi,i in enumerate(tqdm(range(0,len(order),batch),desc=f' epoch {ep+1}/{epochs}',leave=False,unit='batch'),1):
                ix=order[i:i+batch];opt.zero_grad(set_to_none=True);pred,truth=forward(net,trt,ix,device,arm)
                loss=F.smooth_l1_loss(pred/1000.,truth/1000.,beta=.1)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
                losses.append(float(loss.detach()));counts.append(len(ix));mean_loss=float(np.average(losses,weights=counts))
                if time.monotonic()-last_write>=1 or i+batch>=len(order):status(stage='training',epoch=ep+1,batch=bi,train_loss=mean_loss,learning_rate=current_lr);last_write=time.monotonic()
            status(stage='validation');_,err=infer(net,vat,device,arm,batch);val=metrics(err,va)
            row=dict(epoch=ep+1,lr=current_lr,train_loss=mean_loss,val_ADE_m=val['ADE_m'],val_FDE_m=val['FDE_m'])
            history.append(row);improved=best['ADE_m'] is None or val['ADE_m']<best['ADE_m']
            if improved:
                best=dict(epoch=ep+1,ADE_m=val['ADE_m'],FDE_m=val['FDE_m'])
                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            if scheduler is not None:scheduler.step()
            payload=dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
                rng=legacy.rng_state(device),completed_epoch=ep+1,history=history,best_val=copy.deepcopy(best),best_model=best_state,
                config_hash=config_hash,initial_model_hash=init,settings=settings)
            checkpoint(lastpath,payload)
            if improved:checkpoint(bestpath,best_payload(best_state,best,config_hash,init,settings))
            checkpoint(out/'weights'/f'epoch_{ep+1:03d}.pt',dict(model=net.state_dict(),epoch=ep+1,validation=row,config_hash=config_hash,initial_model_hash=init))
            with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:wr=csv.DictWriter(f,list(row));wr.writeheader();wr.writerows(history)
            legacy.event(out,'epoch_complete',**row,best_epoch=best['epoch'],best_val_ADE_m=best['ADE_m'],best_updated=improved)
            status(stage='epoch_complete',completed_epochs=ep+1,val_ADE_m=val['ADE_m'],val_FDE_m=val['FDE_m'],best_val=best)
            print(f'{model} s{seed} {ARM_ZH[arm]} h={hidden} | {ep+1}/{epochs} lr={current_lr:.7f} loss={mean_loss:.6f} | val ADE={val["ADE_m"]:.2f}m FDE={val["FDE_m"]:.2f}m | best epoch {best["epoch"]} ADE={best["ADE_m"]:.2f}m',flush=True)
        last_epoch=dict(epoch=epochs,ADE_m=history[-1]['val_ADE_m'],FDE_m=history[-1]['val_FDE_m'],checkpoint=str(lastpath.resolve()),sha256=sha(lastpath))
        best_val={**best,'checkpoint':str(bestpath.resolve()),'sha256':sha(bestpath)}
        selected=bestpath if policy=='cosine_best' else lastpath;selected_epoch=best['epoch'] if policy=='cosine_best' else epochs
        out_metrics={};last_metrics={}
        # Both records are exported explicitly; last files are never confused with selected files.
        for which,path,prefix,epoch,bank in [('selected',selected,'predictions',selected_epoch,out_metrics),('last_epoch',lastpath,'last_epoch_predictions',epochs,last_metrics)]:
            net.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model']);base=None
            for view in tqdm(VIEWS,desc=f'{model} {arm} {which} validation',unit='view',leave=False):
                status(stage='validation_export',view=view,checkpoint_choice=which)
                pred,err=infer(net,prepare(va,view,arm,norm),device,arm,batch)
                if view=='base':base=pred.copy()
                if view=='neighbor_stress' and arm in ('P0','P1'):np.testing.assert_array_equal(pred,base)
                persist(out,view,pred,err,va,settings,epoch,'val',prefix);bank[view]=metrics(err,va)
        result=dict(status='complete',split='val',model=model,arm=arm,seed=seed,hidden=hidden,policy=policy,settings=settings,
            config_hash=config_hash,initial_model_hash=init,parameter_count=sum(p.numel() for p in net.parameters()),completed_epochs=epochs,expected_epochs=epochs,
            best_val=best_val,last_epoch=last_epoch,selected_epoch=selected_epoch,selected_checkpoint=str(selected.resolve()),
            selected_checkpoint_sha256=sha(selected),views=out_metrics,last_epoch_views=last_metrics,elapsed_seconds=time.monotonic()-start)
        save(out/'result.json',result);status(stage='complete',best_val=best_val,last_epoch=last_epoch);legacy.event(out,'complete',selected_epoch=selected_epoch)
        return result
    except BaseException as exc:
        event_kind='interrupted' if isinstance(exc,KeyboardInterrupt) else 'error'
        legacy.event(out,event_kind,message=str(exc))
        if 'status' in locals():status(stage=event_kind,message=str(exc))
        raise
    finally:lock.unlink(missing_ok=True)

def evaluate(data_dir,out,model_run,split='test',device='cuda:0',freeze_path=None):
    data_dir=Path(data_dir);out=Path(out);model_run=Path(model_run);out.mkdir(parents=True,exist_ok=True)
    result=read(model_run/'result.json');assert result['status']=='complete'
    checkpoint_path=Path(result['selected_checkpoint'])
    if sha(checkpoint_path)!=result['selected_checkpoint_sha256']:raise RuntimeError('Selected checkpoint changed')
    if split=='test':
        if freeze_path is None or not Path(freeze_path).is_file():raise RuntimeError('Test requires checkpoint freeze')
        freeze=read(freeze_path)
        matched=[x for x in freeze.get('models',[]) if x.get('selected_checkpoint_sha256')==result['selected_checkpoint_sha256']
                 and x.get('model')==result['model'] and x.get('seed')==result['seed'] and x.get('arm')==result['arm']]
        if len(matched)!=1:raise RuntimeError('Exactly one matching model/arm/seed/checkpoint entry is required in freeze record')
        if matched[0].get('selected_checkpoint') and Path(matched[0]['selected_checkpoint']).resolve()!=checkpoint_path.resolve():
            raise RuntimeError('Selected checkpoint path differs from freeze entry')
    norm=read(model_run/'normalizer.json');settings=result['settings'];d=load_data(data_dir/f'{split}.npz')
    if legacy.object_hash(norm)!=settings['normalizer_content_hash']:raise RuntimeError('Frozen normalization content changed')
    if d['time_s'].shape[1]!=settings['history_steps'] or d['query_s'].shape[1]!=settings['future_steps']:raise ValueError('Evaluation horizon mismatch')
    identity=dict(model_selected_sha256=sha(checkpoint_path),data_sha256=sha(data_dir/f'{split}.npz'),norm_sha256=sha(model_run/'normalizer.json'),
                  freeze_sha256=sha(freeze_path) if freeze_path else None,split=split,source_sha256=source_hashes())
    marker=out/'evaluation_plan.json'
    if marker.exists() and read(marker)!=identity:raise RuntimeError('Frozen evaluation inputs changed')
    save(marker,identity)
    if (out/'result.json').exists():
        prior=read(out/'result.json')
        if prior.get('status')=='complete':return prior
    dev=configure(device);lock=legacy.acquire_lock(out)
    try:
        model=ForecastModel(result['model'],history_steps=settings['history_steps'],hidden=result['hidden']).to(dev)
        ck=torch.load(checkpoint_path,map_location=dev,weights_only=False);model.load_state_dict(ck['model']);views={};base=None
        for view in tqdm(VIEWS,desc=f'{split} {result["model"]} {result["arm"]} s{result["seed"]}',unit='view'):
            pred,err=infer(model,prepare(d,view,result['arm'],norm),dev,result['arm'],settings['batch_size'])
            if view=='base':base=pred.copy()
            if view=='neighbor_stress' and result['arm'] in ('P0','P1'):np.testing.assert_array_equal(pred,base)
            persist(out,view,pred,err,d,settings,result['selected_epoch'],split);views[view]=metrics(err,d)
        tested={**result,'split':split,'views':views,'last_epoch_views':None,'evaluation_identity':identity,
                'selection_records_note':'best_val and last_epoch refer to base-validation, never test-based selection'}
        save(out/'result.json',tested);return tested
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=MODEL_NAMES,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--hidden-dim',type=int,choices=[64,128,256],default=64);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--epochs',type=int);p.add_argument('--lr',type=float,default=.002);p.add_argument('--policy',choices=['cosine_best','constant_last'],default='cosine_best')
    p.add_argument('--norm',type=Path);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    train(a.data,a.out,a.model,a.seed,a.arm,a.hidden_dim,a.batch_size,a.epochs or (75 if a.model=='itransformer' else 25),a.lr,a.policy,a.norm,a.device)
