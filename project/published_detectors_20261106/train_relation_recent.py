#!/usr/bin/env python3
"""Native unsupervised detectors on the locked relation four-case AIS cohort.

The published models do not accept temporal masks. Resample each valid time
interval to 20 points, preserving all source quartets. Train on controlled
label 0 in the base view; score windows without point adjustment.
"""
import os, sys, argparse, importlib, json, random, hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
sys.path.insert(0,str(ROOT/'heading_alignment_20261030'))
import train_heading as base
base.VARIANT='aligned_embed'
# train_heading imports a local module called models.py; TFT-GCN has its own
# package named models. Clear the cached name before importing that vendor.
sys.modules.pop('models', None)
sys.path[:]=[p for p in sys.path if p != str(ROOT/'framework_validation_v1_20261028')]
VIEWS=('base','natural','suffix','combined_l100_m600')

def read(split):
    if split!='test': d=base.read(split)
    else:
        with np.load(ROOT/'formal_20261101/relation/test_relation_field.npz',allow_pickle=False) as z:
            d={k:z[k] for k in z.files}
        d['mask']=np.arange(20)[None,:] < d['length'][:,None]
    ids,counts=np.unique(d['source'],return_counts=True)
    if not np.all(counts==4): raise ValueError('source must have exactly four cases')
    for src in ids:
        j=d['source']==src
        if sorted(d['case'][j].tolist())!=[0,1,2,3] or np.sum(d['y'][j])!=2:
            raise ValueError('invalid four-case label structure')
        lengths=d['length'][j]
        if not np.all(lengths==lengths[0]): raise ValueError('quartet lengths differ')
        t=d['time_s'][j,:int(lengths[0])]
        if not np.allclose(t,t[:1],rtol=0,atol=1e-6): raise ValueError('quartet time stamps differ')
        if not np.all(np.diff(t,axis=1)>0): raise ValueError('time stamps must strictly increase')
    return d

def evaluate(model,x,d,device,batch):
    model.eval(); scores=[]
    with torch.no_grad():
        for s in range(0,len(x),batch):
            xx=torch.as_tensor(x[s:s+batch],dtype=torch.float32,device=device)
            scores.extend(model.score(xx).detach().cpu().numpy().tolist())
    scores=np.asarray(scores)
    if not np.isfinite(scores).all(): raise ValueError('nonfinite score')
    return scores,{'pooled_AP':float(average_precision_score(d['y'],scores)),
      'AUROC':float(roc_auc_score(d['y'],scores)),
      'normal_mean_score':float(scores[d['y']==0].mean())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['tft_gcn','simad','dmrad'])
    p.add_argument('--arm',required=True,choices=['C0','F0','F1']);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=.0001);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--mode',choices=['train','test'],default='train');p.add_argument('--smoke',action='store_true')
    p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107')
    a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(2);dev=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    if a.method=='tft_gcn' and dev.type!='cuda': raise RuntimeError('upstream TFT-GCN graph requires CUDA')
    tr=read('train');va=read('val')
    if set(tr['mmsi']) & set(va['mmsi']): raise ValueError('target MMSI overlap')
    val_mmsi=set(va['mmsi'])
    # Fit original 35D normalisation on genuine training steps, no padding.
    mu,sd=base.fit(tr)
    def prep(d,view):
        vals=base.prep(d,view,a.arm,mu,sd,'cpu')[0].numpy()
        _,valid=base.features(d,view,a.arm)
        out=np.empty_like(vals)
        for i,n in enumerate(d['length']):
            t=d['time_s'][i,:n].astype(np.float64);new=np.linspace(t[0],t[-1],20)
            for c in range(vals.shape[2]):out[i,:,c]=np.interp(new,t,vals[i,:n,c])
            # A context estimate exists only at an available source step or
            # between TWO consecutive available source steps. Never bridge an
            # unavailable step or interpolate padded zero into the field.
            right=np.searchsorted(t,new,side='left');safe=np.clip(right,0,n-1)
            exact=(right<n) & (np.abs(t[safe]-new)<1e-6)
            left=np.clip(right-1,0,n-1)
            between=(right>0)&(right<n)&valid[i,left]&valid[i,safe]
            ok=np.where(exact,valid[i,safe],between)
            out[i,~ok,base.TARGET:]=0.
            if np.any(out[i,~ok,base.TARGET:] != 0): raise AssertionError('invalid context interpolation')
        return out
    xi=prep(tr,'base');nidx=np.flatnonzero(tr['y']==0)
    if a.smoke: nidx=nidx[:64]
    x=xi[nidx]; xv=prep(va,'base')
    if a.smoke:
        va={k:v[:64] for k,v in va.items()};xv=xv[:64]
    net=importlib.import_module('adapters.'+a.method).Adapter(seq_len=20,channels=35,gpu=a.gpu).to(dev)
    out=Path(a.out)/('relation_smoke' if a.smoke else 'relation_pilot')/a.method/a.arm/f'seed{a.seed}'
    if a.mode=='test':
        ck=torch.load(out/'best.pt',map_location=dev,weights_only=False);net.load_state_dict(ck['model'])
        te=read('test')
        if set(te['mmsi']) & (set(tr['mmsi'])|val_mmsi): raise ValueError('target MMSI overlap test')
        scores={'y':te['y'],'source':te['source'],'case':te['case'],'mmsi':te['mmsi'],'region':te['region']}; results={}
        for view in VIEWS:
            scores[view],results[view]=evaluate(net,prep(te,view),te,dev,a.batch_size)
        np.savez_compressed(out/'test_scores.npz',**scores)
        (out/'test_metrics.json').write_text(json.dumps(results,indent=2),encoding='utf8');print('TEST',json.dumps(results),flush=True);return
    if out.exists() and not a.smoke: raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    spec={'method':a.method,'arm':a.arm,'seed':a.seed,'epochs':a.epochs,'train_normal_windows':len(nidx),'train_sources':len(np.unique(tr['source'])),'validation_sources':len(np.unique(va['source'])),'feature_channels':35,'train_view':'base','source_selection':'all four cases; all original lengths','time_resampling':'20 linearly spaced actual-time points; context zero unless exact valid original or between adjacent valid originals; no gap bridging','normalization':'train only via original 35D relation features before resampling','model_score':'original per-time anomaly score averaged across 20 timestamps','model_selection':'mean score on validation controlled label=0','test':'previously exposed; evaluate separately','scope':'unsupervised adaptation, different objective from supervised historical relation models'}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8');history=[];best=float('inf')
    for epoch in range(a.epochs):
        if hasattr(net,'begin_epoch'):net.begin_epoch(epoch,a.epochs)
        net.train();order=np.random.default_rng(a.seed+10000*epoch).permutation(len(x));losses=[]
        bar=tqdm(range(0,len(order),a.batch_size),file=sys.stdout,desc=f'{a.method}/{a.arm} 轮次 {epoch+1}/{a.epochs}',dynamic_ncols=True)
        for s in bar:
            ids=order[s:s+a.batch_size];batch=torch.as_tensor(x[ids],dtype=torch.float32,device=dev)
            opt.zero_grad(set_to_none=True);loss=net.loss(batch,indices=ids,epoch=epoch)
            if not torch.isfinite(loss): raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();losses.append(float(loss.detach()));bar.set_postfix(loss=f'{losses[-1]:.5f}')
        pred,m=evaluate(net,xv,va,dev,a.batch_size);hist={'epoch':epoch+1,'train_loss':float(np.mean(losses)),**m};history.append(hist)
        ck={'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'spec':spec,'mu':mu,'sd':sd,'metrics':m}
        torch.save(ck,out/'last.pt')
        if m['normal_mean_score']<best:
            best=m['normal_mean_score'];torch.save(ck,out/'best.pt')
            np.savez_compressed(out/'val_scores.npz',score=pred,y=va['y'],source=va['source'],case=va['case'],mmsi=va['mmsi'])
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8');print('VALIDATION',json.dumps(hist),flush=True)

if __name__=='__main__': main()
