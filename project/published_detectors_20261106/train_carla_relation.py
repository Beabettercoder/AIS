#!/usr/bin/env python3
"""CARLA two-stage AIS relation pilot; same four-cell sources and views.

This is an AIS-specific adaptation of the author network and losses. Each
source remains within its target-MMSI split. No test scores choose checkpoints.
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from train_relation_recent import read, base, VIEWS
from train_carla import embeddings, neighbors
from adapters.carla import Adapter

def prep(d,view,arm,mu,sd):
    vals=base.prep(d,view,arm,mu,sd,'cpu')[0].numpy()
    _,valid=base.features(d,view,arm)
    out=np.empty_like(vals)
    for i,n0 in enumerate(d['length']):
        n=int(n0);t=d['time_s'][i,:n].astype(np.float64);new=np.linspace(t[0],t[-1],20)
        for c in range(vals.shape[2]):out[i,:,c]=np.interp(new,t,vals[i,:n,c])
        right=np.searchsorted(t,new,side='left');safe=np.clip(right,0,n-1)
        exact=(right<n)&(np.abs(t[safe]-new)<1e-6);left=np.clip(right-1,0,n-1)
        between=(right>0)&(right<n)&valid[i,left]&valid[i,safe]
        ok=np.where(exact,valid[i,safe],between)
        out[i,~ok,base.TARGET:]=0.
    if not np.isfinite(out).all():raise ValueError('nonfinite features')
    return out

def previous_windows(d):
    # Time must refer to the last VALID observation, never the padded slot.
    lengths=np.asarray(d['length'],dtype=int)
    times=np.asarray(d['time_s'])[np.arange(len(lengths)),lengths-1]
    ids=np.full(len(times),-1,dtype=int)
    for vessel in np.unique(d['mmsi']):
        idx=np.flatnonzero(d['mmsi']==vessel);idx=idx[np.argsort(times[idx],kind='stable')]
        for j in range(1,len(idx)):
            candidates=idx[max(0,j-10):j];candidates=candidates[times[candidates]<times[idx[j]]]
            if len(candidates):ids[idx[j]]=np.random.choice(candidates)
    return ids

def positive(x,prev,ix,device):
    v=torch.as_tensor(x[ix],dtype=torch.float32,device=device);pos=v+torch.randn_like(v)*.01
    has=prev[ix]>=0
    if has.any():pos[torch.as_tensor(has,device=device)]=torch.as_tensor(x[prev[ix[has]]],dtype=torch.float32,device=device)
    return v,pos

@torch.no_grad()
def scores(model,x,device,batch):
    model.eval();r=[]
    for s in range(0,len(x),batch):r.extend(model.score(torch.as_tensor(x[s:s+batch],dtype=torch.float32,device=device)).cpu().numpy())
    r=np.asarray(r)
    if not np.isfinite(r).all():raise ValueError('nonfinite scores')
    return r

def metrics(d,s):return {'AP':float(average_precision_score(d['y'],s)),'AUROC':float(roc_auc_score(d['y'],s))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['C0','F0','F1'],required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda:0')
    p.add_argument('--pretext-epochs',type=int,default=25);p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=50);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--smoke',action='store_true');p.add_argument('--mode',choices=['train','test'],default='train')
    p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107');a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_num_threads(2)
    dev=torch.device(a.device);tr=read('train');va=read('val')
    if set(tr['mmsi']) & set(va['mmsi']):raise ValueError('MMSI train/val overlap')
    mu,sd=base.fit(tr);xx=prep(tr,'base',a.arm,mu,sd);xv=prep(va,'base',a.arm,mu,sd)
    idx=np.flatnonzero(tr['y']==0)
    if a.smoke:idx=idx[:64]
    xx=xx[idx];tn={k:v[idx] for k,v in tr.items()};prev=previous_windows(tn)
    vi=np.flatnonzero(va['y']==0)
    if a.smoke:vi=vi[:64]
    vn={k:v[vi] for k,v in va.items()};xvn=xv[vi];vprev=previous_windows(vn)
    out=Path(a.out)/('carla_relation_smoke' if a.smoke else 'carla_relation_pilot')/a.arm/f'seed{a.seed}'
    model=Adapter(20,35).to(dev)
    if a.mode=='test':
        ck=torch.load(out/'best.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['model']);model.begin_clustering()
        te=read('test')
        if set(te['mmsi'])&(set(tr['mmsi'])|set(va['mmsi'])):raise ValueError('MMSI test overlap')
        saved={k:te[k] for k in ('y','source','case','mmsi','region')};results={}
        for view in VIEWS:
            saved[view]=scores(model,prep(te,view,a.arm,mu,sd),dev,a.batch_size);results[view]=metrics(te,saved[view])
        np.savez_compressed(out/'test_scores.npz',**saved);(out/'test_metrics.json').write_text(json.dumps(results,indent=2),encoding='utf8')
        print('TEST',json.dumps(results),flush=True);return
    out.mkdir(parents=True,exist_ok=False)
    spec={'arm':a.arm,'seed':a.seed,'train_sources':int(len(np.unique(tr['source']))),'val_sources':int(len(np.unique(va['source']))),'train_normal_windows':len(xx),'train_view':'base','evaluation_views':VIEWS,'input':'35D stage cumulative causal target and LOO context, 20 equally spaced observed-time points; no unsupported context interpolation','phase1':'author PretextLoss; earlier same MMSI valid end only, Gaussian fallback','phase2':'author clustering loss; frozen pretext embedding, 2N normal+synthetic neighbors; validation normal clustering loss selection','score':'one minus training-majority cluster softmax','layout':'time-channel transpose for AIS adaptation','test':'previously exposed; separate execution'}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    np.savez_compressed(out/'normalizer.npz',mu=mu,sd=sd)
    print('DATA',json.dumps(spec,ensure_ascii=False),'positives',int((prev>=0).sum()),flush=True)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=.001);prebest=float('inf')
    for epoch in range(a.pretext_epochs):
        model.train();losses=[];order=np.random.permutation(len(xx));bar=tqdm(range(0,len(order),a.batch_size),file=sys.stdout,desc=f'预训练 {epoch+1}/{a.pretext_epochs}')
        for start in bar:
            ix=order[start:start+a.batch_size];x,pos=positive(xx,prev,ix,dev);opt.zero_grad(set_to_none=True)
            loss=model.loss(x,positive=pos)
            if not torch.isfinite(loss):raise ValueError('nonfinite pretext loss')
            loss.backward();opt.step();losses.append(float(loss.detach()));bar.set_postfix(loss=np.mean(losses))
        model.eval();vls=[]
        with torch.no_grad():
            for st in range(0,len(xvn),a.batch_size):
                ii=np.arange(st,min(st+a.batch_size,len(xvn)));x,pos=positive(xvn,vprev,ii,dev)
                z=model.embed(torch.cat([x,pos,model.synthetic(x)],0));model.pretext_loss.bs=len(ii);vls.append(float(model.pretext_loss(z)))
        ck={'model':model.state_dict(),'epoch':epoch+1,'optimizer':opt.state_dict(),'val_loss':float(np.mean(vls)),'margin':model.pretext_loss.margin,'prev_loss':model.prev_loss}
        torch.save(ck,out/'last.pt')
        if ck['val_loss']<prebest:prebest=ck['val_loss'];torch.save(ck,out/'best.pt')
        print(f'预训练 {epoch+1}: loss={np.mean(losses):.5f} val_loss={ck["val_loss"]:.5f}',flush=True)
    ck=torch.load(out/'best.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['model']);model.prev_loss=ck['prev_loss'];model.pretext_loss.margin=ck['margin']
    augmented=[]
    for st in range(0,len(xx),a.batch_size):augmented.append(model.synthetic(torch.as_tensor(xx[st:st+a.batch_size],device=dev)).cpu().numpy())
    cluster_x=np.concatenate([xx,*augmented],axis=0);z=embeddings(model,cluster_x,dev,a.batch_size);n,f=neighbors(z,z,same=True)
    zv=embeddings(model,xvn,dev,a.batch_size);vnear,vfar=neighbors(zv,z)
    model.begin_clustering();opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=.001);best=float('inf');history=[]
    for epoch in range(a.epochs):
        model.train();ls=[];order=np.random.permutation(len(cluster_x));bar=tqdm(range(0,len(order),a.batch_size),file=sys.stdout,desc=f'聚类 {epoch+1}/{a.epochs}')
        for st in bar:
            ii=order[st:st+a.batch_size];nn=n[ii,np.random.randint(n.shape[1],size=len(ii))];ff=f[ii,np.random.randint(f.shape[1],size=len(ii))]
            x=torch.as_tensor(cluster_x[ii],device=dev);near=torch.as_tensor(cluster_x[nn],device=dev);far=torch.as_tensor(cluster_x[ff],device=dev)
            opt.zero_grad(set_to_none=True);loss=model.loss(x,near=near,far=far)
            if not torch.isfinite(loss):raise ValueError('nonfinite cluster loss')
            loss.backward();opt.step();ls.append(float(loss.detach()));bar.set_postfix(loss=np.mean(ls))
        model.eval();vls=[]
        with torch.no_grad():
            for st in range(0,len(xvn),a.batch_size):
                end=min(st+a.batch_size,len(xvn));x=torch.as_tensor(xvn[st:end],device=dev)
                vls.append(float(model.loss(x,near=torch.as_tensor(cluster_x[vnear[st:end,0]],device=dev),far=torch.as_tensor(cluster_x[vfar[st:end,0]],device=dev))))
            logits=[]
            for st in range(0,len(xx),a.batch_size):logits.append(model.cluster(model._reshape(torch.as_tensor(xx[st:st+a.batch_size],device=dev)))[0].argmax(1).cpu())
            model.normal_cluster.copy_(torch.bincount(torch.cat(logits),minlength=10).argmax().to(dev))
        val=metrics(va,scores(model,xv,dev,a.batch_size));vl=float(np.mean(vls));history.append({'epoch':epoch+1,'loss':float(np.mean(ls)),'val_loss':vl,**val})
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'val_loss':vl,'metrics':val}
        torch.save(ck,out/'last.pt')
        if vl<best:best=vl;torch.save(ck,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8');print('VALIDATION',json.dumps(history[-1]),flush=True)

if __name__=='__main__':main()
