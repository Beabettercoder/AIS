"""Run author CARLA's pretext + clustering on the locked AIS single-vessel task."""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import data
from adapters.carla import Adapter

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def previous_windows(d):
    # Restrict time-positive windows to earlier observations from the same vessel.
    # Earlier means a strictly earlier window end; equal-time ties never qualify.
    times=np.asarray(d['time_s'])[:,-1]
    ids=np.full(len(times),-1,dtype=int)
    for m in np.unique(d['mmsi']):
        idx=np.flatnonzero(d['mmsi']==m); idx=idx[np.argsort(times[idx],kind='stable')]
        for j in range(1,len(idx)):
            candidates=idx[max(0,j-10):j]
            candidates=candidates[times[candidates]<times[idx[j]]]
            if len(candidates): ids[idx[j]]=np.random.choice(candidates)
    return ids

@torch.no_grad()
def embeddings(m,x,device,batch):
    m.eval(); z=[]
    for i in range(0,len(x),batch): z.append(m.embed(torch.as_tensor(x[i:i+batch],device=device)).cpu())
    return torch.cat(z)

def neighbors(zq,zb,same=False,k=5):
    if len(zb)<k+1: raise ValueError('not enough training windows for CARLA neighbors')
    near=[];far=[]
    for st in range(0,len(zq),256):
        dist=torch.cdist(zq[st:st+256],zb)
        if same:
            r=torch.arange(len(dist));dist[r,r+st]=float('inf')
        near.append(torch.topk(dist,k,largest=False).indices.numpy())
        if same:
            dist[r,r+st]=float('-inf')
        far.append(torch.topk(dist,k,largest=True).indices.numpy())
    return np.concatenate(near),np.concatenate(far)

@torch.no_grad()
def score(m,x,device,batch):
    m.eval();return np.concatenate([m.score(torch.as_tensor(x[i:i+batch],device=device)).cpu().numpy() for i in range(0,len(x),batch)])

def metrics(d,s):
    return {'AP':float(average_precision_score(d['y'],s)), 'AUROC':float(roc_auc_score(d['y'],s))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['raw','gate','ordinary','reliable'],required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda:0')
    p.add_argument('--pretext-epochs',type=int,default=25);p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=50);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--max-train',type=int,default=0);p.add_argument('--out',default='runs/carla');a=p.parse_args()
    seed_all(a.seed);torch.set_num_threads(2);device=torch.device(a.device)
    tr,va,te=(data.load(s) for s in ('train','val','test'))
    for left,right in ((tr,va),(tr,te),(va,te)):
        if set(left['mmsi']) & set(right['mmsi']): raise ValueError('MMSI split overlap')
    norm=data.fit(tr);xx=data.features(tr,norm,a.arm);xv=data.features(va,norm,a.arm);xt=data.features(te,norm,a.arm)
    normal=np.flatnonzero(tr['y']==0)
    if a.max_train: normal=normal[:a.max_train]
    xx=xx[normal]; train={k:v[normal] for k,v in tr.items()};prev=previous_windows(train)
    val_idx=np.flatnonzero(va['y']==0)
    xvalnormal=xv[val_idx]; val_normal={k:v[val_idx] for k,v in va.items()}
    val_prev=previous_windows(val_normal)
    out=Path(a.out)/a.arm/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
    spec=vars(a)|{'train_samples':len(xx),'same_ship_previous':int((prev>=0).sum()),'fallback_noise':int((prev<0).sum()),'phase1':'author PretextLoss, same-MMSI prior window or author Gaussian noise','phase2':'author ClassificationLoss, train-embedding neighbors','selection':'validation normal unsupervised clustering loss','score':'1 minus majority training cluster probability'}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    np.savez_compressed(out/'normalizer.npz',**norm)
    print('CARLA 正样本：同船先前窗口',spec['same_ship_previous'],'噪声回退',spec['fallback_noise'],flush=True)
    model=Adapter(xx.shape[1],xx.shape[2]).to(device);opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=.001)
    for epoch in range(a.pretext_epochs):
        model.train();losses=[];order=np.random.permutation(len(xx));bar=tqdm(range(0,len(order),a.batch_size),desc=f'预训练 {epoch+1}/{a.pretext_epochs}')
        for start in bar:
            ix=order[start:start+a.batch_size];x=torch.as_tensor(xx[ix],device=device);pos=x+torch.randn_like(x)*.01
            has=prev[ix]>=0
            if has.any(): pos[torch.as_tensor(has,device=device)]=torch.as_tensor(xx[prev[ix[has]]],device=device)
            opt.zero_grad(set_to_none=True);loss=model.loss(x,positive=pos);loss.backward();opt.step();losses.append(float(loss.detach()));bar.set_postfix(loss=np.mean(losses))
        model.eval();vl=[]
        with torch.no_grad():
            for st in range(0,len(xvalnormal),a.batch_size):
                ix=np.arange(st,min(st+a.batch_size,len(xvalnormal)));x=torch.as_tensor(xvalnormal[ix],device=device);pos=x+torch.randn_like(x)*.01
                has=val_prev[ix]>=0
                if has.any(): pos[torch.as_tensor(has,device=device)]=torch.as_tensor(xvalnormal[val_prev[ix[has]]],device=device)
                z=model.embed(torch.cat([x,pos,model.synthetic(x)],0));model.pretext_loss.bs=len(ix)
                vl.append(float(model.pretext_loss(z)))
        torch.save({'phase':'pretext','epoch':epoch+1,'model':model.state_dict(),'optimizer':opt.state_dict(),'margin':model.pretext_loss.margin,'prev_loss':model.prev_loss,'val_loss':float(np.mean(vl))},out/'last.pt')
        print(f'预训练 {epoch+1}: loss={np.mean(losses):.5f} val_loss={np.mean(vl):.5f}',flush=True)
    # Original repository mines over original + synthetic-subsequence windows.
    augmented=[]
    for st in range(0,len(xx),a.batch_size):
        augmented.append(model.synthetic(torch.as_tensor(xx[st:st+a.batch_size],device=device)).cpu().numpy())
    cluster_x=np.concatenate([xx,*augmented],axis=0)
    z=embeddings(model,cluster_x,device,a.batch_size);n,f=neighbors(z,z,same=True);zv=embeddings(model,xvalnormal,device,a.batch_size);vn,vf=neighbors(zv,z)
    model.begin_clustering();opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=.001);best=float('inf')
    for epoch in range(a.epochs):
        model.train();ls=[];order=np.random.permutation(len(cluster_x));bar=tqdm(range(0,len(order),a.batch_size),desc=f'聚类训练 {epoch+1}/{a.epochs}')
        for st in bar:
            ix=order[st:st+a.batch_size];nn=n[ix,np.random.randint(n.shape[1],size=len(ix))];ff=f[ix,np.random.randint(f.shape[1],size=len(ix))]
            x=torch.as_tensor(cluster_x[ix],device=device);near=torch.as_tensor(cluster_x[nn],device=device);far=torch.as_tensor(cluster_x[ff],device=device)
            opt.zero_grad(set_to_none=True);loss=model.loss(x,near=near,far=far);loss.backward();opt.step();ls.append(float(loss.detach()));bar.set_postfix(loss=np.mean(ls))
        model.eval();vls=[]
        with torch.no_grad():
            for st in range(0,len(xvalnormal),a.batch_size):
                end=min(st+a.batch_size,len(xvalnormal));x=torch.as_tensor(xvalnormal[st:end],device=device)
                vls.append(float(model.loss(x,near=torch.as_tensor(cluster_x[vn[st:end,0]],device=device),far=torch.as_tensor(cluster_x[vf[st:end,0]],device=device))))
            logits=[]
            for st in range(0,len(xx),a.batch_size): logits.append(model.cluster(model._reshape(torch.as_tensor(xx[st:st+a.batch_size],device=device)))[0].argmax(1).cpu())
            model.normal_cluster.copy_(torch.bincount(torch.cat(logits),minlength=10).argmax().to(device))
        val=metrics(va,score(model,xv,device,a.batch_size));vl=float(np.mean(vls));checkpoint={'phase':'clustering','epoch':epoch+1,'model':model.state_dict(),'optimizer':opt.state_dict(),'val_loss':vl,'val_metrics':val}
        torch.save(checkpoint,out/'last.pt')
        if vl<best: best=vl;torch.save(checkpoint,out/'best.pt')
        print(f'聚类 {epoch+1}: loss={np.mean(ls):.5f} val_loss={vl:.5f} AP={val["AP"]:.4f} AUROC={val["AUROC"]:.4f}',flush=True)
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model']);s=score(model,xt,device,a.batch_size)
    np.savez_compressed(out/'test_scores.npz',score=s,y=te['y'],mmsi=te['mmsi'],region=te['region'],window_id=te['window_id'])
    result={'best_epoch':ck['epoch'],'validation':ck['val_metrics'],'test':metrics(te,s)}
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('完成',result,flush=True)

if __name__=='__main__':main()
