#!/usr/bin/env python3
"""Eight existing backbone adaptations on the shared AIS feature protocol.

This controlled comparison uses only BCE, not the historical 4D bundle.
Graph13 is a target/reference-node adaptation; all networks see the same slots.
"""
import argparse,json,random,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from train_supervised_recent import prepare,metrics

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
sys.path.insert(0,str(ROOT/'framework_validation_v1_20261028'))
sys.path.insert(0,str(ROOT/'eight_backbone_preparation_20261030'))
from models import make_model,GraphTemporalClassifier,_masked_mean
from relation8 import model_for as relation_model

NAMES=('tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph')

class Graph13(GraphTemporalClassifier):
    """Same two-node GAT operator, now mapped to target 7 / reference 6."""
    def __init__(self):
        super().__init__(12,20,64);self.input_dim=13;self.target_dim=7
        self.target_projection=nn.Linear(7,64);self.reference_projection=nn.Linear(6,64)

def create(name,task,dim):
    if task=='relation':
        if dim!=35:raise ValueError('relation requires 35 channels')
        return relation_model(name)
    if dim!=13:raise ValueError('single requires 13 channels')
    if name=='graph':return Graph13()
    return make_model(name,dim,max_len=20,hidden=64)

def elapsed(d,task):
    times=np.asarray(d['time_s'],dtype=np.float64)
    if task=='single':return (times-times[:,:1]).astype('float32')
    out=np.empty((len(times),20),dtype=np.float32)
    for i,n in enumerate(d['length']):
        t=times[i,:int(n)];out[i]=np.linspace(0.,float(t[-1]-t[0]),20)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['single','relation'],required=True)
    p.add_argument('--arm',required=True);p.add_argument('--name',choices=NAMES,default='tcn')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--mode',choices=['train','test'],default='train');p.add_argument('--smoke',action='store_true');p.add_argument('--audit-all',action='store_true')
    p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107/eight_matched');a=p.parse_args()
    arms={'single':('raw','ordinary','reliable'),'relation':('C0','F0','F1')}
    if a.arm not in arms[a.task]:raise ValueError('invalid task arm')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_num_threads(2)
    device=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu');tr,va,te,make,views=prepare(a.task,a.arm)
    for d,e in ((tr,va),(tr,te),(va,te)):
        if set(d['mmsi'])&set(e['mmsi']):raise ValueError('target MMSI split overlap')
    x=make(tr);xv=make(va);t=elapsed(tr,a.task);tv=elapsed(va,a.task);y=tr['y'].astype('float32');yv=va['y'].astype('int32')
    if a.audit_all:
        # Actual AIS input, one train/backward and one eval/forward per backbone.
        results={};xx=torch.tensor(x[:4],device=device);tt=torch.tensor(t[:4],device=device);yy=torch.tensor(y[:4],device=device);mask=torch.ones(xx.shape[:2],dtype=torch.bool,device=device)
        for name in NAMES:
            net=create(name,a.task,x.shape[-1]).to(device);net.train();pred=net(xx,mask,tt)['logits'];loss=F.binary_cross_entropy_with_logits(pred,yy)
            if not torch.isfinite(loss):raise ValueError(name+' invalid training loss')
            loss.backward();net.eval()
            with torch.no_grad():q=net(xx,mask,tt)['logits']
            if q.shape!=(4,) or not torch.isfinite(q).all():raise ValueError(name+' invalid eval')
            results[name]={'loss':float(loss),'params':sum(p.numel() for p in net.parameters())};del net
        print('AUDIT_ALL',json.dumps(results),flush=True);return
    if a.smoke:x=x[:64];t=t[:64];y=y[:64];xv=xv[:64];tv=tv[:64];yv=yv[:64]
    net=create(a.name,a.task,x.shape[-1]).to(device)
    out=Path(a.out)/('smoke' if a.smoke else a.task)/a.name/a.arm/f'seed{a.seed}'
    @torch.no_grad()
    def predict(arr,times):
        net.eval();scores=[]
        for i in range(0,len(arr),a.batch_size):
            v=torch.as_tensor(arr[i:i+a.batch_size],device=device);tt=torch.as_tensor(times[i:i+a.batch_size],device=device);mask=torch.ones(v.shape[:2],dtype=torch.bool,device=device)
            scores.extend(torch.sigmoid(net(v,mask,tt)['logits']).cpu().numpy())
        return np.asarray(scores)
    if a.mode=='test':
        ck=torch.load(out/'best_val.pt',map_location=device,weights_only=False);net.load_state_dict(ck['model']);pv=predict(xv,tv)
        thresholds={f'{q}%':float(np.quantile(pv[yv==0],1-q/100,method='higher')) for q in (1,5)};saved={'y':te['y'],'mmsi':te['mmsi'],'region':te['region']};results={'val_thresholds':thresholds,'views':{}}
        for view in views:
            saved[view]=predict(make(te,view),elapsed(te,a.task));results['views'][view]=metrics(te['y'],saved[view],thresholds)
        if a.task=='relation':saved.update(source=te['source'],case=te['case'])
        else:saved['window_id']=te['window_id']
        np.savez_compressed(out/'test_scores.npz',**saved);(out/'test_metrics.json').write_text(json.dumps(results,indent=2),encoding='utf8');print('TEST',json.dumps(results),flush=True);return
    out.mkdir(parents=True,exist_ok=False);opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=.01);best=-1;hist=[]
    spec=vars(a)|{'features':'same 13D/35D resampled slots as supervised recent methods','train':'all controlled labelled train windows','loss':'BCE only, no historical 4D bundle and no recent-model auxiliary loss','checkpoint':'best validation AP, matching supervised recent methods','test':'previously exposed','times':'actual relative timestamps; no future outside full offline window','graph':'two-node target/reference, active only when support valid','model':'existing classifiers adapted to shared feature slots, not original frozen historical late-adapter results'}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    for epoch in range(a.epochs):
        net.train();ls=[];order=np.random.default_rng(a.seed+10000*epoch).permutation(len(x))
        for pos in tqdm(range(0,len(order),a.batch_size),file=sys.stdout,desc=f'{a.task}/{a.name}/{a.arm} {epoch+1}/{a.epochs}'):
            ids=order[pos:pos+a.batch_size];xx=torch.as_tensor(x[ids],device=device);tt=torch.as_tensor(t[ids],device=device);yy=torch.as_tensor(y[ids],device=device);mask=torch.ones(xx.shape[:2],dtype=torch.bool,device=device)
            opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(net(xx,mask,tt)['logits'],yy)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();ls.append(float(loss.detach()))
        pv=predict(xv,tv);m=metrics(yv,pv);row={'epoch':epoch+1,'train_loss':float(np.mean(ls)),**m};hist.append(row)
        ck={'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'spec':spec,'val':m};torch.save(ck,out/'last.pt')
        if m['AP']>best:best=m['AP'];torch.save(ck,out/'best_val.pt');np.savez_compressed(out/'val_scores.npz',y=yv,score=pv)
        (out/'history.json').write_text(json.dumps(hist,indent=2),encoding='utf8');print('VALIDATION',json.dumps(row),flush=True)
if __name__=='__main__':main()
