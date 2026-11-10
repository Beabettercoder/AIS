#!/usr/bin/env python3
"""Two-stage CARLA plus BCE: supervised adaptation, not native CARLA."""
import argparse,json,random,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from train_supervised_recent import prepare,Detector,metrics
from train_carla import embeddings,neighbors
from train_carla_relation import previous_windows,positive

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['single','relation'],required=True);p.add_argument('--arm',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',type=int,default=0);p.add_argument('--pretext-epochs',type=int,default=12);p.add_argument('--epochs',type=int,default=13)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=.001);p.add_argument('--aux-weight',type=float,default=.1)
    p.add_argument('--mode',choices=['train','test'],default='train');p.add_argument('--smoke',action='store_true')
    p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107/supervised_carla_twostage');a=p.parse_args()
    valid={'single':('raw','gate','ordinary','reliable'),'relation':('C0','F0','F1')}
    if a.arm not in valid[a.task]:raise ValueError('invalid task/arm')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_num_threads(2)
    dev=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu');tr,va,te,make,views=prepare(a.task,a.arm)
    for x,y in ((tr,va),(tr,te),(va,te)):
        if set(x['mmsi'])&set(y['mmsi']):raise ValueError('target MMSI overlap')
    x=make(tr);xv=make(va);y=tr['y'].astype('float32');yv=va['y'].astype('int32')
    if a.task=='single':tr['length']=np.full(len(y),20)
    if a.smoke:x=x[:64];y=y[:64];tr={k:v[:64] for k,v in tr.items()};xv=xv[:64];yv=yv[:64]
    fingerprints={'train':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),'validation':hashlib.sha256(xv.tobytes()+yv.tobytes()).hexdigest()}
    prev=previous_windows(tr);model=Detector('carla',x.shape[-1],a.gpu).to(dev)
    out=Path(a.out)/('smoke' if a.smoke else a.task)/a.arm/f'seed{a.seed}'
    @torch.no_grad()
    def predict(arr):
        model.eval();z=[]
        for i in range(0,len(arr),a.batch_size):z.extend(torch.sigmoid(model.logits(torch.as_tensor(arr[i:i+a.batch_size],device=dev))).cpu().numpy())
        return np.asarray(z)
    if a.mode=='test':
        ck=torch.load(out/'best_val.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['model'])
        if ck.get('fingerprints')!=fingerprints:raise ValueError('Training or validation inputs changed since checkpoint')
        pv=predict(xv);neg=pv[yv==0];thresholds={f'{q}%':float(np.quantile(neg,1-q/100,method='higher')) for q in (1,5)}
        saved={'y':te['y'],'mmsi':te['mmsi'],'region':te['region']};res={'val_thresholds':thresholds,'validation_operating_points':metrics(yv,pv,thresholds),'checkpoint_phase':ck['phase'],'views':{}}
        for view in views:
            saved[view]=predict(make(te,view));res['views'][view]=metrics(te['y'],saved[view],thresholds)
        if a.task=='relation':saved.update(source=te['source'],case=te['case'])
        else:saved['window_id']=te['window_id']
        np.savez_compressed(out/'test_scores.npz',**saved);(out/'test_metrics.json').write_text(json.dumps(res,indent=2),encoding='utf8');print('TEST',json.dumps(res),flush=True);return
    out.mkdir(parents=True,exist_ok=False);best=-1.;history=[]
    spec=vars(a)|{'label_policy':'all train windows use controlled supervised labels','phase1':'BCE + aux-weight original CARLA pretext loss; strictly earlier same-vessel positive or Gaussian fallback','phase2':'BCE on original samples + aux-weight original CARLA clustering loss on original and synthetic samples; nearest/farthest frozen pretext embedding','checkpoint':'best validation AP across both phases','views':views,'test':'previously exposed; evaluate separately','layout':'AIS time-channel transpose','normalization':'training only','parameter_count':sum(p.numel() for p in model.parameters()),'initial_parameter_sha256':hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest(),'note':'25 BCE epochs default; auxiliary clustering sees original and synthetic samples'}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    print('DATA',json.dumps(spec),'earlier_positives',int((prev>=0).sum()),flush=True)
    def save(phase,epoch,ls,opt):
        nonlocal best
        val=metrics(yv,predict(xv));row={'phase':phase,'epoch':epoch+1,'loss':float(np.mean(ls)),**val};history.append(row)
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'phase':phase,'val':val,'spec':spec,'margin':model.adapter.pretext_loss.margin,'prev_loss':model.adapter.prev_loss,'fingerprints':fingerprints}
        torch.save(ck,out/'last.pt')
        if val['AP']>best:best=val['AP'];torch.save(ck,out/'best_val.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8');print('VALIDATION',json.dumps(row),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    for epoch in range(a.pretext_epochs):
        model.train();ls=[];order=np.random.default_rng(a.seed+10000*epoch).permutation(len(x))
        for pos in tqdm(range(0,len(order),a.batch_size),file=sys.stdout,desc=f'预训练+BCE {epoch+1}/{a.pretext_epochs}'):
            ids=order[pos:pos+a.batch_size];xx,pp=positive(x,prev,ids,dev);yy=torch.as_tensor(y[ids],device=dev)
            opt.zero_grad(set_to_none=True);bce=F.binary_cross_entropy_with_logits(model.logits(xx),yy);aux=model.adapter.loss(xx,positive=pp)
            loss=bce+a.aux_weight*aux
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();ls.append(float(loss.detach()))
        save('pretext',epoch,ls,opt)
    ck=torch.load(out/'best_val.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['model']);model.adapter.prev_loss=ck['prev_loss'];model.adapter.pretext_loss.margin=ck['margin']
    augmented=[]
    for st in range(0,len(x),a.batch_size):augmented.append(model.adapter.synthetic(torch.as_tensor(x[st:st+a.batch_size],device=dev)).cpu().numpy())
    cluster_x=np.concatenate([x,*augmented],axis=0);z=embeddings(model.adapter,cluster_x,dev,a.batch_size);near,far=neighbors(z,z,same=True)
    model.adapter.begin_clustering();opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    for epoch in range(a.epochs):
        model.train();ls=[];order=np.random.default_rng(a.seed+10000*(a.pretext_epochs+epoch)).permutation(len(cluster_x))
        for st in tqdm(range(0,len(order),a.batch_size),file=sys.stdout,desc=f'聚类+BCE {epoch+1}/{a.epochs}'):
            ids=order[st:st+a.batch_size];n=near[ids,np.random.randint(near.shape[1],size=len(ids))];f=far[ids,np.random.randint(far.shape[1],size=len(ids))]
            xx=torch.as_tensor(cluster_x[ids],device=dev);nx=torch.as_tensor(cluster_x[n],device=dev);fx=torch.as_tensor(cluster_x[f],device=dev)
            original=ids<len(x);opt.zero_grad(set_to_none=True);aux=model.adapter.loss(xx,near=nx,far=fx)
            if original.any():bce=F.binary_cross_entropy_with_logits(model.logits(xx[torch.as_tensor(original,device=dev)]),torch.as_tensor(y[ids[original]],device=dev))
            else:bce=aux.new_zeros(())
            loss=bce+a.aux_weight*aux
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();ls.append(float(loss.detach()))
        save('clustering',epoch,ls,opt)
if __name__=='__main__':main()
