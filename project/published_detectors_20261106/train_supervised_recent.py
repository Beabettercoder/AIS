"""Supervised adaptation of recent detectors; native auxiliary objectives retained."""
import argparse, importlib, json, random, sys, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

def prepare(task, arm):
    if task == 'single':
        import data_resampled as data
        tr,va,te=(data.load(s) for s in ('train','val','test'))
        norm=data.fit(tr,supervised=True)
        def make(d,view='base'): return data.features(d,norm,arm)
        return tr,va,te,make,('base',)
    from train_relation_recent import read,base,VIEWS
    from train_carla_relation import prep
    tr,va,te=(read(s) for s in ('train','val','test'))
    for d in (tr,va,te):
        # Needed for CARLA's strictly earlier same-vessel positive windows.
        d['time_s_full']=d['time_s'].copy()
    mu,sd=base.fit(tr)
    def make(d,view='base'):return prep(d,view,arm,mu,sd)
    return tr,va,te,make,VIEWS

class Detector(nn.Module):
    def __init__(self, method, channels, gpu):
        super().__init__();self.method=method
        self.adapter=importlib.import_module('adapters.'+method).Adapter(seq_len=20,channels=channels,gpu=gpu)
        dim=128 if method=='carla' else 20*channels
        self.head=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,1))
    def representation(self,x):
        if self.method=='carla':return self.adapter.embed(x)
        y=self.adapter.model(x)
        if isinstance(y,tuple):y=y[0]
        return y.reshape(x.size(0),-1)
    def logits(self,x):return self.head(self.representation(x)).squeeze(-1)

def metrics(y,s,thresholds=None):
    result={'AP':float(average_precision_score(y,s)),'AUROC':float(roc_auc_score(y,s))}
    if thresholds:
        for k,t in thresholds.items():
            result['FAR@'+k]=float(np.mean(s[y==0]>=t));result['Recall@'+k]=float(np.mean(s[y==1]>=t))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['tft_gcn','simad','dmrad','carla']);p.add_argument('--task',required=True,choices=['single','relation']);p.add_argument('--arm',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--aux-weight',type=float,default=.1);p.add_argument('--smoke',action='store_true');p.add_argument('--mode',choices=['train','test'],default='train');p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107/supervised')
    a=p.parse_args();valid={'single':('raw','gate','ordinary','reliable'),'relation':('C0','F0','F1')}
    if a.arm not in valid[a.task]:raise ValueError('invalid task arm')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_num_threads(2)
    device=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    tr,va,te,make,views=prepare(a.task,a.arm)
    for x,y in ((tr,va),(tr,te),(va,te)):
        if set(x['mmsi']) & set(y['mmsi']):raise ValueError('Target MMSI overlap')
    x=make(tr);xv=make(va);yt=tr['y'].astype('float32');yv=va['y'].astype('int32')
    if a.smoke:x=x[:64];yt=yt[:64];xv=xv[:64];yv=yv[:64]
    fingerprints={'train':hashlib.sha256(x.tobytes()+yt.tobytes()).hexdigest(),'validation':hashlib.sha256(xv.tobytes()+yv.tobytes()).hexdigest()}
    model=Detector(a.method,x.shape[-1],a.gpu).to(device)
    out=Path(a.out)/('smoke' if a.smoke else a.task)/a.method/a.arm/f'seed{a.seed}'
    @torch.no_grad()
    def predict(arr):
        model.eval();z=[]
        for i in range(0,len(arr),a.batch_size):z.extend(torch.sigmoid(model.logits(torch.as_tensor(arr[i:i+a.batch_size],device=device))).cpu().numpy())
        return np.asarray(z)
    if a.mode=='test':
        ck=torch.load(out/'best_val.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model'])
        if ck.get('fingerprints')!=fingerprints:raise ValueError('Training or validation inputs changed since checkpoint')
        pv=predict(xv);neg=pv[yv==0];thresholds={f'{p}%':float(np.quantile(neg,1-p/100,method='higher')) for p in (1,5)}
        scores={'y':te['y'],'mmsi':te['mmsi'],'region':te['region']};res={'val_thresholds':thresholds,'validation_operating_points':metrics(yv,pv,thresholds),'views':{}}
        for view in views:
            s=predict(make(te,view));scores[view]=s;res['views'][view]=metrics(te['y'],s,thresholds)
        if a.task=='relation':scores.update(source=te['source'],case=te['case'])
        else:scores['window_id']=te['window_id']
        np.savez_compressed(out/'test_scores.npz',**scores);(out/'test_metrics.json').write_text(json.dumps(res,indent=2),encoding='utf8');print('TEST',res,flush=True);return
    if out.exists() and not a.smoke:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);opt=torch.optim.AdamW(model.parameters(),lr=.002 if a.task=='single' else .001,weight_decay=.01)
    init_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()
    spec=vars(a)|{'model':'supervised head on original reconstruction output (CARLA: embedding)','train':'all labelled train windows','objective':'binary BCE + aux_weight native objective','checkpoint':'best validation pooled AP; fixed last also retained','test':'previously exposed test','preprocessing':'20-point full-window offline interpolation, train-only normalization','initial_parameter_sha256':init_hash,'parameter_count':sum(p.numel() for p in model.parameters()),'note':'new preprocessing, checkpoint selection and auxiliary objectives differ from earlier eight-backbone experiments','native_model_difference':'CARLA supervised uses original pretext loss and encoder; does not include clustering phase'}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8');best=-1.;history=[]
    for epoch in range(a.epochs):
        if hasattr(model.adapter,'begin_epoch'):model.adapter.begin_epoch(epoch,a.epochs)
        model.train();order=np.random.default_rng(a.seed+epoch*10000).permutation(len(x));ls=[]
        for pos in tqdm(range(0,len(order),a.batch_size),desc=f'{a.task}/{a.method}/{a.arm} {epoch+1}/{a.epochs}',file=sys.stdout):
            ids=order[pos:pos+a.batch_size];xx=torch.as_tensor(x[ids],device=device);yy=torch.as_tensor(yt[ids],device=device)
            opt.zero_grad(set_to_none=True);log=model.logits(xx);bce=F.binary_cross_entropy_with_logits(log,yy)
            if a.method=='carla':
                # Author's Gaussian positive fallback; pretext preserved, clustering not claimed.
                aux=model.adapter.loss(xx,positive=xx+torch.randn_like(xx)*.01)
            else:aux=model.adapter.loss(xx,indices=ids,epoch=epoch)
            loss=bce+a.aux_weight*aux
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();ls.append(float(loss.detach()))
        pv=predict(xv);m=metrics(yv,pv);row={'epoch':epoch+1,'train_loss':float(np.mean(ls)),**m};history.append(row)
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'spec':spec,'val':m,'fingerprints':fingerprints};torch.save(ck,out/'last.pt')
        if m['AP']>best:best=m['AP'];torch.save(ck,out/'best_val.pt');np.savez_compressed(out/'val_scores.npz',y=yv,score=pv)
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8');print('VALIDATION',row,flush=True)
if __name__=='__main__':main()
