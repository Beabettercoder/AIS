#!/usr/bin/env python3
"""Matched TCN bridge: fixed thirteen-slot backbone, change one component at a time.

The original formal architecture used a seven-slot TCN. Here the late-stream
encoder sees the same seven channels plus six fixed zeros; this retains the
new 13-slot backbone parameterization and isolates the early/late injection.
Inputs and valid masks are cached source artefacts; no test use in this script.
"""
import argparse,hashlib,json,random,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from tqdm import tqdm
import data_resampled as resampled

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
sys.path.insert(0,str(ROOT/'framework_validation_v1_20261028'))
sys.path.insert(0,str(ROOT/'original_mainline_pilot_20261029'))
sys.path.insert(0,str(ROOT/'formal_20261101/single'))
from models import make_model,_prepare,_masked_mean
from pilot_historical_loss import historical_bundle
from pilot_features import build_field_inputs

ARMS={'raw':('resampled','raw','none',False),
      'gate':('resampled','target','none',False),
      'early':('resampled','target','early',False),
      'late':('resampled','target','late',False),
      'late_bundle':('resampled','target','late',True),
      'late_bundle_irregular':('irregular','target','late',True),
      'late_bundle_irregular_historical_rho':('irregular','target','late_historical_rho',True)}

def load(split):
    base=ROOT/'framework_validation_v1_20261028';org=ROOT/'original_mainline_pilot_20261029'
    with np.load(base/'data/individual'/f'{split}.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
    with np.load(base/'extra_data'/f'field_{split}.npz',allow_pickle=False) as z:e={k:z[k] for k in z.files}
    with np.load(org/f'rho_{split}.npz',allow_pickle=False) as z:hist_rho=z['rho']
    a=build_field_inputs(d,e,'F0');b=build_field_inputs(d,e,'F1')
    if not np.array_equal(a['valid'],b['valid']):raise ValueError('field masks differ')
    N=len(d['y']);times=d['time_s'].astype(np.float64);mask=d['mask'].astype(bool);valid=b['valid']
    state=b['state']; rho=a['rho'];res=resampled.load(split)
    for k in ('window_id','mmsi','region','y'):
        if not np.array_equal(d[k],res[k]):raise ValueError('pair mismatch '+k)
    state_res=np.zeros((N,20,4),np.float32)
    for i in range(N):
        keep=mask[i];old=times[i,keep];old=old-old[0];new=np.linspace(old[0],old[-1],20)
        state_res[i],valid_state=resampled._field_on_grid(old,state[i,keep],valid[i,keep],new)
        if not np.array_equal(res['valid'][i],valid_state):raise ValueError('support mismatch')
    cats={v:i for i,v in enumerate(['turn_back','deviation','stall','sinuous','speed_drop'],1)}
    yc=np.asarray([cats.get(str(s),0) for s in d['behavior_type']],dtype=np.int64)
    if not np.array_equal(yc>0,d['y']>0):raise ValueError('unknown behaviour labels')
    irregular=dict(raw=d['base__C0'][...,:7],target=a['self'],f0=a['field'],f1=b['field'],
                   rho=rho,rho_hist=hist_rho,valid=valid,mask=mask,times=times-times[:,:1],state=state)
    res.update(mask=np.ones_like(res['valid']),times=res['time_s']-res['time_s'][:,:1],state=state_res)
    meta={k:d[k] for k in ('y','mmsi','region','window_id')};meta['yc']=yc
    return {'irregular':irregular,'resampled':res,'meta':meta}

class TCNBridge(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=make_model('tcn',13,max_len=20,hidden=64)
        self.adapter=nn.Sequential(nn.Linear(4,64),nn.GELU(),nn.Linear(64,64))
        self.alpha=nn.Parameter(torch.tensor(0.));self.typehead=nn.Linear(64,5);self.ctrl=nn.Linear(4,2)
    def forward(self,x,mask,times,field,rho,late):
        x,mask,times,_=_prepare(x,mask,times,13,20);v=mask.to(x.dtype)
        seq=self.backbone.input(x).masked_fill(~mask.unsqueeze(-1),0)
        for block in self.backbone.blocks:seq=block(seq,mask)
        emb=_masked_mean(seq,mask)
        if late:
            ctx=(self.adapter(field)*(rho*v).unsqueeze(-1)).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
            emb=emb+self.alpha*ctx
        return self.backbone.head(emb).squeeze(-1),self.typehead(emb),emb,seq

def convert(dataset,norm,arm,dev):
    mode,target,place,bundle=ARMS[arm];d=dataset[mode];meta=dataset['meta'];mask=d['mask'];valid=d['valid']
    x=((d[target]-norm['sm'])/norm['ss']).astype(np.float32);x[~mask]=0
    f=((d['f1']-norm['fm'])/norm['fs']).astype(np.float32);f[~valid]=0
    rho=d['rho_hist'] if place=='late_historical_rho' else d['rho']
    extra=np.zeros((*x.shape[:2],6),np.float32)
    if place=='early':extra[:,:,:4]=f;extra[:,:,4]=rho;extra[:,:,5]=valid
    x=np.concatenate((x,extra),-1);t=np.asarray(d['times'],np.float32);t[~mask]=0
    return {k:torch.as_tensor(v,device=dev) for k,v in {'x':x,'mask':mask,'times':t,'field':f,'rho':rho,'state':d['state'],'y':meta['y'].astype(np.float32),'yc':meta['yc']}.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--arms',default='raw,gate,early,late,late_bundle,late_bundle_irregular');p.add_argument('--smoke',action='store_true');p.add_argument('--mode',choices=['train','test'],default='train');p.add_argument('--out',default='/home/ps/AIS_bridge_20261107/tcn');a=p.parse_args()
    torch.set_num_threads(2);dev=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    tr=load('train');va=load('val');assert not(set(tr['meta']['mmsi'])&set(va['meta']['mmsi']))
    norm=resampled.fit(tr['resampled'],supervised=True)
    test=None; lock=None
    if a.mode=='test':
        lock=json.loads((Path(a.out)/'frozen_validation.lock.json').read_text(encoding='utf8'))
        expected=lock['code_sha256'];actual=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if expected!=actual:raise ValueError('bridge code changed after freezing')
        test=load('test')
        if set(test['meta']['mmsi'])&(set(tr['meta']['mmsi'])|set(va['meta']['mmsi'])):raise ValueError('MMSI leakage')
    counts=np.bincount(tr['meta']['yc'][tr['meta']['yc']>0]-1,minlength=5);cw=torch.as_tensor(1./np.maximum(counts,1),dtype=torch.float32,device=dev)
    for arm in a.arms.split(','):
        if arm not in ARMS:raise ValueError(arm)
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        model=TCNBridge().to(dev);initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        train=convert(tr,norm,arm,dev);val=convert(va,norm,arm,dev);train_n=64 if a.smoke else len(train['y']);val_n=128 if a.smoke else len(val['y'])
        out=Path(a.out)/('smoke' if a.smoke else 'validation')/arm/f'seed{a.seed}'
        if a.mode=='test':
            key=f'{arm}/seed{a.seed}'
            if key not in lock['runs']:raise ValueError('run not in frozen lock')
            tt=convert(test,norm,arm,dev);saved={k:test['meta'][k] for k in ('y','mmsi','region','window_id')};result={}
            for ckname in ('best_val','last'):
                path=out/(ckname+'.pt');digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if digest!=lock['runs'][key][ckname]:raise ValueError('checkpoint changed after freeze')
                model.load_state_dict(torch.load(path,map_location=dev,weights_only=False)['state_dict']);model.eval();pred=[]
                with torch.no_grad():
                    for i in range(0,len(tt['y']),256):
                        sl=slice(i,i+256);log=model(tt['x'][sl],tt['mask'][sl],tt['times'][sl],tt['field'][sl],tt['rho'][sl],ARMS[arm][2].startswith('late'))[0];pred.extend(torch.sigmoid(log).cpu().numpy())
                pred=np.asarray(pred);y=test['meta']['y'];saved[ckname]=pred;result[ckname]={'pooled_AP':float(average_precision_score(y,pred)),'AUROC':float(roc_auc_score(y,pred))}
            np.savez_compressed(out/'test_scores.npz',**saved);(out/'test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('TEST',arm,a.seed,result,flush=True);continue
        out.mkdir(parents=True,exist_ok=a.smoke)
        input_hash=lambda z:hashlib.sha256(z['x'].detach().cpu().numpy().tobytes()+z['mask'].detach().cpu().numpy().tobytes()+z['field'].detach().cpu().numpy().tobytes()+z['rho'].detach().cpu().numpy().tobytes()+z['y'].detach().cpu().numpy().tobytes()).hexdigest()
        backbone_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.backbone.parameters())).hexdigest()
        spec={'arm':arm,'seed':a.seed,'epochs':a.epochs,'batch_size':a.batch_size,'optimizer':'AdamW 0.001 wd 0.01','selection':'best validation pooled AP; save last separately; no test','initial_parameter_sha256':initial,'initial_backbone_sha256':backbone_hash,'train_input_sha256':input_hash(train),'validation_input_sha256':input_hash(val),'parameters':sum(p.numel() for p in model.parameters()),'architecture':'fixed 13-slot TCN; late arms feed 7 target features + 6 zero slots into backbone; 4D field through zero initialized historical adapter with rho','normalizer':'all training raw+gate features, F0+F1 valid features in resampled view, identical across all arms','rho':'count-only except explicitly named historical rho arm','data':'same 7840 training, 1680 validation windows and MMSI isolation','bundle':'formal historical_bundle with physical state, class labels, augmented dropout and type consistency; original code','differences_from_formal':'13 slots instead of 7, count rho except named arm, fixed new training budget and normalization'}
        (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8');best=-1.;history=[]
        for epoch in range(a.epochs):
            model.train();order=np.random.default_rng(a.seed+10000*epoch).permutation(train_n);losses=[]
            for st in tqdm(range(0,len(order),a.batch_size),desc=f'TCN {arm} seed {a.seed} {epoch+1}/{a.epochs}',file=sys.stdout):
                ix=torch.as_tensor(order[st:st+a.batch_size],device=dev);x=train['x'][ix];mask=train['mask'][ix];times=train['times'][ix];field=train['field'][ix];rho=train['rho'][ix];y=train['y'][ix];yc=train['yc'][ix]
                opt.zero_grad(set_to_none=True);late=ARMS[arm][2].startswith('late');pb,pc,h,seq=model(x,mask,times,field,rho,late)
                if not ARMS[arm][3]:loss=F.binary_cross_entropy_with_logits(pb,y)
                else:
                    keep=(torch.rand(x.shape[:2],device=dev)>.15).to(x.dtype);x2=x*keep.unsqueeze(-1);lengths=mask.sum(1);jj=torch.floor(torch.rand(len(x),device=dev)*lengths).long();x2=x2*mask.unsqueeze(-1);x2[torch.arange(len(x),device=dev),jj,0]+=2.;ret=(keep*mask).sum(1)/lengths.clamp_min(1)
                    _,_,h2,_=model(x2,mask,times,field,rho,late)
                    loss,_=historical_bundle(model.backbone.head,model.typehead,model.ctrl,pb,pc,h2,seq,train['state'][ix],mask,y,yc,cw,ret)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            model.eval();pred=[]
            with torch.no_grad():
                for st in range(0,val_n,256):
                    ix=slice(st,min(val_n,st+256));outp=model(val['x'][ix],val['mask'][ix],val['times'][ix],val['field'][ix],val['rho'][ix],ARMS[arm][2].startswith('late'))[0];pred.extend(torch.sigmoid(outp).cpu().numpy())
            pred=np.asarray(pred);y=va['meta']['y'][:val_n];ap=float(average_precision_score(y,pred));auc=float(roc_auc_score(y,pred))
            m={'epoch':epoch+1,'loss':float(np.mean(losses)),'pooled_AP':ap,'AUROC':auc,'alpha':float(model.alpha.detach())};history.append(m)
            ck={'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'spec':spec,'norm':norm,'metrics':m};torch.save(ck,out/'last.pt')
            if ap>best:best=ap;torch.save(ck,out/'best_val.pt');np.savez_compressed(out/'val_scores.npz',score=pred,y=y,mmsi=va['meta']['mmsi'][:val_n],region=va['meta']['region'][:val_n],window_id=va['meta']['window_id'][:val_n])
            (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8');print('VALIDATION',json.dumps(m),flush=True)
if __name__=='__main__':main()
