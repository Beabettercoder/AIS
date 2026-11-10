"""Formal single-vessel 8-backbone, fixed epoch25, train/val only; requested 2026-11-01."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,json,hashlib,time,argparse,random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
root=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028')
origin=Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')
out=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single')
sys.path.insert(0,str(root))
sys.path.insert(0,str(origin))
from models import make_model,_prepare,_masked_mean
from pilot_features import build_field_inputs
from pilot_historical_loss import historical_bundle

from eight_backbone_adapter import Model8 as Model

def load(split):
 with np.load(root/'data/individual'/f'{split}.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
 with np.load(root/'extra_data'/f'field_{split}.npz',allow_pickle=False) as z: e={k:z[k] for k in z.files}
 with np.load(origin/f'rho_{split}.npz') as z:rho=z['rho']
 a=build_field_inputs(d,e,'F0');b=build_field_inputs(d,e,'F1');assert np.array_equal(a['valid'],b['valid'])
 assert np.all(rho[~b['valid']]==0) and np.all(rho[b['valid']]>0)
 cat={x:i for i,x in enumerate(['turn_back','deviation','stall','sinuous','speed_drop'],1)};yc=np.array([cat.get(str(x),0) for x in d['behavior_type']],np.int64);assert np.array_equal(yc>0,d['y']>0)
 t=d['time_s'].copy();t=t-t[:,:1];t[~d['mask']]=0
 return dict(raw=d['base__C0'][...,:7],self=a['self'],mask=d['mask'],times=t.astype(np.float32),y=d['y'].astype(np.float32),yc=yc,region=d['region'].astype(str),mmsi=d['mmsi'].astype(str),ids=d['window_id'].astype(str),f0=a['field'],f1=b['field'],state0=a['state'],state1=b['state'],rho=rho,valid=b['valid'])

def fit(d):
 mask=d['mask'];sf=np.concatenate([d['raw'][mask],d['self'][mask]],axis=0).astype(np.float64);f=np.concatenate([d['f0'][d['valid']],d['f1'][d['valid']]],axis=0).astype(np.float64)
 def stats(x):
  m=x.mean(0);s=x.std(0);s[s<1e-7]=1.;return m,s
 sm,ss=stats(sf);fm,fs=stats(f)
 return dict(sm=sm,ss=ss,fm=fm,fs=fs)
def convert(d,n,device):
 r={}
 for key in ('raw','self'):
  x=((d[key]-n['sm'])/n['ss']).astype(np.float32);x[~d['mask']]=0;r[key]=torch.tensor(x,device=device)
 for key in ('f0','f1'):
  x=((d[key]-n['fm'])/n['fs']).astype(np.float32);x[~d['valid']]=0;r[key]=torch.tensor(x,device=device)
 for key in ('mask','times','y','yc','state0','state1','rho'):r[key]=torch.tensor(d[key],device=device)
 return r

def metrics(y,p,region):
 return dict(pooled_AP=float(average_precision_score(y,p)),macro_AP=float(np.mean([average_precision_score(y[region==r],p[region==r]) for r in np.unique(region)])),AUROC=float(roc_auc_score(y,p)))
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def main(name,gpu,run_seed):
 seed(run_seed);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;dev=torch.device('cuda:'+str(gpu));tr,va=load('train'),load('val');norm=fit(tr);trt,vat=convert(tr,norm,dev),convert(va,norm,dev)
 print('READY',name,'train',len(tr['y']),'val',len(va['y']),'support',tr['valid'].sum()/tr['mask'].sum(),flush=True)
 cnt=np.bincount(tr['yc'][tr['yc']>0]-1,minlength=5);cw=torch.tensor(1./np.maximum(cnt,1),dtype=torch.float32,device=dev)
 spec={'name':name,'seed':run_seed,'arms':['raw','causal_bundle','f0_bundle','f1_bundle'],'epochs':25,'checkpoint':'fixed epoch25 primary; best validation exploratory','normalizer':'pooled raw and causal training target; pooled F0/F1 valid training field; no clip','state':'physical units, no normalization; mean over valid time; first or previous-unavailable speed change zero','rho':'min(n/4,1) times mean member w_abs on shared valid fitted support','smooth':'masked second difference; iTransformer variable-token/input proxy, PatchTST overlapping patch-token proxy; others per-step hidden states; GraphTemporal uses 7 self features plus 5 zero graph-context slots','bundle':'step35 adapted to this model: type CE, dropout 0.15 plus +2 standardized slot0, epsilon [.05,.25], type JS, aug BCE, temporal smooth; controller jointly minimizes','validation':'macro-AP primary, 8 regions, previously exposed','target_gate':'gated7_legacy, existing audited causal package; observed position remains field query','optimizer':'AdamW 0.002 weight_decay .01','batch':128,'batch_order':'numpy default_rng(300000+epoch); same across seeds as original pilot','user_authorization':'2026-11-01 explicit full run; no inference of previous gate passage'}
 outdir=out/'runs'/name/('seed'+str(run_seed));outdir.mkdir(parents=True,exist_ok=False);(outdir/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
 for arm in spec['arms']:
  seed(run_seed);model=Model(name).to(dev);no_decay=[p for p in model.parameters() if getattr(p,'_no_weight_decay',False)]; decay=[p for p in model.parameters() if not getattr(p,'_no_weight_decay',False)]; opt=torch.optim.AdamW([{'params':decay,'weight_decay':.01},{'params':no_decay,'weight_decay':0.}],lr=.002);folder=outdir/arm;folder.mkdir(exist_ok=False)
  init=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest();israw=arm=='raw';hasfield=arm.startswith('f');fk='f0' if arm.startswith('f0') else 'f1';sk='state0' if fk=='f0' else 'state1';key='raw' if israw else 'self';hist=[];best=-1;bestepoch=0
  def predict():
   model.eval();p=[]
   with torch.no_grad():
    for k in range(0,len(va['y']),256):
     z=slice(k,k+256);log,_,_,_=model(vat[key][z],vat['mask'][z],vat['times'][z],vat[fk][z],vat['rho'][z],hasfield);p.extend(torch.sigmoid(log).cpu().numpy())
   return np.array(p)
  for epoch in range(25):
   model.train();order=np.random.default_rng(300000+epoch).permutation(len(tr['y']));losses=[]
   for start in range(0,len(order),128):
    ix=torch.tensor(order[start:start+128],device=dev);x=trt[key][ix];mask=trt['mask'][ix];tm=trt['times'][ix];field=trt[fk][ix];rho=trt['rho'][ix];y=trt['y'][ix];yc=trt['yc'][ix]
    opt.zero_grad(set_to_none=True);pb,pc,h,seq=model(x,mask,tm,field,rho,hasfield)
    if israw: loss=F.binary_cross_entropy_with_logits(pb,y)
    else:
     keep=(torch.rand(x.shape[:2],device=dev)>.15).to(x.dtype);x2=x*keep.unsqueeze(-1);lengths=mask.sum(1);jj=torch.floor(torch.rand(len(x),device=dev)*lengths).long();x2=x2*mask.unsqueeze(-1);x2[torch.arange(len(x),device=dev),jj,0]+=2.;ret=(keep*mask).sum(1)/lengths.clamp_min(1)
     _,_,h2,_=model(x2,mask,tm,field,rho,hasfield)
     loss,_=historical_bundle(model.backbone.head,model.typehead,model.ctrl,pb,pc,h2,seq,trt[sk][ix],mask,y,yc,cw,ret)
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
   p=predict();m=metrics(va['y'],p,va['region']);hist.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**m));
   if m['macro_AP']>best:best=m['macro_AP'];bestepoch=epoch+1;np.savez_compressed(folder/'best_val.npz',score=p,y=va['y'],mmsi=va['mmsi'],region=va['region'],window_id=va['ids']);torch.save(model.state_dict(),folder/'best.pt')
   if epoch%5==0: print('EPOCH',name,arm,epoch+1,m['macro_AP'],flush=True)
  np.savez_compressed(folder/'final_val.npz',score=p,y=va['y'],mmsi=va['mmsi'],region=va['region'],window_id=va['ids']);torch.save(model.state_dict(),folder/'final.pt')
  result={'arm':arm,'name':name,'seed':run_seed,'init':init,'final':hist[-1],'best_epoch':bestepoch,'best_macro_AP':best,'history':hist,'alpha':float(model.alpha.detach()),'valid_train_steps':int(tr['valid'].sum()),'train_steps':int(tr['mask'].sum()),'valid_val_steps':int(va['valid'].sum()),'val_steps':int(va['mask'].sum())};(folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('DONE',name,arm,hist[-1]['macro_AP'],bestepoch,best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph']);p.add_argument('--gpu',required=True,type=int);p.add_argument('--seed',required=True,type=int,choices=[0,1,2]);args=p.parse_args();main(args.name,args.gpu,args.seed)



