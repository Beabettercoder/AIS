"""Heading side-evidence development pilot, equal-width paired controls.

Only train/val are read. No input may use target or neighbour clean/source SOG,
position, or contamination labels. Heading is a separate AIS field, not proven
independent of other measurements. This is an offline fixed-cohort experiment.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import argparse, hashlib, json, random, sys
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
OUT=ROOT/'itrans_sourcebalanced_20261031'
CACHE=ROOT/'heading_evidence_pilot_20261030'
SOURCE=ROOT/'heading_alignment_20261030'
sys.path.insert(0,str(SOURCE))
import train_heading as base
VIEWS=('base','natural','suffix','combined_l100_m600')
ARMS=('C0','F0_heading','F1_noheading','F1_heading')
NF=40

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def read(split):
 d=base.read(split)
 with np.load(CACHE/f'{split}_heading_features.npz',allow_pickle=False) as z:
  h={k:z[k] for k in z.files}
 for k in ('source','case','y','mmsi'):
  if not np.array_equal(d[k].astype(str),h[k].astype(str)): raise ValueError('alignment '+k)
 d.update({'heading_'+k:v for k,v in h.items() if k not in ('source','case','y','mmsi')})
 return d

def heading_extra(d,view,arm):
 oldarm='F0' if arm=='F0_heading' else 'F1'
 x=d[f'heading_{view}__{oldarm}'].astype(np.float32)
 if x.shape[:2]!=d['mask'].shape or x.shape[-1]!=5: raise ValueError('heading shape')
 active=np.stack((x[...,2]>0,x[...,2]>0,np.ones_like(x[...,2],dtype=bool),x[...,4]>0,np.ones_like(x[...,4],dtype=bool)),axis=-1)
 return x,active

def fit(tr):
 mu35,sd35=base.fit(tr)
 vals=[]
 for arm in ('F0_heading','F1_heading'):
  x,a=heading_extra(tr,'base',arm);vals.append((x,a & tr['mask'][...,None]))
 mu=np.zeros(NF,np.float64);sd=np.ones(NF,np.float64);mu[:35]=mu35;sd[:35]=sd35
 for j in (0,1,3):
  z=np.concatenate([x[...,j][a[...,j]] for x,a in vals]).astype(np.float64)
  if not len(z): raise ValueError('no heading available: '+str(j))
  mu[35+j]=z.mean();sd[35+j]=max(z.std(),1e-7)
 return mu,sd

def prep(d,view,arm,mu,sd,device):
 oldarm={'C0':'C0','F0_heading':'F0','F1_noheading':'F1','F1_heading':'F1'}[arm]
 x,mask,time=base.prep(d,view,oldarm,mu[:35],sd[:35],device)
 extra=torch.zeros((*x.shape[:2],5),dtype=x.dtype,device=device)
 if arm in ('F0_heading','F1_heading'):
  z,active=heading_extra(d,view,arm)
  z=((z-mu[35:])/sd[35:]).astype(np.float32)
  z[~active]=0;z[~d['mask']]=0
  extra=torch.tensor(z,device=device)
 return torch.cat((x,extra),dim=-1),mask,time

def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

def metrics(y,p,region):
 return {'pooled_AP':float(average_precision_score(y,p)),
  'AUROC':float(roc_auc_score(y,p)),
  'macro_AP':float(np.mean([average_precision_score(y[region==r],p[region==r]) for r in np.unique(region)]))}

def run(name,n,gpu):
 seed(n);torch.set_num_threads(2);dev=torch.device('cuda:'+str(gpu))
 tr,va=read('train'),read('val');mu,sd=fit(tr)
 epochs=75 if name=='itransformer' else 25
 dest=OUT/'runs'/name/f'seed{n}';dest.mkdir(parents=True,exist_ok=False)
 spec={'name':name,'seed':n,'epochs':epochs,'lr':.001,'batch_size':128,
  'arms':ARMS,'candidate':'distance/reliability weighted heading-COG circular angle and valid coverage; previous heading angular rate and coverage',
  'control':'equal-width F1_noheading with appended five zeros; F0_heading and F1_heading both access the same heading evidence',
  'normalization':'original 35 from original train/base fit; heading train/base F0/F1 pooled; fractions unscaled; missing set zero after normalization',
  'training':'base only, fixed 75 epochs, BCE; 32 complete four-case sources per batch; source-ID-only random ordering shared across arms; no labels used for batching',
  'scope':'offline development; observations and heading not guaranteed independent; no test',
  'views':VIEWS,'mean':mu.tolist(),'std':sd.tolist(),
  'sha256':{str(p):sha(p) for p in [Path(__file__),SOURCE/'train_heading.py',CACHE/'train_heading_features.npz',CACHE/'val_heading_features.npz',CACHE/'feature_builder.py',ROOT/'relation_repair_20261030/train_relation_field.npz',ROOT/'relation_repair_20261030/val_relation_field.npz']}}
 (dest/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
 for arm in ARMS:
  seed(n);model=base.get_model(name,NF,name=='itransformer').to(dev)
  initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest()
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
  x,m,t=prep(tr,'base',arm,mu,sd,dev);y=torch.tensor(tr['y'],dtype=torch.float32,device=dev)
  hist=[];folder=dest/arm;folder.mkdir(exist_ok=False)
  source=tr['source'].astype(str);_,inv,counts=np.unique(source,return_inverse=True,return_counts=True)
  if not np.all(counts==4): raise ValueError('Every source must have exactly 4 cases')
  source_groups=[np.flatnonzero(inv==j) for j in range(len(counts))]
  if sum(map(len,source_groups))!=len(y): raise ValueError('Source groups must partition training windows')
  for epoch in range(epochs):
   model.train();order=np.random.default_rng(300000+epoch).permutation(len(source_groups));losses=[]
   for i in range(0,len(order),32):
    idx=np.concatenate([source_groups[j] for j in order[i:i+32]])
    ix=torch.tensor(idx,device=dev);opt.zero_grad(set_to_none=True)
    log=model(x[ix],m[ix],t[ix])['logits'];loss=F.binary_cross_entropy_with_logits(log,y[ix])
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
   hist.append(float(np.mean(losses)))
   if epoch%10==0: print('EPOCH',name,n,arm,epoch+1,hist[-1],flush=True)
  model.eval();result={};saved={}
  for view in VIEWS:
   xv,mv,tv=prep(va,view,arm,mu,sd,dev);pred=[]
   with torch.no_grad():
    for i in range(0,len(xv),256): pred.extend(torch.sigmoid(model(xv[i:i+256],mv[i:i+256],tv[i:i+256])['logits']).cpu().numpy())
   pred=np.asarray(pred);result[view]=metrics(va['y'],pred,va['region']);saved[view+'_score']=pred
  np.savez_compressed(folder/'val_scores.npz',y=va['y'],mmsi=va['mmsi'],region=va['region'],source=va['source'],case=va['case'],**saved)
  torch.save(model.state_dict(),folder/'final.pt')
  (folder/'result.json').write_text(json.dumps({'arm':arm,'seed':n,'init_sha256':initial,'last_epoch':epochs,'loss_history':hist,'views':result,'prediction_sha256':sha(folder/'val_scores.npz'),'weights_sha256':sha(folder/'final.pt')},indent=2),encoding='utf8')
  print('DONE',name,n,arm,result,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',choices=['tcn','gru','transformer','itransformer'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();run(a.name,a.seed,a.gpu)
