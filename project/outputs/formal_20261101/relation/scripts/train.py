#!/usr/bin/env python3
"""Frozen relation study: 35D stage cumulative, fixed 25/75 epoch, paired C0/F0/F1."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import argparse,hashlib,json,random,sys,time
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
OUT=ROOT/'formal_20261101/relation'
PRE=ROOT/'eight_backbone_preparation_20261030'
HEADING=ROOT/'heading_alignment_20261030'
sys.path.insert(0,str(PRE));sys.path.insert(0,str(HEADING))
from relation8 import model_for
import train_heading as base

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def state_hash(m):
 return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(m.state_dict().items()))).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph']);p.add_argument('--seed',required=True,type=int,choices=[0,1,2]);p.add_argument('--gpu',required=True,type=int,choices=[3,5,6,7]);a=p.parse_args()
 lock=json.loads((OUT/'protocol.lock.json').read_text(encoding='utf8'))
 for k,v in lock['source_sha256'].items():
  if sha(Path(k))!=v:raise ValueError('source hash changed '+k)
 seed(a.seed);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
 dev=torch.device('cuda:'+str(a.gpu));base.VARIANT='aligned_embed'
 tr,va=base.read('train'),base.read('val');mu,sd=base.fit(tr)
 assert len(tr['y'])==8136 and len(va['y'])==1088
 assert not(set(tr['mmsi'].astype(str)) & set(va['mmsi'].astype(str)))
 groups=None
 if a.name in ('itransformer','patchtst'):
  _,inv,n=np.unique(tr['source'].astype(str),return_inverse=True,return_counts=True)
  assert np.all(n==4)
  groups=[np.flatnonzero(inv==j) for j in range(len(n))]
 epochs=75 if a.name=='itransformer' else 25
 dest=OUT/'runs'/a.name/f'seed{a.seed}';dest.mkdir(parents=True,exist_ok=False)
 spec={'name':a.name,'seed':a.seed,'epochs':epochs,'arms':['C0','F0','F1'],'input_dim':35,'variant':'aligned_embed','ais_heading_channels':False,'optimizer':'AdamW lr .001 wd .01; gradient clip 1','train_view':'base only; original phase cumulative field','batch':'source quartet groups 32' if groups is not None else '128 windows','order':'np.random.default_rng(300000+epoch), shared across seeds and arms','mask':'existing cohort and input schema','selection':'fixed last epoch, no test use','seed_policy':'matched random initialization per arm','mean':mu.tolist(),'std':sd.tolist(),'code_data_lock_sha256':sha(OUT/'protocol.lock.json'),'device':str(dev),'torch':torch.__version__}
 (dest/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
 for arm in ('C0','F0','F1'):
  seed(a.seed);model=model_for(a.name).to(dev);initial=state_hash(model)
  decay=[p for p in model.parameters() if not getattr(p,'_no_weight_decay',False)];nodecay=[p for p in model.parameters() if getattr(p,'_no_weight_decay',False)]
  opt=torch.optim.AdamW([{'params':decay,'weight_decay':.01}]+([{'params':nodecay,'weight_decay':0.}] if nodecay else []),lr=.001)
  x,m,t=base.prep(tr,'base',arm,mu,sd,dev);y=torch.tensor(tr['y'],dtype=torch.float32,device=dev)
  folder=dest/arm;folder.mkdir(exist_ok=False);hist=[];start=time.time()
  for epoch in range(epochs):
   model.train();order=np.random.default_rng(300000+epoch).permutation(len(groups) if groups is not None else len(y));ls=[]
   for ix0 in range(0,len(order),32 if groups is not None else 128):
    ids=np.concatenate([groups[j] for j in order[ix0:ix0+32]]) if groups is not None else order[ix0:ix0+128]
    ix=torch.as_tensor(ids,device=dev);opt.zero_grad(set_to_none=True)
    loss=F.binary_cross_entropy_with_logits(model(x[ix],m[ix],t[ix])['logits'],y[ix])
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();ls.append(float(loss.detach()))
   hist.append(float(np.mean(ls)))
   if epoch%10==0:print('EPOCH',a.name,a.seed,arm,epoch+1,hist[-1],flush=True)
  model.eval();results={};saved={}
  with torch.no_grad():
   for view in base.VIEWS:
    xv,mv,tv=base.prep(va,view,arm,mu,sd,dev);pred=[]
    for i in range(0,len(xv),256):pred.extend(torch.sigmoid(model(xv[i:i+256],mv[i:i+256],tv[i:i+256])['logits']).cpu().numpy())
    pred=np.asarray(pred);results[view]=base.metrics(va['y'],pred,va['region']);saved[view+'_score']=pred
  np.savez_compressed(folder/'val_scores.npz',y=va['y'],mmsi=va['mmsi'],region=va['region'],source=va['source'],case=va['case'],**saved)
  torch.save(model.state_dict(),folder/'final.pt')
  result={'name':a.name,'arm':arm,'seed':a.seed,'init_sha256':initial,'last_epoch':epochs,'loss_history':hist,'views':results,'elapsed_seconds':time.time()-start,'prediction_sha256':sha(folder/'val_scores.npz'),'weights_sha256':sha(folder/'final.pt')}
  (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
  print('DONE',a.name,a.seed,arm,results,flush=True)
 if len({json.loads((dest/arm/'result.json').read_text())['init_sha256'] for arm in ('C0','F0','F1')})!=1:raise ValueError('initial weight mismatch')
if __name__=='__main__':main()
