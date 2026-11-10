"""Target-vessel heading extra evidence; train/validation only, offline relation task."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys, json, random, hashlib, argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch.nn import functional as F

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
PILOT=ROOT/'heading_evidence_pilot_20261030'
COVER=ROOT/'heading_coverage_20261030'
OUT=ROOT/'target_heading_20261031'
sys.path.insert(0,str(PILOT))
import train_heading_evidence as h
VIEWS=('base','natural','suffix','combined_l100_m600')
ARMS=('C0_target','F0_target','F1_no_target','F1_target')
NF=45

def sha(p):
 m=hashlib.sha256()
 with open(p,'rb') as f:
  for q in iter(lambda:f.read(1048576),b''):m.update(q)
 return m.hexdigest()

def load(split):
 d=h.read(split)
 with np.load(COVER/f'heading_{split}.npz',allow_pickle=False) as z:
  ids=z['window_id'].astype(str); mapping={x:i for i,x in enumerate(ids)}
  if len(mapping)!=len(ids):raise ValueError('duplicate source heading')
  ix=np.array([mapping[str(s)] for s in d['source']],np.int64)
  ht=z['target_t'][ix]
  if not np.array_equal(ht,d['time_s']): raise ValueError('target time mismatch')
  if not np.array_equal(z['target_mmsi'][ix].astype(str),d['mmsi'].astype(str)):raise ValueError('target MMSI mismatch')
  if not np.array_equal(z['target_length'][ix],d['length']):raise ValueError('target length mismatch')
  value=z['target_heading'][ix].astype('float64'); valid=z['target_valid'][ix] & d['mask']
 if not np.isfinite(value[valid]).all() or not np.all((value[valid]>=0)&(value[valid]<360)):raise ValueError('invalid heading')
 # Source-based heading is shared across 4 variants; COG is generated/corrupted view.
 heading=np.deg2rad(np.where(valid,value,0.0))
 s,c=d['causal7'][...,5],d['causal7'][...,6]
 angle_sin=np.sin(heading)*c-np.cos(heading)*s
 angle_cos=np.cos(heading)*c+np.sin(heading)*s
 dt=np.diff(d['time_s'],axis=1)
 rate_valid=valid[:,1:]&valid[:,:-1]&(dt>0)&(dt<=300)
 delta=(value[:,1:]-value[:,:-1]+180.)%360.-180.
 rate=np.zeros_like(value);np.divide(delta,dt,out=rate[:,1:],where=rate_valid)
 rate_valid=np.column_stack((np.zeros(len(valid),bool),rate_valid))
 x=np.stack((angle_sin,angle_cos,valid,rate,rate_valid),axis=-1).astype(np.float32)
 active=np.stack((valid,valid,d['mask'],rate_valid,d['mask']),axis=-1)
 x[~active]=0;x[~d['mask']]=0
 assert np.isfinite(x).all()
 d['target_heading_feature']=x
 d['target_heading_active']=active
 return d

def fit(d):
 m,s=h.fit(d)
 x,a=d['target_heading_feature'],d['target_heading_active']
 mt=np.zeros(5);st=np.ones(5)
 for j in (0,1,3):
  z=x[...,j][a[...,j]].astype(np.float64)
  if len(z)==0:raise ValueError('empty valid heading')
  mt[j]=z.mean();st[j]=max(z.std(),1e-7)
 return np.r_[m,mt],np.r_[s,st]

def prep(d,view,arm,mu,sd,dev):
 underlying={'C0_target':'C0','F0_target':'F0_heading','F1_no_target':'F1_heading','F1_target':'F1_heading'}[arm]
 x,mask,time=h.prep(d,view,underlying,mu[:40],sd[:40],dev)
 extra=torch.zeros((*x.shape[:2],5),device=dev,dtype=x.dtype)
 if arm!='F1_no_target':
  z=(d['target_heading_feature']-mu[40:])/sd[40:]
  z=z.astype(np.float32);z[~d['target_heading_active']]=0;z[~d['mask']]=0
  extra=torch.tensor(z,device=dev)
 return torch.cat((x,extra),-1),mask,time

def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

def run(name,n,gpu,balance):
 seed(n);torch.set_num_threads(2);dev=torch.device('cuda:'+str(gpu))
 tr,va=load('train'),load('val');mu,sd=fit(tr)
 target=OUT/'runs'/balance/name/f'seed{n}';target.mkdir(parents=True,exist_ok=False)
 epochs=75 if name=='itransformer' else 25
 deps=[Path(__file__),PILOT/'train_heading_evidence.py',PILOT/'train_heading_features.npz',PILOT/'val_heading_features.npz',COVER/'heading_train.npz',COVER/'heading_val.npz',ROOT/'relation_repair_20261030/train_relation_field.npz',ROOT/'relation_repair_20261030/val_relation_field.npz',ROOT/'heading_alignment_20261030/train_heading.py',ROOT/'framework_validation_v1_20261028/models.py']
 spec={'name':name,'seed':n,'epochs':epochs,'balance':balance,'arms':ARMS,'learning_rate':.001,'optimizer':'AdamW wd .01 clip1 batch128','features':'target observed heading minus paired target COG, sin/cos; target heading rate deg/s and validity masks; Heading only from source records at exact MMSI and UTC; common neighbor Heading present in F0/F1','heading_missing':'zero with validity masks; train-only normalization','scope':'exploratory offline validation, shared target heading across quartet; natural headings not proven independent nor physically synchronized with synthetic trajectories','hashes':{str(p):sha(p) for p in deps},'normalizer_mean':mu.tolist(),'normalizer_std':sd.tolist(),'fixed_last_epoch':True}
 (target/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
 src=tr['source'].astype(str);unique,inv,counts=np.unique(src,return_inverse=True,return_counts=True)
 if not np.all(counts==4):raise ValueError('source count not 4')
 groups=np.stack([np.flatnonzero(inv==j) for j in range(len(unique))])
 for arm in ARMS:
  seed(n);model=h.base.get_model(name,NF,name=='itransformer').to(dev)
  initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest()
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
  x,m,t=prep(tr,'base',arm,mu,sd,dev);y=torch.tensor(tr['y'],dtype=torch.float32,device=dev)
  folder=target/arm;folder.mkdir(exist_ok=False);hist=[]
  for epoch in range(epochs):
   model.train();rng=np.random.default_rng(300000+epoch)
   order=groups[rng.permutation(len(groups))].reshape(-1) if balance=='source' else rng.permutation(len(y))
   losses=[]
   for i in range(0,len(order),128):
    ix=torch.tensor(order[i:i+128],device=dev);opt.zero_grad(set_to_none=True)
    log=model(x[ix],m[ix],t[ix])['logits'];loss=F.binary_cross_entropy_with_logits(log,y[ix])
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
   hist.append(float(np.mean(losses)))
   if epoch%10==0:print('EPOCH',name,n,arm,epoch+1,hist[-1],flush=True)
  model.eval();result={};saved={}
  for view in VIEWS:
   xv,mv,tv=prep(va,view,arm,mu,sd,dev);pred=[]
   with torch.no_grad():
    for i in range(0,len(xv),256):pred.extend(torch.sigmoid(model(xv[i:i+256],mv[i:i+256],tv[i:i+256])['logits']).cpu().numpy())
   pred=np.asarray(pred);result[view]=h.metrics(va['y'],pred,va['region']);saved[view+'_score']=pred
  np.savez_compressed(folder/'val_scores.npz',y=va['y'],mmsi=va['mmsi'],region=va['region'],source=va['source'],case=va['case'],**saved)
  torch.save(model.state_dict(),folder/'final.pt')
  (folder/'result.json').write_text(json.dumps({'arm':arm,'seed':n,'init_sha256':initial,'last_epoch':epochs,'loss_history':hist,'views':result,'prediction_sha256':sha(folder/'val_scores.npz'),'weights_sha256':sha(folder/'final.pt')},indent=2),encoding='utf8')
  print('DONE',name,n,arm,result,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',choices=['tcn','gru','transformer','itransformer'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--balance',choices=['source','window'],default='window');a=p.parse_args();run(a.name,a.seed,a.gpu,a.balance)
