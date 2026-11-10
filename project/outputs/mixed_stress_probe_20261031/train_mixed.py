"""Development-only: source-paired observation-view exposure, baseline-compatible."""
from pathlib import Path
import os,sys,json,hashlib,argparse,random,importlib.util
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
OUT=ROOT/'mixed_stress_probe_20261031'
H=ROOT/'heading_evidence_pilot_20261030'
P=H/'train_heading_evidence.py'
spec=importlib.util.spec_from_file_location('heading_ref',P)
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
VIEWS=('base','natural','suffix','combined_l100_m600')
TRAIN_VIEWS=VIEWS[:3]
ARMS=('F0_heading','F1_heading')

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def historical(name,seed,arm):
 p=(ROOT/'itrans_sourcebalanced_20261031' if name=='itransformer' else H)/'runs'/name/('seed'+str(seed))/arm
 if not p.joinpath('result.json').is_file():raise FileNotFoundError(str(p))
 return p

def run(name,n,gpu):
 ref.seed(n);torch.set_num_threads(2);dev=torch.device('cuda:'+str(gpu))
 tr,va=ref.read('train'),ref.read('val');mu,sd=ref.fit(tr)
 epochs=75 if name=='itransformer' else 25
 dest=OUT/'runs'/name/f'seed{n}';dest.mkdir(parents=True,exist_ok=False)
 src=tr['source'].astype(str);uni,inv,cnt=np.unique(src,return_inverse=True,return_counts=True)
 if not np.all(cnt==4):raise ValueError('source must contain exactly four cases')
 groups=[np.flatnonzero(inv==j) for j in range(len(uni))]
 masks={}
 for arm in ARMS:
  ref.seed(n);model=ref.base.get_model(name,40,name=='itransformer').to(dev)
  initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest()
  old=historical(name,n,arm); baseline=json.loads((old/'result.json').read_text())
  if baseline['init_sha256']!=initial:raise ValueError('model initialization differs from baseline')
  if baseline['last_epoch']!=epochs:raise ValueError('baseline budget mismatch')
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
  ts=[ref.prep(tr,view,arm,mu,sd,dev) for view in TRAIN_VIEWS]
  if not all(torch.equal(ts[0][1],z[1]) and torch.equal(ts[0][2],z[2]) for z in ts[1:]):raise ValueError('mask/time inconsistent views')
  X=torch.stack([z[0] for z in ts],dim=0);m,t=ts[0][1:]
  y=torch.tensor(tr['y'],dtype=torch.float32,device=dev)
  fold=dest/arm;fold.mkdir(exist_ok=False);hist=[];counts=[]
  for epoch in range(epochs):
   # Draw one view per source, independently from initialization and batch RNG.
   assignment=np.random.default_rng(920000+epoch).integers(0,3,size=len(uni),dtype=np.int64)
   vid=torch.tensor(assignment[inv],device=dev)
   # Keep the exact baseline's order and grouping for this backbone.
   if name=='itransformer':
    order=np.random.default_rng(300000+epoch).permutation(len(groups))
    batches=(np.concatenate([groups[j] for j in order[i:i+32]]) for i in range(0,len(order),32))
   else:
    order=np.random.default_rng(300000+epoch).permutation(len(y))
    batches=(order[i:i+128] for i in range(0,len(order),128))
   counts.append(np.bincount(assignment,minlength=3).tolist());losses=[];model.train()
   for ixnp in batches:
    ix=torch.tensor(ixnp,device=dev);opt.zero_grad(set_to_none=True)
    logits=model(X[vid[ix],ix],m[ix],t[ix])['logits'];loss=F.binary_cross_entropy_with_logits(logits,y[ix])
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
   hist.append(float(np.mean(losses)))
   if epoch%10==0:print('EPOCH',name,n,arm,epoch+1,hist[-1],flush=True)
  model.eval();results={};saved={}
  for view in VIEWS:
   xv,mv,tv=ref.prep(va,view,arm,mu,sd,dev);p=[]
   with torch.no_grad():
    for i in range(0,len(xv),256):p.extend(torch.sigmoid(model(xv[i:i+256],mv[i:i+256],tv[i:i+256])['logits']).cpu().numpy())
   p=np.asarray(p);results[view]=ref.metrics(va['y'],p,va['region']);saved[view+'_score']=p
  np.savez_compressed(fold/'val_scores.npz',y=va['y'],mmsi=va['mmsi'],region=va['region'],source=va['source'],case=va['case'],**saved)
  torch.save(model.state_dict(),fold/'final.pt')
  result={'arm':arm,'seed':n,'last_epoch':epochs,'init_sha256':initial,'view_source_counts':counts,'loss_history':hist,'views':results,'prediction_sha256':sha(fold/'val_scores.npz'),'weights_sha256':sha(fold/'final.pt'),'baseline_prediction_sha256':sha(old/'val_scores.npz'),'baseline_weights_sha256':sha(old/'final.pt')}
  (fold/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('DONE',name,n,arm,results,flush=True)
 spec={'name':name,'seed':n,'epochs':epochs,'lr':.001,'batch_size':128,'arms':ARMS,'model_input':'40dim legacy35 plus neighbor heading 5','normalization':'train base only original baseline pooled F0/F1','view_rng':'source sorted np.unique, random uniform 0/1/2 via default_rng(920000+epoch), shared across arms, no labels','views':list(TRAIN_VIEWS),'batch':'itransformer source-grouped32 else original window permutation; default_rng(300000+epoch)','no_test':True,'hashes':{str(p):sha(p) for p in [Path(__file__),P,H/'train_heading_features.npz',H/'val_heading_features.npz',ROOT/'relation_repair_20261030/train_relation_field.npz',ROOT/'relation_repair_20261030/val_relation_field.npz']}}
 (dest/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',choices=['tcn','gru','transformer','itransformer'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();run(a.name,a.seed,a.gpu)
