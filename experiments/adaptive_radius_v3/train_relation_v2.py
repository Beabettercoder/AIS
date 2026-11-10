"""Four-case relation detection with conditionally rebuilt 35D inputs."""
from pathlib import Path
import argparse,hashlib,json,random,sys
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from tqdm import tqdm
from radius_core import ARMS
from build_relation_v2 import VIEWS,REGIONS,ROOT,sha
sys.path.insert(0,str(ROOT/'heading_alignment_20261030'));sys.path.insert(0,str(ROOT/'eight_backbone_preparation_20261030'))
import train_heading as base
from relation8 import model_for
base.VARIANT='aligned_embed'

def data(root,split):
 rows=[];hashes={}
 for region in REGIONS:
  p=root/'relation'/f'{split}_{region}.npz';meta=p.with_suffix('.json');d=json.loads(meta.read_text())
  if sha(p)!=d['sha256']:raise ValueError('relation checksum mismatch '+str(p))
  hashes[str(p)]=d['sha256'];hashes[str(meta)]=sha(meta)
  with np.load(p,allow_pickle=False) as z:rows.append({k:z[k] for k in z.files})
 all={k:np.concatenate([r[k] for r in rows]) for k in rows[0]};all['mask']=np.arange(20)[None,:]<all['length'][:,None]
 return all,hashes
def feature(d,view,arm):
 # Reuse prior heading/stage-cumulative feature construction, but direct it to
 # this arm's actual field and rho. F1 is just the old routine's input slot.
 key=view+'__'+arm;proxy={k:d[k] for k in ('mask','causal7','time_s')};x=d[key+'__field']
 proxy[view+'__field']=np.stack((x,x),axis=1);proxy[view+'__count']=d[key+'__count']
 return base.features(proxy,view,'F1')
def normalization(d):
 x,_=feature(d,'base','target_only');target=x[d['mask'],:8].astype(np.float64);parts=[]
 for arm in ('tight_plain','wide_plain','adaptive_plain_nofallback','adaptive_plain_fallback','adaptive_plain_deg6'):
  x,valid=feature(d,'base',arm);parts.append(x[valid,8:].astype(np.float64))
 context=np.concatenate(parts);mean=np.zeros(35);std=np.ones(35)
 mean[:8]=target.mean(0);std[:8]=target.std(0);mean[8:]=context.mean(0);std[8:]=context.std(0);mean[22:24]=0;std[22:24]=1;std[std<1e-7]=1
 return {'mean':mean,'std':std}
def prepare(d,view,arm,n,device):
 x,valid=feature(d,view,arm);x=(x-n['mean'])/n['std'];x[...,8:]*=valid[...,None];x[~d['mask']]=0
 t=d['time_s'].astype(np.float64)-d['time_s'][:,:1];t[~d['mask']]=0
 if not np.isfinite(x).all():raise ValueError('nonfinite input')
 return torch.as_tensor(x,dtype=torch.float32,device=device),torch.as_tensor(d['mask'],device=device),torch.as_tensor(t,dtype=torch.float32,device=device)
def predict(model,values,batch):
 model.eval();out=[];x,m,t=values
 with torch.no_grad():
  for st in range(0,len(x),batch):out.extend(torch.sigmoid(model(x[st:st+batch],m[st:st+batch],t[st:st+batch])['logits']).cpu().numpy())
 return np.asarray(out)
def metrics(y,p,thresholds=None):
 out={'AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p))}
 if thresholds:
  for k,t in thresholds.items():out['FAR@'+k]=float(np.mean(p[y==0]>=t));out['recall@'+k]=float(np.mean(p[y==1]>=t))
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',choices=('tcn','gru','transformer','graph'),required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--root',type=Path,default=Path('/home/ps/AIS_adaptive_radius_relation_v3_20261109'));p.add_argument('--mode',choices=('train','test'),default='train');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_num_threads(2);dev=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
 tr,trhash=data(a.root,'train');va,vahash=data(a.root,'val')
 if set(tr['mmsi'].astype(str))&set(va['mmsi'].astype(str)):raise ValueError('target MMSI train/val overlap')
 norm=normalization(tr);xt,mt,tt=prepare(tr,'base',a.arm,norm,dev);xv,mv,tv=prepare(va,'base',a.arm,norm,dev)
 if a.smoke:xt,mt,tt=xt[:128],mt[:128],tt[:128];xv,mv,tv=xv[:128],mv[:128],tv[:128]
 y=tr['y'][:len(xt)];yv=va['y'][:len(xv)];model=model_for(a.name).to(dev);out=a.root/('smoke_runs' if a.smoke else 'runs')/'relation'/a.name/a.arm/f'seed{a.seed}'
 if a.mode=='test':
  manifest=json.loads((a.root/'frozen_manifest.json').read_text());key=f'relation/{a.name}/{a.arm}/seed{a.seed}'
  if key not in manifest['runs']:raise ValueError('run not frozen')
  for path,digest in manifest['source_sha256'].items():
   if sha(path)!=digest:raise ValueError('source changed '+path)
  for path,digest in (trhash|vahash).items():
   if manifest['input_sha256'].get(path)!=digest:raise ValueError('input changed '+path)
  te,tehash=data(a.root,'test')
  for path,digest in tehash.items():
   if manifest['input_sha256'].get(path)!=digest:raise ValueError('test changed '+path)
  if set(te['mmsi'].astype(str))&(set(tr['mmsi'].astype(str))|set(va['mmsi'].astype(str))):raise ValueError('target MMSI test overlap')
  for ckname in ('last','best_val'):
   path=out/f'{ckname}.pt'
   if sha(path)!=manifest['runs'][key][ckname]:raise ValueError('checkpoint changed '+key)
   model.load_state_dict(torch.load(path,map_location=dev,weights_only=False)['model']);pv=predict(model,(xv,mv,tv),256);neg=pv[yv==0];th={f'{q}%':float(np.quantile(neg,1-q/100,method='higher')) for q in (1,5)}
   saved={k:te[k] for k in ('source','case','y','mmsi','region','length','sparse_source','base_count8')};res={'validation':metrics(yv,pv,th),'thresholds':th,'views':{}}
   for view in VIEWS:
    scores=predict(model,prepare(te,view,a.arm,norm,dev),256);saved[view]=scores;saved[view+'_hit']=te[view+'__hit'];res['views'][view]=metrics(te['y'],scores,th)
   np.savez_compressed(out/f'test_{ckname}_scores.npz',**saved);(out/f'test_{ckname}_metrics.json').write_text(json.dumps(res,indent=2),encoding='utf8');print('TEST',a.name,a.arm,a.seed,ckname,json.dumps(res),flush=True)
  return
 if out.exists() and not a.smoke:raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=True);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);init=hashlib.sha256(b''.join(q.detach().cpu().numpy().tobytes() for q in model.parameters())).hexdigest()
 spec={'name':a.name,'arm':a.arm,'seed':a.seed,'epochs':a.epochs,'batch_size':a.batch_size,'initial_parameter_sha256':init,'same_seed_order':'source-balanced quartets; np.random.default_rng(300000+epoch)','optimizer':'AdamW .001 wd .01; clip1','objective':'BCE on all controlled quartet variants','context':'35D prior stage-cumulative feature mapping; per-arm qualification/radius/fallback','normalization':'train-only pooled target and ordinary arm context; no neighbor reliability in ordinary normalization','primary_checkpoint':'fixed last epoch25; best validation pooled AP saved','scope':'historically selected quartet cohort; re-query all geometrically admissible ambient ships within 8nm, no reliability screening for ordinary field','test':'previously used'}
 (out/'protocol.json').write_text(json.dumps(spec,indent=2));best=-1;hist=[];Y=torch.as_tensor(y,dtype=torch.float32,device=dev)
 # Quartets are built consecutively, keeping same-source controlled cases in the same batch.
 if len(y)%4:raise ValueError('incomplete source quartet')
 groups=np.arange(len(y)).reshape(-1,4)
 for epoch in range(a.epochs):
  model.train();order=np.random.default_rng(300000+epoch).permutation(len(groups));losses=[];groups_per_batch=max(1,a.batch_size//4)
  bar=tqdm(range(0,len(order),groups_per_batch),desc=f'relation/{a.name}/{a.arm} seed={a.seed} {epoch+1}/{a.epochs}',file=sys.stdout,dynamic_ncols=True)
  for st in bar:
   ids=groups[order[st:st+groups_per_batch]].reshape(-1);ix=torch.as_tensor(ids,device=dev);opt.zero_grad(set_to_none=True)
   loss=F.binary_cross_entropy_with_logits(model(xt[ix],mt[ix],tt[ix])['logits'],Y[ix])
   if not torch.isfinite(loss):raise ValueError('nonfinite loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()));bar.set_postfix(loss=f'{losses[-1]:.4f}')
  pv=predict(model,(xv,mv,tv),256);m=metrics(yv,pv);entry={'epoch':epoch+1,'loss':float(np.mean(losses)),**m};hist.append(entry)
  ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'norm':norm,'spec':spec};torch.save(ck,out/'last.pt')
  if m['AP']>best:best=m['AP'];torch.save(ck,out/'best_val.pt');np.savez_compressed(out/'val_scores.npz',score=pv,y=yv,mmsi=va['mmsi'][:len(yv)],region=va['region'][:len(yv)],source=va['source'][:len(yv)],case=va['case'][:len(yv)])
  (out/'history.json').write_text(json.dumps(hist,indent=2));print('VALIDATION',json.dumps(entry),flush=True)
if __name__=='__main__':main()
