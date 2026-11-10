"""Evaluate every predeclared final checkpoint once after validation freeze."""
from pathlib import Path
import json,hashlib,csv,sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=ROOT/'causal_gate_main_v1_20261027';BASE=ROOT/'labeling/labeled_v1/features_scene';FOLDER=OUT/'test_background'
import train_controls as train
from gate_builder import causal
regions=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
KNOT=1852/3600
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prep():
 cap=json.loads((OUT/'gate_cap.json').read_text())['cap_mps'];parts=[];proof=[]
 for region in regions:
  p=BASE/region/'test.npz';q=FOLDER/region/'test.npz'
  with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
  with np.load(q,allow_pickle=False) as z:f={k:z[k] for k in z.files}
  assert np.array_equal(d['window_id'],f['window_id']); mask=d['mask'];N,T=mask.shape
  raw=np.zeros((N,T,7),np.float32);raw[...,:2]=(d['target_pos']-d['target_pos'][:,0:1,:]).astype(np.float32)
  angle=np.deg2rad(d['target_cog_deg']);speed=d['target_sog_kn'];raw[...,2]=speed*KNOT*np.sin(angle);raw[...,3]=speed*KNOT*np.cos(angle);raw[...,4]=speed;raw[...,5]=np.sin(angle);raw[...,6]=np.cos(angle);raw[~mask]=0
  w=d['self_gate_weight'].astype(np.float32).copy();w[~mask]=0;valid=d['kinematic_valid'].astype(np.float32);common=f['common_valid'].astype(np.float32)
  old=raw.copy();comp=raw.copy();cand=raw.copy()
  for i in range(N):
   n=int(mask[i].sum());dp=np.diff(raw[i,:,:2],axis=0,prepend=np.zeros((1,2)))*w[i,:,None];old[i,:,:2]=np.cumsum(dp,axis=0)
   for j in range(1,T):
    if w[i,j]<.2:old[i,j,2:7]=old[i,j-1,2:7]
   comp[i]=old[i];cand[i]=old[i];current=np.zeros(2)
   for j in range(1,n):
    dx=raw[i,j,:2].astype(float)-raw[i,j-1,:2].astype(float);delta=d['time_s'][i,j]-d['time_s'][i,j-1];step=dx*float(w[i,j])
    if valid[i,j] and np.isfinite(delta) and delta>0 and np.isfinite(raw[i,j-1:j+1,2:4]).all():
     step+=(1-float(w[i,j]))*.5*(raw[i,j-1,2:4].astype(float)+raw[i,j,2:4].astype(float))*delta
    current+=step;comp[i,j,:2]=current
   comp[i,n:,:2]=current;new,_=causal(raw[i],w[i],mask[i],valid[i],d['time_s'][i],cap);cand[i,:,:2]=new[:,:2]
  zero=np.zeros_like(w);zero2=np.zeros((N,T,2),np.float32)
  def pack(x,field,ww,vv,av):return np.concatenate([x,field,ww[...,None],vv[...,None],av[...,None]],-1).astype(np.float32)
  features={'raw':pack(raw,zero2,zero,zero,zero),'old':pack(old,zero2,w,valid,zero),
            'comp':pack(comp,zero2,w,valid,zero),'causal':pack(cand,zero2,w,valid,zero),
            'causal_common':pack(cand,zero2,w,valid,common)}
  for name in ('affine','mean','median','weighted_median'):
   features[name]=pack(cand,f['residual_'+name+'_mps'],w,valid,common)
  for name in ('causal_common','affine','weighted_median'):features[name+'_bundle']=features[name]
  part={k:d[k] for k in ('window_id','base_window_id','mmsi','region','y','mask','behavior_type','noise_type','target_noise_effective')};part['features']=features;parts.append(part)
  proof.append({'region':region,'data_sha256':sha(p),'field_sha256':sha(q),'windows':N})
 data={k:np.concatenate([d[k] for d in parts]) for k in parts[0] if k!='features'}
 data['features']={k:np.concatenate([d['features'][k] for d in parts]) for k in train.arms}
 assert len(data['y'])==1680 and len(set(data['window_id']))==1680
 return data,proof

def metric(y,s,regions,thresholds):
 by={r:float(average_precision_score(y[regions==r],s[regions==r])) for r in sorted(set(regions))}
 out={'pooled_AP':float(average_precision_score(y,s)),'pooled_AUROC':float(roc_auc_score(y,s)),'macro_AP':float(np.mean(list(by.values()))),'region_AP':by}
 for alpha,ob in thresholds.items():
  t=ob['threshold'];out[alpha]={'threshold':t,'actual_FAR':float(np.mean(s[y==0]>=t)),'recall':float(np.mean(s[y==1]>=t)),'false_alarms':int(np.sum(s[y==0]>=t)),'negative_windows':int(np.sum(y==0)),'positive_windows':int(np.sum(y==1))}
 return out

def boot_weights(y,mmsi,B=2000):
 ids,inv=np.unique(mmsi,return_inverse=True);rng=np.random.default_rng(20261027)
 counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=B).astype(np.float64)
 return counts[:,inv],ids

def bootstrap_one(y,s,weights,thresholds):
 order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];ww=weights[:,order];ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
 cum=np.cumsum(ww,axis=1)[:,ends];tp=np.cumsum(ww*yy[None,:],axis=1)[:,ends];prev=np.concatenate([np.zeros((len(weights),1)),tp[:,:-1]],axis=1)
 ap=np.sum((tp-prev)*np.divide(tp,cum,out=np.zeros_like(tp),where=cum>0),axis=1)/np.maximum(tp[:,-1],1)
 out={'AP':ap}
 for alpha,ob in thresholds.items():
  pp=(s>=ob['threshold']);numpos=np.sum(weights*(y==1),axis=1);numneg=np.sum(weights*(y==0),axis=1)
  out['recall_'+alpha]=np.divide(np.sum(weights*((y==1)&pp),axis=1),numpos,out=np.zeros(len(weights)),where=numpos>0)
  out['FAR_'+alpha]=np.divide(np.sum(weights*((y==0)&pp),axis=1),numneg,out=np.zeros(len(weights)),where=numneg>0)
 return out

def main():
 planp=OUT/'frozen_test_plan.json';plan=json.loads(planp.read_text());assert plan['status']=='frozen_before_test'
 assert plan['test_evaluator_sha256']==sha(Path(__file__)) and plan['test_background_builder_sha256']==sha(OUT/'build_backgrounds_test.py')
 dest=OUT/'formal_test_results';dest.mkdir(exist_ok=False);data,proof=prep();y=data['y'].astype(int);weights,ids=boot_weights(y,data['mmsi'].astype(str));torch.set_num_threads(4)
 output=[];predictions=[];boots={}
 for run in plan['runs']:
  ckpt=Path(run['checkpoint']);assert sha(ckpt)==run['checkpoint_sha256'];saved=torch.load(ckpt,map_location='cpu',weights_only=False)
  model=train.hm.HistoricalModel(run['architecture'],run['arm']);model.load_state_dict(saved['state_dict']);model.eval()
  x=(data['features'][run['arm']]-saved['normalizer_mean'].numpy())/saved['normalizer_std'].numpy();p=[]
  with torch.no_grad():
   for lo in range(0,len(y),128):
    xb=torch.from_numpy(x[lo:lo+128]);m=torch.from_numpy(data['mask'][lo:lo+128]);score,*_=model(xb,m);p.extend(torch.sigmoid(score).numpy().tolist())
  s=np.asarray(p);predictions.append(s);met=metric(y,s,data['region'].astype(str),run['thresholds']);bb=bootstrap_one(y,s,weights,run['thresholds'])
  # Exact AP implementation check against scikit-learn on three fixed cluster draws.
  for j in (0,1,2):assert abs(bb['AP'][j]-average_precision_score(y,s,sample_weight=weights[j]))<1e-10
  met['bootstrap_95']={k:np.quantile(v,[.025,.975]).tolist() for k,v in bb.items()}
  key=(run['architecture'],run['seed'],run['arm']);boots[key]=bb
  output.append({'architecture':run['architecture'],'seed':run['seed'],'arm':run['arm'],**met})
  print('TEST',*key,round(met['macro_AP'],5),flush=True)
 comparisons=[]
 for arch in ('tcn','gru'):
  for seed in (0,1,2):
   for a,b in [('causal','raw'),('weighted_median','causal_common'),('weighted_median','affine'),('weighted_median_bundle','weighted_median')]:
    A=boots[(arch,seed,a)];B=boots[(arch,seed,b)];diff=A['AP']-B['AP']
    comparisons.append({'architecture':arch,'seed':seed,'a':a,'b':b,'pooled_AP_difference_CI95':np.quantile(diff,[.025,.975]).tolist()})
 np.savez_compressed(dest/'all_test_predictions.npz',scores=np.asarray(predictions),window_id=data['window_id'],mmsi=data['mmsi'],region=data['region'],y=y,
                     architecture=np.array([r['architecture'] for r in plan['runs']]),seed=np.array([r['seed'] for r in plan['runs']]),arm=np.array([r['arm'] for r in plan['runs']]))
 out={'status':'complete','scope':'historically exposed test split, not untouched blind test', 'frozen_plan_sha256':sha(planp),'bootstrap_unit':'target MMSI','bootstrap_draws':len(weights),'test_unique_target_mmsi':len(ids),'test_windows':len(y),'proof':proof,'results':output,'paired_AP_intervals':comparisons}
 (dest/'test_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');print('TEST_ALL_COMPLETE',flush=True)
if __name__=='__main__':main()
