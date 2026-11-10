"""Frozen observational-model cross-over: no retraining and no test access."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
OUT=ROOT/'mixed_context_swap_20261031'
MIX=ROOT/'mixed_stress_probe_20261031'
H=ROOT/'heading_evidence_pilot_20261030'
spec=importlib.util.spec_from_file_location('heading_frozen',H/'train_heading_evidence.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
VIEWS=('base','natural','suffix','combined_l100_m600')
ARMS=('F0_heading','F1_heading')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()

def weighted_ap(y,score,weight):
 ix=np.argsort(-score,kind='mergesort');s=score[ix];w=weight[ix].astype(float); yy=y[ix].astype(float)
 ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
 tp=np.cumsum(w*yy)[ends];den=np.cumsum(w)[ends]
 if tp[-1]<=0:return np.nan
 return float((np.diff(np.r_[0.,tp])*np.divide(tp,den,out=np.zeros_like(tp),where=den>0)).sum()/tp[-1])

def main():
 torch.set_num_threads(2);dev=torch.device('cuda:5')
 tr=ref.read('train'); va=ref.read('val');mu,sd=ref.fit(tr)
 meta={'y':va['y'],'mmsi':va['mmsi'],'region':va['region'],'source':va['source'],'case':va['case']}
 uniq,inv=np.unique(meta['mmsi'].astype(str),return_inverse=True)
 draws=np.random.default_rng(20261031).integers(0,len(uniq),size=(2000,len(uniq)))
 combos={'F0_on_F0':('F0_heading','F0_heading'),'F0_on_F1':('F0_heading','F1_heading'),
         'F1_on_F0':('F1_heading','F0_heading'),'F1_on_F1':('F1_heading','F1_heading')}
 contrasts={'F0_model_switch_field':('F0_on_F1','F0_on_F0'),'F1_model_switch_field':('F1_on_F1','F1_on_F0'),
            'F0_field_switch_model':('F1_on_F0','F0_on_F0'),'F1_field_switch_model':('F1_on_F1','F0_on_F1'),
            'native_F1_minus_F0':('F1_on_F1','F0_on_F0')}
 result={}
 for name in ('tcn','gru','transformer','itransformer'):
  seeds=(0,1,2) if name in ('tcn','itransformer') else (0,)
  scores={}; hashinfo={};diagonal_max=0.
  prepared={}
  for view in VIEWS:
   for arm in ARMS:
    prepared[(view,arm)]=ref.prep(va,view,arm,mu,sd,dev)
   assert torch.equal(prepared[(view,ARMS[0])][1],prepared[(view,ARMS[1])][1])
   assert torch.equal(prepared[(view,ARMS[0])][2],prepared[(view,ARMS[1])][2])
  for seed in seeds:
   for netarm in ARMS:
    folder=MIX/'runs'/name/f'seed{seed}'/netarm
    rec=json.loads((folder/'result.json').read_text())
    assert rec['last_epoch']==(75 if name=='itransformer' else 25)
    assert rec['weights_sha256']==sha(folder/'final.pt') and rec['prediction_sha256']==sha(folder/'val_scores.npz')
    with np.load(folder/'val_scores.npz',allow_pickle=False) as z:
     stored={k:z[k] for k in z.files}
    for k,v in meta.items():np.testing.assert_array_equal(stored[k].astype(str),v.astype(str))
    model=ref.base.get_model(name,40,name=='itransformer').to(dev)
    model.load_state_dict(torch.load(folder/'final.pt',map_location=dev,weights_only=True));model.eval()
    for view in VIEWS:
     for featurearm in ARMS:
      x,m,t=prepared[(view,featurearm)];p=[]
      with torch.no_grad():
       for j in range(0,len(x),256):p.extend(torch.sigmoid(model(x[j:j+256],m[j:j+256],t[j:j+256])['logits']).cpu().numpy())
      p=np.asarray(p); scores[(seed,netarm,featurearm,view)]=p
      if featurearm==netarm:
       diff=float(np.max(np.abs(p-stored[view+'_score'])))
       diagonal_max=max(diagonal_max,diff)
       if diff>1e-5:raise AssertionError(f'native prediction mismatch {name} {seed} {netarm} {view}: {diff}')
    hashinfo[f'{seed}:{netarm}']={'checkpoint':sha(folder/'final.pt'),'native_prediction':sha(folder/'val_scores.npz'),'init':rec['init_sha256']}
  save={k:np.asarray(v) for k,v in meta.items()}
  for (seed,netarm,featurearm,view),score in scores.items():save[f's{seed}__{netarm}__{featurearm}__{view}']=score
  dest=OUT/f'{name}_scores.npz';np.savez_compressed(dest,**save)
  out={'seeds':list(seeds),'n_mmsi':len(uniq),'n_windows':len(meta['y']),'native_max_abs':diagonal_max,
       'scores_sha256':sha(dest),'upstream':hashinfo,'views':{}}
  y=np.asarray(meta['y']).astype(int)
  for view in VIEWS:
   val={}; observations={}
   for seed in seeds:
    for label,(network,feature) in combos.items():
     p=scores[(seed,network,feature,view)];val[(seed,label)]=float(average_precision_score(y,p));observations[(seed,label)]=p
     assert abs(weighted_ap(y,p,np.ones(len(y)))-val[(seed,label)])<1e-11
   boot={k:[] for k in contrasts}
   for draw in draws:
    w=np.bincount(draw,minlength=len(uniq))[inv]
    aps={k:weighted_ap(y,p,w) for k,p in observations.items()}
    for label,(a,b) in contrasts.items():
     per=[aps[(s,a)]-aps[(s,b)] for s in seeds]
     if np.isfinite(per).all():boot[label].append(float(np.mean(per)))
   out['views'][view]={'mean_ap':{label:float(np.mean([val[(s,label)] for s in seeds])) for label in combos},
       'per_seed_ap':{f'{s}:{label}':val[(s,label)] for s in seeds for label in combos},
       'contrasts':{label:{'per_seed':[val[(s,a)]-val[(s,b)] for s in seeds],
                            'mean':float(np.mean([val[(s,a)]-val[(s,b)] for s in seeds])),
                            'ci95':list(map(float,np.quantile(boot[label],[.025,.975])))}
                    for label,(a,b) in contrasts.items()}}
  result[name]=out
  (OUT/'summary_partial.json').write_text(json.dumps(result,indent=2),encoding='utf8')
  print('DONE',name,'native_max',diagonal_max,flush=True)
 (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__':main()
