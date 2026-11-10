"""Paired target-MMSI cluster bootstrap on fixed seed0 validation predictions."""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from pilot_train import load,fit
P=Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')
R=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
np.savez_compressed(P/'normalizer_train_only.npz',**fit(load('train')))
comps=[('f1_bundle','causal_bundle'),('f1_bundle','f0_bundle')];rng=np.random.default_rng(20261029);output=[]
for name in ['tcn','gru','transformer','itransformer']:
 zs={a:np.load(P/'runs'/name/a/'final_val.npz') for a in ('f1_bundle','f0_bundle','causal_bundle')};z=zs['f1_bundle'];y=z['y'];region=z['region'];ids,inv=np.unique(z['mmsi'],return_inverse=True);splits=[]
 for region_name in np.unique(region):
  ix=np.flatnonzero(region==region_name);sections={}
  for arm,a in zs.items():
   order=ix[np.argsort(-a['score'][ix],kind='stable')];score=a['score'][order];end=np.r_[np.flatnonzero(score[1:]!=score[:-1]),len(score)-1];sections[arm]=(order,end)
  splits.append((region_name,sections))
 def macro(weights,arm):
  ap=[]
  for _,section in splits:
   o,end=section[arm];w=weights[inv[o]];pos=w*y[o];cum=np.cumsum(pos);den=np.cumsum(w);tot=cum[-1]
   if not tot:return np.nan
   c=cum[end];prev=np.r_[0,c[:-1]];ap.append(float(np.sum(((c-prev)/tot)*(c/np.maximum(den[end],1e-20)))))
  return np.mean(ap)
 ones=np.ones(len(ids),np.float64)
 for arm in zs:
  calc=macro(ones,arm);r=json.loads((P/'runs'/name/arm/'result.json').read_text());assert abs(calc-r['final']['macro_AP'])<1e-12
  for region_name,_ in splits:
   ix=region==region_name;assert np.isfinite(average_precision_score(y[ix],zs[arm]['score'][ix]))
 distributions={str(c):[] for c in comps};invalid=0
 for k in range(2000):
  weights=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids)).astype(float)
  scores={a:macro(weights,a) for a in zs}
  if not all(np.isfinite(x) for x in scores.values()):invalid+=1;continue
  for a,b in comps:distributions[str((a,b))].append(scores[a]-scores[b])
 for a,b in comps:
  dif=macro(ones,a)-macro(ones,b);arr=np.array(distributions[str((a,b))]);ci=np.quantile(arr,[.025,.975]);output.append(dict(backbone=name,contrast=a+' minus '+b,delta=diff if False else dif,ci=ci.tolist(),valid_draws=len(arr),invalid_draws=invalid,clusters=len(ids)))
  print(name,a,b,'delta_pp',dif*100,'CI_pp',ci*100,'valid',len(arr),flush=True)
files=[P/'pilot_train.py',P/'pilot_historical_loss.py',P/'pilot_features.py',P/'build_rho.py',P/'rho_train.npz',P/'rho_val.npz',P/'normalizer_train_only.npz',R/'data/individual/train.npz',R/'data/individual/val.npz',R/'extra_data/field_train.npz',R/'extra_data/field_val.npz']
for name in ['tcn','gru','transformer','itransformer']:
 for arm in ['raw','causal_bundle','f0_bundle','f1_bundle']:
  files.extend([P/'runs'/name/arm/'final.pt',P/'runs'/name/arm/'final_val.npz',P/'runs'/name/arm/'result.json'])
manifest={str(x):sha(x) for x in files};(P/'audit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
(P/'pilot_bootstrap.json').write_text(json.dumps({'method':'paired cluster resampling of target MMSI; 2000 draws; fixed seed0 models; region-macro AP; percentile interval','results':output},indent=2),encoding='utf8')
