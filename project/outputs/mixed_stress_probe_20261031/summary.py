"""Independently compute fixed-epoch paired AP and target-MMSI bootstrap."""
from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OUT=ROOT/'mixed_stress_probe_20261031'
VIEWS=('base','natural','suffix','combined_l100_m600')
ARMS=('F0_heading','F1_heading')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ap_weighted(y,score,weights):
 ix=np.argsort(-score,kind='mergesort');s=score[ix];w=weights[ix].astype(float);y=y[ix].astype(float)
 ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1];tp=np.cumsum(w*y)[ends];tot=np.cumsum(w)[ends]
 if tp[-1]<=0:return float('nan')
 return float((np.diff(np.r_[0.,tp])*np.divide(tp,tot,out=np.zeros_like(tp),where=tot>0)).sum()/tp[-1])
report={}
for name in ('tcn','gru','transformer','itransformer'):
 seeds=(0,1,2) if name in ('tcn','itransformer') else (0,)
 data={};runs={};ref=None;source_views={}
 for seed in seeds:
  mdir=OUT/'runs'/name/f'seed{seed}'
  cfg=json.loads((mdir/'protocol.json').read_text())
  for arm in ARMS:
   oldroot=ROOT/('itrans_sourcebalanced_20261031' if name=='itransformer' else 'heading_evidence_pilot_20261030')
   old=oldroot/'runs'/name/f'seed{seed}'/arm;new=mdir/arm
   for strategy,p in [('base',old),('mixed',new)]:
    result=json.loads((p/'result.json').read_text())
    assert result['last_epoch']==(75 if name=='itransformer' else 25)
    assert result['prediction_sha256']==sha(p/'val_scores.npz') and result['weights_sha256']==sha(p/'final.pt')
    with np.load(p/'val_scores.npz',allow_pickle=False) as z:dat={k:z[k] for k in z.files}
    if ref is None:ref=dat
    for key in ('y','mmsi','region','source','case'):np.testing.assert_array_equal(ref[key],dat[key])
    data[(seed,strategy,arm)]=dat;runs[(seed,strategy,arm)]=result
   assert runs[(seed,'mixed',arm)]['init_sha256']==runs[(seed,'base',arm)]['init_sha256']
  assert runs[(seed,'mixed',ARMS[0])]['init_sha256']==runs[(seed,'mixed',ARMS[1])]['init_sha256']
  assert runs[(seed,'mixed',ARMS[0])]['view_source_counts']==runs[(seed,'mixed',ARMS[1])]['view_source_counts']
 y=ref['y'].astype(int);ids,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True);rng=np.random.default_rng(20261031);draw=rng.integers(0,len(ids),size=(2000,len(ids)))
 pairs={'mixed_f1_minus_f0':(('mixed','F1_heading'),('mixed','F0_heading')),'base_f1_minus_f0':(('base','F1_heading'),('base','F0_heading')),'f1_mixed_minus_base':(('mixed','F1_heading'),('base','F1_heading')),'f0_mixed_minus_base':(('mixed','F0_heading'),('base','F0_heading'))}
 result={'seeds':seeds,'mmsi':len(ids),'sources':len(np.unique(ref['source'])),'windows':len(y),'views':{},'note':'fixed trained seeds; 2000 target MMSI bootstrap; no seed-population, shared background or multiplicity coverage'}
 for view in VIEWS:
  values={k:float(average_precision_score(y,v[view+'_score'])) for k,v in data.items()}
  cache={k:v[view+'_score'] for k,v in data.items()}
  for k,s in cache.items():assert abs(ap_weighted(y,s,np.ones(len(y)))-values[k])<1e-12
  boot={k:[] for k in pairs}
  for sample in draw:
   weights=np.bincount(sample,minlength=len(ids))[inv]
   aps={k:ap_weighted(y,score,weights) for k,score in cache.items()}
   for key,((sa,aa),(sb,ab)) in pairs.items():
    z=[aps[(s,sa,aa)]-aps[(s,sb,ab)] for s in seeds]
    if np.isfinite(z).all():boot[key].append(float(np.mean(z)))
  diffs={key:[values[(s,sa,aa)]-values[(s,sb,ab)] for s in seeds] for key,((sa,aa),(sb,ab)) in pairs.items()}
  result['views'][view]={'mean_ap':{f'{strategy}_{arm}':float(np.mean([values[(s,strategy,arm)] for s in seeds])) for strategy in ('base','mixed') for arm in ARMS},'per_seed_ap':{f'{s}_{strategy}_{arm}':v for (s,strategy,arm),v in values.items()},'per_seed_diff':diffs,'mean_diff':{k:float(np.mean(v)) for k,v in diffs.items()},'bootstrap95':{k:list(map(float,np.quantile(v,[.025,.975]))) for k,v in boot.items()}}
 report[name]=result;print('DONE',name,flush=True)
(OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
