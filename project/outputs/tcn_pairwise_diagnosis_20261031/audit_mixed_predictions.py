"""Independent metric/bootstrap check of frozen mixed exposure development study."""
from pathlib import Path
import json, hashlib
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
M=ROOT/'mixed_stress_probe_20261031'
O=ROOT/'tcn_pairwise_diagnosis_20261031'
VIEWS=('base','natural','suffix','combined_l100_m600')
ARMS=('F0_heading','F1_heading')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()

def prepared(y,p):
 order=np.argsort(-p,kind='stable'); s=p[order]; label=y[order]
 starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
 return order,label,starts

def weighted_ap(prep,weights):
 order,label,starts=prep
 sorted_weight=weights[order].astype(float)
 w=np.add.reduceat(sorted_weight,starts)
 wp=np.add.reduceat(sorted_weight*label,starts)
 total=wp.sum()
 if total<=0:return float('nan')
 den=np.cumsum(w);ratio=np.divide(np.cumsum(wp),den,out=np.zeros_like(den),where=den>0)
 return float(np.dot(wp,ratio)/total)

summary=json.loads((M/'summary.json').read_text())
out={'models':{},'bootstrap_sklearn_crosschecks':0,'scope':'independent train/val-only evaluation; precomputed score model randomness not resampled; no multiplicity correction'}
for name in ('tcn','gru','transformer','itransformer'):
 seeds=[0,1,2] if name in ('tcn','itransformer') else [0]
 reference=None; pred={}
 for seed in seeds:
  for arm in ARMS:
   oldroot=ROOT/('itrans_sourcebalanced_20261031' if name=='itransformer' else 'heading_evidence_pilot_20261030')
   dirs={'base':oldroot/'runs'/name/f'seed{seed}'/arm,'mixed':M/'runs'/name/f'seed{seed}'/arm}
   for treatment,path in dirs.items():
    r=json.loads((path/'result.json').read_text());assert r['last_epoch']==(75 if name=='itransformer' else 25)
    assert sha(path/'val_scores.npz')==r['prediction_sha256'] and sha(path/'final.pt')==r['weights_sha256']
    with np.load(path/'val_scores.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    if reference is None:reference=d
    for k in ('y','mmsi','region','source','case'):assert np.array_equal(reference[k],d[k]),(name,seed,arm,treatment,k)
    for view in VIEWS:
     p=d[view+'_score']; m=r['views'][view]
     assert abs(average_precision_score(d['y'],p)-m['pooled_AP'])<1e-12
     assert abs(roc_auc_score(d['y'],p)-m['AUROC'])<1e-12
    pred[seed,treatment,arm]=d
   a=json.loads((dirs['base']/'result.json').read_text());b=json.loads((dirs['mixed']/'result.json').read_text())
   assert a['init_sha256']==b['init_sha256']
 y=reference['y'].astype(int);ids,inv=np.unique(reference['mmsi'].astype(str),return_inverse=True)
 draw=np.random.default_rng(20261031).integers(0,len(ids),size=(2000,len(ids)))
 pairs={'mixed_f1_minus_f0':(('mixed','F1_heading'),('mixed','F0_heading')),'base_f1_minus_f0':(('base','F1_heading'),('base','F0_heading')),'f1_mixed_minus_base':(('mixed','F1_heading'),('base','F1_heading')),'f0_mixed_minus_base':(('mixed','F0_heading'),('base','F0_heading'))}
 checks={}
 for view in VIEWS:
  prep={k:prepared(y,z[view+'_score']) for k,z in pred.items()}
  boot={k:[] for k in pairs}
  for k,counts in enumerate(draw):
   weights=np.bincount(counts,minlength=len(ids))[inv]
   aps={key:weighted_ap(item,weights) for key,item in prep.items()}
   if k<3:
    for key in aps:
     check=average_precision_score(y,pred[key][view+'_score'],sample_weight=weights)
     assert abs(check-aps[key])<1e-12;out['bootstrap_sklearn_crosschecks']+=1
   for label,((a,b),(c,d)) in pairs.items():boot[label].append(np.mean([aps[s,a,b]-aps[s,c,d] for s in seeds]))
  for label,values in boot.items():
   independent=np.quantile(values,[.025,.975]);existing=summary[name]['views'][view]['bootstrap95'][label]
   assert np.allclose(independent,existing,atol=1e-12),(name,view,label,independent,existing)
  checks[view]={label:np.quantile(values,[.025,.975]).tolist() for label,values in boot.items()}
 out['models'][name]={'seed_count':len(seeds),'runs_checked':len(pred),'views':checks}
 print('OK',name,flush=True)
(O/'mixed_predictions_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print('independent checks passed',out['bootstrap_sklearn_crosschecks'])
