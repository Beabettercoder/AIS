from pathlib import Path
import json,hashlib,numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
base=Path('/mnt/sda/WangZijia/AIS/outputs'); parent=base/'neighbor_gate_safe_20261030'; root=parent/'fixed_normalizer'; out=base/'gate_step_diagnostic_20261030'
V=('base','natural','suffix','combined_l100_m600'); arms=('C0','F0','F1');summary=json.loads((root/'paired_summary.json').read_text());results={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('tcn','gru','transformer','itransformer'):
 seeds=(0,1,2) if name in ('tcn','itransformer') else (0,);pred={};init={};mx=0;ref=None
 for s in seeds:
  for arm in arms:
   if arm=='F1': path=root/'runs/aligned_embed'/('seed'+str(s))/name/arm
   elif name in ('tcn','gru','transformer'):
    path=(base/'heading_alignment_20261030/runs/aligned_embed'/name/arm if s==0 else base/'tcn_suffix_stability_20261030/runs/aligned_embed'/('seed'+str(s))/'tcn'/arm)
   elif arm=='C0':path=parent/'runs/aligned_embed'/('seed'+str(s))/name/arm
   else:path=base/'itrans_stability_20261030/extended75/runs'/('seed'+str(s))/'aligned_embed/itransformer'/arm
   r=json.loads((path/'result.json').read_text())
   with np.load(path/'val_scores.npz',allow_pickle=False) as z:arr={k:z[k] for k in z.files}
   pred[s,arm]=arr;init.setdefault(s,[]).append(r['init_sha256']);assert r['seed']==s and r['last_epoch']==(75 if name=='itransformer' else 25) and len(r['loss_history'])==r['last_epoch']
   assert r['prediction_sha256']==sha(path/'val_scores.npz') and r['weights_sha256']==sha(path/'final.pt')
   if ref is None:ref=arr
   for k in ('source','case','y','mmsi','region'):assert np.array_equal(arr[k],ref[k])
   for v in V:
    ap=average_precision_score(arr['y'],arr[v+'_score']);au=roc_auc_score(arr['y'],arr[v+'_score']);mx=max(mx,abs(ap-r['views'][v]['pooled_AP']),abs(au-r['views'][v]['AUROC']));assert abs(ap-r['views'][v]['pooled_AP'])<1e-10
 assert all(len(set(v))==1 for v in init.values());assert mx<1e-10
 ids,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True);yy=ref['y'].astype(int);rng=np.random.default_rng(20261029);draws=rng.integers(0,len(ids),size=(2000,len(ids)));C={}
 for v in V:
  groups={}
  for s in seeds:
   for arm in arms:
    a=pred[s,arm][v+'_score'];o=np.argsort(-a,kind='stable');sv=a[o];last=np.r_[np.flatnonzero(sv[:-1]!=sv[1:]),len(o)-1];groups[s,arm]=(o,last)
  def weighted(s,arm,w):
   o,end=groups[s,arm];wg=w[o];pos=np.cumsum(wg*yy[o])[end];tot=np.cumsum(wg)[end];inc=np.diff(np.r_[0.,pos]);return float(np.sum((pos/np.maximum(tot,1e-300))*inc)/pos[-1])
  vals={a:[] for a in ('F0','C0')}
  for sample in draws:
   w=np.bincount(sample,minlength=len(ids))[inv]
   for arm in vals: vals[arm].append(np.mean([weighted(s,'F1',w)-weighted(s,arm,w) for s in seeds]))
  C[v]={}
  for arm in vals:
   ci=[float(np.quantile(vals[arm],.025)),float(np.quantile(vals[arm],.975))]
   target=summary[name]['views'][v]['bootstrap95']['safe_f1_'+arm.lower()]
   err=float(np.max(np.abs(np.asarray(ci)-target)));assert err<1e-10,(name,v,arm,ci,target,err)
   C[v][arm]={'ci':ci,'max_difference_from_summary':err}
 results[name]={'seed_count':len(seeds),'model_count':len(seeds)*3,'max_metric_error':mx,'intervals':C,'paired_init_identical':True,'predictions_target_pair_equal':True}
(out/'fixed_normalizer_predictions_independent_audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
