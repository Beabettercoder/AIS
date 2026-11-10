"""Posthoc paired uncertainty for already frozen primary vs raw; no model selection."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score
from evaluate_test import boot_weights,bootstrap_one
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/causal_gate_main_v1_20261027')
plan=json.loads((OUT/'frozen_test_plan.json').read_text());test=json.loads((OUT/'formal_test_results/test_results.json').read_text())
with np.load(OUT/'formal_test_results/all_test_predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
y=d['y'].astype(int);mmsi=d['mmsi'].astype(str);region=d['region'].astype(str);weights,ids=boot_weights(y,mmsi);primary=plan['primary_arm']
out={'status':'complete','scope':'posthoc interval for primary fixed before test vs raw, no new model selection',
     'unit':'target MMSI; shared-background correlation and seed sampling variation not covered','draws':len(weights),'unique_mmsi':len(ids),'primary':primary,'per_seed':[],'three_seed_mean':[]}
def calc(s):
 pooled=bootstrap_one(y,s,weights,{})['AP'];regional=[]
 for r in sorted(set(region)):
  m=region==r; regional.append(bootstrap_one(y[m],s[m],weights[:,m],{})['AP'])
 return pooled,np.mean(regional,axis=0)
for arch in ('tcn','gru'):
 diffs=[]
 for seed in (0,1,2):
  def score(arm):
   ix=np.flatnonzero((d['architecture']==arch)&(d['seed']==seed)&(d['arm']==arm));assert len(ix)==1;return d['scores'][ix[0]]
  a=score(primary);b=score('raw');ap,am=calc(a);bp,bm=calc(b);D=ap-bp;M=am-bm;diffs.append((D,M))
  for s in (a,b):
   x=calc(s)[0]
   for j in (0,1,2):assert abs(x[j]-average_precision_score(y,s,sample_weight=weights[j]))<1e-10
  def macro(v):return float(np.mean([average_precision_score(y[region==r],v[region==r]) for r in set(region)]))
  out['per_seed'].append({'backbone':arch,'seed':seed,'pooled_AP_difference':float(average_precision_score(y,a)-average_precision_score(y,b)),
   'pooled_AP_difference_CI95':np.quantile(D,[.025,.975]).tolist(), 'macro_AP_difference':macro(a)-macro(b),
   'macro_AP_difference_CI95':np.quantile(M,[.025,.975]).tolist()})
 pooled=np.mean([v[0] for v in diffs],axis=0);macro=np.mean([v[1] for v in diffs],axis=0)
 out['three_seed_mean'].append({'backbone':arch,'pooled_AP_difference':float(np.mean([r['pooled_AP_difference'] for r in out['per_seed'] if r['backbone']==arch])),
  'pooled_AP_difference_CI95':np.quantile(pooled,[.025,.975]).tolist(),
  'macro_AP_difference':float(np.mean([r['macro_AP_difference'] for r in out['per_seed'] if r['backbone']==arch])),
  'macro_AP_difference_CI95':np.quantile(macro,[.025,.975]).tolist()})
path=OUT/'formal_test_results/posthoc_primary_bootstrap.json'
with path.open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out['three_seed_mean'],indent=2),flush=True)
