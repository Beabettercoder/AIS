from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score

root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/causal_gate_main_v1_20261027');out=root.parent/'independent_audit_causal_v1_20261027'
reported=json.loads((root/'formal_test_results/posthoc_primary_bootstrap.json').read_text());plan=json.loads((root/'frozen_test_plan.json').read_text())
with np.load(root/'formal_test_results/all_test_predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
y=d['y'];r=d['region'].astype(str);ids,inv=np.unique(d['mmsi'].astype(str),return_inverse=True);rng=np.random.default_rng(20261027);counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000);weights=counts[:,inv];R=np.unique(r)

def ap_draws(y,s,weights):
 order=np.argsort(-s,kind='stable');s=s[order];y=y[order];w=weights[:,order].astype(float);end=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
 cumw=np.cumsum(w,axis=1)[:,end];pos=np.cumsum(w*y,axis=1)[:,end];prev=np.c_[np.zeros(len(w)),pos[:,:-1]]
 return ((pos-prev)*np.divide(pos,cumw,out=np.zeros_like(pos),where=cumw>0)).sum(1)/np.maximum(pos[:,-1],1)

maxerr=0.;maxspot=0.;produced=[]
for arch in ('tcn','gru'):
 all_p=[];all_m=[];diffpoints=[]
 for seed in (0,1,2):
  A=np.flatnonzero((d['architecture']==arch)&(d['seed']==seed)&(d['arm']==plan['primary_arm']));B=np.flatnonzero((d['architecture']==arch)&(d['seed']==seed)&(d['arm']=='raw'));assert len(A)==len(B)==1
  a=d['scores'][A[0]];b=d['scores'][B[0]];dp=ap_draws(y,a,weights)-ap_draws(y,b,weights);dm=np.mean([ap_draws(y[r==g],a[r==g],weights[:,r==g])-ap_draws(y[r==g],b[r==g],weights[:,r==g]) for g in R],axis=0)
  for j in (0,1,2,99,999):maxspot=max(maxspot,abs(dp[j]-(average_precision_score(y,a,sample_weight=weights[j])-average_precision_score(y,b,sample_weight=weights[j]))))
  old=next(q for q in reported['per_seed'] if q['backbone']==arch and q['seed']==seed)
  maxerr=max(maxerr,float(np.max(np.abs(np.quantile(dp,[.025,.975])-old['pooled_AP_difference_CI95']))),float(np.max(np.abs(np.quantile(dm,[.025,.975])-old['macro_AP_difference_CI95']))))
  dp0=average_precision_score(y,a)-average_precision_score(y,b);dm0=np.mean([average_precision_score(y[r==g],a[r==g])-average_precision_score(y[r==g],b[r==g]) for g in R])
  maxerr=max(maxerr,abs(dp0-old['pooled_AP_difference']),abs(dm0-old['macro_AP_difference']));all_p.append(dp);all_m.append(dm);diffpoints.append((dp0,dm0))
 old=next(q for q in reported['three_seed_mean'] if q['backbone']==arch);dp=np.mean(all_p,axis=0);dm=np.mean(all_m,axis=0);points=np.mean(diffpoints,axis=0)
 maxerr=max(maxerr,float(np.max(np.abs(np.quantile(dp,[.025,.975])-old['pooled_AP_difference_CI95']))),float(np.max(np.abs(np.quantile(dm,[.025,.975])-old['macro_AP_difference_CI95']))),abs(points[0]-old['pooled_AP_difference']),abs(points[1]-old['macro_AP_difference']))
 produced.append({'architecture':arch,'macro_difference':float(points[1]),'macro_interval':np.quantile(dm,[.025,.975]).tolist()})
assert maxerr<1e-10 and maxspot<1e-10
result={'status':'passed','bootstrap_draws':2000,'sampled_target_mmsi':len(ids),'max_report_error':maxerr,'max_sklearn_spot_error':maxspot,'three_seed_means':produced,
 'limitation':'posthoc paired cluster intervals conditional on fixed training seeds; do not capture training-seed uncertainty or shared-background dependence'}
(out/'audit_primary_bootstrap.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
