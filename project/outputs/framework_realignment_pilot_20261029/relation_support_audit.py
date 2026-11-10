"""Report effective support and real-time centroid spacing by pressure view."""
import json
from pathlib import Path
import numpy as np
P=Path('/mnt/sda/WangZijia/AIS/outputs/framework_realignment_pilot_20261029')
out={}
for split in ('train','val'):
 with np.load(P/(split+'_relation_field.npz'),allow_pickle=False) as z:d={k:z[k] for k in z.files}
 out[split]={}
 for view in ('base','natural','suffix','combined_l100_m600'):
  rho=d[view+'__field'][:,0,:,6];mask=np.arange(20)[None,:]<d['length'][:,None];idx=np.arange(20)[None,:];early=mask&(idx<d['length'][:,None]//2);late=mask&~early
  w1=rho*early;w2=rho*late;n1=w1.sum(1);n2=w2.sum(1);ok=(n1>0)&(n2>0)
  times=d['time_s']-d['time_s'][:,:1];a=(times*w1).sum(1)/np.maximum(n1,1e-12);b=(times*w2).sum(1)/np.maximum(n2,1e-12);dt=b-a;ok &= dt>0
  sources=np.unique(d['source'].astype(str));source_with_support=len(set(d['source'][ok].astype(str)))
  out[split][view]={'cases':len(ok),'sources':len(sources),'cases_with_two_supported_halves':int(ok.sum()),'sources_with_any_case_two_supported_halves':source_with_support,'span_seconds_median':float(np.median(dt[ok])) if ok.any() else None,'span_seconds_min':float(np.min(dt[ok])) if ok.any() else None,'span_seconds_max':float(np.max(dt[ok])) if ok.any() else None}
with (P/'relation_support_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
