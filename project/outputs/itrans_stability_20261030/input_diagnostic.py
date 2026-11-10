from pathlib import Path
import os,sys,json
os.environ['STABILITY_SEED']='0'
import numpy as np
from sklearn.metrics import average_precision_score
import train_stability as m
out={}
for split in ('train','val'):
 d=m.read(split); out[split]={}
 for view in m.VIEWS:
  if split=='train' and view!='base':continue
  values={}
  for arm in ('F0','F1'):
   x,valid=m.features(d,view,arm); lengths=d['length']; ix=np.arange(len(lengths)); j=lengths-1
   vals=x[ix,j,32]; rr=x[ix,j,34];
   values[arm]={'abs_difference_AP':float(average_precision_score(d['y'],np.abs(vals))),'abs_difference_rate_AP':float(average_precision_score(d['y'],np.abs(rr))),'support_fraction':float((valid & d['mask']).sum()/d['mask'].sum()),'feature_rms':float(np.sqrt((x[valid,8:]**2).mean()))}
  x0,z0=m.features(d,view,'F0');x1,z1=m.features(d,view,'F1');good=z0&z1
  values['paired']={'support_identical':bool(np.array_equal(z0,z1)),'mean_abs_diff':float(np.abs(x0[good,8:]-x1[good,8:]).mean()),'fraction_context_rows_differ':float(np.any(np.abs(x0[good,8:]-x1[good,8:])>1e-6,axis=1).mean())}
  out[split][view]=values
Path('/mnt/sda/WangZijia/AIS/outputs/itrans_stability_20261030/input_diagnostic.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
