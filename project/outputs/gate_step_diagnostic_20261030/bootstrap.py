import json
import numpy as np
from pathlib import Path
p=Path('/mnt/sda/WangZijia/AIS/outputs/gate_step_diagnostic_20261030')
z=np.load(p/'train_sample_records.npz',allow_pickle=False)
rng=np.random.default_rng(20261030)
results={}
def comparison(a,select,left,right, reps=2000):
 q=a[select]
 if not len(q): return None
 ids,rev=np.unique(q['mmsi'],return_inverse=True)
 base=np.bincount(rev,minlength=len(ids))
 l=np.bincount(rev,weights=(q[left]<.3),minlength=len(ids))
 r=np.bincount(rev,weights=(q[right]<.3),minlength=len(ids))
 n=len(ids); ix=rng.integers(0,n,(reps,n))
 draws=np.sum(l[ix]-r[ix],axis=1)/np.sum(base[ix],axis=1)
 return {'records':len(q),'target_mmsi':int(n),'left_suppressed':float(np.sum(l)/np.sum(base)),'right_suppressed':float(np.sum(r)/np.sum(base)),'difference_left_minus_right_pp':float(100*(l.sum()-r.sum())/base.sum()),'mmsi_cluster_ci95_pp':[float(i*100) for i in np.quantile(draws,[.025,.975])]}
for view in ('suffix','natural','combined'):
 a=z[view]
 groups={
  'changed_all':a['kind']==1,
  'changed_lt2':(a['kind']==1)&(a['speed']<2),
  'changed_2to5':(a['kind']==1)&(a['speed']>=2)&(a['speed']<5),
  'changed_ge10':(a['kind']==1)&(a['speed']>=10),
  'adjacent_speed_unchanged':a['kind']==2,
  'other_unchanged_lt5':(a['kind']==0)&(a['speed']<5),
  'position_either_changed':(a['poschange']|a['prevposchange'])>0,
  'unchanged_turn_ge45':(a['kind']==0)&(a['turn']>=45),
  'unchanged_dt_ge120':(a['kind']==0)&(a['dt']>=120)
 }
 results[view]={}
 for name,sel in groups.items():
  results[view][name]={
   'safe_vs_absolute':comparison(a,sel,'ws','wa'),
   'hybrid_vs_absolute':comparison(a,sel,'wh','wa'),
   'hybrid_base_vs_absolute_base':comparison(a,sel,'bwh','bwa')}
doc={'bootstrap':'paired target MMSI resampling 2000, seed 20261030, fixed 400 source training subsample, same target MMSI clusters but background vessels shared; conditional raw reports rather than labels of true faults','description':'suppression is w<0.3, not a calibrated fault classifier; synthetic changed category available for audit only','results':results}
(p/'bootstrap_summary.json').write_text(json.dumps(doc,indent=2),encoding='utf8')
print(json.dumps(doc,indent=2))
