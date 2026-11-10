from pathlib import Path
import json
import numpy as np
base=Path('/mnt/sda/WangZijia/AIS/outputs')
out=base/'low_speed_conditioned_20261030'
with np.load(base/'gate_step_diagnostic_20261030/train_sample_records.npz') as z:a=z['suffix']
# Original speed and perturbation membership are oracle diagnostic variables only.
a=a[(a['speed']<2)&np.isin(a['kind'],[0,1])]
units=np.unique(a['mmsi']);groups=[np.flatnonzero(a['mmsi']==u) for u in units]
rng=np.random.default_rng(20261030); rows=[]
for _ in range(2000):
 idx=np.concatenate([groups[i] for i in rng.integers(0,len(units),len(units))]); z=a[idx];n=z[z['kind']==0];p=z[z['kind']==1]
 if not len(n) or not len(p): continue
 q95,q99=np.quantile(n['e0'],[.95,.99]); old=np.mean(p['wa']<.3)
 rows.append([q95,q99,np.mean(p['e0']>q95),np.mean(p['e0']>q99),old])
r=np.asarray(rows); labels=['unchanged_e0_q95_mps','unchanged_e0_q99_mps','changed_fraction_above_q95','changed_fraction_above_q99','changed_original_weight_below_0_3']
o={'scope':'Exploratory same training sample; target MMSI cluster resampled; percentile and detection measured on same bootstrap sample, no heldout, original speed is oracle for diagnostics only, altered data not true AIS fault labels.', 'bootstrap_draws':len(r),'unique_targets':len(units),'counts':{'changed':int(np.sum(a['kind']==1)),'unmodified':int(np.sum(a['kind']==0))},'values':{}}
for i,k in enumerate(labels):o['values'][k]={'median_bootstrap':float(np.median(r[:,i])),'ci95':np.quantile(r[:,i],[.025,.975]).tolist()}
(out/'low_speed_bootstrap.json').write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(o,ensure_ascii=False,indent=2))
