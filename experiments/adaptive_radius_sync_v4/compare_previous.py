from pathlib import Path
import json,csv,sys,numpy as np
from sklearn.metrics import average_precision_score
sys.path.insert(0,'/mnt/sda/WangZijia/AIS/published_detectors_20261106')
from summarize_eight_matched import paired_bootstrap
old=Path('/home/ps/AIS_adaptive_radius_relation_v3_20261109');new=Path('/home/ps/AIS_adaptive_sync_20261110');rows=[]
for name in ('tcn','gru','transformer','graph'):
 for arm in ('target_only','tight_plain','wide_plain','adaptive_plain_fallback','adaptive_credible_fallback'):
  scores=[]
  for root in (old,new):
   q=[]
   for seed in range(3):
    p=root/'runs'/'relation'/name/arm/f'seed{seed}'/'test_last_scores.npz'
    with np.load(p,allow_pickle=False) as z:q.append({k:z[k] for k in z.files})
   scores.append(q)
  for a in scores[0]+scores[1]:
   for key in ('source','case','y','mmsi'):
    if not np.array_equal(a[key],scores[0][0][key]):raise ValueError('unpaired '+key)
  y=scores[0][0]['y'];m=scores[0][0]['mmsi']
  for view in ('base','natural','suffix','combined_l100_m600'):
   bef=[x[view] for x in scores[0]];aft=[x[view] for x in scores[1]];ci=paired_bootstrap(y,aft,bef,m,2000,1729)
   rows.append({'model':name,'arm':arm,'view':view,'old_AP':float(np.mean([average_precision_score(y,x) for x in bef])),'new_AP':float(np.mean([average_precision_score(y,x) for x in aft])),**ci})
p=new/'reports/protocol_comparison.csv'
with p.open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(new/'reports/protocol_comparison.json').write_text(json.dumps({'status':'complete','rows':len(rows),'interpretation':'same target quartet identity; synchronization, six-neighbor rule and stress selection changed together; not a single-factor causal effect','old':str(old),'new':str(new)},indent=2));print('PROTOCOL_COMPARISON_COMPLETE',len(rows),flush=True)
