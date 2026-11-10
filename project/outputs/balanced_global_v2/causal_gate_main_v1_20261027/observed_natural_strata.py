"""Observation-based stratification of unmodified target trajectories (no behavioral ground truth)."""
from pathlib import Path
import json
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
BASE=ROOT/'labeling/labeled_v1/features_scene'
OUT=ROOT/'causal_gate_main_v1_20261027'
regions=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
rows={r['window_id']:r for r in json.loads((OUT/'trajectory_validation_rows.json').read_text())}
groups={name:[] for name in ('all_observed','turn_45_deg','speed_change_3_kn')}
for region in regions:
 with np.load(BASE/region/'val.npz',allow_pickle=False) as z:
  for i in np.flatnonzero((z['y']==0)&(~z['target_noise_effective'])):
   n=int(z['mask'][i].sum());c=z['target_cog_deg'][i,:n];s=z['target_sog_kn'][i,:n]
   # Maximum observable difference from first valid report, not a manual behavior label.
   turn=float(np.max(np.abs((c-c[0]+180)%360-180)))
   speed=float(np.max(np.abs(s-s[0])))
   item={**rows[str(z['window_id'][i])],'observed_max_heading_change_deg':turn,'observed_max_speed_change_kn':speed}
   groups['all_observed'].append(item)
   if turn>=45:groups['turn_45_deg'].append(item)
   if speed>=3:groups['speed_change_3_kn'].append(item)
summary={}
for name,items in groups.items():
 if not items:continue
 v={'windows':len(items),'unique_target_mmsi':len({x['mmsi'] for x in items})}
 for method in ('raw','old','comp','causal'):
  for metric in ('endpoint','mean_path'):
   a=np.array([x[method+'_'+metric+'_m'] for x in items]);v[method+'_'+metric]={'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.quantile(a,.9))}
 summary[name]=v
(OUT/'observed_natural_strata.json').write_text(json.dumps({'selection':'y=0, no controlled target observation noise, observed variations vs initial point; no manually confirmed behavior or sensor truth','thresholds':'heading difference >=45 degrees; speed difference >=3kn','summary':summary,'groups':groups},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False),flush=True)
