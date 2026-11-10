"""Verify original relation view alignment without reading model scores."""
import json
from pathlib import Path
import numpy as np
from collections import Counter
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
OUT=Path('/home/ps/AIS_adaptive_radius_relation_v2p1_20261109/reports')
entries=[r for r in json.loads((ROOT/'dataset_manifest_combined.json').read_text())['samples'] if r['version']=='relaxed'];counts=Counter()
same_keys=('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','y')
for r in entries:
 data={}
 for view in ('base','natural','suffix'):
  with np.load(ROOT/r['files'][view]['path'],allow_pickle=False) as z:data[view]={k:z[k] for k in (*same_keys,'neighbor_sog_kn')}
 y=data['base']['y']
 if y.shape!=(4,) or int(y.sum())!=2:raise ValueError('quartet label structure')
 for view in ('natural','suffix'):
  for k in same_keys:
   if not np.array_equal(data['base'][k],data[view][k]):raise ValueError(f'{r["split"]}/{r["region"]}/{r["window_id"]}/{view} differs {k}')
  counts[view+'_changed_speed_sources']+=int(not np.array_equal(data['base']['neighbor_sog_kn'],data[view]['neighbor_sog_kn']))
 counts[r['split']+'_sources']+=1
OUT.mkdir(parents=True,exist_ok=True);result={'status':'passed','source_count':len(entries),'counts':dict(counts),'checked':'same target positions, velocities, timestamps, labels, neighbor identity/time/course/position across base/natural/suffix; speed may change; no model outputs accessed'}
(OUT/'relation_source_views_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2),flush=True)
