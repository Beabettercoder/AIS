from pathlib import Path
import json
import numpy as np
import pandas as pd
from build_candidates import REGIONS, OUT

report={}
config=json.loads((OUT/'source_registry.json').read_text())['regions'] if (OUT/'source_registry.json').exists() else None
sources=REGIONS if config is None else {r:(m['canonical_file'],m['core_bbox']) for r,m in config.items()}
for region,(rel,bbox) in sources.items():
 if not (OUT/f'candidates/{region}.parquet').exists():continue
 d=pd.read_parquet(OUT/rel).reset_index(drop=True)
 arrays={k:d[k].to_numpy() for k in ['timestamp','lat','lon','sog','cog']}
 c=pd.read_parquet(OUT/f'candidates/{region}.parquet')
 c=c[c.duration_s>=660]
 errors={f:[] for f in [.0514444444,.5144444444,1.0]}; gaps=[]; gates=[]
 for row in c.itertuples():
  ix=np.array(json.loads(row.row_indices),dtype=np.int64)
  t=arrays['timestamp'][ix];lat=arrays['lat'][ix];lon=arrays['lon'][ix];sog=arrays['sog'][ix];cog=np.deg2rad(arrays['cog'][ix])
  dt=np.diff(t); dx=np.diff(lon)*111195*np.cos(np.deg2rad((lat[1:]+lat[:-1])/2));dy=np.diff(lat)*111195
  good=(dt>0)&(dt<=300)&(sog[1:]>=3)&(sog[:-1]>=3)
  measured=np.column_stack((dx,dy))/dt[:,None]
  uv=np.column_stack((np.sin(cog)*sog,np.cos(cog)*sog))
  for factor in errors:errors[factor].extend(np.linalg.norm(measured-(uv[1:]+uv[:-1])/2*factor,axis=1)[good].tolist())
  gaps.extend(dt.tolist())
  e=np.linalg.norm(measured-(uv[1:]+uv[:-1])/2*.5144444444,axis=1)
  gates.extend(np.exp(-np.maximum(0,e[dt<=300]-2.5)**2/8).tolist())
 def stats(a):
  a=np.array(a);return {'n':len(a),'p10':float(np.quantile(a,.1)),'p50':float(np.median(a)),'p90':float(np.quantile(a,.9)),'p99':float(np.quantile(a,.99))}
 report[region]={'moving_pair_error_mps_by_sog_scale':{str(k):stats(v) for k,v in errors.items()},'sample_gap_seconds':stats(gaps),'fraction_pairs_over_300s':float(np.mean(np.array(gaps)>300)),'kinematic_reliability':stats(gates),'fraction_valid_pairs_w_below_03':float(np.mean(np.array(gates)<.3)),'ships_with_nonstandard_mmsi_prefix':int(d.loc[~d.mmsi.between(200000000,799999999),'mmsi'].nunique())}
 print(region,json.dumps(report[region]),flush=True)
(OUT/'unit_quality_audit.json').write_text(json.dumps(report,indent=2))
