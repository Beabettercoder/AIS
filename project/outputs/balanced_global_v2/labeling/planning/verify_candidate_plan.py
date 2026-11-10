"""Independent source-observation validation of the provisional candidate plan."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

BASE=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=BASE/'labeling/planning';SOURCE=BASE/'release_20261026'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=pd.read_parquet(OUT/'label_plan.parquet');c=pd.read_parquet(OUT/'candidate_generation_plan.parquet')
source=pd.read_parquet(SOURCE/'selected_windows.parquet');registry=json.loads((SOURCE/'source_registry.json').read_text())['regions']
assert len(p)==21600 and len(c)==19044
assert not p.window_id.duplicated().any() and not c.window_id.duplicated().any()
assert set(p.window_id)==set(source.window_id)
assert set(c.window_id)==set(p.loc[p.main_pool_candidate,'window_id'])
assert not p.labels_frozen.any() and not c.labels_frozen.any()
assert all(k not in p.columns for k in ['y','behavior_anomaly','behavior_class'])
assert set(c.latent_kind)=={'turn_back','deviation','stall','sinuous','speed_drop'}
assert p.groupby('mmsi').split.nunique().max()==1
report={'status':'passed','source_windows':len(p),'counterfactual_candidates':len(c),'nominal_extra':len(p)-len(c),'final_binary_labels_frozen':False,'regions':{},'validated_target_steps':0,'input_hashes':{name:sha(OUT/name) for name in ['eligibility.parquet','label_plan.parquet','candidate_generation_plan.parquet','protocol_plan.json','build_label_plan.py']}}
for region,g in c.groupby('region',sort=True):
 meta=registry[region];sp=Path(meta['canonical_file']);assert sha(sp)==meta['sha256']
 d=pd.read_parquet(sp,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
 a={k:d[k].to_numpy() for k in d.columns};box=meta['core_bbox'];lon0=(box[0]+box[2])/2;lat0=(box[1]+box[3])/2
 for row in g.itertuples():
  ii=np.asarray(json.loads(row.row_indices),dtype=np.int64);tt=a['timestamp'][ii].astype(float);ss=a['sog'][ii].astype(float);cc=np.deg2rad(a['cog'][ii])
  assert np.array_equal(tt,json.loads(row.timestamps)) and np.all(a['mmsi'][ii]==row.mmsi)
  xy=np.column_stack(((a['lon'][ii]-lon0)*111195*np.cos(np.deg2rad(lat0)),(a['lat'][ii]-lat0)*111195))
  prefix=tt<row.onset_utc_s;post=tt>=row.onset_utc_s
  assert prefix.sum()>=3 and post.sum()>=3 and tt[-1]-row.onset_utc_s>=180
  assert prefix.sum()==row.prefix_obs and post.sum()==row.post_obs
  pt=tt[prefix];ps=ss[prefix];pp=xy[prefix];pc=cc[prefix]
  assert np.median(ps)>=3 and (ps>=3).sum()>=3 and ps[-1]>=3
  dt=np.diff(pt);dp=np.diff(pp,axis=0)
  assert np.all(dt>0) and np.all(dt<=300)
  assert np.linalg.norm(dp,axis=1).sum()>=500
  assert np.linalg.norm(pp[-1]-pp[0])>=500
  vv=ps[:,None]*(1852/3600)*np.column_stack((np.sin(pc),np.cos(pc)))
  reported=np.sum((ps[:-1]+ps[1:])/2*(1852/3600)*dt)
  assert reported>=500
  ekin=np.linalg.norm(dp/dt[:,None]-(vv[:-1]+vv[1:])/2,axis=1)
  weights=np.exp(-np.maximum(0,ekin-2.5)**2/8)
  assert np.all(weights>=.3)
  assert np.isfinite(row.onset_utc_s) and row.latent_kind_index in range(5)
  assert all(0<=getattr(row,k)<2**63 for k in ['behavior_seed','noise_seed','group_seed','label_assignment_seed'])
  assert row.sign in [-1,1] if row.latent_kind in ['turn_back','deviation','sinuous'] else row.sign==0
  report['validated_target_steps']+=len(ii)
 stats={}
 for split,s in g.groupby('split',sort=True):
  sizes=s.latent_kind.value_counts();assert sizes.max()-sizes.min()<=1
  stats[split]={'windows':len(s),'target_mmsi':int(s.mmsi.nunique()),'latent_kinds':sizes.to_dict(),'n_obs_bins':s.generation_stratum.str.extract(r'^(n[^_]+_[^_]+)',expand=False).value_counts().to_dict()}
 report['regions'][region]=stats
 print('VERIFIED',region,len(g),flush=True)
(OUT/'candidate_plan_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='regions'},indent=2))
