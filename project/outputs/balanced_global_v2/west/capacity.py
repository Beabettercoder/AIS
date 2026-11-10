import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS')
OUT=ROOT/'outputs/balanced_global_v2/west'
OUT.mkdir(parents=True,exist_ok=True)
REGIONS={
 'houston': {'input':'houston_202407','lat':[29.2,29.8],'lon':[-95.1,-94.6]},
 'ny':{'input':'ny','lat':[40.35,40.95],'lon':[-74.35,-73.65]},
 'la':{'input':'la','lat':[33.6,34.05],'lon':[-118.55,-117.85]},
 'puget':{'input':'puget','lat':[47.5,48.5],'lon':[-124,-122.3]},
 'goldengate':{'input':'goldengate','lat':[37.7,37.9],'lon':[-122.6,-122.4]}}

def candidates(df, region):
 rows=[]
 for m,g in df.groupby('mmsi',sort=False):
  t=g.timestamp.to_numpy(np.int64)
  lat=g.lat.to_numpy();lon=g.lon.to_numpy();sog=g.sog.to_numpy()
  core=g.in_core_roi.to_numpy(bool)
  # Break AIS gaps over 30 min; no target window crosses such a gap.
  starts=np.r_[0,np.flatnonzero(np.diff(t)>1800)+1]
  stops=np.r_[starts[1:],len(t)]
  for seg,(a,z) in enumerate(zip(starts,stops)):
   # Fixed 23-minute half-open cells anchored at the start of each segment.
   begin=int(t[a]); endmax=int(t[z-1])
   while begin<=endmax:
    lo=np.searchsorted(t,begin,'left');hi=min(z,np.searchsorted(t,begin+1380,'left'))
    if hi-lo>=8:
     selected=[];last=-10**12
     for ix in range(lo,hi):
      if t[ix]-last>=60:selected.append(ix);last=t[ix]
     selected=selected[-20:]
     if len(selected)>=8 and core[selected[-1]] and float(sog[selected].max())>=3:
      moving=int((sog[selected]>=2).sum())
      rows.append({'region':region,'mmsi':int(m),'segment_id':int(seg),'start_utc_s':int(t[selected[0]]),'end_utc_s':int(t[selected[-1]]),'cell_start_utc_s':begin,'n_obs':len(selected),'moving_points_ge2':moving,'timestamps':json.dumps(t[selected].tolist()),'row_indices':json.dumps(g.index.to_numpy()[selected].tolist())})
    begin+=1380
 return pd.DataFrame(rows)

report={}
for region,meta in REGIONS.items():
 p=ROOT/'data/interim'/f'ais_{meta["input"]}_enriched.parquet'
 d=pd.read_parquet(p,columns=['mmsi','timestamp','lat','lon','sog','cog','heading','in_core_roi'])
 d=d[(d.timestamp>=1721001600)&(d.timestamp<1721260800)].copy()
 raw_n=len(d)
 d=d[d.mmsi.between(100000000,999999999)&d.sog.between(0,102.2)&d.cog.between(0,359.999999)&d.lat.between(-90,90)&d.lon.between(-180,180)]
 d=d.dropna(subset=['mmsi','timestamp','lat','lon','sog','cog']).sort_values(['mmsi','timestamp'],kind='stable').drop_duplicates(['mmsi','timestamp']).reset_index(drop=True)
 c=candidates(d,region)
 c.to_parquet(OUT/f'{region}_candidates.parquet',index=False)
 counts=c.groupby('mmsi').size() if len(c) else pd.Series(dtype=int)
 strict=c[c.moving_points_ge2>=4]
 counts_s=strict.groupby('mmsi').size()
 def stat(v):
  return {'ships_ge1':int((v>=1).sum()),'ships_ge5':int((v>=5).sum()),'ships_ge10':int((v>=10).sum()),'ships_ge20':int((v>=20).sum()),'all_windows':int(v.sum()),'top300_cap20_windows':int(v.clip(upper=20).nlargest(300).sum()),'all_ships_cap20_windows':int(v.clip(upper=20).sum())}
 report[region]={'roi':meta,'date_start':'2024-07-15T00:00:00Z','date_end_exclusive':'2024-07-18T00:00:00Z','input':str(p),'input_rows_in_dates':raw_n,'standard_valid_rows':len(d),'all_states_ships':int(d.mmsi.nunique()),'baseline':stat(counts),'moving_ge4':stat(counts_s)}
 print(region,json.dumps(report[region]),flush=True)
 (OUT/'capacity_3day.json').write_text(json.dumps(report,indent=2))
