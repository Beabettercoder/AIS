from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS')
OUT=ROOT/'outputs/balanced_global_v2/west'
ns={}
exec((OUT/'capacity.py').read_text().split('\nreport={}')[0],ns)
candidates=ns['candidates']
report={}
for region in ['houston','ny','la','puget']:
 p=OUT/f'{region}_raw_standard.parquet'
 d=pd.read_parquet(p)
 assert d.timestamp.between(1721001600,1721260799).all()
 assert not d.duplicated(['mmsi','timestamp']).any()
 assert d.groupby('mmsi').timestamp.apply(lambda x:x.is_monotonic_increasing).all()
 assert d.source_file.str.contains('AIS_2024_07_',regex=False).all()
 assert d.source_row.ge(0).all()
 c=candidates(d,region)
 c=c[c.moving_points_ge2>=4].copy()
 c.to_parquet(OUT/f'{region}_standard_candidates.parquet',index=False)
 counts=c.groupby('mmsi').size()
 # Nonoverlap is checked on actual selected sample times, not only cell IDs.
 for _,g in c.groupby('mmsi'):
  g=g.sort_values('start_utc_s')
  assert np.all(g.start_utc_s.to_numpy()[1:]>g.end_utc_s.to_numpy()[:-1])
  for r in g.itertuples():
   tt=json.loads(r.timestamps)
   assert 8<=len(tt)<=20 and np.min(np.diff(tt))>=60 and max(tt)-min(tt)<1380
 report[region]={'rows':len(d),'all_states_ships':int(d.mmsi.nunique()),'core_rows':int(d.in_core_roi.sum()),'timestamp_min':int(d.timestamp.min()),'timestamp_max':int(d.timestamp.max()),'ships_ge1':int((counts>=1).sum()),'ships_ge5':int((counts>=5).sum()),'ships_ge10':int((counts>=10).sum()),'ships_ge20':int((counts>=20).sum()),'all_windows':int(counts.sum()),'top300_cap20_windows':int(counts.clip(upper=20).nlargest(300).sum()),'all_ships_cap20_windows':int(counts.clip(upper=20).sum()),'standard_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checks':'UTC seconds, valid date range, target-time deduplication, sorting, source lineage, nonoverlapping windows, 60s minimum observation intervals: PASS'}
 print(region,json.dumps(report[region]),flush=True)
 (OUT/'standard_validation.json').write_text(json.dumps(report,indent=2))
