from pathlib import Path
import pandas as pd
import numpy as np
import json
out=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
ds={p.stem:pd.read_parquet(p) for p in (out/'candidates').glob('*.parquet')}
ds={r:c[c.duration_s>=660] for r,c in ds.items()}
ids=np.array(sorted(set().union(*[set(c.mmsi) for c in ds.values()])))
np.random.default_rng(20261025).shuffle(ids)
size=len(ids);a=int(size*.7);b=int(size*.85)
assignment={int(m):s for s,x in [('train',ids[:a]),('val',ids[a:b]),('test',ids[b:])] for m in x}
result={}
for r,c in ds.items():
 v=c.groupby('mmsi').size().clip(upper=20)
 rec={'ships':len(v),'cap20':int(v.sum()),'top300cap20':int(v.nlargest(300).sum()),'all_core_ships':int(c[c.all_points_core].mmsi.nunique()),'all_core_top300cap20':int(c[c.all_points_core].groupby('mmsi').size().clip(upper=20).nlargest(300).sum())}
 for s,k,w in [('train',210,1890),('val',45,405),('test',45,405)]:
  vs=v[[assignment[int(m)]==s for m in v.index]]
  rec[s]={'ships':len(vs),'top_target_ships_cap20':int(vs.nlargest(k).sum()),'enough':len(vs)>=k and int(vs.nlargest(k).sum())>=w}
 result[r]=rec
print(json.dumps({'assignment':'Illustrative fixed RNG permutation seed20261025, no files modified','global_candidate_ships':size,'regions':result},indent=2))
