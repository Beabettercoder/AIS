"""Synthetic-only isolated fixture to test 48-run report generation."""
import json
from pathlib import Path
import numpy as np
from recent_late_provenance import sha

root=Path('/home/ps/AIS_published_detectors_20261107/recent_late_SYNTHETIC_SUMMARY_FIXTURE')
root.mkdir(parents=True,exist_ok=False)
methods=('tft_gcn','simad','dmrad','carla');arms=('raw','causal_bundle','f0_bundle','f1_bundle')
runs={};n=1680;y=np.arange(n)%2;mmsi=(np.arange(n)//20).astype(str);region=np.full(n,'fixture');wid=np.arange(n).astype(str)
for method in methods:
 for arm in arms:
  for seed in range(3):
   key=f'{method}/{arm}/seed{seed}';path=root/'runs'/key;path.mkdir(parents=True);runs[key]={}
   for ck in ('best_val','last'):
    f=path/(ck+'.pt');f.write_bytes(('SYNTHETIC '+key+ck).encode());runs[key][ck]={'sha256':sha(f)}
    score=np.random.default_rng(seed+len(arm)).random(n)
    np.savez_compressed(path/f'test_{ck}_scores.npz',score=score,y=y,mmsi=mmsi,region=region,window_id=wid)
    metric={'pooled_AP':.5,'AUROC':.5,'macro_AP':.5,'FAR@1%':.01,'recall@1%':.1,'FAR@5%':.05,'recall@5%':.2}
    (path/f'test_{ck}_metrics.json').write_text(json.dumps({'epoch':25,'phase':'fixture','checkpoint_sha256':sha(f),f'test_{ck}':metric}))
(root/'frozen_manifest.json').write_text(json.dumps({'seeds':[0,1,2],'runs':runs}))
print('SYNTHETIC FIXTURE CREATED',root)
