from pathlib import Path
import numpy as np, json
from sklearn.metrics import average_precision_score
p=Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030/runs/aligned_embed/tcn');a=np.load(p/'F0/val_scores.npz');b=np.load(p/'F1/val_scores.npz');np.testing.assert_array_equal(a['mmsi'],b['mmsi']);np.testing.assert_array_equal(a['y'],b['y']);ms=np.unique(a['mmsi']);which=[np.flatnonzero(a['mmsi']==s) for s in ms];rng=np.random.default_rng(20261029); vals=[]
for draw in rng.integers(0,len(ms),size=(2000,len(ms))):
 ix=np.concatenate([which[k] for k in draw]); vals.append(average_precision_score(b['y'][ix],b['suffix_score'][ix])-average_precision_score(a['y'][ix],a['suffix_score'][ix]))
vals=np.array(vals);out={'n_unique_target_mmsi':len(ms),'draws':len(vals),'point_pp':float(100*(average_precision_score(b['y'],b['suffix_score'])-average_precision_score(a['y'],a['suffix_score']))),'interval95_pp':list((100*np.quantile(vals,[.025,.975])).astype(float))};Path('/mnt/sda/WangZijia/AIS/outputs/framework_repair_diagnostics_20261030/heading_tcn_suffix_bootstrap.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

