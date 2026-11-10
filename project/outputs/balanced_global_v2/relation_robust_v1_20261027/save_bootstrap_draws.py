"""Persist the MMSI selections and AP differences behind all 95% intervals."""
from pathlib import Path
import csv,json
import numpy as np
from sklearn.metrics import average_precision_score
P=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_robust_v1_20261027')
def read(n):
    with (P/n).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
main=read('statistics_scores.csv');extra=read('absolute_gate_robust_scores.csv');assert len(main)==len(extra)
for a,b in zip(main,extra):
    assert (a['view'],a['source'],a['case'])==(b['view'],b['source'],b['case']);a.update({k:b[k] for k in ('absolute_weighted_median','absolute_report_weighted_median')})
ids=sorted({r['mmsi'] for r in main if r['split']=='val'});rng=np.random.default_rng(20261027);draws=rng.integers(0,len(ids),(2000,len(ids)))
with (P/'bootstrap_mmsi_draws.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.writer(f);w.writerow(['draw','position','mmsi']);w.writerows((i,j,ids[v]) for i,d in enumerate(draws) for j,v in enumerate(d))
pairs=[('mixed_weighted_median','median'),('mixed_report_weighted_median','median'),('mixed_weighted_mean','absolute_weighted_mean'),('mixed_weighted_median','absolute_weighted_median'),('mixed_report_weighted_median','absolute_report_weighted_median')];out=[]
for view in ('base','existing_speed','high_speed'):
    r=[x for x in main if x['split']=='val' and x['view']==view];groups={i:np.array([j for j,x in enumerate(r) if x['mmsi']==i]) for i in ids};y=np.array([int(x['y']) for x in r]);indices=[np.concatenate([groups[ids[j]] for j in d]) for d in draws]
    for a,b in pairs:
        x=np.array([float(t[a]) for t in r]);z=np.array([float(t[b]) for t in r]);
        for n,ix in enumerate(indices):out.append((view,a,b,n,average_precision_score(y[ix],x[ix])-average_precision_score(y[ix],z[ix])))
with (P/'bootstrap_ap_differences.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.writer(f);w.writerow(['view','method_a','method_b','draw','difference_AP']);w.writerows(out)
print('saved',len(draws),'draws and',len(out),'paired differences')
