#!/usr/bin/env python3
"""Paired cluster bootstrap on frozen bridge scores; three seeds conditional."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from summarize_eight_matched import paired_bootstrap
root=Path('/home/ps/AIS_bridge_20261107/tcn')
arms=('raw','gate','early','late','late_bundle','late_bundle_irregular','late_bundle_irregular_historical_rho')
summ={};data={}
for arm in arms:
 data[arm]={}
 for seed in range(3):
  p=root/'validation'/arm/f'seed{seed}'
  with np.load(p/'test_scores.npz',allow_pickle=False) as z:data[arm][seed]={k:z[k] for k in z.files}
ref=data['raw'][0]
for arm in arms:
 for seed in range(3):
  for key in ('y','mmsi','region','window_id'):
   if not np.array_equal(data[arm][seed][key],ref[key]):raise ValueError(f'unpaired {arm} {seed} {key}')
for ck in ('best_val','last'):
 report={'arms':{},'pairs':{}}
 for arm in arms:
  ap=[float(average_precision_score(ref['y'],data[arm][seed][ck])) for seed in range(3)];auc=[float(roc_auc_score(ref['y'],data[arm][seed][ck])) for seed in range(3)]
  report['arms'][arm]={'AP_seeds':ap,'AP_mean':float(np.mean(ap)),'AP_seed_std':float(np.std(ap,ddof=1)),'AUROC_seeds':auc,'AUROC_mean':float(np.mean(auc))}
 for old,new in zip(arms[:-1],arms[1:]):
  aa=[data[new][s][ck] for s in range(3)];bb=[data[old][s][ck] for s in range(3)]
  report['pairs'][new+' minus '+old]=paired_bootstrap(ref['y'],aa,bb,ref['mmsi'],nboot=2000,seed=1729)
 summ[ck]=report
summ['note']='test set previously exposed; paired bootstrap of MMSI conditional on three trained seed models; direct concatenation vs 13-slot zero-target with late 4D field adapter is architectural; not exact 7D formal reproduction'
(root/'paired_test_summary.json').write_text(json.dumps(summ,indent=2),encoding='utf8');print(json.dumps(summ,indent=2))
