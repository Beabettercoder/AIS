"""Independently recompute every metric and test the relation target-only control."""
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from radius_core import ARMS
import train_single_v2 as single
import train_relation_v2 as relation

def main():
 root=Path('/home/ps/AIS_adaptive_radius_relation_v3_20261109');manifest=json.loads((root/'frozen_manifest.json').read_text());scores_hash={};checked=0;max_c0=0
 for task,loader in (('relation',relation.data),):
  base,_=loader(root,'test');keys=('y','mmsi','region','window_id') if task=='single' else ('y','mmsi','region','source','case')
  for name in ('tcn','gru','transformer','graph'):
   for arm in ARMS:
    for seed in (0,1,2):
     key=f'{task}/{name}/{arm}/seed{seed}';p=root/'runs'/key;hist=json.loads((p/'history.json').read_text());valkey='pooled_AP' if task=='single' else 'AP'
     if torch.load(p/'best_val.pt',map_location='cpu',weights_only=False)['epoch']!=max(hist,key=lambda v:v[valkey])['epoch']:raise ValueError('best checkpoint mismatch '+key)
     for ck in ('last','best_val'):
      f=p/f'test_{ck}_scores.npz';m=json.loads((p/f'test_{ck}_metrics.json').read_text())
      with np.load(f,allow_pickle=False) as z:
       for k in keys:
        if not np.array_equal(z[k],base[k]):raise ValueError('score identity mismatch '+key+k)
       y=z['y']
       for view in ('base','natural','suffix','combined_l100_m600'):
        pp=z[view];expected=m['views'][view];observed={valkey:float(average_precision_score(y,pp)),'AUROC':float(roc_auc_score(y,pp))}
        for threshold,t in m['thresholds'].items():observed['FAR@'+threshold]=float(np.mean(pp[y==0]>=t));observed['recall@'+threshold]=float(np.mean(pp[y==1]>=t))
        if any(abs(value-float(expected[k]))>1e-12 for k,value in observed.items()):raise ValueError('metric mismatch '+key+'/'+view)
        if task=='relation' and arm=='target_only':max_c0=max(max_c0,abs(observed['AP']-.5),abs(observed['AUROC']-.5))
        checked+=len(observed)
      scores_hash[key+'/'+ck]=single.sha(f)
 if len(scores_hash)!=216 or max_c0>1e-7:raise ValueError('missing predictions or target-only control not 0.5')
 dest=root/'reports/independent_results_audit.json';out={'status':'passed','runs':108,'checkpoints':len(scores_hash),'recomputed_metrics':checked,'relation_target_only_max_abs_difference_from_0p5':max_c0,'frozen_manifest_sha256':single.sha(root/'frozen_manifest.json'),'score_sha256':scores_hash,'note':'test previously used; source cohort is historically selected; ambient candidates were independently re-queried'}
 dest.write_text(json.dumps(out,indent=2));print('INDEPENDENT_RESULTS_PASSED',len(scores_hash),'checkpoints',checked,'metrics','target_only_error',max_c0,flush=True)
if __name__=='__main__':main()
