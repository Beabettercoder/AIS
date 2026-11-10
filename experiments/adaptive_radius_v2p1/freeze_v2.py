"""Freeze all paired configurations before any model test scoring."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from radius_core import ARMS
import train_single_v2 as single
import train_relation_v2 as relation
from build_relation_v2 import ROOT,sha

def main():
 root=Path('/home/ps/AIS_adaptive_radius_relation_v2p1_20261109');dest=root/'frozen_manifest.json'
 if dest.exists():raise FileExistsError(dest)
 audit=json.loads((root/'reports/pretrain_audit.json').read_text())
 if audit['status']!='passed':raise ValueError('pretrain audit failed')
 origin=json.loads((root/'reports/relation_source_views_audit.json').read_text())
 if origin['status']!='passed' or origin['source_count']!=2526:raise ValueError('relation view audit failed')
 norm_audit=json.loads((root/'reports/normalization_independence.json').read_text())
 if any(norm_audit[t]['status']!='passed' for t in ('single','relation')):raise ValueError('ordinary normalization uses credible features')
 hashes={};reference={};sets={}
 for task,loader in (('single',single.data),('relation',relation.data)):
  sets[task]={}
  for split in ('train','val','test'):
   d,h=loader(root,split);sets[task][split]=d;hashes.update(h)
  if task=='single':
   for view in ('natural','suffix','combined_l100_m600'):
    s,h=single.data(root,'test',view);hashes.update(h)
    for k in ('window_id','mmsi','region','y','mask','time_s','target'):
     if not np.array_equal(s[k],sets[task]['test'][k]):raise ValueError('single view mismatch '+view+k)
 for path,digest in hashes.items():
  if audit['input_sha256'].get(path)!=digest:raise ValueError('feature source changed after audit '+path)
 runs={};initial={};norms={}
 for task in ('single','relation'):
  for name in ('tcn','gru','transformer','graph'):
   for seed in (0,1,2):
    for arm in ARMS:
     key=f'{task}/{name}/{arm}/seed{seed}';p=root/'runs'/key;spec=json.loads((p/'protocol.json').read_text());hist=json.loads((p/'history.json').read_text())
     if spec['name']!=name or spec['arm']!=arm or spec['seed']!=seed or len(hist)!=25 or [r['epoch'] for r in hist]!=list(range(1,26)):raise ValueError('run incomplete '+key)
     if initial.setdefault((task,name,seed),spec['initial_parameter_sha256'])!=spec['initial_parameter_sha256']:raise ValueError('unequal initial weights '+key)
     ck=torch.load(p/'last.pt',map_location='cpu',weights_only=False)
     if ck['epoch']!=25:raise ValueError('invalid last checkpoint '+key)
     chosen=torch.load(p/'best_val.pt',map_location='cpu',weights_only=False)
     metric='pooled_AP' if task=='single' else 'AP'
     if chosen['epoch']!=max(hist,key=lambda x:x[metric])['epoch']:raise ValueError('best validation epoch mismatch '+key)
     norm=ck['norm'];keys=('sm','ss','fm','fs') if task=='single' else ('mean','std')
     digest=hashlib.sha256(b''.join(np.ascontiguousarray(norm[k]).tobytes() for k in keys)).hexdigest()
     if norms.setdefault(task,digest)!=digest:raise ValueError('normalization differs '+key)
     with np.load(p/'val_scores.npz',allow_pickle=False) as z:
      for k in (('y','mmsi','region','window_id') if task=='single' else ('y','mmsi','region','source','case')):
       if not np.array_equal(z[k],sets[task]['val'][k]):raise ValueError('val metadata mismatch '+key+k)
     runs[key]={'last':sha(p/'last.pt'),'best_val':sha(p/'best_val.pt'),'validation_scores':sha(p/'val_scores.npz'),'history':sha(p/'history.json')}
 source=[root/f for f in ('protocol.json','stress_protocol.json','training_protocol_v2p1.json','radius_core.py','test_radius_core.py','build_relation_v2.py','build_single_v2.py','build_single_stress_v2.py','train_single_v2.py','train_relation_v2.py','audit_v2.py','audit_relation_views_source.py','audit_normalization_independence.py','freeze_v2.py')]
 source += [ROOT/'framework_validation_v1_20261028/models.py',ROOT/'framework_validation_v1_20261028/data_builder.py',ROOT/'framework_validation_v1_20261028/field_stress_builder.py',ROOT/'formal_20261101/single/eight_backbone_adapter.py',ROOT/'original_mainline_pilot_20261029/pilot_historical_loss.py',ROOT/'heading_alignment_20261030/train_heading.py',ROOT/'eight_backbone_preparation_20261030/relation8.py',ROOT/'relation_repair_20261030/builder.py']
 if len(runs)!=216:raise ValueError('expected 216 paired runs')
 m={'status':'frozen','runs':runs,'input_sha256':hashes,'source_sha256':{str(p):sha(p) for p in source},'normalization_sha256':norms,'selection':'epoch25 primary, best validation AP supplementary; all configs frozen before test scores','target_MMSI_disjoint':True,'test_scope':'previously used test; synthetic labels and observation interventions','relation_candidate_scope':'prior frozen cohort, not fresh 8nm search','group_definition':'geography and overlapping low-density strata'}
 dest.write_text(json.dumps(m,indent=2),encoding='utf8');print('FROZEN',len(runs),sha(dest),flush=True)
if __name__=='__main__':main()
