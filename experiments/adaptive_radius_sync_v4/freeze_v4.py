from pathlib import Path
import json,hashlib,numpy as np,torch
from radius_core import ARMS
from build_relation_v2 import ROOT,sha
import train_relation_v2 as rel
root=Path('/home/ps/AIS_adaptive_sync_20261110');dest=root/'frozen_manifest.json'
if dest.exists():raise FileExistsError(dest)
audit=json.loads((root/'reports/pretrain_audit.json').read_text());normaudit=json.loads((root/'reports/normalization_independence.json').read_text())
if audit['status']!='passed' or normaudit['relation']['status']!='passed':raise ValueError('audit failed')
sets={};hashes={}
for split in ('train','val','test'):
 sets[split],h=rel.data(root,split);hashes.update(h)
for path,digest in hashes.items():
 if audit['input_sha256'].get(path)!=digest:raise ValueError('input changed after audit '+path)
runs={};initial={};norms={}
for name in ('tcn','gru','transformer','graph'):
 for seed in (0,1,2):
  for arm in ARMS:
   key=f'relation/{name}/{arm}/seed{seed}';p=root/'runs'/key;spec=json.loads((p/'protocol.json').read_text());hist=json.loads((p/'history.json').read_text())
   if spec['name']!=name or spec['arm']!=arm or spec['seed']!=seed or [r['epoch'] for r in hist]!=list(range(1,26)):raise ValueError('incomplete '+key)
   if initial.setdefault((name,seed),spec['initial_parameter_sha256'])!=spec['initial_parameter_sha256']:raise ValueError('initialization mismatch '+key)
   last=torch.load(p/'last.pt',map_location='cpu',weights_only=False);best=torch.load(p/'best_val.pt',map_location='cpu',weights_only=False)
   if last['epoch']!=25 or best['epoch']!=max(hist,key=lambda r:r['AP'])['epoch']:raise ValueError('checkpoint invalid '+key)
   n=last['norm'];digest=hashlib.sha256(b''.join(np.ascontiguousarray(n[k]).tobytes() for k in ('mean','std'))).hexdigest()
   if norms.setdefault('relation',digest)!=digest:raise ValueError('normalization mismatch '+key)
   with np.load(p/'val_scores.npz',allow_pickle=False) as z:
    for k in ('y','mmsi','region','source','case'):
     if not np.array_equal(z[k],sets['val'][k]):raise ValueError('validation identity mismatch '+key+'/'+k)
   runs[key]={'last':sha(p/'last.pt'),'best_val':sha(p/'best_val.pt'),'validation_scores':sha(p/'val_scores.npz'),'history':sha(p/'history.json')}
source=[root/f for f in ('protocol.json','radius_core.py','test_radius_core.py','build_relation_v2.py','build_relation_sync.py','train_relation_v2.py','audit_v4.py','audit_normalization_independence.py','freeze_v4.py')]
source += [ROOT/'heading_alignment_20261030/train_heading.py',ROOT/'eight_backbone_preparation_20261030/relation8.py',ROOT/'relation_repair_20261030/builder.py']
if len(runs)!=84:raise ValueError('expected 84 runs')
out={'status':'frozen','runs':runs,'input_sha256':hashes,'source_sha256':{str(p):sha(p) for p in source},'normalization_sha256':norms,'selection':'fixed epoch25 primary; best validation AP secondary; frozen before test scoring','target_MMSI_disjoint':True,'test_scope':'previously used controlled quartet test','relation_candidate_scope':'geometric ambient candidates synchronized to quartet group intervention; original selected quartet cohort'}
dest.write_text(json.dumps(out,indent=2));print('FROZEN',len(runs),sha(dest),flush=True)
