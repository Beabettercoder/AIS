#!/usr/bin/env python3
"""Freeze all prescribed validation-selected TCN bridge checkpoints before test."""
import json,hashlib
from pathlib import Path
import torch
root=Path('/home/ps/AIS_bridge_20261107/tcn')
code=Path(__file__).with_name('bridge_tcn_20261107.py')
arms=('raw','gate','early','late','late_bundle','late_bundle_irregular','late_bundle_irregular_historical_rho')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
runs={};per_seed_backbone={};norm_hashes=set()
for arm in arms:
 for seed in range(3):
  p=root/'validation'/arm/f'seed{seed}'
  spec=json.loads((p/'protocol.json').read_text());history=json.loads((p/'history.json').read_text())
  if spec['epochs']!=25 or len(history)!=25:raise ValueError(f'incomplete run {p}')
  for name in ('best_val.pt','last.pt','val_scores.npz'):
   if not (p/name).is_file():raise ValueError(f'missing {p/name}')
  per_seed_backbone.setdefault(seed,set()).add(spec['initial_backbone_sha256'])
  checkpoint=torch.load(p/'last.pt',map_location='cpu',weights_only=False)
  norm=checkpoint['norm'];norm_hashes.add(hashlib.sha256(b''.join(norm[k].tobytes() for k in sorted(norm))).hexdigest())
  runs[f'{arm}/seed{seed}']={'best_val':sha(p/'best_val.pt'),'last':sha(p/'last.pt'),'validation_scores':sha(p/'val_scores.npz')}
if any(len(v)!=1 for v in per_seed_backbone.values()) or len(norm_hashes)!=1:raise ValueError('backbone initialization or normalization mismatch')
dependencies=[code.with_name('data_resampled.py'),Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029/pilot_features.py'),Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029/pilot_historical_loss.py'),Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028/models.py')]
base=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028')
data_files=[base/'data/individual'/f'{s}.npz' for s in ('train','val')]+[base/'extra_data'/f'field_{s}.npz' for s in ('train','val')]+[Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')/f'rho_{s}.npz' for s in ('train','val')]
lock={'note':'fixed 21 configurations, 25 epochs, selection by validation AP; old test previously exposed','code_sha256':sha(code),'dependency_sha256':{str(p):sha(p) for p in dependencies},'training_validation_data_sha256':{str(p):sha(p) for p in data_files},'normalizer_sha256':next(iter(norm_hashes)),'per_seed_initial_backbone_sha256':{str(k):next(iter(v)) for k,v in per_seed_backbone.items()},'runs':runs}
dest=root/'frozen_validation.lock.json'
if dest.exists():raise FileExistsError('Refusing to change an existing frozen lock')
dest.write_text(json.dumps(lock,indent=2),encoding='utf8');print('FROZEN',dest,len(runs))
