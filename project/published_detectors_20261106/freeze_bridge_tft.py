"""Freeze every prespecified TFT arm after validation, before any test scoring."""
import hashlib,json
from pathlib import Path
import numpy as np

root=Path('/home/ps/AIS_bridge_20261107/tft_eval128')
groups=[('aux0_count',x) for x in ('raw','gate','direct','late','bundle','original')]+[('aux0_legacy','original'),('aux0.1_count','direct')]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
records=[];reference=None;initial={}
for condition,arm in groups:
 for seed in range(3):
  d=root/condition/arm/f'seed{seed}'
  spec=json.loads((d/'protocol.json').read_text(encoding='utf8'));hist=json.loads((d/'history.json').read_text(encoding='utf8'))
  if len(hist)!=25 or [r['epoch'] for r in hist]!=list(range(1,26)):raise ValueError(f'Incomplete history: {d}')
  if spec['source_sha256']!=sha(Path(__file__).with_name('bridge_tft.py')) or spec['batch_size']!=128:raise ValueError(f'Source or evaluation batch mismatch: {d}')
  if spec['initial_model_sha256']!=initial.setdefault(seed,spec['initial_model_sha256']):raise ValueError(f'Initial weights mismatch: {d}')
  with np.load(d/'val_scores.npz',allow_pickle=False) as z:
   keys=('y','mmsi','region','window_id');metadata={k:z[k].copy() for k in keys}
  if reference is None:reference=metadata
  elif not all(np.array_equal(reference[k],metadata[k]) for k in keys):raise ValueError(f'Validation window mismatch: {d}')
  if len(metadata['y'])!=1680:raise ValueError('Unexpected validation count')
  best=max(hist,key=lambda x:x['AP'])
  records.append({'condition':condition,'arm':arm,'seed':seed,'best_epoch':best['epoch'],'best_AP':best['AP'],'checkpoint_sha256':sha(d/'best_val.pt'),'protocol_sha256':sha(d/'protocol.json')})
with np.load('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028/data/individual/train.npz',allow_pickle=False) as z:
 train={k:z[k].copy() for k in ('window_id','mmsi','region','y')}
if len(train['y'])!=7840 or set(train['mmsi'])&set(reference['mmsi']):raise ValueError('Unexpected training IDs or train/validation MMSI overlap')
np.savez_compressed(root/'training_sample_identity.npz',**train)
manifest={'selection':'each arm best validation AP; all configurations evaluated, no test-based selection','count':len(records),'source_sha256':sha(Path(__file__).with_name('bridge_tft.py')),'training_sample_identity_sha256':sha(root/'training_sample_identity.npz'),'training_window_count':len(train['y']),'validation_window_count':len(reference['y']),'records':records}
if len(records)!=24:raise ValueError('Expected 24 independent runs')
path=root/'validation_frozen.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('FROZEN:',len(records),'runs, validation IDs aligned; no test scored',flush=True)
