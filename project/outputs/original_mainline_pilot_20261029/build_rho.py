"""Build continuous reliability support from frozen neighbor provenance."""
import gzip,json,hashlib
from pathlib import Path
import numpy as np
root=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028')
out=Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')
data={};lookup={}
for split in ['train','val']:
 with np.load(root/'data/individual'/f'{split}.npz',allow_pickle=False) as z:
  ids=z['window_id'];data[split]=np.zeros(z['mask'].shape,np.float32);lookup[split]={str(x):i for i,x in enumerate(ids)}
for line in gzip.open(root/'extra_data/field_member_provenance.jsonl.gz','rt',encoding='utf8'):
 r=json.loads(line);s=r['split']
 if s not in data or not r['shared_field_valid']:continue
 w=np.array([m['w_abs'] if m['kinematic_valid'] else 0. for m in r['members']],dtype=float)
 # Mean per-member reliability, scaled by availability; no true noise labels.
 value=float(w.mean()*min(len(w)/4.,1.)) if len(w) else 0.
 data[s][lookup[s][r['window_id']],int(r['step'])]=value
for split in data:
 with np.load(root/'extra_data'/f'field_{split}.npz') as z:
  assert np.array_equal(data[split]>0,z['field_common_valid'])
 np.savez_compressed(out/f'rho_{split}.npz',rho=data[split])
 print(split,'nonzero',int((data[split]>0).sum()),'continuous',int(((data[split]>0)&(data[split]<.99999)).sum()),flush=True)
