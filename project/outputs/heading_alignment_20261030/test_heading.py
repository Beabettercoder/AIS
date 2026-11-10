import importlib.util
from pathlib import Path
import numpy as np
import torch

path=Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030/train_heading.py')
spec=importlib.util.spec_from_file_location('pilot',path);pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
tr=pilot.read('train'); va=pilot.read('val')
assert not set(tr['mmsi'])&set(va['mmsi']) and not set(tr['source'])&set(va['source'])
mu,sd=pilot.fit(tr)
for view in pilot.VIEWS:
 for arm in pilot.ARMS:
  x,m,t=pilot.prep(va,view,arm,mu,sd,'cpu');assert torch.isfinite(x).all()
  if arm=='C0': assert not x[:,:,pilot.TARGET:].count_nonzero()
  d={k:v.copy() for k,v in va.items()};d['y']=1-d['y'];other,_=pilot.features(d,view,arm);orig,_=pilot.features(va,view,arm);np.testing.assert_array_equal(other,orig)
  if arm=='F1':
   theta=.71;ct=np.cos(theta);st=np.sin(theta);rot=np.array([[ct,-st],[st,ct]])
   d['causal7'][:,:,2:4]=np.einsum('ij,btj->bti',rot,d['causal7'][:,:,2:4]);d['causal7'][:,:,5:7]=np.einsum('ij,btj->bti',rot,d['causal7'][:,:,5:7])
   d[view+'__field'][:,:,:,:2]=np.einsum('ij,bktj->bkti',rot,d[view+'__field'][:,:,:,:2]);rotated,_=pilot.features(d,view,arm)
   np.testing.assert_allclose(rotated[:,:,24:35],orig[:,:,24:35],rtol=2e-4,atol=2e-5)
for name in ['tcn','gru','transformer','itransformer']:
 model=pilot.get_model(name,pilot.NF,True);x,m,t=pilot.prep(va,'base','F1',mu,sd,'cpu'); y=model(x[:8],m[:8],t[:8])['logits'];y.sum().backward();assert torch.isfinite(y).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
print('PASS pairing, train-only normalization, finite all arms/views, C0 zeros, label-blind features, rotation-invariant extra features, 4 backbone forward/backward')
