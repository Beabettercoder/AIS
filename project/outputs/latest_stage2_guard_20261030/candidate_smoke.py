"""Train/val-only dry run; no optimizer, no held-out access, no gate creation."""
import sys,importlib.util,json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
C=Path('/mnt/sda/WangZijia/AIS/outputs/relation_latest_report_20261030')
P=Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')
H=Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030')
sys.path.insert(0,str(P));sys.path.insert(0,str(H))
from candidate_integrity import check_candidate
check_candidate(('train','val'))
import train_heading as base
base.OLD=C;base.VARIANT='aligned_embed'
from relation8 import model_for
spec=importlib.util.spec_from_file_location('locked_candidate',C/'train.py')
locked=importlib.util.module_from_spec(spec);spec.loader.exec_module(locked)
locked.VARIANT='aligned_embed'
T=base.read('train'); V=base.read('val')
assert set(T['mmsi']).isdisjoint(set(V['mmsi'])) and set(T['source']).isdisjoint(set(V['source']))
for data in (T,V):
 for view in base.VIEWS:
  for arm in ('C0','F0','F1'):
   a,am=base.features(data,view,arm);b,bm=locked.features(data,view,arm)
   assert np.array_equal(a,b) and np.array_equal(am,bm),(view,arm)
mu,sd=base.fit(T);lm,ls=locked.fit(T)
assert np.array_equal(mu,lm) and np.array_equal(sd,ls)
torch.set_num_threads(2)
for name in ('tcn','gru','transformer','itransformer'):
 torch.manual_seed(0);x=model_for(name)
 torch.manual_seed(0);y=locked.get_model(name,35,True)
 assert set(x.state_dict())==set(y.state_dict()),name
 for k in x.state_dict(): assert torch.equal(x.state_dict()[k],y.state_dict()[k]),(name,k)
idx=np.arange(min(8,len(T['y']))); sample={k:(v[idx] if isinstance(v,np.ndarray) and len(v)==len(T['y']) else v) for k,v in T.items()}
results={}
for name in ('tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'):
 for arm in ('C0','F0','F1'):
  torch.manual_seed(0);model=model_for(name); x,m,t=base.prep(sample,'base',arm,mu,sd,torch.device('cpu'))
  labels=torch.tensor(sample['y'],dtype=torch.float32)
  pred=model(x,m,t)['logits'];loss=F.binary_cross_entropy_with_logits(pred,labels)
  loss.backward();assert torch.isfinite(loss) and all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters()),(name,arm)
  results[name+'_'+arm]={'loss':float(loss.detach()),'finite_grads':True}
out=Path(__file__).with_suffix('.json')
out.write_text(json.dumps({'scope':'no test, no optimization; feature and initialization equivalence and CPU train smoke','counts':[len(T['y']),len(V['y'])],'normalization_exact':True,'features_exact':True,'four_pilot_initial_models_exact':True,'runs':results},indent=2),encoding='utf8')
print('SMOKE PASSED',out)
