from pathlib import Path
import sys, importlib.util
import numpy as np,torch
root=Path('/mnt/sda/WangZijia/AIS/outputs')
p=root/'itrans_sourcebalanced_20261031/train.py';s=importlib.util.spec_from_file_location('candidate',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
oldp=root/'heading_evidence_pilot_20261030/train_heading_evidence.py';s=importlib.util.spec_from_file_location('old',oldp);old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
tr=m.read('train');mo,so=old.fit(tr);mu,sd=m.fit(tr);np.testing.assert_array_equal(mu,mo);np.testing.assert_array_equal(sd,so)
for arm in m.ARMS:
 x,mask,t=m.prep(tr,'base',arm,mu,sd,'cpu');xo,mo_,to=old.prep(tr,'base',arm,mo,so,'cpu');assert torch.equal(x,xo) and torch.equal(mask,mo_) and torch.equal(t,to)
 m.seed(0);model=m.base.get_model('itransformer',40,True);out=model(x[:16],mask[:16],t[:16])['logits'];loss=torch.nn.functional.binary_cross_entropy_with_logits(out,torch.tensor(tr['y'][:16],dtype=torch.float32));loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 print(arm,'preflight input exactly equal, finite forward/backward',flush=True)
