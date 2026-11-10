import sys,json,torch,numpy as np
from pathlib import Path
sys.path.insert(0,'/mnt/sda/WangZijia/AIS/outputs/target_heading_20261031')
import train_target as t
tr=t.load('train');va=t.load('val');mu,sd=t.fit(tr)
a,mask,time=t.prep(tr,'base','F1_target',mu,sd,'cpu');a=a[:4];mask=mask[:4];time=time[:4]
results={}
for name in ['tcn','gru','transformer','itransformer']:
 t.seed(0);m=t.h.base.get_model(name,45,name=='itransformer');out=m(a,mask,time)['logits'];loss=torch.nn.functional.binary_cross_entropy_with_logits(out,torch.tensor(tr['y'][:4],dtype=torch.float32));loss.backward();grads=[p.grad for p in m.parameters() if p.grad is not None];assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads)
 results[name]=float(loss.detach())
p=Path('/mnt/sda/WangZijia/AIS/outputs/target_heading_20261031/model_smoke.json');p.write_text(json.dumps(results,indent=2));print(results)
