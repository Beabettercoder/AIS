import sys,json
import torch,numpy as np
from pathlib import Path
R=Path('/mnt/sda/WangZijia/AIS/outputs');sys.path[:0]=[str(R/'eight_backbone_preparation_20261030'),str(R/'heading_alignment_20261030')]
import train_heading as base
from relation8 import model_for
from torch.nn.functional import binary_cross_entropy_with_logits
base.VARIANT='aligned_embed';tr=base.read('train');va=base.read('val');mu,sd=base.fit(tr)
dev='cuda:3';res={}
for name in ('tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'):
 model=model_for(name).to(dev)
 for arm in ('C0','F0','F1'):
  x,m,t=base.prep(tr,'base',arm,mu,sd,dev);x,m,t=x[:8],m[:8],t[:8]
  y=torch.tensor(tr['y'][:8],dtype=torch.float32,device=dev);model.zero_grad();p=model(x,m,t)['logits'];loss=binary_cross_entropy_with_logits(p,y);loss.backward()
  assert torch.isfinite(loss) and all(torch.isfinite(z.grad).all() for z in model.parameters() if z.grad is not None)
  res[name+'_'+arm]={'loss':float(loss.detach()),'params':sum(p.numel() for p in model.parameters())}
print(json.dumps(res,indent=2))
