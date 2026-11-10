import sys,torch,numpy as np
sys.path.insert(0,'/mnt/sda/WangZijia/AIS/outputs/heading_evidence_pilot_20261030')
import train_heading_evidence as p
d=p.read('train');mu,sd=p.fit(d);device=torch.device('cpu')
xx={}
for arm in p.ARMS:
 x,m,t=p.prep(d,'base',arm,mu,sd,device);xx[arm]=(x[:8],m[:8],t[:8]);assert torch.isfinite(x).all()
assert torch.all(xx['F1_noheading'][0][...,35:]==0) and torch.all(xx['C0'][0][...,8:]==0)
assert torch.all((xx['F1_heading'][0][...,37]>=0)&(xx['F1_heading'][0][...,37]<=1))
for name in ('tcn','gru','transformer','itransformer'):
 p.seed(0);a=p.base.get_model(name,40,name=='itransformer');p.seed(0);b=p.base.get_model(name,40,name=='itransformer')
 assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
 for arm in p.ARMS:
  x,m,t=xx[arm];out=a(x,m,t)['logits'];loss=out.square().mean();loss.backward();assert all(torch.isfinite(v.grad).all() for v in a.parameters() if v.grad is not None);a.zero_grad()
 print(name,'4arms forward/backward, identical initialization, finite',flush=True)
