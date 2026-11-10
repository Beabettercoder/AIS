import sys, numpy as np, torch
from pathlib import Path
S=Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')
sys.path.insert(0,str(S))
import scale_train as s
tr=s.load('train'); n=s.fit(tr); dev=torch.device('cuda:7');torch.set_num_threads(2)
t=s.convert(tr,n,dev)
q=t['self'].abs().amax(dim=(1,2)); ix=torch.topk(q,128).indices
print('max_abs_std',float(q.max()),'top128_min',float(q[ix].min()),'n',len(ix),flush=True)
cw=torch.ones(5,device=dev)
for name in ('tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'):
 torch.manual_seed(0);model=s.Model(name).to(dev);model.train();x=t['self'][ix];m=t['mask'][ix];tm=t['times'][ix];f=t['f1'][ix];rho=t['rho'][ix];y=t['y'][ix];yc=t['yc'][ix];state=t['state1'][ix]
 pb,pc,h,seq=model(x,m,tm,f,rho,True);_,_,h2,_=model(x*.9,m,tm,f,rho,True)
 loss,_=s.historical_bundle(model.backbone.head,model.typehead,model.ctrl,pb,pc,h2,seq,state,m,y,yc,cw,torch.ones(len(y),device=dev))
 print(name,'loss',float(loss),'logits_max',float(pb.abs().max()),'finite',bool(torch.isfinite(loss)),flush=True)
 if torch.isfinite(loss):
  loss.backward();bad=[k for k,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()];print(name,'bad_grad',bad,flush=True)
 del model
