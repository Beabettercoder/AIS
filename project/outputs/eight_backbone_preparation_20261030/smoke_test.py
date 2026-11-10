import sys,json,hashlib
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
ORIG=Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ORIG))
from eight_backbone_adapter import Model8
from pilot_train import Model as Pilot4
from pilot_historical_loss import historical_bundle

torch.manual_seed(72);torch.set_num_threads(2)
dev=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
x=torch.randn(3,20,7,device=dev);mask=(torch.arange(20,device=dev)[None,:]<torch.tensor([20,17,12],device=dev)[:,None]);x=x*mask[...,None]
time=torch.arange(20,device=dev)[None].expand(3,-1).float()*70;time=time*mask
field=torch.randn(3,20,4,device=dev);rho=torch.rand(3,20,device=dev)*mask
out={}
for name in ['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph']:
 torch.manual_seed(11);model=Model8(name).to(dev);model.eval()
 with torch.no_grad():
  y,t,emb,seq=model(x,mask,time,field,rho,True)
  y0,_,base,_=model(x,mask,time,field,torch.zeros_like(rho),True)
  yr,_,_,_=model(x,mask,time,field,rho,False)
  assert torch.equal(y,y0) and torch.equal(y0,yr),name+' zero alpha/zero support'
  model.alpha.fill_(0.71)
  yz,_,_,_=model(x,mask,time,field,torch.zeros_like(rho),True)
  yr2,_,_,_=model(x,mask,time,field,rho,False)
  assert torch.equal(yz,yr2),name+' trained nonzero alpha no-support fallback'
  model.alpha.fill_(0.)
  actual=model.backbone(torch.cat([x,x.new_zeros(3,20,5)],-1) if model.graph else x,mask,time)['embedding']
  assert torch.allclose(actual,base,atol=1e-6,rtol=1e-5),(name,float((actual-base).abs().max()))
  if name in ('tcn','gru','transformer','itransformer'):
   old=Pilot4(name).to(dev);old.load_state_dict(model.state_dict());old.eval()
   result=old(x,mask,time,field,rho,True)
   for a,b in zip((y,t,emb,seq),result): assert torch.equal(a,b),name+' pilot mismatch'
 model.train();out1,typ,embedding,seq=model(x,mask,time,field,rho,True)
 h2=model(x*0.9,mask,time,field,rho,True)[2]
 state=torch.rand(3,20,4,device=dev);labels=torch.tensor([0.,1.,1.],device=dev);classid=torch.tensor([0,1,2],device=dev);cw=torch.ones(5,device=dev);ret=torch.ones(3,device=dev)
 loss,_=historical_bundle(model.backbone.head,model.typehead,model.ctrl,out1,typ,h2,seq,state,mask,labels,classid,cw,ret)
 assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),name+' gradient'
 out[name]={'status':'pass','params':sum(p.numel() for p in model.parameters()),'embedding_match':True,'initial_and_no_field_exact_fallback':True,'gradients_finite':True,'old_four_exact': name in ('tcn','gru','transformer','itransformer')}
 print(name,out[name],flush=True)
print(json.dumps(out,indent=2))

