"""Dry run: real data compatibility and single backward pass, never optimizer.step."""
import sys
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import scale_train as s
ORIG=Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')
sys.path.insert(0,str(ORIG))
import pilot_train as prior
tr=s.load('train');va=s.load('val')
for data in (tr,va):
 m=data['mask'];assert np.all(np.diff(m.astype('int8'),axis=1)<=0)
 assert len(set(data['ids']))==len(data['ids'])
assert not (set(tr['mmsi'])&set(va['mmsi']))
new=s.fit(tr);old=prior.fit(prior.load('train'))
for key in old: np.testing.assert_array_equal(old[key],new[key])
D=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
t=torch.from_numpy
x=t(((tr['self'][:4]-new['sm'])/new['ss']).astype('float32')).to(D); mask=t(tr['mask'][:4]).to(D);x[~mask]=0
f=t(((tr['f1'][:4]-new['fm'])/new['fs']).astype('float32')).to(D);valid=t(tr['valid'][:4]).to(D);f[~valid]=0
rho=t(tr['rho'][:4]).to(D);tm=t(tr['times'][:4]).to(D)
y=t(tr['y'][:4]).to(D);yc=t(tr['yc'][:4]).to(D);state=t(tr['state1'][:4]).to(D)
for name in ('tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'):
 torch.manual_seed(0);model=s.Model(name).to(D);model.train()
 pb,pc,h,seq=model(x,mask,tm,f,rho,True)
 _,_,h2,_=model(x*.9,mask,tm,f,rho,True)
 loss,_=s.historical_bundle(model.backbone.head,model.typehead,model.ctrl,pb,pc,h2,seq,state,mask,y,yc,torch.ones(5,device=D),torch.ones(len(y),device=D))
 assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 print(name,'real-data backward pass',float(loss.detach()),flush=True)
