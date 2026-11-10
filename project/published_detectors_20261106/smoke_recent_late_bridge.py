"""Unit/integration smoke on frozen AIS training windows. Not an AP experiment."""
import argparse,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
sys.path.insert(0,str(ROOT/'formal_20261101/single'))
import train_single as historical
from recent_late_bridge import RecentLateBridge
# The original AIS baseline imports a top-level models.py; TFT-GCN uses a
# separate package with the same name. Both are already available to callers.
sys.modules.pop('models',None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    torch.manual_seed(3);torch.set_num_threads(2);dev=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    d=historical.load('train');n=historical.fit(d);batch=8;sl=slice(0,batch)
    mask=torch.tensor(d['mask'][sl],device=dev)
    x=torch.tensor(((d['self'][sl]-n['sm'])/n['ss']).astype('float32'),device=dev);x=x.masked_fill(~mask[...,None],0)
    f=torch.tensor(((d['f1'][sl]-n['fm'])/n['fs']).astype('float32'),device=dev)
    valid=torch.tensor(d['valid'][sl],device=dev);f=f.masked_fill(~valid[...,None],0.)
    rho=torch.tensor(d['rho'][sl],device=dev);state=torch.tensor(d['state1'][sl],device=dev)
    y=torch.tensor(d['y'][sl],device=dev);yc=torch.tensor(d['yc'][sl],device=dev)
    model=RecentLateBridge(a.method,gpu=a.gpu).to(dev);model.eval()
    with torch.no_grad():
        h,seq=model.encode(x,mask)
        if a.method=='carla': ref=model.target_encoder(model.adapter.embed(x))
        else:
            out=model.adapter.model(x)
            if isinstance(out,tuple):out=out[0].reshape(batch,20,7)
            ref=model.target_encoder(out.masked_fill(~mask[...,None],0.).reshape(batch,-1))
        diff=(h-ref).abs().max().item()
        if diff>1e-5:raise AssertionError(f'upstream forward mismatch {diff}')
        z0=model(x,mask,f,rho,False)[0];z1=model(x,mask,f,rho,True)[0]
        if not torch.equal(z0,z1): raise AssertionError('alpha=0 branch changes output')
        model.alpha.fill_(1.)
        blank=model(x,mask,f,torch.zeros_like(rho),True)[0]
        if not torch.equal(z0,blank):raise AssertionError('rho=0 fallback changes output')
        model.alpha.fill_(0.)
    model.train();loss,parts=model.bundle_loss(x,mask,f,rho,True,state,y,yc,torch.ones(5,device=dev));loss.backward()
    if not torch.isfinite(loss):raise AssertionError('nonfinite historical loss')
    model.zero_grad(set_to_none=True)
    if a.method=='carla':
        native=model.native_loss(x,mask,positive=x+0.01*torch.randn_like(x));native.backward();model.zero_grad(set_to_none=True)
        model.adapter.begin_clustering()
        native2=model.native_loss(x,mask,near=x.roll(1,0),far=x.roll(2,0));native2.backward()
        if not torch.isfinite(native2):raise AssertionError('nonfinite CARLA cluster loss')
    else:
        if hasattr(model.adapter,'begin_epoch'):model.adapter.begin_epoch(0,1)
        native=model.native_loss(x,mask,indices=np.arange(batch),epoch=0);native.backward()
    if not torch.isfinite(native):raise AssertionError('nonfinite native loss')
    print('PASS',a.method,'forward_native_max_diff',diff,'target',tuple(h.shape),'sequence',tuple(seq.shape),
          'sequence_type',model.sequence_provenance,'bundle',float(loss),'native',float(native),
          'mask_counts',mask.sum(1).tolist(),'valid_field_steps',int((rho>0).sum()),flush=True)
if __name__=='__main__':main()
