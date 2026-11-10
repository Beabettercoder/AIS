"""Frozen developer pilot. Four backbones, seed0, base train, no test evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import sys,json,hashlib,random,argparse
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/relation_repair_20261030')
PRE=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028')
sys.path.insert(0,str(PRE))
from models import make_model
VIEWS=('base','natural','suffix','combined_l100_m600')
ARMS=('C0','F0','F1')
NF=24; TARGET=8

def read(split):
    with np.load(OUT/(split+'_relation_field.npz'),allow_pickle=False) as z: d={k:z[k] for k in z.files}
    d['mask']=np.arange(20)[None,:]<d['length'][:,None]
    return d

def features(d,view,arm):
    # 8 target channels and 16 context channels. C0 has the identical target path, zero context.
    mask=d['mask']; b,t=mask.shape; out=np.zeros((b,t,NF),np.float32); cm=np.zeros((b,t),bool)
    # Four grid setting is offline; per step all quantities only use records available <= that step.
    selfx=d['causal7']; tm=d['time_s']; v=selfx[:,:,2:4]; sp=selfx[:,:,4]; a=selfx[:,:,5:7]
    out[:,:,:TARGET]=np.concatenate((v,sp[...,None],a,v-v[:,:1],(sp-sp[:,:1])[...,None]),axis=-1)
    if arm!='C0':
        f=d[view+'__field'][:,0 if arm=='F0' else 1]; count=d[view+'__count']
        bg=f[:,:,:2]; valid=mask & (f[:,:,7]>0); cm=valid
        res=v-bg
        first=np.argmax(valid,axis=1); rows=np.arange(b); anchorbg=bg[rows,first]; anchorr=res[rows,first]; at=tm[rows,first]
        dt=tm-at[:,None]; can=valid & (dt>0)
        bgdelta=np.where(valid[...,None],bg-anchorbg[:,None,:],0)
        rdelta=np.where(valid[...,None],res-anchorr[:,None,:],0)
        ratebg=np.divide(bgdelta,dt[...,None],out=np.zeros_like(bgdelta),where=can[...,None])
        rater=np.divide(rdelta,dt[...,None],out=np.zeros_like(rdelta),where=can[...,None])
        # derivative gradients zero for zero-order mode; mode is explicit.
        support=np.minimum(count/4.,1.).astype(np.float32) if arm=='F0' else f[:,:,6]
        c=np.concatenate((bg,bgdelta,ratebg,res,rdelta,rater,f[:,:,4:6],support[...,None],(f[:,:,7]==2)[...,None].astype(np.float32)),axis=-1)
        assert c.shape[-1]==16
        out[:,:,TARGET:]=np.where(valid[...,None],c,0)
    out[~mask]=0
    return out,cm

def fit(train):
    x0,_=features(train,'base','C0'); x1,m1=features(train,'base','F0');x2,m2=features(train,'base','F1')
    means=np.zeros(NF);std=np.ones(NF)
    z=x0[train['mask'],:TARGET].astype(np.float64);means[:TARGET]=z.mean(0);std[:TARGET]=z.std(0)
    context=np.concatenate((x1[m1,TARGET:-2],x2[m2,TARGET:-2])).astype(np.float64)
    means[TARGET:-2]=context.mean(0);std[TARGET:-2]=context.std(0)
    std[std<1e-7]=1
    return means,std

def prep(d,view,arm,mu,sd,device):
    x,cm=features(d,view,arm); mask=d['mask']; x=(x-mu)/sd
    x[...,TARGET:]*=cm[...,None];x[~mask]=0
    assert np.isfinite(x).all()
    time=d['time_s'].astype(np.float64).copy();time-=time[:,:1];time[~mask]=0
    return torch.tensor(x,dtype=torch.float32,device=device),torch.tensor(mask,device=device),torch.tensor(time,dtype=torch.float32,device=device)

def seed():random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
def metrics(y,p,region):
    return {'pooled_AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),'macro_AP':float(np.mean([average_precision_score(y[region==r],p[region==r]) for r in np.unique(region)]))}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def run(name,gpu):
    seed();torch.set_num_threads(2);dev=torch.device('cuda:'+str(gpu));tr=read('train');va=read('val');mu,sd=fit(tr)
    print('data ready',name,len(tr['y']),len(va['y']),flush=True)
    dest=OUT/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
    spec={'seed':0,'epochs':25,'lr':.001,'batch_size':128,'optimizer':'AdamW wd .01; clip 1.0','training':'BCE on base only; fixed epoch25','target':'causal7 velocity, speed, heading and changes from first; 8 channels shared across arms','context':'background velocity, change and rate from first supported step, target velocity residual, change and rate, D,C, support, affine availability; 16 channels','standardization':'target from train/base C0; post-feature construction F0/F1 context pooled train/base supported valid steps; support and affine flag unstandardized; zero std floor 1; missing context zero','F0':'causal target plus distance-weighted segment mean field','F1':'causal target plus reliability-weighted segment report mean and distance*member-reliability weighted field','C0':'same causal target, context zero','offline':'full-window model, no online early-warning claim','views':VIEWS,'label_use':'only supervised BCE and metrics','input_sha256':{str(p):sha(p) for p in [Path(__file__),OUT/'builder.py',OUT/'train_relation_field.npz',OUT/'val_relation_field.npz',PRE/'models.py']},'means':mu.tolist(),'stds':sd.tolist()}
    (dest/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    for arm in ARMS:
        seed();model=make_model(name,NF,max_len=20,hidden=64).to(dev);initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest()
        decay=[p for p in model.parameters() if not getattr(p,'_no_weight_decay',False)];nodecay=[p for p in model.parameters() if getattr(p,'_no_weight_decay',False)]
        groups=[{'params':decay,'weight_decay':.01}]+([{'params':nodecay,'weight_decay':0.}] if nodecay else [])
        opt=torch.optim.AdamW(groups,lr=.001)
        x,m,t=prep(tr,'base',arm,mu,sd,dev); y=torch.tensor(tr['y'],dtype=torch.float32,device=dev)
        hist=[]; folder=dest/arm;folder.mkdir(exist_ok=True)
        for epoch in range(25):
            model.train();order=np.random.default_rng(300000+epoch).permutation(len(y));losses=[]
            for i in range(0,len(y),128):
                ix=torch.tensor(order[i:i+128],device=dev);opt.zero_grad(set_to_none=True)
                log=model(x[ix],m[ix],t[ix])['logits'];loss=F.binary_cross_entropy_with_logits(log,y[ix])
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
            hist.append(float(np.mean(losses)))
            if epoch%5==0:print('EPOCH',name,arm,epoch+1,hist[-1],flush=True)
        model.eval(); result={}; saved={}
        for view in VIEWS:
            xv,mv,tv=prep(va,view,arm,mu,sd,dev);pred=[]
            with torch.no_grad():
                for i in range(0,len(xv),256):pred.extend(torch.sigmoid(model(xv[i:i+256],mv[i:i+256],tv[i:i+256])['logits']).cpu().numpy())
            pred=np.asarray(pred); result[view]=metrics(va['y'],pred,va['region']);saved[view+'_score']=pred
        np.savez_compressed(folder/'val_scores.npz',y=va['y'],mmsi=va['mmsi'],region=va['region'],source=va['source'],case=va['case'],**saved)
        torch.save(model.state_dict(),folder/'final.pt')
        (folder/'result.json').write_text(json.dumps({'arm':arm,'seed':0,'init_sha256':initial,'last_epoch':25,'loss_history':hist,'views':result,'prediction_sha256':sha(folder/'val_scores.npz'),'weights_sha256':sha(folder/'final.pt')},indent=2),encoding='utf8')
        print('DONE',name,arm,result,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['tcn','gru','transformer','itransformer'],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();run(a.name,a.gpu)
