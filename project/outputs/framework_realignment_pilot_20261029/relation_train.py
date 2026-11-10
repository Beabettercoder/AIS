"""Four encoders / one seed, relation dataset, development validation only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,argparse,json,random,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
OUT=ROOT/'framework_realignment_pilot_20261029'
sys.path.insert(0,str(ROOT/'original_mainline_pilot_20261029'))
from pilot_train import Model
VIEWS=('base','natural','suffix','combined_l100_m600')
def load(split):
    with np.load(OUT/(split+'_relation_field.npz'),allow_pickle=False) as d: a={k:d[k] for k in d.files}
    a['mask']=np.arange(20)[None,:]<a['length'][:,None]
    time=a['time_s'].copy();time-=time[:,:1];time[~a['mask']]=0;a['times']=time.astype(np.float32)
    for view in VIEWS:
        for i,arm in enumerate(('f0','f1')):
            z=a[view+'__field'][:,i].copy();rho=z[:,:,6];valid=(rho>0)&a['mask']; raw=z[:,:,2:6].copy()
            raw[:,:,0:2]=a['causal7'][:,:,2:4]-z[:,:,0:2];raw[~valid]=0
            magnitude=np.linalg.norm(z[:,:,:2],axis=2); change=np.zeros_like(magnitude);dt=np.diff(a['time_s'],axis=1);good=(dt>0)&valid[:,1:]&valid[:,:-1];change[:,1:][good]=np.abs(np.diff(magnitude,axis=1)[good])/dt[good]
            state=np.stack([magnitude,change,z[:,:,4],z[:,:,5]],axis=-1);state[~valid]=0
            a[view+'_'+arm]=raw;a[view+'_'+arm+'_state']=state;a[view+'_rho']=rho;a[view+'_valid']=valid
    return a
def normalize(train,val):
    f=np.concatenate([train['base_f0'][train['base_valid']],train['base_f1'][train['base_valid']]])
    x=np.concatenate([train['raw7'][train['mask']],train['causal7'][train['mask']]])
    def stats(z):
        m=z.mean(0);s=z.std(0);s[s<1e-7]=1.;return m,s
    sm,ss=stats(x);fm,fs=stats(f)
    for a in (train,val):
        for k in ('raw7','causal7'):
            a[k]=((a[k]-sm)/ss).astype('float32');a[k][~a['mask']]=0
        for view in VIEWS:
            for arm in ('f0','f1'):
                k=view+'_'+arm;a[k]=((a[k]-fm)/fs).astype('float32');a[k][~a[view+'_valid']]=0
    return {'target_mean':sm.tolist(),'target_std':ss.tolist(),'field_mean':fm.tolist(),'field_std':fs.tolist()}
def tensors(a,device):
    keys=['raw7','causal7','mask','times','y']+[v+'_'+k for v in VIEWS for k in ('f0','f1','f0_state','f1_state','rho')]
    return {k:torch.as_tensor(a[k],device=device) for k in keys}
def train_loss(model,pb,h2,seq,state,mask,y,ret):
    # Adapted step-35: no fabricated five-class label/type JS. Binary JS substitutes type JS.
    mf=mask.to(state.dtype);ctrl=model.ctrl((state*mf.unsqueeze(-1)).sum(1)/mf.sum(1,keepdim=True).clamp_min(1))
    eps=.05+.2*(torch.tanh(ctrl[:,0])+1)/2;mu=h2.mean(0,keepdim=True).detach();sd=h2.std(0,keepdim=True).detach().clamp_min(1e-5)
    h=mu+sd*(((h2-mu)/sd)*(1+eps[:,None])+.02*torch.tanh(ctrl[:,1])[:,None]);pb2=model.backbone.head(h).squeeze(-1)
    p=torch.sigmoid(pb).clamp(1e-6,1-1e-6);q=torch.sigmoid(pb2).clamp(1e-6,1-1e-6);m=(p+q)/2
    kl=lambda a,b:a*torch.log(a/b)+(1-a)*torch.log((1-a)/(1-b))
    js=((kl(p,m)+kl(q,m))*.5*ret).mean()
    delta=seq[:,2:]-2*seq[:,1:-1]+seq[:,:-2];v=mask[:,2:]&mask[:,1:-1]&mask[:,:-2];smooth=(delta.square()*v.unsqueeze(-1)).sum()/(v.sum()*seq.shape[-1]).clamp_min(1)
    return F.binary_cross_entropy_with_logits(pb,y)+.2*js+.3*F.binary_cross_entropy_with_logits(pb2,y)+.01*smooth
def seed():random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=['tcn','gru','transformer','itransformer']);p.add_argument('--gpu',type=int,required=True);args=p.parse_args();torch.set_num_threads(2)
    tr,va=load('train'),load('val');norm=normalize(tr,va);dev=torch.device('cuda:'+str(args.gpu));trt=tensors(tr,dev);vat=tensors(va,dev)
    dest=OUT/'relation_runs'/args.name;dest.mkdir(parents=True,exist_ok=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protocol={'task':'four-cell controlled relationship','seed':0,'epochs':25,'train_view':'base','validation_views':VIEWS,'comparison':'raw target BCE, causal no-field with adapted bundle, causal F0 and F1 with adapted bundle','bundle':'step35 adapted for binary task: no type CE, binary rather than type JS; same ctrl, perturbation, aug BCE, smoothness','rho':'same continuous mean member w for F0/F1','predictions':'validation only, existing exposed sources, not independent final test','normalization':norm,'sha256':{str(p):sha(p) for p in [Path(__file__),OUT/'relation_field_builder.py',OUT/'train_relation_field.npz',OUT/'val_relation_field.npz',ROOT/'original_mainline_pilot_20261029/pilot_train.py']}}
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    for arm in ('raw','causal_bundle','f0_bundle','f1_bundle'):
        seed();model=Model(args.name).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);hasfield=arm.startswith('f');fk='f0' if arm.startswith('f0') else 'f1';target='raw7' if arm=='raw' else 'causal7'
        initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest();folder=dest/arm;folder.mkdir(exist_ok=True)
        hist=[]
        for epoch in range(25):
            model.train();order=np.random.default_rng(300000+epoch).permutation(len(tr['y']));losses=[]
            for start in range(0,len(order),128):
                ix=torch.tensor(order[start:start+128],device=dev);x=trt[target][ix];mask=trt['mask'][ix];tm=trt['times'][ix];f=trt['base_'+fk][ix];rho=trt['base_rho'][ix];y=trt['y'][ix].float();opt.zero_grad(set_to_none=True)
                pb,pc,h,seq=model(x,mask,tm,f,rho,hasfield)
                if arm=='raw':loss=F.binary_cross_entropy_with_logits(pb,y)
                else:
                    keep=(torch.rand(x.shape[:2],device=dev)>.15).to(x.dtype);x2=x*keep.unsqueeze(-1);length=mask.sum(1);jj=torch.floor(torch.rand(len(x),device=dev)*length).long();x2=x2*mask.unsqueeze(-1);x2[torch.arange(len(x),device=dev),jj,0]+=2.;ret=(keep*mask).sum(1)/length.clamp_min(1)
                    _,_,h2,_=model(x2,mask,tm,f,rho,hasfield);loss=train_loss(model,pb,h2,seq,trt['base_'+fk+'_state'][ix],mask,y,ret)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
            hist.append(float(np.mean(losses)))
            if epoch%5==0:print('EPOCH',args.name,arm,epoch+1,hist[-1],flush=True)
        model.eval();results={};saved={}
        for view in VIEWS:
            predictions=[]
            with torch.no_grad():
                for start in range(0,len(va['y']),256):
                    z=slice(start,start+256);p,_,_,_=model(vat[target][z],vat['mask'][z],vat['times'][z],vat[view+'_'+fk][z],vat[view+'_rho'][z],hasfield);predictions.extend(torch.sigmoid(p).cpu().numpy())
            predictions=np.asarray(predictions);y=va['y'];r=va['region'];results[view]={'pooled_AP':float(average_precision_score(y,predictions)),'macro_AP':float(np.mean([average_precision_score(y[r==region],predictions[r==region]) for region in np.unique(r)])),'AUROC':float(roc_auc_score(y,predictions)),'supported_steps':int(va[view+'_valid'].sum()),'total_steps':int(va['mask'].sum())};saved[view+'_score']=predictions
        np.savez_compressed(folder/'val_scores.npz',y=va['y'],mmsi=va['mmsi'],region=va['region'],source=va['source'],case=va['case'],**saved);torch.save(model.state_dict(),folder/'final.pt')
        (folder/'result.json').write_text(json.dumps({'arm':arm,'seed':0,'init_sha256':initial,'last_epoch':25,'loss_history':hist,'alpha':float(model.alpha.detach()),'views':results},indent=2),encoding='utf8');print('DONE',args.name,arm,results,flush=True)
if __name__=='__main__':main()
