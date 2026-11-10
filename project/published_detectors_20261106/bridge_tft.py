"""TFT-GCN controlled bridge: fixed 13-channel network and historical late field."""
import argparse, hashlib, json, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from tqdm import tqdm
import data_resampled as data
from adapters.tft_gcn import Adapter

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
sys.path.insert(0,str(ROOT/'original_mainline_pilot_20261029'))
from pilot_features import build_field_inputs
from pilot_historical_loss import historical_bundle

def raw_load(split):
    root=ROOT/'framework_validation_v1_20261028'
    with np.load(root/'data/individual'/f'{split}.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(root/'extra_data'/f'field_{split}.npz',allow_pickle=False) as z:f={k:z[k] for k in z.files}
    rho_path=(ROOT/'eight_backbone_preparation_20261030'/'rho_test.npz' if split=='test' else ROOT/'original_mainline_pilot_20261029'/f'rho_{split}.npz')
    with np.load(rho_path,allow_pickle=False) as z:rho=z['rho'].copy()
    feat=build_field_inputs(d,f,'F1')
    if rho.shape!=d['mask'].shape or not np.array_equal(rho>0,feat['valid']):raise ValueError(f'{split} legacy rho support mismatch')
    if len(np.unique(d['window_id']))!=len(d['window_id']):raise ValueError(f'{split} duplicate window ID')
    return d,feat,rho

def load(split,original=False,rho_mode='count'):
    d,feat,rho=raw_load(split);res=data.load(split)
    state=np.zeros_like(feat['state']);rho_grid=np.zeros_like(rho)
    if original:
        res.update(raw=d['base__C0'][...,:7],target=feat['self'],f1=feat['field'],
          f0=feat['field'],valid=feat['valid'],rho=rho if rho_mode=='legacy' else feat['rho'],mask=d['mask'],state=feat['state'])
        return res
    for i in range(len(state)):
        v=d['mask'][i];t=d['time_s'][i,v];t=t-t[0];g=np.linspace(t[0],t[-1],20);ok=feat['valid'][i,v]
        state[i],support=data._field_on_grid(t,feat['state'][i,v],ok,g)
        r,support2=data._field_on_grid(t,rho[i,v,None],ok,g);rho_grid[i]=r[:,0]
        assert np.array_equal(support,support2) and np.array_equal(support,res['valid'][i])
    if rho_mode=='legacy':res['rho']=rho_grid
    res.update(mask=np.ones_like(d['mask']),state=state)
    return res

class Model(nn.Module):
    def __init__(self,gpu):
        super().__init__();self.net=Adapter(20,13,gpu=gpu);self.hidden=nn.Linear(260,64)
        self.head=nn.Linear(64,1);self.typehead=nn.Linear(64,5);self.ctrl=nn.Linear(4,2)
        self.field_adapter=nn.Sequential(nn.Linear(4,64),nn.GELU(),nn.Linear(64,64));self.alpha=nn.Parameter(torch.tensor(0.));self.seq=None
        self.net.model.t_model.layer_norm.register_forward_hook(self._hook)
    def _hook(self,module,inputs,output):self.seq=output
    def forward(self,selfx,field,rho,mask,direct=False,late=False):
        extra=torch.zeros((*selfx.shape[:2],6),dtype=selfx.dtype,device=selfx.device)
        if direct:extra=torch.cat((field,rho[...,None],(rho>0).to(field.dtype)[...,None]),dim=-1)
        x=torch.cat((selfx,extra),dim=-1);out=self.net.model(x);sequence=self.seq
        h=F.gelu(self.hidden(out.reshape(len(out),-1)))
        if late:
            weight=(rho*mask.to(rho.dtype)).unsqueeze(-1)
            h=h+self.alpha*(self.field_adapter(field)*weight).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return self.head(h).squeeze(-1),self.typehead(h),h,sequence,x,out

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['raw','gate','direct','late','bundle','original'],required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--aux',type=float,default=0.);p.add_argument('--rho-mode',choices=['count','legacy'],default='count');p.add_argument('--smoke',action='store_true');p.add_argument('--mode',choices=['train','test'],default='train');p.add_argument('--out',default='/home/ps/AIS_bridge_20261107/tft_eval128')
    a=p.parse_args();torch.set_num_threads(2);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device(f'cuda:{a.gpu}')
    # Identical training-only stats for both time representations.
    normal_train=data.load('train');norm=data.fit(normal_train,supervised=True)
    tr,va=(load(s,a.arm=='original',a.rho_mode) for s in ('train','val'))
    for x,y in ((tr,va),):
        if set(x['mmsi'])&set(y['mmsi']):raise ValueError('MMSI overlap')
    def tensors(d):
        mask=d['mask'].astype(bool);key='raw' if a.arm=='raw' else 'target'
        x=((d[key]-norm['sm'])/norm['ss']).astype('float32');x[~mask]=0
        field=np.where(d['valid'][...,None],(d['f1']-norm['fm'])/norm['fs'],0).astype('float32');field[~mask]=0
        cats={'turn_back':1,'deviation':2,'stall':3,'sinuous':4,'speed_drop':5}
        meta,_,_=raw_load('train' if d is tr else 'val' if d is va else 'test')
        yc=np.array([cats.get(str(s),0) for s in meta['behavior_type']],np.int64)
        if not np.array_equal(yc>0,d['y']>0):raise ValueError('type labels mismatch')
        return {k:torch.as_tensor(v,device=device) for k,v in dict(x=x,field=field,rho=d['rho'].astype('float32'),mask=mask,state=d['state'].astype('float32'),y=d['y'].astype('float32'),yc=yc).items()}
    ttr,tva=tensors(tr),tensors(va)
    if a.smoke:
        for t in (ttr,tva):
            for k in t:t[k]=t[k][:64]
    def fingerprint(d,t):
        h=hashlib.sha256()
        for k in ('window_id','mmsi','y'):h.update(np.asarray(d[k][:len(t['y'])]).tobytes())
        for k in ('x','field','rho','mask','state','y','yc'):h.update(t[k].detach().cpu().numpy().tobytes())
        return h.hexdigest()
    fingerprints={'train':fingerprint(tr,ttr),'validation':fingerprint(va,tva)}
    model=Model(a.gpu).to(device);out=Path(a.out)/('smoke' if a.smoke else f'aux{a.aux:g}_{a.rho_mode}')/a.arm/f'seed{a.seed}'
    direct=a.arm=='direct';late=a.arm in ('late','bundle','original');bundle=a.arm in ('bundle','original')
    def forward(b):return model(b['x'],b['field'],b['rho'],b['mask'],direct,late)
    def pred(t):
        model.eval();z=[]
        with torch.no_grad():
            for i in range(0,len(t['y']),a.batch_size):
                b={k:v[i:i+a.batch_size] for k,v in t.items()};z.extend(torch.sigmoid(forward(b)[0]).cpu().numpy())
        return np.asarray(z)
    if a.mode=='test':
        ck=torch.load(out/'best_val.pt',map_location=device,weights_only=False)
        if ck['fingerprints']!=fingerprints:raise ValueError('Training or validation input changed since checkpoint')
        model.load_state_dict(ck['model']);te=load('test',a.arm=='original',a.rho_mode);tt=tensors(te);z=pred(tt);y=te['y']
        np.savez_compressed(out/'test_scores.npz',score=z,y=y,mmsi=te['mmsi'],region=te['region'],window_id=te['window_id'])
        print('TEST',a.arm,float(average_precision_score(y,z)),float(roc_auc_score(y,z)),flush=True);return
    if out.exists() and not a.smoke:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);best=-1.;history=[]
    initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    spec=vars(a)|{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fixed_shape':'13 channels; identical architecture and seed for all arms','initial_model_sha256':initial,'data_sha256':fingerprints,'late_fusion':'alpha=0 initialized; rho-weighted field adapter; count-only for strict direct-vs-late pair','physical_state_interpolation':'real-time interpolation only between adjacent supported steps, no bridging unsupported gaps; original arm uses original points','auxiliary':'original TFT-GCN time+FFT reconstruction; optional, default 0 for BCE-only; mask-weighted time and zero-padded FFT','bundle':'historical_bundle with actual final temporal layer sequence; physical state before normalization','normalization':'fixed shared resampled train-only stats across sampling arms','original_sampling':'right-zero-padding with original mask; TFT-GCN internal blocks cannot mask padding or consume timestamps, not identical to historical TCN','evaluation_batch':'fixed 128 in ordered windows; TFT selects periodic frequencies across each batch, so score may depend on batch composition','selection':'validation pooled AP','test':'previously exposed, not used for model selection'}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    cnt=np.bincount(ttr['yc'][ttr['yc']>0].cpu().numpy()-1,minlength=5);cw=torch.as_tensor(1/np.maximum(cnt,1),dtype=torch.float32,device=device)
    for epoch in range(a.epochs):
        model.train();order=np.random.default_rng(300000+epoch).permutation(len(ttr['y']));losses=[]
        for pos in tqdm(range(0,len(order),a.batch_size),desc=f'TFT {a.arm} seed{a.seed} {epoch+1}/{a.epochs}',file=sys.stdout):
            ids=torch.as_tensor(order[pos:pos+a.batch_size],device=device);b={k:v[ids] for k,v in ttr.items()};opt.zero_grad(set_to_none=True)
            logits,types,h,seq,x,recon=forward(b);bce=F.binary_cross_entropy_with_logits(logits,b['y']);mask=b['mask']
            if bundle:
                keep=(torch.rand(x.shape[:2],device=device)>.15).float();b2=b.copy();b2['x']=b['x']*keep.unsqueeze(-1);n=mask.sum(1);j=torch.floor(torch.rand(len(x),device=device)*n).long();b2['x'][torch.arange(len(x),device=device),j,0]+=2.
                _,_,h2,_,_,_=forward(b2);ret=(keep*mask).sum(1)/n.clamp_min(1)
                bce,_=historical_bundle(model.head,model.typehead,model.ctrl,logits,types,h2,seq,b['state'],mask,b['y'],b['yc'],cw,ret)
            mf=mask.unsqueeze(-1);d=(recon-x)*mf;freq=torch.fft.fft(d.transpose(1,2),norm='forward');aux=(d.square().sum()/(mf.sum()*13).clamp_min(1))+freq.real.square().mean()+freq.imag.square().mean();loss=bce+a.aux*aux
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        z=pred(tva);y=tva['y'].cpu().numpy();ap=float(average_precision_score(y,z));row={'epoch':epoch+1,'loss':float(np.mean(losses)),'AP':ap,'alpha':float(model.alpha.detach())};history.append(row)
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'spec':spec,'epoch':epoch+1,'metrics':row,'fingerprints':fingerprints};torch.save(ck,out/'last.pt')
        if ap>best:
            best=ap;torch.save(ck,out/'best_val.pt')
            np.savez_compressed(out/'val_scores.npz',score=z,y=y,mmsi=va['mmsi'][:len(y)],region=va['region'][:len(y)],window_id=va['window_id'][:len(y)])
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8');print('VALIDATION',json.dumps(row),flush=True)
if __name__=='__main__':main()
