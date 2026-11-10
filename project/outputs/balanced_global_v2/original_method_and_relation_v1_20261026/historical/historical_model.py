"""Historical Table-1 architecture with explicit target-padding controls.

The complete graph architecture is loaded verbatim from the archived step-35
source. No step-60 corrected graph is substituted. The graph-off backbone and
heads match the step-34 baseline architecture and initialization order.
"""
from pathlib import Path
import importlib.util
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'vendor/35_graph_ours_final.py'
spec=importlib.util.spec_from_file_location('frozen_historical_step35',SOURCE)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

ARMS={
 'raw_incremental_compat':dict(key='raw_incremental7',nf=7,graph=False,recipe=False,masked=False),
 'raw_legacy_compat':dict(key='raw_legacy7',nf=7,graph=False,recipe=False,masked=False),
 'kg_compat':dict(key='kg_compat8',nf=8,graph=False,recipe=False,masked=False),
 'kglf_compat':dict(key='kglf_compat9',nf=9,graph=False,recipe=False,masked=False),
 'kglf_psa_compat':dict(key='kglf_compat9',nf=9,graph=False,recipe=True,masked=False),
 'full35_compat':dict(key='kglf_compat9',nf=9,graph=True,recipe=True,masked=False),
 'raw_legacy_masked':dict(key='raw_legacy7',nf=7,graph=False,recipe=False,masked=True),
 'full35_masked':dict(key='kglf_masked9',nf=9,graph=True,recipe=True,masked=True),
}


class PlainNet(nn.Module):
    def __init__(self,kind,nf,recipe=False):
        super().__init__()
        self.enc=legacy.Enc(kind,nf,64)
        self.bin=nn.Linear(self.enc.dim,1)
        self.cls=nn.Linear(self.enc.dim,5)
        if recipe:self.ctrl=nn.Linear(4,2)
    def forward(self,x):
        h=self.enc(x)
        return self.bin(h).squeeze(-1),self.cls(h),h,torch.zeros_like(h)


class HistoricalModel(nn.Module):
    def __init__(self,kind,arm):
        super().__init__();self.kind=kind;self.arm=arm;self.config=ARMS[arm].copy()
        self.core=legacy.Net(kind,self.config['nf'],'ours',64) if self.config['graph'] else PlainNet(kind,self.config['nf'],self.config['recipe'])
    @property
    def bin(self):return self.core.bin
    @property
    def cls(self):return self.core.cls
    @property
    def ctrl(self):return self.core.ctrl
    def sequence(self,x,mask):
        if not self.config['masked'] or bool(mask.all()):return self.core.enc.seq(x)
        lengths=mask.sum(1);parts=[];indices=[];T=x.shape[1]
        for n in torch.unique(lengths,sorted=True).tolist():
            ix=torch.where(lengths==n)[0]
            seq=self.core.enc.seq(x[ix,:n])
            parts.append(F.pad(seq,(0,T-n)));indices.append(ix)
        all_indices=torch.cat(indices);all_sequences=torch.cat(parts)
        return all_sequences[torch.argsort(all_indices)]
    def representation(self,x,mask):
        if not self.config['masked'] or bool(mask.all()):return self.core.enc(x)
        seq=self.sequence(x,mask);valid=mask.unsqueeze(1)
        mean=(seq*valid).sum(-1)/mask.sum(1).clamp_min(1).unsqueeze(-1)
        maximum=seq.masked_fill(~valid,float('-inf')).max(-1).values
        return torch.cat([mean,maximum],-1)
    def forward(self,x,mask,nseq=None,nxy=None,nmask=None,nw=None,rf=None):
        if not self.config['masked'] or bool(mask.all()):
            if self.config['graph']:
                return self.core(x,nseq,nxy,nmask,nw,rf[:,-1])
            return self.core(x)
        h=self.representation(x,mask)
        if self.config['graph']:
            last=mask.sum(1)-1
            ri=rf[torch.arange(len(rf),device=rf.device),last]
            h,ctx=self.core.graph(h,nseq,nxy,nmask,nw,ri)
        else:ctx=torch.zeros_like(h)
        return self.bin(h).squeeze(-1),self.cls(h),h,ctx


def epsilon_and_perturbed_hidden(model,h2,ep,mask):
    cfg=model.config
    summary=(ep*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp_min(1).unsqueeze(-1) if cfg['masked'] and not bool(mask.all()) else ep.mean(1)
    e=model.ctrl(summary)
    eps=.05+.2*(torch.tanh(e[:,0])+1)/2
    mu=h2.mean(0,keepdim=True).detach();sd=h2.std(0,keepdim=True).detach().clamp_min(1e-5)
    hp=mu+sd*(((h2-mu)/sd)*(1+eps[:,None])+.02*torch.tanh(e[:,1])[:,None])
    return eps,hp


def augment_features(x,mask,masked):
    keep=(torch.rand(x.shape[:2],device=x.device)>.15).to(x.dtype)
    augmented=x*keep.unsqueeze(-1)
    if masked and not bool(mask.all()):
        lengths=mask.sum(1)
        jj=torch.floor(torch.rand(len(x),device=x.device)*lengths).long()
        augmented=augmented*mask.unsqueeze(-1)
        retention=(keep*mask).sum(1)/lengths.clamp_min(1)
    else:
        jj=torch.randint(0,x.shape[1],(len(x),),device=x.device)
        retention=keep.mean(1)
    augmented[torch.arange(len(x),device=x.device),jj,0]+=2.
    return augmented,retention,jj,keep


def smooth_penalty(sequence,mask,masked):
    diff=sequence[:,:,2:]-2*sequence[:,:,1:-1]+sequence[:,:,:-2]
    if not masked or bool(mask.all()):return diff.square().mean()
    valid=mask[:,2:]&mask[:,1:-1]&mask[:,:-2]
    return (diff.square()*valid.unsqueeze(1)).sum()/(valid.sum()*sequence.shape[1]).clamp_min(1)


def training_loss(model,batch,class_weights):
    x,mask=batch['x'],batch['mask'];yb=batch['y'];yc=batch['label_class']
    args=(mask,batch['nb_seq'],batch['nb_xy'],batch['nb_mask'],batch['nb_w'],batch['rf'])
    pb,pc,h,_=model(x,*args)
    binary=F.binary_cross_entropy_with_logits(pb,yb)
    positive=yb>0
    typed=F.cross_entropy(pc[positive],yc[positive]-1,weight=class_weights) if positive.any() else binary*0
    loss=binary+.5*typed
    logs=dict(binary=float(binary.detach()),type_ce=float(typed.detach()),js=0.,aug_binary=0.,hidden_smooth=0.,epsilon_mean=0.)
    if model.config['recipe']:
        x2,retention,_,_=augment_features(x,mask,model.config['masked'])
        _,_,h2,_=model(x2,*args)
        eps,hp=epsilon_and_perturbed_hidden(model,h2,batch['ep'],mask)
        aug_b=F.binary_cross_entropy_with_logits(model.bin(hp).squeeze(-1),yb)
        p1,p2=F.softmax(pc,-1),F.softmax(model.cls(hp),-1);middle=.5*(p1+p2)
        kl=lambda a,b:(a*(torch.log(a+1e-8)-torch.log(b+1e-8))).sum(-1)
        js=((.5*kl(p1,middle)+.5*kl(p2,middle))*retention).mean()
        smooth=smooth_penalty(model.sequence(x,mask),mask,model.config['masked'])
        loss=loss+.2*js+.3*aug_b+.01*smooth
        logs.update(js=float(js.detach()),aug_binary=float(aug_b.detach()),hidden_smooth=float(smooth.detach()),epsilon_mean=float(eps.mean().detach()))
    logs['total']=float(loss.detach())
    return loss,logs
