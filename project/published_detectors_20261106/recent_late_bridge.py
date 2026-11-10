"""Recent published models with historical single-vessel 4-D field side branch.

This is an architecture adapter, not reproduction of historical trained weights.
Original model-native auxiliary losses stay in ``adapter.loss`` and are NOT
hidden inside the historical bundle. The caller must add the intended native
loss explicitly and, for CARLA, preserve the pretext/clustering protocol.

These upstream models do not support variable-length attention/convolution
masks.  For native unresampled samples, masked trailing slots are zero padded;
the published backbone can still see the padding.  Masking the output and
historical smoothness term is not equivalent to a mask-aware backbone.
"""
import importlib
import sys
import types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ORIGINAL=Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')
if str(ORIGINAL) not in sys.path: sys.path.insert(0,str(ORIGINAL))
from pilot_historical_loss import historical_bundle

class RecentLateBridge(nn.Module):
    """7D target stream -> 64D head; 4D LOO side stream, zero-init scalar.

    The field and rho are separate inputs so the exact same target backbone,
    target channel count and parameters can be used for raw/gate/LOO arms.
    The caller controls source sequence treatment, target gate and F0/F1.
    """
    def __init__(self, method, seq_len=20, gpu=0):
        super().__init__()
        if method not in ('tft_gcn','simad','dmrad','carla'): raise ValueError(method)
        self.method=method;self.seq_len=seq_len
        if method=='tft_gcn':
            # The historical AIS sources expose a top-level models.py; the TFT
            # vendor's models directory is a namespace package. Isolate that
            # name for construction in this one-method-per-process workflow.
            prior=sys.modules.get('models'); prior_sub=sys.modules.get('models.model')
            pkg=types.ModuleType('models');pkg.__path__=[str(Path(__file__).resolve().parent/'vendor'/'TFT-GCN'/'models')]
            sys.modules['models']=pkg;sys.modules.pop('models.model',None)
            try:self.adapter=importlib.import_module('adapters.'+method).Adapter(seq_len=seq_len,channels=7,gpu=gpu)
            finally:
                if prior is None:sys.modules.pop('models',None)
                else:sys.modules['models']=prior
                if prior_sub is None:sys.modules.pop('models.model',None)
                else:sys.modules['models.model']=prior_sub
        else:self.adapter=importlib.import_module('adapters.'+method).Adapter(seq_len=seq_len,channels=7,gpu=gpu)
        target_dim=128 if method=='carla' else seq_len*7
        self.target_encoder=nn.Sequential(nn.Linear(target_dim,64),nn.GELU())
        # Real time branch hidden states for TFT; CARLA convolutional states;
        # SimAD patch states repeated across their 4 constituent steps; DMRAD
        # learned decoder trend+residual components at original 20 steps.
        # Do not add a trainable projection solely for smoothness: such a layer
        # could collapse to zero and trivially bypass the historical penalty.
        self.sequence_proxy=nn.Identity()
        self.sequence_provenance={'tft_gcn':'native temporal hidden states',
            'simad':'native patch hidden states repeated to time steps (patch-resolution proxy)',
            'dmrad':'native trend/residual decoder components at each output step',
            'carla':'native convolutional hidden states'}[method]
        self.field_adapter=nn.Sequential(nn.Linear(4,64),nn.GELU(),nn.Linear(64,64))
        self.alpha=nn.Parameter(torch.tensor(0.))
        self.bin_head=nn.Linear(64,1);self.type_head=nn.Linear(64,5);self.controller=nn.Linear(4,2)

    def encode(self,x,mask):
        if x.ndim!=3 or x.shape[1:]!=(self.seq_len,7) or mask.shape!=x.shape[:2]:
            raise ValueError('Expected x [B,20,7] and corresponding mask')
        if not mask.any(1).all(): raise ValueError('Empty trajectory window')
        x=x.masked_fill(~mask[...,None],0.)
        if self.method=='carla':
            # Equivalent to ContrastiveModel.forward but retain conv states.
            states=self.adapter.pretext.backbone.layers(self.adapter._reshape(x))
            pooled=states.mean(-1)
            representation=F.normalize(self.adapter.pretext.contrastive_head(pooled),dim=1)
            seq=self.sequence_proxy(states.transpose(1,2))
        else:
            model=self.adapter.model
            if self.method=='tft_gcn':
                captured=[]
                handle=model.t_model.layer_norm.register_forward_hook(lambda module,inputs,output:captured.append(output))
                try: out=model(x)
                finally:handle.remove()
                if not captured:raise RuntimeError('TFT temporal hidden states not captured')
                states=captured[-1]
            elif self.method=='simad':
                z=model.embedding(x);z=model.to_patch_embedding(z);states=model.encoder(z)
                out=model.projection(states).reshape(x.shape[0],self.seq_len,7)
                # Each encoder token describes a patch, not an individual point.
                states=states.repeat_interleave(model.patch_size,dim=1)
            elif self.method=='dmrad':
                normalized=model.revin_layer(x);res,main=model.Decompose(normalized)
                main=model.Linear_main(main.permute(0,2,1)).permute(0,2,1)
                res=model.Linear_res(res.permute(0,2,1)).permute(0,2,1)
                states=torch.cat((main,res),dim=-1)
                out=model.revin_layer.inverse_normalize(main+res)
            if out.shape!=x.shape: raise ValueError(f'unexpected reconstructed shape {out.shape}')
            out=out.masked_fill(~mask[...,None],0.)
            representation=out.reshape(x.shape[0],-1)
            seq=self.sequence_proxy(states)
        return self.target_encoder(representation),seq.masked_fill(~mask[...,None],0.)

    def forward(self,x,mask,field,rho,use_field):
        emb,seq=self.encode(x,mask)
        if use_field:
            if field.shape!=(*x.shape[:2],4) or rho.shape!=x.shape[:2]:raise ValueError('Expected field [B,T,4], rho [B,T]')
            valid=mask.to(x.dtype)
            ctx=(self.field_adapter(field)*(rho*valid).unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True).clamp_min(1.)
            emb=emb+self.alpha*ctx
        return self.bin_head(emb).squeeze(-1),self.type_head(emb),emb,seq

    def bundle_loss(self,x,mask,field,rho,use_field,physical_state,labels,label_class,class_weights):
        """Invoke the unchanged step-35 historical loss and exact augmentation.

        physical_state must be [B,T,4] in physical units and historical order:
        background speed, speed-change rate, divergence and curl. No target label
        is used to fit field; caller supplies original frozen features and rho.
        """
        if physical_state.shape!=(*x.shape[:2],4):raise ValueError('Expected physical state [B,T,4]')
        # The original augmentation indexes [0,length), so require prefix masks.
        lengths=mask.sum(1)
        expected=torch.arange(mask.shape[1],device=mask.device)[None,:] < lengths[:,None]
        if not torch.equal(mask,expected):raise ValueError('historical augmentation requires prefix mask')
        binary_logits,type_logits,_,seq=self(x,mask,field,rho,use_field)
        keep=(torch.rand(x.shape[:2],device=x.device)>.15).to(x.dtype)
        x2=x*keep.unsqueeze(-1);jj=torch.floor(torch.rand(len(x),device=x.device)*lengths).long()
        x2=x2*mask.unsqueeze(-1);x2[torch.arange(len(x),device=x.device),jj,0]+=2.
        retention=(keep*mask).sum(1)/lengths.clamp_min(1)
        _,_,h2,_=self(x2,mask,field,rho,use_field)
        return historical_bundle(self.bin_head,self.type_head,self.controller,binary_logits,type_logits,
                                 h2,seq,physical_state,mask,labels,label_class,class_weights,retention)

    def native_loss(self,x,mask,indices=None,epoch=0,**kwargs):
        """Keep upstream objective separate, with caller-specified weight.

        DMRAD requires stable indices and begin_epoch. CARLA phase-specific
        positive/near/far kwargs are mandatory: no fabricated replacements.
        """
        return self.adapter.loss(x.masked_fill(~mask[...,None],0.),indices=indices,epoch=epoch,**kwargs)
