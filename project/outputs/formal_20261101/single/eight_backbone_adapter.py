"""Provisional 8-backbone single-vessel stream for a conditional follow-up.

Adaptation of the frozen four-backbone pilot, not a historical Step-35 model.
No training or experimental selection is performed in this module.
"""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028')
sys.path.insert(0, str(ROOT))
from models import make_model, _prepare, _masked_mean, cde_rk4_step

class Model8(nn.Module):
    """Same late field adapter and heads as pilot for 4 existing architectures.

    For GraphTemporal the backbone consumes seven self features and five zero
    contextual slots; all external field enters only the late adapter. This is
    a target-only graph adaptation, not an active graph of multiple vessels.
    """
    def __init__(self, name):
        super().__init__()
        self.name = name.lower().replace('-', '').replace('_', '')
        self.graph = self.name in ('graphtemporal', 'graph')
        self.backbone = make_model(name, 12 if self.graph else 7)
        self.adapter = nn.Sequential(nn.Linear(4,64), nn.GELU(), nn.Linear(64,64))
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.typehead = nn.Linear(64,5)
        self.ctrl = nn.Linear(4,2)

    def encode(self,x,mask,times):
        x,mask,times,lengths = _prepare(x,mask,times,7,20)
        b = self.backbone
        if self.name == 'tcn':
            h = b.input(x).masked_fill(~mask.unsqueeze(-1),0)
            for block in b.blocks: h=block(h,mask)
            return _masked_mean(h,mask),h
        if self.name == 'gru':
            packed = nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
            po, final = b.gru(packed)
            seq,_ = nn.utils.rnn.pad_packed_sequence(po,batch_first=True,total_length=20)
            return final[-1],seq
        if self.name == 'transformer':
            seq=b.encoder(b.input(x)+b.position,src_key_padding_mask=~mask)
            return _masked_mean(seq,mask),seq
        if self.name == 'itransformer':
            tokens=b.temporal_projection(x.transpose(1,2))+b.mask_projection(mask.to(x.dtype)).unsqueeze(1)
            tokens=b.encoder(tokens)
            return b.readout(tokens.flatten(1)), torch.einsum('btd,bdh->bth',x,tokens)/7.
        if self.name == 'patchtst':
            batch,_,dim=x.shape
            xp=F.pad(x.transpose(1,2),(0,b.padded_len-b.max_len))
            m=F.pad(mask,(0,b.padded_len-b.max_len)).unfold(-1,b.patch_len,b.stride)
            patches=xp.unfold(-1,b.patch_len,b.stride).reshape(batch*dim,b.patch_count,b.patch_len)
            patch_mask=m.any(-1)[:,None].expand(-1,dim,-1).reshape(batch*dim,b.patch_count)
            support=m[:,None].expand(-1,dim,-1,-1).reshape_as(patches).to(x.dtype)
            tokens=b.patch_projection(patches)+b.mask_projection(support)+b.position
            tokens=b.encoder(tokens,src_key_padding_mask=~patch_mask)
            embedding=b.readout(_masked_mean(tokens,patch_mask).reshape(batch,dim*b.hidden))
            # Time-indexed proxy from overlapping channel-averaged patch tokens.
            p=tokens.reshape(batch,dim,b.patch_count,b.hidden).mean(1)
            accum=torch.zeros(batch,b.padded_len,b.hidden,device=x.device,dtype=x.dtype)
            denom=torch.zeros(batch,b.padded_len,1,device=x.device,dtype=x.dtype)
            for k in range(b.patch_count):
                s=k*b.stride
                active=m[:,k,:,None].to(x.dtype)
                accum[:,s:s+b.patch_len] += p[:,k,None,:]*active
                denom[:,s:s+b.patch_len] += active
            seq=(accum/denom.clamp_min(1))[:,:20].masked_fill(~mask.unsqueeze(-1),0)
            return embedding,seq
        if self.name == 'mamba':
            seq=b.input(x).masked_fill(~mask.unsqueeze(-1),0)
            for block in b.blocks: seq=block(seq,mask)
            seq=b.norm(seq).masked_fill(~mask.unsqueeze(-1),0)
            return _masked_mean(seq,mask),seq
        if self.name == 'neuralcde':
            elapsed=(times-times[:,:1])/b.time_unit_seconds
            control=torch.cat((elapsed.unsqueeze(-1),x),-1)
            z=b.initial(control[:,0]); states=[z]
            for j in range(1,x.shape[1]):
                inc=(control[:,j]-control[:,j-1]).masked_fill(~mask[:,j,None],0)
                step=inc/b.rk4_substeps
                for _ in range(b.rk4_substeps): z=cde_rk4_step(z,step,b.vector_field)
                states.append(z)
            return z,torch.stack(states,1).masked_fill(~mask.unsqueeze(-1),0)
        if self.graph:
            # Same GraphTemporal.encode with its reference node inactive.
            full=torch.cat((x,x.new_zeros(x.shape[0],x.shape[1],5)),-1)
            active=(full[...,-1]>0)&mask
            target=b.target_projection(full[...,:b.target_dim])
            ref=b.reference_projection(full[...,b.target_dim:]).masked_fill(~active.unsqueeze(-1),0)
            nodes=torch.stack((target,ref),2)
            for layer in b.graph_layers: nodes=layer(nodes,active)
            seq=nodes[:,:,0].masked_fill(~mask.unsqueeze(-1),0)
            for layer in b.temporal_layers: seq=layer(seq,mask)
            return _masked_mean(seq,mask),seq
        raise ValueError(self.name)

    def forward(self,x,mask,times,field,rho,use_field):
        emb,seq=self.encode(x,mask,times)
        if use_field:
            v=mask.to(x.dtype)
            ctx=(self.adapter(field)*(rho*v).unsqueeze(-1)).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
            emb=emb+self.alpha*ctx
        return self.backbone.head(emb).squeeze(-1),self.typehead(emb),emb,seq

