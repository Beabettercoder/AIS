"""Finite forward/backward & graph reference-gating smoke checks, train samples only."""
import sys
from pathlib import Path
import numpy as np
import torch

HEADING = Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030')
sys.path.insert(0,str(HEADING))
import train_heading as base
from relation8 import model_for

torch.set_num_threads(2)
base.VARIANT='aligned_embed'
tr=base.read('train');mu,sd=base.fit(tr)
for name in ['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph']:
    model=model_for(name);model.train()
    x,m,t=base.prep(tr,'base','F1',mu,sd,torch.device('cpu'))
    out=model(x[:4],m[:4],t[:4])['logits']
    loss=torch.nn.functional.binary_cross_entropy_with_logits(out,torch.tensor(tr['y'][:4],dtype=torch.float32))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(name,'finite-loss-and-gradient',flush=True)

model=model_for('graph').eval()
x,m,t=base.prep(tr,'base','C0',mu,sd,torch.device('cpu'));x=x[:4].clone();m=m[:4];t=t[:4]
assert x.shape[-1]==35 and model.target_projection.in_features==8 and model.reference_projection.in_features==27
with torch.no_grad():
    original=model(x,m,t)['logits']
    x[...,8:]=torch.randn_like(x[...,8:]);x[...,22]=0
    perturbed=model(x,m,t)['logits']
assert torch.equal(original,perturbed)
print('graph: 8 target channels, 27 reference channels, no-support exact fallback',flush=True)
