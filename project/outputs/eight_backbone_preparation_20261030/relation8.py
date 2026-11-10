"""Reference-node adapter for 35D trajectory context classification.

First eight channels belong to the target; channel 22 is reference support.
This custom two-node graph is an adaptation, not the original full-vessel GAT.
"""
import sys
from pathlib import Path
import torch
from torch import nn

PRE = Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028')
HEADING = Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030')
sys.path.insert(0, str(PRE))
sys.path.insert(0, str(HEADING))
from models import GraphTemporalClassifier, make_model, _masked_mean
from train_heading import EmbeddedITransformer

class RelationGraph35(GraphTemporalClassifier):
    def __init__(self, hidden=64):
        super().__init__(12, 20, hidden)
        self.input_dim = 35
        self.target_dim = 8
        self.target_projection = nn.Linear(8, hidden)
        self.reference_projection = nn.Linear(27, hidden)

    def encode(self, x, mask, times, lengths):
        active = (x[...,22] > 0) & mask
        target = self.target_projection(x[...,:8])
        ref = self.reference_projection(x[...,8:]).masked_fill(~active.unsqueeze(-1), 0)
        nodes = torch.stack((target,ref),2)
        for layer in self.graph_layers:
            nodes = layer(nodes,active)
        seq = nodes[:,:,0].masked_fill(~mask.unsqueeze(-1),0)
        for layer in self.temporal_layers:
            seq = layer(seq,mask)
        return _masked_mean(seq,mask)

def model_for(name):
    if name == 'itransformer': return EmbeddedITransformer(35,20,64)
    if name == 'graph': return RelationGraph35()
    return make_model(name,35,max_len=20,hidden=64)
