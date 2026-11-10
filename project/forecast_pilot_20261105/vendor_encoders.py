"""Four retained encoder bodies from framework_validation_v1_20261028/models.py.

Only torch is required. These are auditable small classification adaptations, not
claims to reproduce the original forecasting papers' complete training recipes.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ("TCN", "GRU", "Transformer", "iTransformer")
IMPLEMENTATION_VERSION = "framework_encoders_v1"


def _masked_mean(x, mask):
    return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def _prepare(x, mask, times, input_dim, max_len):
    """Compact valid observations in their original order, then canonical pad.

    Invalid input values (including NaNs) never reach an encoder. Interior gaps
    are omitted; supplied physical timestamps retain their actual spacing.
    """
    if x.ndim != 3 or x.shape[-1] != input_dim:
        raise ValueError(f"Expected [B,T,{input_dim}], got {tuple(x.shape)}")
    if x.shape[1] == 0 or tuple(mask.shape) != tuple(x.shape[:2]):
        raise ValueError("mask must be [B,T] and time axis must be nonempty")
    mask = mask.to(device=x.device, dtype=torch.bool)
    lengths = mask.sum(1)
    if bool((lengths == 0).any()) or bool((lengths > max_len).any()):
        raise ValueError(f"Each example needs 1..{max_len} valid observations")
    if times is None:
        times = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[None].expand(x.shape[0], -1) * 60.0
    if tuple(times.shape) != tuple(mask.shape):
        raise ValueError("times must be [B,T] in seconds")
    times = times.to(device=x.device, dtype=x.dtype)
    indices = torch.arange(x.shape[1], device=x.device)[None].expand_as(mask)
    order = (indices + (~mask).long() * x.shape[1]).argsort(dim=1)
    take = min(x.shape[1], max_len)
    order = order[:, :take]
    safe_x = x.masked_fill(~mask.unsqueeze(-1), 0)
    safe_t = times.masked_fill(~mask, 0)
    x = safe_x.gather(1, order.unsqueeze(-1).expand(-1, -1, input_dim))
    times = safe_t.gather(1, order)
    x = F.pad(x, (0, 0, 0, max_len - take))
    times = F.pad(times, (0, max_len - take))
    mask = torch.arange(max_len, device=x.device)[None] < lengths[:, None]
    x = x.masked_fill(~mask.unsqueeze(-1), 0)
    times = times.masked_fill(~mask, 0)
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(times).all()):
        raise ValueError("Valid observations and timestamps must be finite")
    if bool(((times[:, 1:] < times[:, :-1]) & mask[:, 1:]).any()):
        raise ValueError("Valid timestamps must be nondecreasing")
    return x, mask, times, lengths


class TrajectoryClassifier(nn.Module):
    def __init__(self, input_dim, max_len, hidden):
        super().__init__()
        self.input_dim, self.max_len, self.hidden = input_dim, max_len, hidden
        # The exact same classification head is used by every encoder.
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x, mask, times=None):
        x, mask, times, lengths = _prepare(x, mask, times, self.input_dim, self.max_len)
        embedding = self.encode(x, mask, times, lengths)
        return {"logits": self.head(embedding).squeeze(-1), "embedding": embedding}


class CausalResidualBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout=0.1):
        super().__init__()
        self.padding = 2 * dilation
        self.conv1 = nn.Conv1d(hidden, hidden, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, dilation=dilation)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.conv1(F.pad(x.transpose(1, 2), (self.padding, 0))).transpose(1, 2)
        x = self.dropout(F.gelu(self.norm1(x))).masked_fill(~mask.unsqueeze(-1), 0)
        x = self.conv2(F.pad(x.transpose(1, 2), (self.padding, 0))).transpose(1, 2)
        x = self.dropout(F.gelu(self.norm2(x)))
        return (x + residual).masked_fill(~mask.unsqueeze(-1), 0)


class TCNClassifier(TrajectoryClassifier):
    def __init__(self, input_dim, max_len, hidden):
        super().__init__(input_dim, max_len, hidden)
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([CausalResidualBlock(hidden, d) for d in (1, 2, 4)])

    def encode(self, x, mask, times, lengths):
        x = self.input(x).masked_fill(~mask.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, mask)
        return _masked_mean(x, mask)


class GRUClassifier(TrajectoryClassifier):
    def __init__(self, input_dim, max_len, hidden):
        super().__init__(input_dim, max_len, hidden)
        self.gru = nn.GRU(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)

    def encode(self, x, mask, times, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, final = self.gru(packed)
        return final[-1]


def _transformer(hidden):
    layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden, dropout=0.1,
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)


class TransformerClassifier(TrajectoryClassifier):
    def __init__(self, input_dim, max_len, hidden):
        super().__init__(input_dim, max_len, hidden)
        self.input = nn.Linear(input_dim, hidden)
        self.position = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.normal_(self.position, std=0.02)
        self.encoder = _transformer(hidden)

    def encode(self, x, mask, times, lengths):
        x = self.encoder(self.input(x) + self.position, src_key_padding_mask=~mask)
        return _masked_mean(x, mask)


class ITransformerClassifier(TrajectoryClassifier):
    """Inverted attention over variables; temporal projection precedes attention."""
    def __init__(self, input_dim, max_len, hidden):
        super().__init__(input_dim, max_len, hidden)
        self.temporal_projection = nn.Linear(max_len, hidden)
        self.mask_projection = nn.Linear(max_len, hidden, bias=False)
        self.encoder = _transformer(hidden)
        self.readout = nn.Linear(input_dim * hidden, hidden)

    def encode(self, x, mask, times, lengths):
        tokens = self.temporal_projection(x.transpose(1, 2))
        tokens = tokens + self.mask_projection(mask.to(x.dtype)).unsqueeze(1)
        tokens = self.encoder(tokens)
        # Preserve variable identity; symmetric mean pooling would lose it.
        return self.readout(tokens.flatten(1))




def make_encoder(name, input_dim=9, max_len=12, hidden=64):
    key = name.lower().replace('-', '').replace('_', '')
    registry = {'tcn': TCNClassifier, 'gru': GRUClassifier,
                'transformer': TransformerClassifier, 'itransformer': ITransformerClassifier}
    if key not in registry:
        raise ValueError(f'Unknown encoder: {name}')
    model = registry[key](input_dim, max_len, hidden)
    # Encoder architecture is reused; classification head is not part of forecasting.
    model.head = nn.Identity()
    return model
