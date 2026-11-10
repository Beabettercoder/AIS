"""Eight explicit trajectory classification adaptations; see MODEL_IMPLEMENTATIONS.md.

Only torch is required. These are auditable small classification adaptations, not
claims to reproduce the original forecasting papers' complete training recipes.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ("TCN", "GRU", "Transformer", "iTransformer", "PatchTST", "Mamba", "Neural-CDE", "GraphTemporal")
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


class PatchTSTClassifier(TrajectoryClassifier):
    """Shared channel-independent temporal patch encoder and classification readout."""
    def __init__(self, input_dim, max_len, hidden):
        super().__init__(input_dim, max_len, hidden)
        self.patch_len, self.stride = 4, 2
        if max_len < self.patch_len:
            raise ValueError("PatchTST requires max_len >= 4")
        self.patch_count = math.ceil((max_len - self.patch_len) / self.stride) + 1
        self.padded_len = (self.patch_count - 1) * self.stride + self.patch_len
        self.patch_projection = nn.Linear(self.patch_len, hidden)
        self.mask_projection = nn.Linear(self.patch_len, hidden, bias=False)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, hidden))
        nn.init.normal_(self.position, std=0.02)
        self.encoder = _transformer(hidden)
        self.readout = nn.Linear(input_dim * hidden, hidden)

    def encode(self, x, mask, times, lengths):
        b, _, d = x.shape
        x = F.pad(x.transpose(1, 2), (0, self.padded_len - self.max_len))
        m = F.pad(mask, (0, self.padded_len - self.max_len)).unfold(-1, self.patch_len, self.stride)
        patches = x.unfold(-1, self.patch_len, self.stride).reshape(b * d, self.patch_count, self.patch_len)
        patch_mask = m.any(-1)[:, None].expand(-1, d, -1).reshape(b * d, self.patch_count)
        support = m[:, None].expand(-1, d, -1, -1).reshape_as(patches).to(x.dtype)
        tokens = self.patch_projection(patches) + self.mask_projection(support) + self.position
        tokens = self.encoder(tokens, src_key_padding_mask=~patch_mask)
        channel_embedding = _masked_mean(tokens, patch_mask).reshape(b, d * self.hidden)
        return self.readout(channel_embedding)


def selective_scan_reference(u, delta, A, B, C, D, mask):
    """Real diagonal input-selective SSM, matching Mamba's reference recurrence.

    u, delta [batch,time,inner]; A [inner,state]; B,C [batch,time,state].
    Mamba uses exp(delta*A) for the state transition and delta*B*u as
    the input term. Invalid observations leave the hidden state unchanged.
    """
    state = u.new_zeros(u.shape[0], u.shape[-1], A.shape[-1])
    outputs = []
    for j in range(u.shape[1]):
        dt = delta[:, j, :, None]
        candidate = torch.exp(dt * A[None]) * state + dt * B[:, j, None, :] * u[:, j, :, None]
        state = torch.where(mask[:, j, None, None], candidate, state)
        y = (state * C[:, j, None, :]).sum(-1) + D * u[:, j]
        outputs.append(y * mask[:, j, None])
    return torch.stack(outputs, 1)


class SelectiveMambaBlock(nn.Module):
    """Mamba-1 mixer math with a transparent scan instead of fused CUDA kernels."""
    def __init__(self, hidden, state_dim=16, expansion=2):
        super().__init__()
        self.inner, self.state_dim, self.dt_rank = expansion * hidden, state_dim, math.ceil(hidden / 16)
        self.norm = nn.LayerNorm(hidden)
        self.in_projection = nn.Linear(hidden, 2 * self.inner, bias=False)
        self.conv = nn.Conv1d(self.inner, self.inner, 4, groups=self.inner, padding=3)
        self.selectors = nn.Linear(self.inner, self.dt_rank + 2 * state_dim, bias=False)
        self.delta_projection = nn.Linear(self.dt_rank, self.inner)
        nn.init.uniform_(self.delta_projection.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(self.inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.delta_projection.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, state_dim + 1).float().log().repeat(self.inner, 1))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.out_projection = nn.Linear(self.inner, hidden, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        u, gate = self.in_projection(self.norm(x)).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2))
        dt, b, c = torch.split(self.selectors(u), [self.dt_rank, self.state_dim, self.state_dim], -1)
        delta = F.softplus(self.delta_projection(dt))
        y = selective_scan_reference(u, delta, -self.A_log.exp(), b, c, self.D, mask)
        y = self.out_projection(y * F.silu(gate))
        return (x + self.dropout(y)).masked_fill(~mask.unsqueeze(-1), 0)


class MambaClassifier(TrajectoryClassifier):
    def __init__(self, input_dim, max_len, hidden):
        super().__init__(input_dim, max_len, hidden)
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([SelectiveMambaBlock(hidden) for _ in range(2)])
        self.norm = nn.LayerNorm(hidden)

    def encode(self, x, mask, times, lengths):
        x = self.input(x).masked_fill(~mask.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, mask)
        return _masked_mean(self.norm(x), mask)


def cde_rk4_step(z, control_increment, vector_field):
    """One RK4 step of dz/du=f(z) dX/du on a straight control segment."""
    def flow(state):
        return torch.einsum("bhc,bc->bh", vector_field(state), control_increment)
    k1 = flow(z)
    k2 = flow(z + k1 / 2)
    k3 = flow(z + k2 / 2)
    k4 = flow(z + k3)
    return z + (k1 + 2 * k2 + 2 * k3 + k4) / 6


class CDEVectorField(nn.Module):
    def __init__(self, hidden, control_dim):
        super().__init__()
        self.hidden, self.control_dim = hidden, control_dim
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden * control_dim), nn.Tanh())

    def forward(self, z):
        return self.net(z).reshape(-1, self.hidden, self.control_dim)


class NeuralCDEClassifier(TrajectoryClassifier):
    """Neural CDE with linear control interpolation and segment-aligned RK4."""
    def __init__(self, input_dim, max_len, hidden):
        super().__init__(input_dim, max_len, hidden)
        self.control_dim = input_dim + 1
        self.time_unit_seconds = 60.0
        self.rk4_substeps = 4
        self.initial = nn.Linear(self.control_dim, hidden)
        self.vector_field = CDEVectorField(hidden, self.control_dim)

    def encode(self, x, mask, times, lengths):
        elapsed = (times - times[:, :1]) / self.time_unit_seconds
        control = torch.cat((elapsed.unsqueeze(-1), x), -1)
        z = self.initial(control[:, 0])
        for j in range(1, x.shape[1]):
            increment = (control[:, j] - control[:, j - 1]).masked_fill(~mask[:, j, None], 0)
            step = increment / self.rk4_substeps
            for _ in range(self.rk4_substeps):
                z = cde_rk4_step(z, step, self.vector_field)
        return z


class TwoNodeGraphAttention(nn.Module):
    """Additive multihead GAT with explicit two-node adjacency masking."""
    def __init__(self, hidden, heads=4):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.projection = nn.Linear(hidden, hidden, bias=False)
        self.source_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.destination_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.destination_attention)
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.1)

    def forward(self, nodes, reference_active):
        # nodes [B,T,2,H]; valid target and optional aggregate-reference node.
        b, t, _, h = nodes.shape
        transformed = self.projection(nodes).reshape(b, t, 2, self.heads, self.head_dim)
        source = (transformed * self.source_attention).sum(-1)
        dest = (transformed * self.destination_attention).sum(-1)
        score = F.leaky_relu(source.unsqueeze(3) + dest.unsqueeze(2), negative_slope=0.2)
        node_valid = torch.stack((torch.ones_like(reference_active), reference_active), -1)
        adjacency = node_valid.unsqueeze(-1) & node_valid.unsqueeze(-2)
        # Inactive reference rows get a harmless self loop to avoid all -inf.
        adjacency[..., 1, 1] = True
        attention = score.masked_fill(~adjacency.unsqueeze(-1), -torch.inf).softmax(dim=3)
        messages = torch.einsum("btijh,btjhd->btihd", attention, transformed).reshape(b, t, 2, h)
        result = self.norm(nodes + self.dropout(F.elu(self.out(messages))))
        return result.masked_fill(~node_valid.unsqueeze(-1), 0)


class GraphTemporalClassifier(TrajectoryClassifier):
    """Custom target/reference GAT + TCN, not a native full-vessel graph model."""
    def __init__(self, input_dim, max_len, hidden):
        super().__init__(input_dim, max_len, hidden)
        schema = {12: 7, 7: 4}
        if input_dim not in schema:
            raise ValueError("GraphTemporal requires registered 12D individual or 7D relation schema")
        self.target_dim = schema[input_dim]
        self.target_projection = nn.Linear(self.target_dim, hidden)
        self.reference_projection = nn.Linear(input_dim - self.target_dim, hidden)
        self.graph_layers = nn.ModuleList([TwoNodeGraphAttention(hidden) for _ in range(2)])
        self.temporal_layers = nn.ModuleList([CausalResidualBlock(hidden, d) for d in (1, 2, 4)])

    def encode(self, x, mask, times, lengths):
        reference_active = (x[..., -1] > 0) & mask
        target = self.target_projection(x[..., :self.target_dim])
        reference = self.reference_projection(x[..., self.target_dim:]).masked_fill(~reference_active.unsqueeze(-1), 0)
        nodes = torch.stack((target, reference), 2)
        for layer in self.graph_layers:
            nodes = layer(nodes, reference_active)
        temporal = nodes[:, :, 0].masked_fill(~mask.unsqueeze(-1), 0)
        for layer in self.temporal_layers:
            temporal = layer(temporal, mask)
        return _masked_mean(temporal, mask)


def make_model(name: str, input_dim: int, max_len: int = 20, hidden: int = 64):
    if hidden < 4 or hidden % 4:
        raise ValueError("hidden must be positive and divisible by four")
    key = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    registry = {
        "tcn": TCNClassifier, "gru": GRUClassifier, "transformer": TransformerClassifier,
        "itransformer": ITransformerClassifier, "patchtst": PatchTSTClassifier,
        "mamba": MambaClassifier, "neuralcde": NeuralCDEClassifier,
        "graphtemporal": GraphTemporalClassifier, "graph": GraphTemporalClassifier,
    }
    if key not in registry:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    model = registry[key](input_dim, max_len, hidden)
    model.model_name = name
    model.implementation_version = IMPLEMENTATION_VERSION
    return model
