"""AIS window interface for the pinned upstream SimAD implementation.

The architecture and timestamp scoring come from vendor/SimAD. Its training
objective follows trainer_v2/trainer2.py, including the negative dissimilarity
term and stop-gradient. Window score is the mean of the upstream point scores;
that aggregation is an explicit adaptation to AIS window-level labels.
"""
from pathlib import Path
import importlib.util
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

_path = Path(__file__).resolve().parents[1] / 'vendor/SimAD/models_v2/model2_lazy.py'
_spec = importlib.util.spec_from_file_location('_pinned_simad_model', _path)
_upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_upstream)


class Adapter(nn.Module):
    def __init__(self, seq_len, channels, **kwargs):
        super().__init__()
        self.seq_len, self.channels = int(seq_len), int(channels)
        requested_patch = int(kwargs.get('patch_size', 4))
        if self.seq_len % requested_patch:
            raise ValueError(f'SimAD seq_len={seq_len} must be divisible by patch_size={requested_patch}')
        self.patch_size = requested_patch
        self.noise_level = float(kwargs.get('noise_level', 0.1))
        self.warmup_steps = int(kwargs.get('warmup_steps', 20000))
        self.warmup_max_ratio = float(kwargs.get('warmup_max_ratio', 0.2))
        self.register_buffer('_steps', torch.zeros((), dtype=torch.long))
        self.model = _upstream.ContAD_wo_ci(
            c_dim=self.channels, seq_len=self.seq_len, patch_size=self.patch_size,
            d_model=int(kwargs.get('d_model', 128)), n_layers=int(kwargs.get('n_layers', 3)),
            d_feed_foward_scale=int(kwargs.get('ffn_scale', 4)),
            n_head=int(kwargs.get('n_head', 4)), dropout=float(kwargs.get('dropout', 0.1)),
            query_len=int(kwargs.get('query_len', 8)))

    def forward(self, x):
        return self.model(x)

    def loss(self, x, indices=None, epoch=0):
        patch = rearrange(x, 'b (l p) c -> b l (p c)', p=self.patch_size)
        out, sim = self.model(x)
        out2, sim2 = self.model(x + torch.randn_like(x) * self.noise_level)
        rec = F.mse_loss(out, patch) + F.mse_loss(out2, patch)
        sim, sim2 = F.normalize(sim, dim=-1), F.normalize(sim2, dim=-1)
        discrepancy = (F.mse_loss(sim, sim2.detach()) +
                       (1-F.cosine_similarity(sim, sim2.detach(), dim=-1)).mean() +
                       F.mse_loss(sim2, sim.detach()) +
                       (1-F.cosine_similarity(sim2, sim.detach(), dim=-1)).mean())
        warmup = min((int(self._steps.item()) + 1) / (self.warmup_steps + 1e-8),
                     self.warmup_max_ratio)
        if self.training:
            self._steps.add_(1)
        return rec - discrepancy * warmup

    def point_score(self, x):
        out, _ = self.model(x)
        patch = rearrange(x, 'b (l p) c -> b l (p c)', p=self.patch_size)
        return _upstream.rec_score_func2(out, patch, self.channels)

    def score(self, x):
        return self.point_score(x).mean(dim=-1)
