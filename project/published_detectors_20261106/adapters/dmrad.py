"""AIS bridge for the public DMRAD repository, commit 060495e.

Run each vendor in its own Python process: their top-level `model` and `utils`
packages are not mutually compatible. The upstream memory uses loader position;
this bridge keys it by stable training-window index, so shuffling is safe.
"""
from pathlib import Path
import sys
import math
import numpy as np
import torch
from torch import nn

_vendor = Path(__file__).resolve().parents[1] / "vendor" / "DMRAD"
sys.path.insert(0, str(_vendor))
from model.MYdetector import MYdetector


class Adapter(nn.Module):
    def __init__(self, seq_len, channels, *, d_model=128, e_layers=3,
                 kernel_size=25, warm_epochs=1, fp=2.0, **kwargs):
        super().__init__()
        self.model = MYdetector(win_size=seq_len, enc_in=channels, c_out=channels,
                                channel=channels, d_model=d_model, e_layers=e_layers,
                                kernel_size=kernel_size)
        t = torch.arange(seq_len, dtype=torch.float32)
        weights = torch.exp(-0.5 * ((t - (seq_len - 1) / 2) / (seq_len / 6)) ** 2)
        self.register_buffer('time_weights', weights / weights.sum())
        self.warm_epochs = warm_epochs
        self.fp = fp
        self.history = {}
        self.flagged = set()
        self.mean = 0.0
        self.std = 0.0

    def begin_epoch(self, epoch, total_epochs):
        self.flagged = set()
        if epoch < self.warm_epochs or not self.history:
            return
        ew = np.arange(1, epoch + 1, dtype=float) / total_epochs
        if any(len(v) != epoch for v in self.history.values()):
            raise ValueError('DMRAD requires each stable training sample exactly once per epoch')
        means = {k: float(np.mean(np.asarray(v) * ew)) for k, v in self.history.items()}
        self.mean, self.std = float(np.mean(list(means.values()))), float(np.std(list(means.values())))
        self.flagged = {k for k, v in means.items() if v >= self.mean + self.fp * self.std}

    def loss(self, x, indices=None, epoch=0):
        if indices is None:
            raise ValueError('DMRAD needs stable training sample indices')
        indices = [int(i) for i in torch.as_tensor(indices).detach().cpu().tolist()]
        if len(set(indices)) != len(indices):
            raise ValueError('duplicate indices within DMRAD training batch')
        pred = self.model(x)
        per_window = (((pred - x) ** 2).mean(dim=-1) * self.time_weights).sum(dim=-1)
        vals = per_window.detach().cpu().numpy()
        weights = np.ones(len(indices), dtype=np.float32)
        for j, (i, value) in enumerate(zip(indices, vals)):
            self.history.setdefault(i, []).append(float(value))
            if epoch >= self.warm_epochs and i in self.flagged and self.std > 1e-12:
                z = (float(value) - self.mean) / self.std
                weights[j] = math.exp(-0.5 * z * z) / (self.std * math.sqrt(2 * math.pi))
        return (per_window * torch.as_tensor(weights, device=x.device)).mean()

    def score(self, x):
        # Upstream per-timestamp score = channel-mean reconstruction MSE;
        # mean over timestamps is the declared window-level AIS aggregation.
        return ((self.model(x) - x) ** 2).mean(dim=(-1, -2))

    def get_extra_state(self):
        return {'history': self.history, 'flagged': list(self.flagged),
                'mean': self.mean, 'std': self.std}

    def set_extra_state(self, state):
        self.history = {int(k): v for k, v in state['history'].items()}
        self.flagged = set(state['flagged'])
        self.mean, self.std = state['mean'], state['std']
