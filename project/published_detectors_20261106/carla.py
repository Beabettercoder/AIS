"""CARLA adapter. The caller supplies same-vessel temporal positive samples.

Uses the author's unmodified network and losses. Two stages are mandatory:
pretext triplets, then clustering with frozen embedding-neighbor indices.
"""
from pathlib import Path
import importlib.util
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1] / 'vendor' / 'CARLA'
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

net = _load('carla_original_models', 'models/models.py')
resnet = _load('carla_original_resnet', 'models/resent_time.py')
losses = _load('carla_original_losses', 'losses/losses.py')
augment = _load('carla_original_augment', 'data/augment.py')

class Adapter(nn.Module):
    def __init__(self, seq_len, channels, **kwargs):
        super().__init__()
        if seq_len < 10 or channels < 4:
            raise ValueError('CARLA original augmentation needs seq_len >= 10 and channels >= 4')
        self.seq_len, self.channels = seq_len, channels
        backbone = resnet.resnet_ts(in_channels=channels, mid_channels=kwargs.get('mid_channels', 4))
        self.pretext = net.ContrastiveModel(backbone, features_dim=kwargs.get('features_dim', 128))
        self.cluster = net.ClusteringModel(backbone, nclusters=kwargs.get('clusters', 10), nheads=1)
        self.classification_loss = losses.ClassificationLoss(entropy_weight=2, inconsistency_weight=0)
        self.pretext_loss = losses.PretextLoss(bs=1, temperature=kwargs.get('temperature', .1))
        self.phase = 'pretext'
        self.register_buffer('normal_cluster', torch.tensor(-1, dtype=torch.long))
        self.prev_loss = None

    def _reshape(self, x):
        # Match the author's train_utils.py reshape for (batch,time,channel).
        return x.reshape(x.shape[0], self.channels, self.seq_len)

    def embed(self, x):
        return self.pretext(self._reshape(x))

    def synthetic(self, x):
        # Source SubAnomaly operates sample-wise; retaining it avoids inventing a replacement.
        op = augment.SubAnomaly(portion_len=.99)
        return torch.stack([op(row.detach().cpu()) for row in x]).to(x.device, x.dtype)

    def begin_clustering(self):
        self.phase = 'clustering'

    def loss(self, x, indices=None, epoch=0, positive=None, negative=None,
             near=None, far=None):
        if self.phase == 'pretext':
            if positive is None:
                raise ValueError('Supply positive from a previous window of the same MMSI; use documented noise fallback only if unavailable')
            if negative is None:
                negative = self.synthetic(x)
            z = self.embed(torch.cat([x, positive, negative], dim=0))
            self.pretext_loss.bs = len(x)
            loss = self.pretext_loss(z, self.prev_loss)
            self.prev_loss = float(loss.detach())
            return loss
        if near is None or far is None:
            raise ValueError('Supply nearest and farthest training windows from frozen pretext embedding')
        a = self.cluster(self._reshape(x))[0]
        n = self.cluster(self._reshape(near))[0]
        f = self.cluster(self._reshape(far))[0]
        return self.classification_loss(a, n, f)[0]

    @torch.no_grad()
    def select_normal_cluster(self, train_x):
        """Set majority cluster using training windows only (not evaluation labels)."""
        logits = self.cluster(self._reshape(train_x))[0]
        self.normal_cluster.copy_(torch.bincount(logits.argmax(1), minlength=logits.shape[1]).argmax())

    @torch.no_grad()
    def score(self, x):
        if self.phase != 'clustering' or self.normal_cluster.item() < 0:
            raise RuntimeError('Complete both phases and set majority training cluster before scoring')
        return 1 - self.cluster(self._reshape(x))[0].softmax(-1)[:, self.normal_cluster]
