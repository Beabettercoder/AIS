#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 51: 1-seed 训练并逐个落盘权重（后续所有图纯前向）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
OUT = BASE / "vis_weights"
OUT.mkdir(parents=True, exist_ok=True)


def load(d, split):
    z = np.load(Path(d) / f"{split}.npz")
    raw = z["raw"].astype(np.float32)
    w = z["w_self"].astype(np.float32)
    gated = raw.copy()
    dp = np.diff(gated[:, :, 0:2], axis=1, prepend=0.0) * w[..., None]
    gated[:, :, 0:2] = np.cumsum(dp, axis=1)
    for ch in (2, 3, 4, 5, 6):
        v = gated[:, :, ch]; low = w < 0.2
        for t in range(1, v.shape[1]):
            m = low[:, t]; v[m, t] = v[m, t - 1]
        gated[:, :, ch] = v
    n = len(raw)
    nb_seq = z["nb_seq"].astype(np.float32) if "nb_seq" in z.files else np.zeros((n, 32, 5, 4), np.float32)
    nb_xy = z["nb_xy"].astype(np.float32) if "nb_xy" in z.files else np.zeros((n, 32, 2), np.float32)
    nb_w = z["nb_w"].astype(np.float32) if "nb_w" in z.files else np.ones((n, 32), np.float32)
    nb_m = z["nb_mask"].astype(bool) if "nb_mask" in z.files else np.zeros((n, 32), bool)
    return dict(raw=np.nan_to_num(raw), gated=np.nan_to_num(gated), w=w, rf=z["r_field"],
                mask=z["mask"], y=z["label_class"], cell=z["label_cell"],
                nb_seq=np.nan_to_num(nb_seq), nb_xy=np.nan_to_num(nb_xy), nb_w=nb_w, nb_mask=nb_m)


class Enc(nn.Module):
    def __init__(self, kind, nf, nh=64):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(nf, nh, batch_first=True, bidirectional=True); self.dim = 4 * nh
        elif kind == "transformer":
            self.proj = nn.Linear(nf, nh)
            layer = nn.TransformerEncoderLayer(nh, 4, 2 * nh, batch_first=True, dropout=0.1)
            self.enc = nn.TransformerEncoder(layer, 2); self.dim = 2 * nh
        else:
            self.net = nn.Sequential(nn.Conv1d(nf, nh, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(nh, nh, 5, padding=4, dilation=2), nn.GELU(),
                                     nn.Conv1d(nh, nh, 5, padding=8, dilation=4), nn.GELU())
            self.dim = 2 * nh

    def forward(self, x):
        if self.kind == "gru":
            h, _ = self.rnn(x); return torch.cat([h.mean(1), h.max(1).values], -1)
        if self.kind == "transformer":
            h = self.enc(self.proj(x)); return torch.cat([h.mean(1), h.max(1).values], -1)
        h = self.net(x.transpose(1, 2))
        return torch.cat([h.mean(-1), h.max(-1).values], -1)


class Net(nn.Module):
    def __init__(self, kind, nf, use_graph=False, nh=64):
        super().__init__()
        self.enc = Enc(kind, nf, nh)
        d = self.enc.dim
        self.use_graph = use_graph
        self.bin = nn.Linear(d, 1); self.cls = nn.Linear(d, 5)
        if use_graph:
            self.rnn = nn.GRU(4, 64, batch_first=True)
            self.gproj = nn.Linear(66, d)
            self.alpha = nn.Parameter(torch.tensor(1.0)); self.beta = nn.Parameter(torch.tensor(1.0))
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, nseq=None, nxy=None, nm=None, nw=None, ri=None):
        h = self.enc(x)
        if self.use_graph and nseq is not None:
            b, k, s, f = nseq.shape
            g, _ = self.rnn(nseq.reshape(b * k, s, f))
            g = self.gproj(torch.cat([g[:, -1].reshape(b, k, -1), (nxy / 1852.0).to(x.dtype)], -1)) * nm[..., None].to(x.dtype)
            d = torch.norm(g, dim=-1) * 0.1
            sc = -(torch.tanh(self.alpha * d) + torch.tanh(self.beta * torch.abs(ri[:, None] - d)))
            a = torch.softmax(sc, -1) * nm.to(x.dtype) * nw
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
            rho = torch.clamp(nm.to(x.dtype).sum(1) / 4.0, max=1.0)
            h = h + self.gamma * torch.einsum("bk,bkh->bh", a, g) * rho[:, None]
        return self.bin(h).squeeze(-1), self.cls(h), h


def build(d, use_kg, use_lf):
    if not use_kg:
        return d["raw"]
    parts = [d["gated"]]
    if use_lf:
        parts.append(d["rf"][..., None])
    parts.append(d["w"][..., None])
    return np.concatenate(parts, -1)


def train(kind, tr, dev, use_kg, use_lf, psa, graph, epochs=25, clean=False, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    X = build(tr, use_kg, use_lf).copy()
    if clean:
        dp = np.diff(X[:, :, 0:2], axis=1, prepend=0.0)
        sp = np.linalg.norm(dp, axis=-1)
        med = np.median(sp, axis=1, keepdims=True)
        bad = sp > np.maximum(3 * med, 50.0)
        for b in range(X.shape[0]):
            idx = np.flatnonzero(~bad[b])
            if len(idx) > 2:
                for ch in range(X.shape[2]):
                    X[b, :, ch] = np.interp(np.arange(X.shape[1]), idx, X[b, idx, ch])
    mu = X[tr["mask"]].mean(0, keepdims=True); sd = X[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd
    model = Net(kind, X.shape[-1], use_graph=graph).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    for _ in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 256):
            ii = idx[b:b + 256]
            xb = torch.tensor(Xn[ii]).to(dev)
            args = ()
            if graph:
                args = (torch.tensor(tr["nb_seq"][ii]).to(dev), torch.tensor(tr["nb_xy"][ii]).to(dev),
                        torch.tensor(tr["nb_mask"][ii]).to(dev), torch.tensor(tr["nb_w"][ii]).to(dev),
                        torch.tensor(tr["rf"][ii][:, -1]).to(dev))
            pb, pc, h = model(xb, *args)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            if psa:
                keep = (torch.rand(xb.shape[0], xb.shape[1], device=dev) > 0.15).float()
                xb2 = xb * keep[..., None]
                pb2, pc2, h2 = model(xb2, *args)
                m_ = h2.mean(0, keepdim=True).detach(); s_ = h2.std(0, keepdim=True).detach().clamp_min(1e-5)
                h2p = m_ + s_ * (((h2 - m_) / s_) * 1.1 + 0.05)
                p1, p2 = F.softmax(pc, -1), F.softmax(model.cls(h2p), -1)
                mm = 0.5 * (p1 + p2)
                kl = lambda a, bb: (a * (torch.log(a + 1e-8) - torch.log(bb + 1e-8))).sum(-1)
                con = (0.5 * kl(p1, mm) + 0.5 * kl(p2, mm)) * keep.mean(1)
                loss = loss + 0.20 * con.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model, mu, sd, X.shape[-1]


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = load(BASE / "eval_v2_graph2", "train")
    specs = [
        ("ablation_0_raw_tcn", "tcn", dict(use_kg=False, use_lf=False, psa=False, graph=False)),
        ("ablation_1_kg", "tcn", dict(use_kg=True, use_lf=False, psa=False, graph=False)),
        ("ablation_2_kglf", "tcn", dict(use_kg=True, use_lf=True, psa=False, graph=False)),
        ("ablation_3_psa", "tcn", dict(use_kg=True, use_lf=True, psa=True, graph=False)),
        ("ablation_4_full_cdg", "tcn", dict(use_kg=True, use_lf=True, psa=True, graph=True)),
        ("base_gru", "gru", dict(use_kg=False, use_lf=False, psa=False, graph=False)),
        ("base_transformer", "transformer", dict(use_kg=False, use_lf=False, psa=False, graph=False)),
        ("base_aisclean", "transformer", dict(use_kg=False, use_lf=False, psa=False, graph=False, clean=True)),
        ("base_stgv", "tcn", dict(use_kg=False, use_lf=False, psa=False, graph=True)),
    ]
    for name, kind, kw in specs:
        model, mu, sd, nf = train(kind, tr, dev, **kw)
        torch.save({"sd": model.state_dict(), "stats": (mu, sd), "kind": kind,
                    "use_graph": bool(kw["graph"]), "nf": int(nf), "cfg": kw}, OUT / f"{name}.pt")
        print("saved", name, flush=True)


if __name__ == "__main__":
    main()
