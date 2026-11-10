#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 60: 修正版双距离校准图 Gψ（物理速度距离 + τ 进 softmax + LN 残差），重训 1 seed 并重扫 τ/K。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


class Net2(nn.Module):
    """修正版：物理速度距离 + τ 在 softmax 内 + LayerNorm 残差（不可自我关闭）。"""

    def __init__(self, kind, nf, nh=64):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(nf, nh, batch_first=True, bidirectional=True); dim = 4 * nh
        else:
            self.net = nn.Sequential(nn.Conv1d(nf, nh, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(nh, nh, 5, padding=4, dilation=2), nn.GELU(),
                                     nn.Conv1d(nh, nh, 5, padding=8, dilation=4), nn.GELU())
            dim = 2 * nh
        self.dim = dim
        self.nbrnn = nn.GRU(4, 32, batch_first=True)
        self.nbproj = nn.Linear(34, 64)
        self.q = nn.Linear(dim, 64)
        self.vproj = nn.Linear(64, dim)
        self.ln = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.bin = nn.Linear(dim, 1)
        self.cls = nn.Linear(dim, 5)

    def enc(self, x):
        if self.kind == "gru":
            h, _ = self.rnn(x); return torch.cat([h.mean(1), h.max(1).values], -1)
        h = self.net(x.transpose(1, 2))
        return torch.cat([h.mean(-1), h.max(-1).values], -1)

    def graph(self, h, nb_seq, nb_v, nb_xy, nb_mask, nb_w, r_i, v_i, tau=1.0, k=None):
        b, kk, s, f = nb_seq.shape
        g, _ = self.nbrnn(nb_seq.reshape(b * kk, s, f))
        g = self.nbproj(torch.cat([g[:, -1].reshape(b, kk, -1), nb_xy / 1852.0], -1))
        mask = nb_mask.clone()
        if k is not None:
            mask[:, k:] = False
        d_abs = torch.norm(nb_v - v_i[:, None, :], dim=-1) / 5.0          # 物理速度差 m/s
        mu = (nb_v * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        r_nb = torch.norm(nb_v - mu[:, None, :], dim=-1) / 5.0
        d_rel = torch.abs(r_i[:, None] - r_nb * 5.0) / 5.0
        s = self.alpha * d_abs + self.beta * d_rel
        a = torch.softmax(-s / tau, -1) * mask.float() * nb_w
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
        ctx = torch.einsum("bk,bkh->bh", a, self.vproj(g))
        return h + self.ln(ctx), a

    def forward(self, x, nb_seq, nb_v, nb_xy, nb_mask, nb_w, r_i, v_i, tau=1.0, k=None):
        h = self.enc(x)
        h, a = self.graph(h, nb_seq, nb_v, nb_xy, nb_mask, nb_w, r_i, v_i, tau, k)
        return self.bin(h).squeeze(-1), self.cls(h), h, a


def tensors(d, ii, dev):
    return (torch.tensor(d["nb_seq"][ii]).to(dev), torch.tensor(d["nb_v"][ii]).to(dev),
            torch.tensor(d["nb_xy"][ii]).to(dev), torch.tensor(d["nb_mask"][ii]).to(dev),
            torch.tensor(d["nb_w"][ii]).to(dev), torch.tensor(d["rf"][ii][:, -1]).to(dev),
            torch.tensor(d["raw"][ii][:, -1, 2:4]).to(dev))


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = sw.load(BASE / "eval_v2_graph2", "train")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    for _d, _s in ((tr, "train"), (te, "test")):
        _d["nb_v"] = np.load(BASE / "eval_v2_graph2" / f"{_s}.npz")["nb_v"].astype(np.float32)
    Xtr = np.concatenate([tr["gated"], tr["rf"][..., None], tr["w"][..., None]], -1)
    Xte = np.concatenate([te["gated"], te["rf"][..., None], te["w"][..., None]], -1)
    mu = Xtr[tr["mask"]].mean(0, keepdims=True); sd = Xtr[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

    torch.manual_seed(0); np.random.seed(0)
    model = Net2("tcn", Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2]); yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    for ep in range(25):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 256):
            ii = idx[b:b + 256]
            xb = torch.tensor(Xtr[ii]).to(dev)
            pb, pc, _, _ = model(xb, *tensors(tr, ii, dev))
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()

    torch.save({"sd": model.state_dict(), "stats": (mu, sd), "kind": "tcn", "nf": int(Xtr.shape[-1])},
               W / "graph_fixed.pt")

    model.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte).to(dev)
        args = tensors(te, np.arange(len(xb)), dev)
        pb, _, h, a = model(xb, *args)
        base = torch.sigmoid(pb).cpu().numpy()
    y = te["y"]; cell = te["cell"]
    print("clean AP =", round(ap_score(base, y > 0), 3))
    print("attention: entropy", round(float((-(a.clamp_min(1e-9).log()) * a).sum(-1).mean() / np.log(32)), 3),
          " max", round(float(a.max(-1).values.mean()), 3),
          " std", round(float(a.std(-1).mean()), 3))

    taus = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    acc_t = []
    for t in taus:
        with torch.no_grad():
            pb, _, _, _ = model(xb, *args, tau=t)
        acc_t.append(round(ap_score(torch.sigmoid(pb).cpu().numpy(), y > 0), 4))
    ks = [4, 8, 12, 16, 24, 32]
    acc_k = []
    for k in ks:
        with torch.no_grad():
            pb, _, _, _ = model(xb, *args, k=k)
        acc_k.append(round(ap_score(torch.sigmoid(pb).cpu().numpy(), y > 0), 4))
    print("tau sweep", dict(zip(taus, acc_t)))
    print("K sweep", dict(zip(ks, acc_k)))
    (BASE / "graph_fixed_sweep.json").write_text(json.dumps({"tau": dict(zip(map(str, taus), acc_t)),
                                                              "k": dict(zip(map(str, ks), acc_k)),
                                                              "clean": float(ap_score(base, y > 0))}, indent=2))


if __name__ == "__main__":
    main()
