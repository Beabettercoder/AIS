#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 66: 群体相对异常——当训练里存在群体异常时，图是否就有效？（受控对照）"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
sp = importlib.util.spec_from_file_location("gf", "/mnt/sda/WangZijia/AIS/scripts/60_graph_fix.py")
gf = importlib.util.module_from_spec(sp); sp.loader.exec_module(gf)


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def shift(d, factors, rng):
    out = {k: v for k, v in d.items()}
    f = np.asarray(factors, dtype=np.float32)
    out["nb_v"] = d["nb_v"] * f[:, None, None]
    out["nb_seq"] = d["nb_seq"].copy()
    out["nb_seq"][:, :, :, 2:4] *= f[:, None, None, None]
    return out


def build_pair(d, rng, n_pos=None):
    """返回 (X_norm_set, X_pos_set)，目标船特征完全相同，只有邻船速度不同。"""
    n = len(d["raw"])
    base = dict(d)
    f = rng.uniform(0.60, 0.85, size=n).astype(np.float32)
    pos = shift(d, f, rng)
    return base, pos


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = sw.load(BASE / "eval_v2_graph2", "train")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    for d, s in ((tr, "train"), (te, "test")):
        d["nb_v"] = np.load(BASE / "eval_v2_graph2" / f"{s}.npz")["nb_v"].astype(np.float32)

    Xtr = np.concatenate([tr["gated"], tr["rf"][..., None], tr["w"][..., None]], -1)
    Xte = np.concatenate([te["gated"], te["rf"][..., None], te["w"][..., None]], -1)
    mu = Xtr[tr["mask"]].mean(0, keepdims=True); sd = Xtr[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    rng = np.random.default_rng(0)
    tr_neg, tr_pos = build_pair(tr, rng)
    te_neg, te_pos = build_pair(te, rng)

    def train_graph(epochs=20):
        torch.manual_seed(0)
        m = gf.Net2("tcn", Xtr.shape[-1]).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
        n = len(Xtr)
        for _ in range(epochs):
            m.train()
            perm = np.random.permutation(n)
            for b in range(0, n, 256):
                ii = perm[b:b + 256]
                xb = torch.tensor(Xtr[ii]).to(dev)
                pb, _, _, _ = m(xb, *gf.tensors(tr_neg, ii, dev))
                pp, _, _, _ = m(xb, *gf.tensors(tr_pos, ii, dev))
                loss = F.binary_cross_entropy_with_logits(pb, torch.zeros_like(pb)) + \
                       F.binary_cross_entropy_with_logits(pp, torch.ones_like(pp))
                opt.zero_grad(); loss.backward(); opt.step()
        return m

    def train_plain(epochs=20):
        torch.manual_seed(0)
        m = sw.Net("tcn", Xtr.shape[-1], use_graph=False).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
        n = len(Xtr)
        for _ in range(epochs):
            m.train()
            perm = np.random.permutation(n)
            for b in range(0, n, 256):
                ii = perm[b:b + 256]
                xb = torch.tensor(Xtr[ii]).to(dev)
                pb, _, _ = m(xb); pp, _, _ = m(xb)     # 目标特征完全相同
                loss = F.binary_cross_entropy_with_logits(pb, torch.zeros_like(pb)) + \
                       F.binary_cross_entropy_with_logits(pp, torch.ones_like(pp))
                opt.zero_grad(); loss.backward(); opt.step()
        return m

    mg = train_graph(); mp = train_plain()
    mg.eval(); mp.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte).to(dev)
        pn, _, _, _ = mg(xb, *gf.tensors(te_neg, np.arange(len(xb)), dev))
        pp, _, _, _ = mg(xb, *gf.tensors(te_pos, np.arange(len(xb)), dev))
        sg = torch.sigmoid(torch.cat([pn, pp])).cpu().numpy()
        bn, _, _ = mp(xb); bp, _, _ = mp(xb)
        sp_ = torch.sigmoid(torch.cat([bn, bp])).cpu().numpy()
    y = np.r_[np.zeros(len(xb)), np.ones(len(xb))]
    print("no-graph AP =", round(ap_score(sp_, y), 3))
    print("graph    AP =", round(ap_score(sg, y), 3))
    torch.save({"sd": mg.state_dict(), "stats": (mu, sd)}, W / "group_anomaly_graph.pt")


if __name__ == "__main__":
    main()
