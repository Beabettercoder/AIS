#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 31: 架构基准（Table 1 Grand Benchmark, part 1）。

统一数据集（冻结 Houston 4500 窗）、统一双头、统一评测口径，只换编码器：
  gru / tcn / transformer / itransformer / ours (tcn+PSA+gate) / ours_graph (加邻船注意力)
Mamba / Neural-CDE / ST-GAT / Interval-GNN / STGVAD 需要额外依赖或图结构，列入 scaffold 未跑。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_MAX, EPS_MIN = 0.3, 0.01


def ap_score(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def load(d, split):
    z = np.load(Path(d) / f"{split}.npz")
    raw = z["raw"].astype(np.float32)
    w = z["w_self"].astype(np.float32)
    dp = np.diff(raw[:, :, 0:2], axis=1, prepend=0.0) * w[..., None]
    raw[:, :, 0:2] = np.cumsum(dp, axis=1)
    for ch in (2, 3, 4, 5, 6):
        v = raw[:, :, ch]
        low = w < 0.2
        for t in range(1, v.shape[1]):
            m = low[:, t]
            v[m, t] = v[m, t - 1]
        raw[:, :, ch] = v
    X = np.concatenate([raw, z["r_field"][..., None], w[..., None]], -1)
    return dict(X=np.nan_to_num(X), mask=z["mask"], y=z["label_class"], cell=z["label_cell"],
                ep=z["e_phys"])


class GRUEnc(nn.Module):
    def __init__(self, nf, nh=64):
        super().__init__()
        self.rnn = nn.GRU(nf, nh, batch_first=True, bidirectional=True)

    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.cat([h.mean(1), h.max(1).values], -1)


class TCNEnc(nn.Module):
    def __init__(self, nf, nh=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(nf, nh, 5, padding=2), nn.GELU(),
            nn.Conv1d(nh, nh, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(nh, nh, 5, padding=8, dilation=4), nn.GELU())

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return torch.cat([h.mean(-1), h.max(-1).values], -1)


class TransformerEnc(nn.Module):
    def __init__(self, nf, nh=64, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(nf, nh)
        layer = nn.TransformerEncoderLayer(nh, heads, 2 * nh, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        h = self.enc(self.proj(x))
        return torch.cat([h.mean(1), h.max(1).values], -1)


class ITransformerEnc(nn.Module):
    """iTransformer 风格：把每个变量当作 token（在时间轴上做注意力）。"""
    def __init__(self, nf, nh=64, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(20, nh)
        layer = nn.TransformerEncoderLayer(nh, heads, 2 * nh, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        h = x.transpose(1, 2)              # (B, C, T)
        h = self.enc(self.proj(h))         # (B, C, H)
        return torch.cat([h.mean(1), h.max(1).values], -1)


class Head(nn.Module):
    def __init__(self, din, ncls=5):
        super().__init__()
        self.bin = nn.Linear(din, 1)
        self.cls = nn.Linear(din, ncls)

    def forward(self, h):
        return self.bin(h).squeeze(-1), self.cls(h)


class Model(nn.Module):
    def __init__(self, arch, nf=9, nh=64):
        super().__init__()
        self.arch = arch
        self.enc = {"gru": GRUEnc, "tcn": TCNEnc, "transformer": TransformerEnc,
                    "itransformer": ITransformerEnc}[arch](nf, nh)
        din = (4 if arch == "gru" else 2) * nh + 1   # GRU 双向输出 2*nh，mean+max 再翻倍
        self.head = Head(din)
        self.name = arch

    def forward(self, x, gctx=None):
        h = self.enc(x)
        if gctx is None:
            gctx = torch.zeros(h.shape[0], 1, device=h.device)
        return self.head(torch.cat([h, gctx], -1))


def graph_context(nb_xy, nb_v, nb_mask, device):
    """邻船注意力的简化版：以 8 nm 高斯核加权聚合邻船速度偏差（D_abs 分支）。"""
    d = np.linalg.norm(nb_xy, axis=-1)
    w = np.exp(-(d ** 2) / (2 * (8 * 1852 / 2) ** 2)) * nb_mask
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-6)
    mean_v = np.einsum("bn,bnv->bv", w, nb_v)
    spread = np.linalg.norm(nb_v - mean_v[:, None, :], axis=-1)
    ctx = (spread * w).sum(1, keepdims=True)
    return torch.tensor(ctx, dtype=torch.float32, device=device)


def train_eval(arch, seed, tr, te, epochs=30, use_psa=False, use_graph=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = tr["X"][tr["mask"]].mean(0, keepdims=True)
    sd = tr["X"][tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = (tr["X"] - mu) / sd
    Xte = (te["X"] - mu) / sd
    gtr = gte = None

    model = Model(arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    for _ in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            xb = torch.tensor(Xtr[ii]).to(dev)
            gb = gtr[ii] if gtr is not None else None
            pb, pc = model(xb, gb)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pb, pc = model(torch.tensor(Xte).to(dev), gte)
        pb = torch.sigmoid(pb).cpu().numpy(); pc = F.softmax(pc, -1).cpu().numpy()
    cell, y = te["cell"], te["y"]
    out = {"arch": arch, "seed": seed,
           "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(pb, y > 0)}
    for k, name in enumerate(["turn_back", "deviation", "stall", "sinuous", "speed_drop"], start=1):
        m = cell == 2
        out[f"AP_{name}"] = ap_score(pc[m, k - 1], (y[m] == k))
    neg, pos = pb[cell == 0], pb[cell == 2]
    thr = np.quantile(neg, 0.99)
    out["recall_1pct"] = float((pos > thr).mean())
    n_params = sum(p.numel() for p in model.parameters())
    out["params"] = int(n_params)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    tr, te = load(args.dir, "train"), load(args.dir, "test")
    specs = [("gru", False, False), ("tcn", False, False), ("transformer", False, False),
             ("itransformer", False, False)]
    rows = []
    for arch, psa, graph in specs:
        name = "Ours+Graph" if graph else arch
        for s in range(args.seeds):
            r = train_eval(arch, s, tr, te, args.epochs, psa, graph)
            r["name"] = name
            rows.append(r)
            print(f"{name} seed{s}: clean={r['clean_AP']:.3f} full={r['full_AP']:.3f} R1%={r['recall_1pct']:.3f}", flush=True)

    agg = {}
    for r in rows:
        agg.setdefault(r["name"], []).append(r)
    summary = {k: {m: round(float(np.nanmean([x[m] for x in v])), 3)
                   for m in ("clean_AP", "full_AP", "recall_1pct", "params")} for k, v in agg.items()}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
