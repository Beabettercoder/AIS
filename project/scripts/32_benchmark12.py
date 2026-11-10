#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 32: 12 模型基准（纯 PyTorch 实现，无外部 C++/CUDA 扩展）。

流派一：GRU / TCN / Transformer / iTransformer / Mamba-lite
流派二：Neural-CDE-lite / AISClean+Transformer / PhyLoss-TCN
流派三：ST-GAT / Interval-GNN / STGVAD
流派四：Ours-Full（TCN + 门控 + 场残差 + 校准双距离图 Gψ）
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

K_NB = 32


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
                rf=z["r_field"], nb_xy=z["nb_xy"], nb_v=z["nb_v"], nb_w=z["nb_w"], nb_mask=z["nb_mask"])


# ---------------- encoders ----------------
class TCNEnc(nn.Module):
    def __init__(self, nf, nh=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(nf, nh, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(nh, nh, 5, padding=4, dilation=2), nn.GELU(),
                                 nn.Conv1d(nh, nh, 5, padding=8, dilation=4), nn.GELU())

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return torch.cat([h.mean(-1), h.max(-1).values], -1)


class GRUEnc(nn.Module):
    def __init__(self, nf, nh=64):
        super().__init__()
        self.rnn = nn.GRU(nf, nh, batch_first=True, bidirectional=True)

    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.cat([h.mean(1), h.max(1).values], -1)


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
    def __init__(self, nf, nh=64, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(20, nh)
        layer = nn.TransformerEncoderLayer(nh, heads, 2 * nh, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.enc(self.proj(h))
        return torch.cat([h.mean(1), h.max(1).values], -1)


class MambaEnc(nn.Module):
    """纯 PyTorch selective SSM（Δ/B/C 输入依赖） + SiLU 门控，T=20 逐步扫描。"""
    def __init__(self, nf, nh=64):
        super().__init__()
        self.inp = nn.Linear(nf, nh)
        self.dw = nn.Conv1d(nh, nh, 3, padding=1, groups=nh)
        self.delta = nn.Linear(nh, 1)
        self.B = nn.Linear(nh, nh)
        self.C = nn.Linear(nh, nh)
        self.A_log = nn.Parameter(torch.zeros(nh))
        self.gate = nn.Linear(nh, nh)
        self.out = nn.Linear(nh, 2 * nh)

    def forward(self, x):
        u = self.inp(x)                                  # (B,T,H)
        u = self.dw(u.transpose(1, 2)).transpose(1, 2)
        A = -torch.exp(self.A_log)
        h = torch.zeros(u.shape[0], u.shape[2], device=u.device)
        ys = []
        for t in range(u.shape[1]):
            dt = F.softplus(self.delta(u[:, t])) + 1e-3
            Bt = self.B(u[:, t]); Ct = self.C(u[:, t])
            h = torch.exp(dt * A) * h + dt * Bt * u[:, t]
            ys.append(Ct * h)
        y = torch.stack(ys, 1)
        y = y * F.silu(self.gate(u))
        p = self.out(y)
        return torch.cat([p.mean(1), p.max(1).values], -1)


class CDEEnc(nn.Module):
    """Neural CDE-lite：以线性插值导数驱动 Euler 积分。"""
    def __init__(self, nf, nh=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nh + nf, nh), nn.Tanh())
        self.z0 = nn.Linear(nf, nh)
        self.dx_proj = nn.Linear(nf, nh)
        self.out = nn.Linear(nh, 2 * nh)

    def forward(self, x):
        dx = torch.diff(x, dim=1)
        z = self.z0(x[:, 0])
        zs = []
        for t in range(dx.shape[1]):
            z = z + self.f(torch.cat([z, x[:, t]], -1)) * self.dx_proj(dx[:, t])
            zs.append(z)
        z = torch.stack(zs, 1)
        p = self.out(z)
        return torch.cat([p.mean(1), p.max(1).values], -1)


# ---------------- graph context ----------------
class GraphCtx(nn.Module):
    def __init__(self, nh=64, out_dim=128, use_w=True, use_rel=True, mode="gat"):
        super().__init__()
        self.nb_enc = nn.Sequential(nn.Linear(5, nh), nn.GELU(), nn.Linear(nh, nh))
        self.q = nn.Linear(2 * nh, nh)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.use_w = use_w
        self.use_rel = use_rel
        self.out = nn.Linear(nh, out_dim)
        self.mode = mode
        self.aux = nn.Linear(out_dim + nh, 1) if mode == "interval" else None
        self.recon = nn.Linear(out_dim + nh, 2) if mode == "vad" else None

    def forward(self, h, nb_v, nb_xy, nb_w, nb_mask, r_i):
        # 邻船特征
        feat = torch.cat([nb_v, nb_xy / 1852.0, nb_w[..., None]], -1)
        g = self.nb_enc(feat)                                  # (B,K,H)
        # 距离
        v_nb = nb_v
        v_i = torch.zeros_like(nb_v[:, :1, :])                  # 目标速度已隐含在 h 中，用邻船均值做参照
        d_abs = torch.norm(v_nb - v_nb.mean(1, keepdim=True), dim=-1)
        r_nb = torch.norm(v_nb - v_nb.mean(1, keepdim=True), dim=-1)
        d_rel = torch.abs(r_i[:, None] - r_nb) if self.use_rel else torch.zeros_like(d_abs)
        score = -(torch.tanh(self.alpha * d_abs) + torch.tanh(self.beta * d_rel))
        a = torch.softmax(score, -1) * nb_mask.float()
        if self.use_w:
            a = a * nb_w
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
        ctx = torch.einsum("bk,bkh->bh", a, g)
        extra = None
        if self.mode == "interval":
            extra = self.aux(torch.cat([h, ctx], -1)).squeeze(-1)
        elif self.mode == "vad":
            mv = (nb_v * nb_mask[..., None]).sum(1) / nb_mask.sum(1, keepdim=True).clamp_min(1.0)
            rec = self.recon(torch.cat([h, ctx], -1))
            extra = torch.norm(rec - mv, dim=-1)
            ctx = ctx + extra[:, None]
        return self.gamma * self.out(ctx), extra


def build(name, nf=9):
    """返回 (encoder, head_dim, graph_module or None, aux_mode)."""
    enc = {"gru": GRUEnc, "tcn": TCNEnc, "transformer": TransformerEnc, "itransformer": ITransformerEnc,
           "mamba": MambaEnc, "cde": CDEEnc}[name](nf)
    dim = 256 if name in ("gru", "mamba", "cde") else 128
    return enc, dim


def train_one(name, seed, tr, te, epochs=25, phy=0.0, graph=False, graph_w=True, clean=False, gmode="gat", ours_full=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Xte = tr["X"].copy(), te["X"].copy()

    if clean:   # AISClean：对位移尖峰做中值滤波修复
        for A in (Xtr, Xte):
            d = np.diff(A[:, :, 0:2], axis=1, prepend=0.0)
            med = np.median(d, axis=1, keepdims=True)
            dev_ = np.linalg.norm(d - med, axis=-1, keepdims=True)
            bad = dev_ > np.percentile(dev_, 99, axis=1, keepdims=True)
            d = np.where(bad, med, d)
            A[:, :, 0:2] = np.cumsum(d, axis=1)

    mu = Xtr[tr["mask"]].mean(0, keepdims=True); sd = Xtr[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    enc, dim = build(name)
    enc = enc.to(dev)
    head = nn.Linear(dim, 5).to(dev)
    bin_head = nn.Linear(dim, 1).to(dev)
    gctx = GraphCtx(out_dim=dim, use_w=graph_w, mode=gmode).to(dev) if graph else None

    params = list(enc.parameters()) + list(head.parameters()) + list(bin_head.parameters())
    if gctx is not None:
        params += list(gctx.parameters())
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)

    def nb_tensors(d, ii):
        return (torch.tensor(d["nb_v"][ii]).to(dev), torch.tensor(d["nb_xy"][ii]).to(dev),
                torch.tensor(d["nb_w"][ii]).to(dev), torch.tensor(d["nb_mask"][ii]).to(dev),
                torch.tensor(d["rf"][ii][:, -1]).to(dev))

    for _ in range(epochs):
        enc.train(); head.train(); bin_head.train()
        np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            xb = torch.tensor(Xtr[ii]).to(dev)
            h = enc(xb)
            if gctx is not None:
                nv, nxy, nw, nm, ri = nb_tensors(tr, ii)
                ctx, _aux = gctx(h, nv, nxy, nw, nm, ri)
                h = h + ctx
            pb = bin_head(h).squeeze(-1); pc = head(h)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            if phy > 0 and hasattr(enc, "net"):
                hs = enc.net(xb.transpose(1, 2))
                loss = loss + phy * ((hs[:, :, 2:] - 2 * hs[:, :, 1:-1] + hs[:, :, :-2]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    enc.eval(); head.eval(); bin_head.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte).to(dev)
        h = enc(xb)
        if gctx is not None:
            nv, nxy, nw, nm, ri = nb_tensors(te, np.arange(len(xb)))
            ctx, _aux = gctx(h, nv, nxy, nw, nm, ri)
            h = h + ctx
        pb = torch.sigmoid(bin_head(h).squeeze(-1)).cpu().numpy()
        pc = F.softmax(head(h), -1).cpu().numpy()
    cell, y = te["cell"], te["y"]
    out = {"name": name, "seed": seed,
           "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(pb, y > 0)}
    for k, cn in enumerate(["turn_back", "deviation", "stall", "sinuous", "speed_drop"], start=1):
        m = cell == 2
        out[f"AP_{cn}"] = ap_score(pc[m, k - 1], (y[m] == k))
    neg, pos = pb[cell == 0], pb[cell == 2]
    for far in (1, 5):
        thr = np.quantile(neg, 1 - far / 100)
        out[f"recall_{far}pct"] = float((pos > thr).mean())
    out["params"] = int(sum(p.numel() for p in params))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", default="", help="逗号分隔的模型名子集")
    args = ap.parse_args()

    tr, te = load(args.dir, "train"), load(args.dir, "test")
    specs = [
        ("GRU", dict(name="gru")),
        ("TCN", dict(name="tcn")),
        ("Transformer", dict(name="transformer")),
        ("iTransformer", dict(name="itransformer")),
        ("Mamba-lite", dict(name="mamba")),
        ("Neural-CDE-lite", dict(name="cde")),
        ("AISClean+Transformer", dict(name="transformer", clean=True)),
        ("PhyLoss-TCN", dict(name="tcn", phy=0.05)),
        ("ST-GAT", dict(name="tcn", graph=True, graph_w=False, gmode="gat")),
        ("Interval-GNN", dict(name="tcn", graph=True, graph_w=False, gmode="interval")),
        ("STGVAD", dict(name="tcn", graph=True, graph_w=False, gmode="vad")),
        ("Ours-Full-TCN", dict(name="tcn", graph=True, graph_w=True, gmode="gat", ours_full=True)),
        ("Ours-Full-GRU", dict(name="gru", graph=True, graph_w=True, gmode="gat", ours_full=True)),
    ]
    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    if only:
        specs = [s for s in specs if s[0] in only]
    rows = []
    for label, kw in specs:
        for s in range(args.seeds):
            r = train_one(seed=s, tr=tr, te=te, **kw)
            r["model"] = label
            rows.append(r)
            print(f"{label} s{s}: clean={r['clean_AP']:.3f} full={r['full_AP']:.3f} R1={r['recall_1pct']:.3f}", flush=True)
    agg = {}
    for r in rows:
        agg.setdefault(r["model"], []).append(r)
    summ = {k: {m: round(float(np.nanmean([x[m] for x in v])), 3)
                for m in ("clean_AP", "full_AP", "recall_1pct", "recall_5pct", "params")} for k, v in agg.items()}
    Path(args.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
