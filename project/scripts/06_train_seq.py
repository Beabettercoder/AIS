#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 06: GRU / TCN 序列基线 —— 测 C 格行为 AP 的可达上界。

特征组：
  raw            [dx, dy, sog, cog_sin, cog_cos]
  frenet         raw + [s-s0, d, v_s, v_d, dchi]（沿航道 Frenet 坐标）
  frenet_field   frenet + 决策时刻的留一法场残差 r_i（标量）

评测口径：
  clean  : A vs C（观测干净，纯行为可分性）
  full   : A+B vs C+D（含观测退化窗口，检验误报是否被退化抬高）
  另报 5 类行为各自的 per-class AP（仅 C 格）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NM, RADIUS_M, KNOT = 1852.0, 8.0 * 1852.0, 0.514444
CLASSES = ["normal", "turn_back", "deviation", "stall", "sinuous", "speed_drop"]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def field_residual(nb_xy, nb_v, nb_w, nb_mask):
    """决策时刻的留一法加权局部线性残差（目标速度为 0 的相对量，故取中心速度偏差的模）。"""
    out = np.full(len(nb_xy), np.nan)
    for i in range(len(nb_xy)):
        m = nb_mask[i]
        if m.sum() < 5:
            continue
        px, py = nb_xy[i, m, 0], nb_xy[i, m, 1]
        vx, vy = nb_v[i, m, 0], nb_v[i, m, 1]
        w = np.clip(nb_w[i, m], 0.05, 1.0)
        X = np.column_stack([np.ones_like(px), px / RADIUS_M, py / RADIUS_M])
        sw = np.sqrt(w)[:, None]
        try:
            th, *_ = np.linalg.lstsq(X * sw, np.column_stack([vx, vy]) * sw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        # 目标船相对速度：用中心处背景速度的模作为"群体速度水平"的代理
        out[i] = np.hypot(th[0, 0], th[0, 1])
    return out


def build_features(z, mode):
    xy = z["xy"]; sc = z["sog_cog"]; mask = z["mask"]; fr = z["frenet"]
    n, T, _ = xy.shape
    dx = xy[:, :, 0]
    dy = xy[:, :, 1]
    # 用位置差分得到相对位移速度（把弯道/直道统一到"位移"层面）
    dt = np.diff(z["t"], axis=1, prepend=z["t"][:, :1])
    dt = np.where(dt <= 0, 70.0, dt)
    vx = np.gradient(xy[:, :, 0], axis=1) / dt
    vy = np.gradient(xy[:, :, 1], axis=1) / dt
    feats = [dx, dy, vx, vy, sc[:, :, 0], np.sin(sc[:, :, 1]), np.cos(sc[:, :, 1])]
    if mode in ("frenet", "frenet_field"):
        d_rel = fr[:, :, 1] - fr[:, :, 1][:, :1]
        feats += [fr[:, :, 0], d_rel, fr[:, :, 2], fr[:, :, 3], fr[:, :, 4]]
    X = np.stack(feats, -1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    extra = None
    if mode == "frenet_field":
        extra = field_residual(z["nb_xy"], z["nb_v"], z["nb_w"], z["nb_mask"]).astype(np.float32)
    return X, extra, mask


class GRUNet(nn.Module):
    def __init__(self, nf, nh=64, ncls=6, extra=0):
        super().__init__()
        self.gru = nn.GRU(nf, nh, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * nh + extra, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, ncls))

    def forward(self, x, m, e=None):
        h, _ = self.gru(x)
        last = h[:, -1]
        if e is not None:
            last = torch.cat([last, e], -1)
        return self.head(last)


class TCN(nn.Module):
    def __init__(self, nf, nh=64, ncls=6, extra=0):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(nf, nh, 3, padding=1, dilation=1),
            nn.Conv1d(nh, nh, 3, padding=2, dilation=2),
            nn.Conv1d(nh, nh, 3, padding=4, dilation=4),
        ])
        self.head = nn.Sequential(nn.Linear(2 * nh + extra, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, ncls))

    def forward(self, x, m, e=None):
        h = x.transpose(1, 2)
        for c in self.convs:
            h = F.gelu(c(h))
        pooled = torch.cat([h.mean(-1), h.max(-1).values], -1)
        if e is not None:
            pooled = torch.cat([pooled, e], -1)
        return self.head(pooled)


def ap_score(scores, labels):
    ok = np.isfinite(scores)
    s, y = scores[ok], labels[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", default="frenet", choices=["raw", "frenet", "frenet_field"])
    ap.add_argument("--arch", default="gru", choices=["gru", "tcn"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = Path(args.dir)
    tr, va, te = (np.load(d / f"{s}.npz") for s in ("train", "val", "test"))
    Xtr, Etr, Mtr = build_features(tr, args.mode)
    Xva, Eva, Mva = build_features(va, args.mode)
    Xte, Ete, Mte = build_features(te, args.mode)

    mu = Xtr[Mtr].mean(0, keepdims=True)
    sd = Xtr[Mtr].std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd

    ytr = tr["label_class"]; yva = va["label_class"]; yte = te["label_class"]
    cell_tr, cell_va, cell_te = tr["label_cell"], va["label_cell"], te["label_cell"]

    # 只在 A/C 训练（clean 行为任务）
    sel_tr = np.isin(cell_tr, [0, 2])
    sel_va = np.isin(cell_va, [0, 2])

    def to_t(X, M, E, idx):
        out = [torch.tensor(X[idx]), torch.tensor(M[idx]).float()]
        out.append(torch.tensor(E[idx][:, None]) if E is not None else None)
        return out

    extra_dim = 1 if args.mode == "frenet_field" else 0
    model = (GRUNet(Xtr.shape[-1], extra=extra_dim) if args.arch == "gru"
             else TCN(Xtr.shape[-1], extra=extra_dim)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

    idx_tr = np.flatnonzero(sel_tr)
    cnt = np.bincount(ytr[idx_tr], minlength=6).astype(float)
    cw = torch.tensor(len(idx_tr) / (6.0 * np.maximum(cnt, 1.0)), dtype=torch.float32)
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(idx_tr)
        for b in range(0, len(perm), 64):
            ii = perm[b:b + 64]
            xb, mb, eb = to_t(Xtr, Mtr, Etr, ii)
            yb = torch.tensor(ytr[ii]).to(dev)
            out = model(xb.to(dev), mb.to(dev), eb.to(dev) if eb is not None else None)
            loss = F.cross_entropy(out, yb, weight=cw.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        xa, ma, ea = to_t(Xva, Mva, Eva, np.flatnonzero(sel_va))
        xb, mb, eb = to_t(Xte, Mte, Ete, np.arange(len(Xte)))
        pva = F.softmax(model(xa.to(dev), ma.to(dev), ea.to(dev) if ea is not None else None).cpu(), -1).numpy()
        pte = F.softmax(model(xb.to(dev), mb.to(dev), eb.to(dev) if eb is not None else None).cpu(), -1).numpy()

    any_anom = 1.0 - pte[:, 0]
    res = {
        "mode": args.mode, "arch": args.arch, "epochs": args.epochs,
        "clean_AP_A_vs_C": ap_score(any_anom[np.isin(cell_te, [0, 2])],
                                    (yte[np.isin(cell_te, [0, 2])] > 0)),
        "full_AP_AB_vs_CD": ap_score(any_anom, yte > 0),
        "per_class_AP": {CLASSES[k]: ap_score(pte[cell_te == 2][:, k], (yte[cell_te == 2] == k))
                         for k in range(1, 6)},
        "val_acc": float((pva.argmax(1) == yva[sel_va]).mean()),
        "test_acc_C": float((pte[cell_te == 2].argmax(1) == yte[cell_te == 2]).mean()),
    }
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
