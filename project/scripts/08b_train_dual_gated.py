#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 08b: 双头 TCN + w 软门控掩码 + 决策时刻分层 AP + 1%/5% FAR。

w 掩码（关键）：不改绝对坐标，而是对**步间增量**做软门控
    Δp̃_t = w_t · Δp_t ,   p̃_t = p̃_{t-1} + Δp̃_t
被跳变污染的两步（跳入/跳出）w≈0，于是整段位移平滑续接，不会造出新的跳变。
速度/航向通道在 w<0.2 时用前一步值前向填充。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["normal", "turn_back", "deviation", "stall", "sinuous", "speed_drop"]


def apply_gate(X, w):
    """对增量做软门控，返回重建后的特征。X: (N,T,7)，w: (N,T)。"""
    X = X.copy()
    dp = np.diff(X[:, :, 0:2], axis=1, prepend=0.0)
    dp = dp * w[..., None]
    X[:, :, 0:2] = np.cumsum(dp, axis=1)
    low = w < 0.2
    for ch in (2, 3, 4, 5, 6):
        v = X[:, :, ch]
        for t in range(1, v.shape[1]):
            m = low[:, t]
            v[m, t] = v[m, t - 1]
        X[:, :, ch] = v
    return X


def load(d, split, mode, use_gate=True):
    z = np.load(Path(d) / f"{split}.npz")
    X = z["raw"].astype(np.float32).copy()
    w = z["w_self"].astype(np.float32)
    if mode in ("mean", "ours"):
        X = apply_gate(X, w) if (mode == "ours" and use_gate) else X
        extra = z["r_mean"] if mode == "mean" else z["r_field"]
        X = np.concatenate([X, extra[..., None]], -1)
    if mode == "ours":
        X = np.concatenate([X, w[..., None]], -1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return dict(X=X, mask=z["mask"], y=z["label_class"], cell=z["label_cell"],
                delay=z["label_delay"])


class Net(nn.Module):
    def __init__(self, nf, nh=64):
        super().__init__()
        self.c1 = nn.Conv1d(nf, nh, 5, padding=2)
        self.c2 = nn.Conv1d(nh, nh, 5, padding=4, dilation=2)
        self.c3 = nn.Conv1d(nh, nh, 5, padding=8, dilation=4)
        self.bin = nn.Linear(2 * nh, 1)
        self.cls = nn.Linear(2 * nh, 5)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.gelu(self.c1(h)); h = F.gelu(self.c2(h)); h = F.gelu(self.c3(h))
        p = torch.cat([h.mean(-1), h.max(-1).values], -1)
        return self.bin(p).squeeze(-1), self.cls(p)


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


def run(mode, seed, tr, te, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = tr["X"][tr["mask"]].mean(0, keepdims=True)
    sd = tr["X"][tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr, Xte = (tr["X"] - mu) / sd, (te["X"] - mu) / sd

    model = Net(Xtr.shape[-1]).to(dev)
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
            pb, pc = model(torch.tensor(Xtr[ii]).to(dev))
            lb = torch.tensor(yb[ii]).to(dev)
            loss = F.binary_cross_entropy_with_logits(pb, lb)
            m = lb > 0
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pb, pc = model(torch.tensor(Xte).to(dev))
        pb = torch.sigmoid(pb).cpu().numpy(); pc = F.softmax(pc, -1).cpu().numpy()

    cell, y, dl = te["cell"], te["y"], te["delay"]
    out = {"mode": mode, "seed": seed,
           "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(pb, y > 0)}
    neg = pb[cell == 0]
    for far in (1, 5):
        thr = np.quantile(neg, 1 - far / 100)
        out[f"recall_{far}pct"] = float((pb[cell == 2] > thr).mean())
        for d0 in (180, 300, 600):
            m = (cell == 2) & (dl == d0)
            out[f"recall_{far}pct_d{d0//60}"] = float((pb[m] > thr).mean()) if m.any() else float("nan")
    # 分时刻 AP：正类 = 该 delay 的 C 窗，负类 = 全部 A 窗
    for d0 in (180, 300, 600):
        m = (cell == 2) & (dl == d0)
        idx_sel = m | (cell == 0)
        out[f"AP_d{d0//60}"] = ap_score(pb[idx_sel], (cell[idx_sel] == 2).astype(float))
    for k in range(1, 6):
        m = cell == 2
        out[f"AP_{CLASSES[k]}"] = ap_score(pc[m, k - 1], (y[m] == k))
    return out, {"pb": pb.astype("float32"), "pc": pc.astype("float32"),
                 "cell": cell.astype("int16"), "y": y.astype("int16"), "delay": dl.astype("int16")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", required=True, choices=["raw", "mean", "ours"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_gate", action="store_true")
    args = ap.parse_args()

    tr, te = load(args.dir, "train", args.mode, not args.no_gate), load(args.dir, "test", args.mode, not args.no_gate)
    rows, preds = [], []
    for s in range(args.seeds):
        out_s, p_s = run(args.mode, s, tr, te, args.epochs)
        rows.append(out_s)
        preds.append(p_s)
        pp = Path(args.out).with_name(Path(args.out).stem + f"_s{s}.npz")
        np.savez_compressed(pp, **p_s)
    keys = [k for k in rows[0] if k not in ("mode", "seed")]
    agg = {k: {"mean": float(np.nanmean([r[k] for r in rows])),
               "std": float(np.nanstd([r[k] for r in rows]))} for k in keys}
    Path(args.out).write_text(json.dumps({"mode": args.mode, "runs": rows, "agg": agg}, indent=2), encoding="utf-8")
    print(json.dumps({"mode": args.mode, **{k: round(v["mean"], 3) for k, v in agg.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
