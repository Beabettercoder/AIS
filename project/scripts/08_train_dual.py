#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 08: 解耦双头 TCN + 三种特征组 + 5 seeds（Table 1 / Table 3 原型）。

特征组：
  raw   原始运动特征
  mean  raw + 未加权邻船均值残差序列 r_mean(t)
  ours  raw + 留一法局部线性场残差序列 r_field(t) + 自身可信度 w_self(t)

双头：
  bin_head  BCE（不加权）→ 负责 A+B vs C+D
  cls_head  仅正类（C/D）上的加权 CE → 负责 5 类细分
推理：p_cls = sigmoid(bin) * softmax(cls)
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


def load(d, split, mode):
    z = np.load(Path(d) / f"{split}.npz")
    X = [z["raw"]]
    if mode == "mean":
        X.append(z["r_mean"][..., None])
    elif mode == "ours":
        X.append(z["r_field"][..., None])
        X.append(z["w_self"][..., None])
    X = np.concatenate(X, -1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return dict(X=X, mask=z["mask"], y=z["label_class"], cell=z["label_cell"],
                delay=z["label_delay"], t=z["t"])


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
    s, y = s[ok], y[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def run(mode, seed, tr, va, te, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = tr["X"][tr["mask"]].mean(0, keepdims=True)
    sd = tr["X"][tr["mask"]].std(0, keepdims=True) + 1e-6
    for d in (tr, va, te):
        d["X"] = (d["X"] - mu) / sd

    model = Net(tr["X"].shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    pos = yb[sel] > 0
    cnt = np.bincount(tr["y"][sel][pos] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)

    idx = np.flatnonzero(sel)
    for ep in range(epochs):
        model.train()
        np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            xb = torch.tensor(tr["X"][ii]).to(dev)
            pb, pc = model(xb)
            lb = torch.tensor(yb[ii]).to(dev)
            loss = F.binary_cross_entropy_with_logits(pb, lb)
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                lc = F.cross_entropy(pc[m], torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev), weight=cw)
                loss = loss + 0.5 * lc
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pb_te, pc_te = model(torch.tensor(te["X"]).to(dev))
        pb_te = torch.sigmoid(pb_te).cpu().numpy()
        pc_te = F.softmax(pc_te, -1).cpu().numpy()
    cell, y = te["cell"], te["y"]
    clean = np.isin(cell, [0, 2])
    res = {"mode": mode, "seed": seed,
           "clean_AP": ap_score(pb_te[clean], (y[clean] > 0)),
           "full_AP": ap_score(pb_te, (y > 0))}
    for k in range(1, 6):
        m = cell == 2
        res[f"AP_{CLASSES[k]}"] = ap_score(pc_te[m, k - 1], (y[m] == k))
    # 1% FAR on A
    thr = np.quantile(pb_te[cell == 0], 0.99)
    res["recall_at_1pctFAR"] = float((pb_te[(cell == 2)] > thr).mean())
    for dl in (180, 300, 600):
        m = (cell == 2) & (te["delay"] == dl)
        res[f"recall_delay_{dl//60}min"] = float((pb_te[m] > thr).mean()) if m.any() else float("nan")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", required=True, choices=["raw", "mean", "ours"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tr, va, te = (load(args.dir, s, args.mode) for s in ("train", "val", "test"))
    rows = [run(args.mode, s, tr, va, te, args.epochs) for s in range(args.seeds)]
    keys = [k for k in rows[0] if k not in ("mode", "seed")]
    agg = {k: {"mean": float(np.nanmean([r[k] for r in rows])),
               "std": float(np.nanstd([r[k] for r in rows]))} for k in keys}
    out = {"mode": args.mode, "seeds": args.seeds, "runs": rows, "agg": agg}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"mode": args.mode, **{k: round(v["mean"], 3) for k, v in agg.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
