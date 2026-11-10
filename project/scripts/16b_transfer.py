#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 16b: 跨域迁移（修复版）—— 三种方法 × S1/S2/S3(1-shot/5-shot) × 双向。

S3 修复：
  * 解冻最后一个 TCN block + 双头，低学习率 2e-4，40 epoch
  * L2-SP 锚定：λ‖θ−θ_src‖²，λ=1e-3，只对解冻参数
  * 推理时与冻结源模型做 0.5/0.5 概率融合，防止小样本跑偏
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["normal", "turn_back", "deviation", "stall", "sinuous", "speed_drop"]


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


def load(d, split, method):
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
    if method == "raw":
        X = raw
    elif method == "knn":
        X = np.concatenate([raw, z["r_mean"][..., None]], -1)
    else:
        X = np.concatenate([raw, z["r_field"][..., None], w[..., None]], -1)
    return dict(X=np.nan_to_num(X), mask=z["mask"], y=z["label_class"], cell=z["label_cell"])


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


def fit(model, X, y, cell, dev, epochs, lr=2e-3, params=None, anchor=None, lam=0.0):
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=1e-2)
    sel = np.isin(cell, [0, 2])
    yb = (y > 0).astype(np.float32)
    cnt = np.bincount(y[sel & (y > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    for _ in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 64):
            ii = idx[b:b + 64]
            pb, pc = model(torch.tensor(X[ii]).to(dev))
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(y[ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            if anchor is not None and lam > 0:
                reg = sum(((p - anchor[n]) ** 2).sum() for n, p in model.named_parameters() if n in anchor)
                loss = loss + lam * reg
            opt.zero_grad(); loss.backward(); opt.step()


def predict(model, X, dev):
    model.eval()
    with torch.no_grad():
        pb, pc = model(torch.tensor(X).to(dev))
        return torch.sigmoid(pb).cpu().numpy(), F.softmax(pc, -1).cpu().numpy()


def run(setting, method, seed, src, tgt, shot, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = src["train"]["X"][src["train"]["mask"]].mean(0, keepdims=True)
    sd = src["train"]["X"][src["train"]["mask"]].std(0, keepdims=True) + 1e-6
    nf = src["train"]["X"].shape[-1]

    def norm(d):
        return (d["X"] - mu) / sd

    model = Net(nf).to(dev)
    if setting == "s1":
        fit(model, norm(tgt["train"]), tgt["train"]["y"], tgt["train"]["cell"], dev, epochs)
    else:
        fit(model, norm(src["train"]), src["train"]["y"], src["train"]["cell"], dev, epochs)
        if setting == "s3":
            src_state = {n: p.detach().clone() for n, p in model.named_parameters()}
            frozen = copy.deepcopy(model)
            for p in model.c1.parameters():
                p.requires_grad = False
            for p in model.c2.parameters():
                p.requires_grad = False
            params = list(model.c3.parameters()) + list(model.bin.parameters()) + list(model.cls.parameters())
            tr = tgt["train"]
            rng = np.random.default_rng(seed)
            idx = []
            for k in range(6):
                pool = np.flatnonzero(tr["y"] == k) if k > 0 else np.flatnonzero(tr["cell"] == 0)
                if len(pool):
                    idx.extend(rng.choice(pool, size=min(shot, len(pool)), replace=False).tolist())
            idx = np.array(sorted(set(idx)))
            fit(model, norm(tr)[idx], tr["y"][idx], np.where(tr["y"][idx] > 0, 2, 0), dev,
                epochs=40, lr=2e-4, params=params, anchor=src_state, lam=1e-3)
            Xte = norm(tgt["test"])
            pb_ft, pc_ft = predict(model, Xte, dev)
            pb_src, pc_src = predict(frozen, Xte, dev)
            pb = 0.5 * pb_ft + 0.5 * pb_src
            pc = 0.5 * pc_ft + 0.5 * pc_src
        else:
            pb, pc = predict(model, norm(tgt["test"]), dev)
    if setting != "s3":
        pb, pc = predict(model, norm(tgt["test"]), dev)
    cell, y = tgt["test"]["cell"], tgt["test"]["y"]
    out = {"setting": setting, "method": method, "seed": seed, "shot": shot,
           "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(pb, y > 0)}
    for k in range(1, 6):
        m = cell == 2
        out[f"AP_{CLASSES[k]}"] = ap_score(pc[m, k - 1], (y[m] == k))
    neg, pos = pb[cell == 0], pb[cell == 2]
    for far in (1, 5):
        thr = np.quantile(neg, 1 - far / 100)
        out[f"recall_{far}pct"] = float((pos > thr).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--setting", required=True, choices=["s1", "s2", "s3"])
    ap.add_argument("--method", required=True, choices=["raw", "knn", "ours"])
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = {"train": load(args.src, "train", args.method)}
    tgt = {"train": load(args.tgt, "train", args.method), "test": load(args.tgt, "test", args.method)}
    rows = [run(args.setting, args.method, s, src, tgt, args.shot, args.epochs) for s in range(args.seeds)]
    keys = [k for k in rows[0] if k not in ("setting", "method", "seed", "shot")]
    agg = {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows]))}
           for k in keys}
    Path(args.out).write_text(json.dumps({"setting": args.setting, "method": args.method,
                                          "shot": args.shot, "runs": rows, "agg": agg}, indent=2), encoding="utf-8")
    print(json.dumps({"setting": args.setting, "method": args.method, "shot": args.shot,
                      **{k: round(v["mean"], 3) for k, v in agg.items() if k in
                         ("clean_AP", "full_AP", "recall_1pct", "recall_5pct")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
