#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 21: 物理场重标定（S2+）与原型分类器（S3-proto）。

S2+  ：源域训练，目标域用**无标签**统计量重标定特征（均值/方差）后再推理。
S3   ：源域编码器冻结；用目标域 k-shot 支撑集在编码器上的类原型做余弦分类，
       再与源域双头输出 0.5/0.5 概率融合。
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


def load(d, split, method="ours"):
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
    return dict(X=np.nan_to_num(X), mask=z["mask"], y=z["label_class"], cell=z["label_cell"])


class Net(nn.Module):
    def __init__(self, nf=9, nh=64):
        super().__init__()
        self.c1 = nn.Conv1d(nf, nh, 5, padding=2)
        self.c2 = nn.Conv1d(nh, nh, 5, padding=4, dilation=2)
        self.c3 = nn.Conv1d(nh, nh, 5, padding=8, dilation=4)
        self.bin = nn.Linear(2 * nh, 1)
        self.cls = nn.Linear(2 * nh, 5)

    def encode(self, x):
        h = x.transpose(1, 2)
        h = F.gelu(self.c1(h)); h = F.gelu(self.c2(h)); h = F.gelu(self.c3(h))
        return torch.cat([h.mean(-1), h.max(-1).values], -1)

    def forward(self, x):
        p = self.encode(x)
        return self.bin(p).squeeze(-1), self.cls(p)


def train_source(X, y, cell, dev, epochs=25):
    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(cell, [0, 2])
    yb = (y > 0).astype(np.float32)
    cnt = np.bincount(y[sel & (y > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    for _ in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            pb, pc = model(torch.tensor(X[ii]).to(dev))
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(y[ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = load(args.src, "train")
    tgt_tr = load(args.tgt, "train")
    tgt_te = load(args.tgt, "test")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smu = src["X"][src["mask"]].mean(0, keepdims=True)
    ssd = src["X"][src["mask"]].std(0, keepdims=True) + 1e-6
    tmu = tgt_tr["X"][tgt_tr["mask"]].mean(0, keepdims=True)   # 无标签：只用目标域正常/全部窗口统计
    tsd = tgt_tr["X"][tgt_tr["mask"]].std(0, keepdims=True) + 1e-6

    rows = []
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = train_source((src["X"] - smu) / ssd, src["y"], src["cell"], dev)
        model.eval()
        rng = np.random.default_rng(seed)

        for tag, mu, sd in (("S2_source_norm", smu, ssd), ("S2plus_target_norm", tmu, tsd)):
            Xte = torch.tensor((tgt_te["X"] - mu) / sd).to(dev)
            with torch.no_grad():
                pb, pc = model(Xte)
                pb = torch.sigmoid(pb).cpu().numpy()
                pc = F.softmax(pc, -1).cpu().numpy()
            cell, y = tgt_te["cell"], tgt_te["y"]
            rows.append({"seed": seed, "setting": tag,
                         "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
                         "full_AP": ap_score(pb, y > 0),
                         "recall_1pct": float((pb[cell == 2] > np.quantile(pb[cell == 0], 0.99)).mean())})
            print(f"seed{seed} {tag}: clean={rows[-1]['clean_AP']:.3f} full={rows[-1]['full_AP']:.3f}", flush=True)

        # 原型 S3：目标域 k-shot 支撑集在冻结编码器上的类均值
        for shot in (1, 5):
            idx = []
            for k in range(6):
                pool = np.flatnonzero(tgt_tr["y"] == k) if k > 0 else np.flatnonzero(tgt_tr["cell"] == 0)
                if len(pool):
                    idx.extend(rng.choice(pool, size=min(shot, len(pool)), replace=False).tolist())
            idx = np.array(sorted(set(idx)))
            Xs = torch.tensor((tgt_tr["X"][idx] - tmu) / tsd).to(dev)
            ys = tgt_tr["y"][idx]
            with torch.no_grad():
                Fs = model.encode(Xs).cpu().numpy()
            Fs = Fs / (np.linalg.norm(Fs, axis=1, keepdims=True) + 1e-8)
            cent = np.zeros((6, Fs.shape[1]), dtype=np.float32)
            for k in range(6):
                m = ys == k if k > 0 else np.isin(ys, [0])
                if m.any():
                    cent[k] = Fs[m].mean(0)
            cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)

            Xte = torch.tensor((tgt_te["X"] - tmu) / tsd).to(dev)
            with torch.no_grad():
                Ft = model.encode(Xte).cpu().numpy()
                pb_src, pc_src = model(Xte)
                pb_src = torch.sigmoid(pb_src).cpu().numpy()
                pc_src = F.softmax(pc_src, -1).cpu().numpy()
            Ft = Ft / (np.linalg.norm(Ft, axis=1, keepdims=True) + 1e-8)
            sim = Ft @ cent.T
            proto = np.exp(sim * 8.0); proto = proto / proto.sum(1, keepdims=True)
            pb = 0.5 * pb_src + 0.5 * (1.0 - proto[:, 0])
            cell, y = tgt_te["cell"], tgt_te["y"]
            rows.append({"seed": seed, "setting": f"S3_proto_{shot}shot",
                         "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
                         "full_AP": ap_score(pb, y > 0),
                         "recall_1pct": float((pb[cell == 2] > np.quantile(pb[cell == 0], 0.99)).mean())})
            print(f"seed{seed} proto{shot}: clean={rows[-1]['clean_AP']:.3f} full={rows[-1]['full_AP']:.3f}", flush=True)

    df = {}
    for r in rows:
        df.setdefault(r["setting"], []).append(r)
    summary = {k: {m: round(float(np.nanmean([x[m] for x in v])), 3) for m in ("clean_AP", "full_AP", "recall_1pct")}
               for k, v in df.items()}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
