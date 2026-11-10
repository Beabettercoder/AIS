#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 119: 多源域轮换零样本（每个区域当一次源域，迁移到其余未见区域）。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)

SOURCES = ["ny", "la", "puget", "miami", "gom_open", "atl_open"]
TARGETS = ["la", "mississippi", "honolulu", "miami", "goldengate", "chesapeake", "gom_open", "cpacific"]
MODELS = [("Raw-TCN", False, "tcn"), ("GRU", False, "gru"), ("Ours-TCN", True, "tcn")]


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def gated_of(d):
    g = d["raw"].copy()
    dp = np.diff(g[:, :, 0:2], axis=1, prepend=0.0) * d["w"][..., None]
    g[:, :, 0:2] = np.cumsum(dp, axis=1)
    for ch in (2, 3, 4, 5, 6):
        v = g[:, :, ch]; low = d["w"] < 0.2
        for t in range(1, v.shape[1]):
            m = low[:, t]; v[m, t] = v[m, t - 1]
        g[:, :, ch] = v
    return g


def features(d, ours):
    if not ours:
        return d["raw"]
    return np.concatenate([gated_of(d), d["rf"][..., None], d["w"][..., None]], -1)


def train_model(tr, ours, kind, seed=0, epochs=20):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = features(tr, ours)
    mu = X[tr["mask"]].mean(0, keepdims=True); sd = X[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    torch.manual_seed(seed)
    m = sw.Net(kind, Xn.shape[-1], use_graph=False).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2]); yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    for _ in range(epochs):
        m.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            xb = torch.tensor(Xn[ii]).to(dev)
            pb, pc, _ = m(xb)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            mm = torch.tensor(yb[ii] > 0).to(dev)
            if mm.any():
                yy = torch.tensor(tr["y"][ii][mm.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[mm], yy, weight=cw)
            if ours:
                keep = (torch.rand(xb.shape[0], xb.shape[1], device=dev) > 0.15).float()
                pb2, pc2, _ = m(xb * keep[..., None])
                p1, p2 = F.softmax(pc, -1), F.softmax(pc2, -1)
                mid = 0.5 * (p1 + p2)
                kl = lambda a, bb: (a * (torch.log(a + 1e-8) - torch.log(bb + 1e-8))).sum(-1)
                loss = loss + 0.2 * ((0.5 * kl(p1, mid) + 0.5 * kl(p2, mid)) * keep.mean(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m, (mu, sd), dev


def eval_model(m, stats, ours, d, dev):
    X = ((features(d, ours) - stats[0]) / stats[1]).astype(np.float32)
    with torch.no_grad():
        pb, _, _ = m(torch.tensor(X).to(dev))
        p = torch.sigmoid(pb).cpu().numpy()
    y, cell = d["y"], d["cell"]
    ap = ap_score(p, y > 0)
    thr = np.quantile(p[cell == 0], 0.99)
    r1 = float((p[cell == 2] > thr).mean())
    return ap, r1


def main() -> None:
    out = {}
    for src in SOURCES:
        tr = sw.load(BASE / f"{src}_dataset", "train")
        for name, ours, kind in MODELS:
            m, stats, dev = train_model(tr, ours, kind)
            row = {}
            for tgt in TARGETS:
                if tgt == src:
                    continue
                p = BASE / f"{tgt}_dataset"
                if not (p / "test.npz").exists():
                    continue
                d = sw.load(p, "test")
                ap, r1 = eval_model(m, stats, ours, d, dev)
                row[tgt] = {"ap": round(ap, 3), "r1": round(r1, 3)}
            out.setdefault(name, {})[src] = row
            mean_ap = np.mean([v["ap"] for v in row.values()])
            print(f"src={src:10s} {name:9s} meanAP={mean_ap:.3f} over {len(row)} domains", flush=True)
    (BASE / "multisource_zeroshot.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
