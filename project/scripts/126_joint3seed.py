#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 122: 四区域联合训练 -> 8 个未见区域零样本（S2 源域统计 / S2+ 目标域重标定）。"""
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

SRC = [BASE / "eval_v2_graph2", BASE / "ny_dataset", BASE / "puget_dataset", BASE / "atl_open_dataset"]
TGT = ["la", "mississippi", "honolulu", "miami", "goldengate", "chesapeake", "gom_open", "cpacific"]
MODELS = [("Raw-TCN", "tcn", False), ("GRU", "gru", False), ("Transformer", "transformer", False),
          ("Ours-TCN", "tcn", True), ("Ours-GRU", "gru", True)]


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def merge(dirs):
    parts = [sw.load(p, "train") for p in dirs]
    out = {}
    for k in ("raw", "gated", "rf", "w", "mask", "y", "cell"):
        out[k] = np.concatenate([p[k] for p in parts], 0)
    return out


def build_x(d, ours):
    if not ours:
        return d["raw"]
    return np.concatenate([d["gated"], d["rf"][..., None], d["w"][..., None]], -1)


def train(tr, kind, ours, seed=0, epochs=20):
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = build_x(tr, ours)
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
        for b in range(0, len(idx), 192):
            ii = idx[b:b + 192]
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


def evaluate(m, ours, stats, d, dev):
    X = build_x(d, ours)
    res = {}
    for tag, (mu, sd) in (("S2", stats),
                          ("S2p", (X[d["mask"]].mean(0, keepdims=True),
                                   X[d["mask"]].std(0, keepdims=True) + 1e-6))):
        with torch.no_grad():
            xb = torch.tensor(((X - mu) / sd).astype(np.float32), device=dev)
            pb, _, _ = m(xb)
            p = torch.sigmoid(pb).cpu().numpy()
        ap = ap_score(p, d["y"] > 0)
        norm = p[d["cell"] == 0]; pos = p[d["cell"] == 2]
        r1 = float((pos > np.quantile(norm, 0.99)).mean())
        r05 = float((pos > np.quantile(norm, 0.995)).mean())
        res[tag] = (round(ap, 3), round(r1, 3), round(r05, 3))
    return res


def main() -> None:
    tr = merge(SRC)
    print("merged train:", tr["raw"].shape, flush=True)
    out = {}
    for seed in (0, 1, 2):
        for name, kind, ours in MODELS:
            m, stats, dev = train(tr, kind, ours, seed=seed)
            row = {}
            for t in TGT:
                p = BASE / f"{t}_dataset"
                if not (p / "test.npz").exists():
                    continue
                row[t] = evaluate(m, ours, stats, sw.load(p, "test"), dev)
            out.setdefault(name, {})[str(seed)] = row
            aps = [v["S2"][0] for v in row.values()]; aps2 = [v["S2p"][0] for v in row.values()]
            print(f"seed={seed} {name:12s} S2={np.mean(aps):.3f} S2+={np.mean(aps2):.3f}", flush=True)
            del m
            torch.cuda.empty_cache()
        (BASE / "joint4_3seed.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
