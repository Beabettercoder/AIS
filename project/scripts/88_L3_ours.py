#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 88: L3——MMSI 级划分下训练并评测我们的模型（与 LLM 同 200 窗）。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
D = BASE / "eval_v2_mmsi"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
sp = importlib.util.spec_from_file_location("gf", "/mnt/sda/WangZijia/AIS/scripts/60_graph_fix.py")
gf = importlib.util.module_from_spec(sp); sp.loader.exec_module(gf)


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = sw.load(D, "train"); te = sw.load(D, "test")
    for d, s in ((tr, "train"), (te, "test")):
        d["nb_v"] = np.load(D / f"{s}.npz")["nb_v"].astype(np.float32)
    Xtr = np.concatenate([tr["gated"], tr["rf"][..., None], tr["w"][..., None]], -1)
    Xte = np.concatenate([te["gated"], te["rf"][..., None], te["w"][..., None]], -1)
    mu = Xtr[tr["mask"]].mean(0, keepdims=True); sd = Xtr[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).astype(np.float32); Xte = ((Xte - mu) / sd).astype(np.float32)

    torch.manual_seed(0); np.random.seed(0)
    m = gf.Net2("tcn", Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2]); yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    for _ in range(25):
        m.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 256):
            ii = idx[b:b + 256]
            pb, pc, _, _ = m(torch.tensor(Xtr[ii]).to(dev), *gf.tensors(tr, ii, dev))
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            mm = torch.tensor(yb[ii] > 0).to(dev)
            if mm.any():
                yy = torch.tensor(tr["y"][ii][mm.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[mm], yy, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()

    m.eval()
    with torch.no_grad():
        pb, _, _, _ = m(torch.tensor(Xte).to(dev), *gf.tensors(te, np.arange(len(Xte)), dev))
        p = torch.sigmoid(pb).cpu().numpy()
    y = (te["y"] > 0).astype(int)

    # 与 LLM 完全相同的 200 窗
    sub = np.random.default_rng(0).choice(len(p), size=200, replace=False)
    res = {"setting": "ours, MMSI-level split", "AP_full_test": round(ap_score(p, y), 3),
           "AP_200_subset": round(ap_score(p[sub], y[sub]), 3), "pos_rate_200": float(y[sub].mean())}
    print(res)
    (BASE / "L3_ours_mmsi.json").write_text(json.dumps(res, indent=2))
    np.save(BASE / "L3_subset_idx.npy", sub)


if __name__ == "__main__":
    main()
