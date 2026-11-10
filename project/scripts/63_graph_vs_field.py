#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 63: 图 vs 场——"学习式邻船聚合"能否替代解析式留一法场残差？"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
sp = importlib.util.spec_from_file_location("gf", "/mnt/sda/WangZijia/AIS/scripts/60_graph_fix.py")
gf = importlib.util.module_from_spec(sp); sp.loader.exec_module(gf)


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = sw.load(BASE / "eval_v2_graph2", "train")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    for d, s in ((tr, "train"), (te, "test")):
        d["nb_v"] = np.load(BASE / "eval_v2_graph2" / f"{s}.npz")["nb_v"].astype(np.float32)

    # 无 r_field：只保留 gated(7) + w(1)
    Xtr = np.concatenate([tr["gated"], tr["w"][..., None]], -1)
    Xte = np.concatenate([te["gated"], te["w"][..., None]], -1)
    mu = Xtr[tr["mask"]].mean(0, keepdims=True); sd = Xtr[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

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
            xb = torch.tensor(Xtr[ii]).to(dev)
            pb, pc, _, _ = m(xb, *gf.tensors(tr, ii, dev))
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            mask = torch.tensor(yb[ii] > 0).to(dev)
            if mask.any():
                yy = torch.tensor(tr["y"][ii][mask.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[mask], yy, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()

    m.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte).to(dev)
        pb, _, _, a = m(xb, *gf.tensors(te, np.arange(len(xb)), dev), tau=0.2)
        p = torch.sigmoid(pb).cpu().numpy()
    y = te["y"] > 0
    print("graph-only (no field) AP =", round(ap_score(p, y), 3))
    print("attention entropy =", round(float((-(a.clamp_min(1e-9).log()) * a).sum(-1).mean() / np.log(32)), 3))
    torch.save({"sd": m.state_dict(), "stats": (mu, sd), "kind": "tcn", "nf": int(Xtr.shape[-1])},
               BASE / "vis_weights" / "graph_only_no_field.pt")


if __name__ == "__main__":
    main()
