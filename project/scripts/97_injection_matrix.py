#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 97: 平滑 vs 硬注入——交叉泛化矩阵 + 物理分布对比。"""
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


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def feats(d):
    X = np.concatenate([d["gated"], d["rf"][..., None], d["w"][..., None]], -1)
    return X


def train_eval(src_dir, tgt_dir, seed=0, epochs=25):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = sw.load(src_dir, "train"); te = sw.load(tgt_dir, "test")
    Xtr = feats(tr); Xte = feats(te)
    mu = Xtr[tr["mask"]].mean(0, keepdims=True); sd = Xtr[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).astype(np.float32); Xte = ((Xte - mu) / sd).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    m = sw.Net("tcn", Xtr.shape[-1], use_graph=False).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2]); yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    for _ in range(epochs):
        m.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 256):
            ii = idx[b:b + 256]
            xb = torch.tensor(Xtr[ii]).to(dev)
            pb, pc, _ = m(xb)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            mm = torch.tensor(yb[ii] > 0).to(dev)
            if mm.any():
                yy = torch.tensor(tr["y"][ii][mm.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[mm], yy, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pb, _, _ = m(torch.tensor(Xte).to(dev))
        p = torch.sigmoid(pb).cpu().numpy()
    return ap_score(p, te["y"] > 0)


def main() -> None:
    smooth = BASE / "eval_v2_graph2"
    crude = BASE / "eval_crude"
    res = {}
    res["smooth->smooth"] = train_eval(smooth, smooth)
    res["smooth->crude"] = train_eval(smooth, crude)
    res["crude->crude"] = train_eval(crude, crude)
    res["crude->smooth"] = train_eval(crude, smooth)
    print(res)
    (BASE / "injection_matrix.json").write_text(json.dumps(res, indent=2))

    # 运动学残差分布（真实正常 vs 平滑异常 vs 硬注入）
    zs = np.load(smooth / "test.npz"); zc = np.load(crude / "test.npz")
    def ekin(z, mask_cells):
        g = z["gated"]; m = z["mask"]
        pos = np.cumsum(g[:, :, 0:2], axis=1)
        acc = np.linalg.norm(np.diff(pos, n=2, axis=1), axis=-1) / (70.0 ** 2)
        return acc[z["mask"][:, :-2] & np.isin(z["label_cell"], mask_cells)[:, None]]
    dist = {"real_normal": ekin(zs, [0]), "smooth_anom": ekin(zs, [2]), "crude_anom": ekin(zc, [2])}
    for k, v in dist.items():
        print(k, "median", float(np.median(v)), "p95", float(np.percentile(v, 95)))
    np.savez(BASE / "ekin_dist.npz", **dist)


if __name__ == "__main__":
    main()
