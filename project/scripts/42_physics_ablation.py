#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 42: 纯物理机制基线（无神经网络）在冻结测试集上的判别力。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1/eval_v2_graph2")


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    o = np.argsort(s); ranks = np.empty(len(s)); ranks[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap(s, y):
    s = np.asarray(s, float); y = np.asarray(y, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main() -> None:
    z = np.load(BASE / "test.npz")
    cell, y = z["label_cell"], z["label_class"]
    ep, r_mean, r_field, w = z["e_phys"], z["r_mean"], z["r_field"], z["w_self"]
    T = -1
    scores = {
        "P1 Kinematic Z-score": ep[:, T, 0],
        "P2 Unweighted kNN field": r_mean[:, T],
        "P3 LOO linear field (3.3)": r_field[:, T],
        "P4 Gate + LOO field (3.1+3.3)": r_field[:, T] * (w[:, T] > 0.5),
    }
    rows = []
    for name, s in scores.items():
        obs_y = np.isin(cell, [1, 3])
        beh_y = cell == 2
        clean = np.isin(cell, [0, 2])
        thr = np.quantile(s[cell == 0], 0.95)
        rows.append({"method": name,
                     "auroc_obs": auroc(s, obs_y),
                     "ap_behavior": ap(s[clean], beh_y[clean]),
                     "far_group": float((s[cell == 1] > thr).mean())})
        print(rows[-1], flush=True)
    out = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
    (out / "table3_physics_only.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Table 3 (upper block) — pure physics mechanisms, no neural network", "",
             "| Mechanism | Obs-anomaly AUROC | Behaviour AP | Group-slowdown FAR |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['auroc_obs']:.3f} | {r['ap_behavior']:.3f} | {r['far_group']:.3f} |")
    txt = "\n".join(lines)
    (out / "table3_upper_physics.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
