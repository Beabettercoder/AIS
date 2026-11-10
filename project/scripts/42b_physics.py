#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 42b: 纯物理机制基线（修正口径：门控可信度 + 多步聚合）。"""
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
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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
    cell = z["label_cell"]
    w, r_field, r_mean = z["w_self"], z["r_field"], z["r_mean"]

    obs_score = 1.0 - w.min(1)                     # 最大运动学残差
    rf_mean = r_field.mean(1)
    rf_p90 = np.percentile(r_field, 90, axis=1)
    rm_p90 = np.percentile(r_mean, 90, axis=1)
    w_mean = w.mean(1)

    scores = {
        "P1 Kinematic gate (3.1 only)": obs_score,
        "P2 Unweighted kNN field (P90)": rm_p90,
        "P3 LOO linear field (3.3, P90)": rf_p90,
        "P4 Gate + LOO field (3.1+3.3, P90)": rf_p90 * (w_mean > 0.5),
        "P5 LOO field (3.3, mean)": rf_mean,
        "P6 Gate + LOO (3.1+3.3, mean)": rf_mean * (w_mean > 0.5),
    }
    obs_y = np.isin(cell, [1, 3])
    beh_y = cell == 2
    clean = np.isin(cell, [0, 2])
    rows = []
    for name, s in scores.items():
        thr = np.quantile(s[cell == 0], 0.95)
        rows.append({"method": name, "auroc_obs": auroc(s, obs_y),
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
