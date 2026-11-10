#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 05: 2x2 评测骨架（v1）。

不训练模型，只验证评测链路与三种"无需训练"的打分器在 2x2 四格上的行为：
  A 单船速度 z-score
  B 未加权邻船均值残差
  C 留一法加权局部线性场残差
输出：每格分数分布 + 行为 AP（C/D vs A/B）、观测 FAR（B/D 的观测分）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NM = 1852.0
RADIUS_M = 8.0 * NM
MIN_NBRS = 4


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def score_window(xy, sog, cog, nb_xy, nb_v, nb_w, nb_mask):
    m = np.isfinite(xy[:, 0]).sum()
    if m < 5:
        return None
    v_t = np.array([sog[m - 1] * 0.514444 * np.sin(cog[m - 1]),
                    sog[m - 1] * 0.514444 * np.cos(cog[m - 1])])
    valid = nb_mask
    if valid.sum() < MIN_NBRS:
        return {"A": float(sog[m - 1] - np.median(sog[:m])), "B": np.nan, "C": np.nan, "T": np.nan, "n_nb": int(valid.sum())}
    px, py = nb_xy[valid, 0], nb_xy[valid, 1]
    vx, vy = nb_v[valid, 0], nb_v[valid, 1]
    w = nb_w[valid]
    cnb = np.arctan2(vx, vy)
    ct = np.arctan2(v_t[0], v_t[1])
    same = np.abs(wrap(cnb - ct)) <= np.pi / 4
    if same.sum() < MIN_NBRS:
        return {"A": float(sog[m - 1] - np.median(sog[:m])), "B": np.nan, "C": np.nan, "T": np.nan, "n_nb": int(same.sum())}

    px, py, vx, vy, w = px[same], py[same], vx[same], vy[same], w[same]
    b = float(np.hypot(v_t[0] - vx.mean(), v_t[1] - vy.mean()))

    X = np.column_stack([np.ones_like(px), px / RADIUS_M, py / RADIUS_M])
    sw = np.sqrt(np.clip(w, 0.05, 1.0))[:, None]
    try:
        theta, *_ = np.linalg.lstsq(X * sw, np.column_stack([vx, vy]) * sw, rcond=None)
        c = float(np.hypot(v_t[0] - theta[0, 0], v_t[1] - theta[0, 1]))
    except np.linalg.LinAlgError:
        c = np.nan
    turn = float(abs(wrap(cog[m - 1] - cog[0]))) * 180.0 / np.pi
    dspd = float((sog[m - 1] - sog[0]) / max(sog[0], 1e-3))
    return {"A": float(sog[m - 1] - np.median(sog[:m])), "B": b, "C": c,
            "T": max(turn / 90.0, abs(dspd)), "n_nb": int(same.sum())}


def ap_score(scores, labels):
    ok = np.isfinite(scores)
    s, y = scores[ok], labels[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float(np.sum(prec * y) / y.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.npz)
    cells = z["label_cell"]
    behav = z["label_behavior"]
    rows = {"A": [], "B": [], "C": [], "T": []}
    nn = []
    for i in range(len(cells)):
        xy = z["xy"][i]
        r = score_window(xy, z["sog_cog"][i, :, 0], z["sog_cog"][i, :, 1],
                         z["nb_xy"][i], z["nb_v"][i], z["nb_w"][i], z["nb_mask"][i])
        if r is None:
            for k in rows:
                rows[k].append(np.nan)
            nn.append(0)
            continue
        for k in rows:
            rows[k].append(r[k])
        nn.append(r["n_nb"])

    rows = {k: np.asarray(v, float) for k, v in rows.items()}
    nn = np.asarray(nn)
    report = {"n": int(len(cells)), "cells": {int(c): int((cells == c).sum()) for c in range(4)}}
    for k in ("A", "B", "C", "T"):
        sc = rows[k]
        report[k] = {
            "mean_by_cell": {int(c): (float(np.nanmean(sc[cells == c])) if np.isfinite(sc[cells == c]).any() else None)
                             for c in range(4)},
            "ap_behavior": ap_score(np.abs(sc) if k == "A" else sc, behav.astype(float)),
        }
    far = {}
    for k in ("A", "B", "C", "T"):
        sc = np.abs(rows[k]) if k == "A" else rows[k]
        clean = sc[(cells == 0)]
        if np.isfinite(clean).sum() > 10:
            thr = np.nanpercentile(clean, 99)
            far[k] = {"threshold": float(thr),
                      "far_A": float(np.nanmean(clean > thr)),
                      "far_B": float(np.nanmean(sc[cells == 1] > thr)),
                      "far_D": float(np.nanmean(sc[cells == 3] > thr))}
    report["operating_point_99pct_on_A"] = far
    report["neighbours"] = {"median": float(np.median(nn)), "frac_lt4": float((nn < 4).mean())}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
