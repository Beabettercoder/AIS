#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 02: Fig.2 原型——三种检测器在"群体减速"下的误报率（FAR）曲线。

三种检测器：
  A 单船绝对速度 z-score（相对该船自身历史中位速度）
  B 未加权 kNN 均值残差（8 nm、同向、留一）
  C 本文 8 nm 留一法加权局部线性场残差（分层：场有效 / 回退）

注入两种群体减速：
  uniform  ——走廊内所有船同乘 alpha（含目标船）
  gradient ——沿航道 s0 附近的高斯"排队"，alpha 随 s 变化（模拟队尾减速、队首畅通）

FAR 定义：在"无异常"的窗口里，被判为异常的目标船占比；阈值取 alpha=1.0 时各检测器分数的 99 分位。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NM = 1852.0
RADIUS_M = 8.0 * NM
SIGMA_D = 4.0 * NM
MIN_NBRS = 4
MAX_DCOG = np.radians(45.0)
COND_MAX = 1e3
SPREAD_MIN = 0.15
ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def loo_field(px, py, vx, vy, cog, mask_target, weighted=True):
    """返回每个目标的 (res, valid)；weighted=False 时退化为等权均值残差。"""
    n = int(mask_target.sum())
    res = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return res, valid
    tree = cKDTree(np.column_stack([px, py]))
    idx = np.flatnonzero(mask_target)
    balls = tree.query_ball_point(np.column_stack([px[idx], py[idx]]), r=RADIUS_M)

    for pos, i in enumerate(idx):
        cand = np.asarray(balls[pos], dtype=np.int64)
        if cand.size < MIN_NBRS + 1:
            continue
        dcog = np.abs(wrap(cog[cand] - cog[i]))
        nbr = cand[(cand != i) & (dcog <= MAX_DCOG)]
        if nbr.size < MIN_NBRS:
            continue

        if not weighted:
            res[pos] = np.hypot(vx[i] - vx[nbr].mean(), vy[i] - vy[nbr].mean())
            valid[pos] = True
            continue

        dx = (px[nbr] - px[i]) / RADIUS_M
        dy = (py[nbr] - py[i]) / RADIUS_M
        if np.sqrt(dx.var() + dy.var()) <= SPREAD_MIN:
            continue
        X = np.column_stack([np.ones_like(dx), dx, dy])
        if np.linalg.cond(X) >= COND_MAX:
            continue
        Y = np.column_stack([vx[nbr], vy[nbr]])
        theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res[pos] = np.hypot(vx[i] - theta[0, 0], vy[i] - theta[0, 1])
        valid[pos] = True
    return res, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--centerline", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--s0_nm", type=float, default=24.0)
    ap.add_argument("--sigma_nm", type=float, default=4.0)
    args = ap.parse_args()

    from importlib.machinery import SourceFileLoader
    mod = SourceFileLoader("scan", "/mnt/sda/WangZijia/AIS/scripts/01c_corridor_scan.py").load_module()
    poly = mod.load_centerline(args.centerline)

    cols = ["mmsi", "timestamp", "base_datetime", "sog", "cog", "vx_rep", "vy_rep",
            "x_m", "y_m", "kin_ok", "w_heur"]
    df = pd.read_parquet(args.parquet, columns=cols)
    t0 = df.base_datetime.min()
    df = df[(df.base_datetime < t0 + pd.Timedelta(days=args.days)) & df.kin_ok & df.sog.ge(2.0)].copy()
    df["time_bin"] = (df.timestamp // 600) * 600
    snaps = df.sort_values("timestamp").groupby(["time_bin", "mmsi"], as_index=False).tail(1)

    s, d, _, _ = mod.corridor_coords(snaps.x_m.to_numpy() / NM, snaps.y_m.to_numpy() / NM, poly)
    snaps["s_nm"] = s
    snaps["d_nm"] = d
    snaps = snaps[snaps.d_nm.abs() <= 1.0].reset_index(drop=True)
    print(f"[02] corridor snapshots {len(snaps):,} bins {snaps.time_bin.nunique()}", flush=True)

    # 每船历史中位速度（用于检测器 A）
    base = snaps.groupby("mmsi")["sog"].agg(["median", "count"])
    base = base[base["count"] >= 20]
    med = base["median"]

    rows = []
    store = {}
    for mode in ("uniform", "gradient"):
        for alpha in ALPHAS:
            if mode == "gradient" and alpha == 1.0:
                continue
            scores = {k: [] for k in ("A", "B", "C")}
            for t_bin, grp in snaps.groupby("time_bin"):
                if len(grp) < 10:
                    continue
                a = np.ones(len(grp)) * alpha
                if mode == "gradient":
                    a = 1.0 - (1.0 - alpha) * np.exp(-((grp.s_nm.to_numpy() - args.s0_nm) ** 2) / (2 * args.sigma_nm**2))
                vx = grp.vx_rep.to_numpy() * a
                vy = grp.vy_rep.to_numpy() * a
                sog_new = grp.sog.to_numpy() * a
                px, py = grp.x_m.to_numpy(), grp.y_m.to_numpy()
                cog = np.arctan2(vx, vy)

                ref = grp.mmsi.map(med).to_numpy()
                ok_a = np.isfinite(ref)
                scores["A"].extend((sog_new[ok_a] / ref[ok_a] - 1.0).tolist())

                tgt = np.ones(len(grp), dtype=bool)
                r_b, ok_b = loo_field(px, py, vx, vy, cog, tgt, weighted=False)
                r_c, ok_c = loo_field(px, py, vx, vy, cog, tgt, weighted=True)
                scores["B"].extend(r_b[ok_b].tolist())
                scores["C"].extend(r_c[ok_c].tolist())

            out = {"mode": mode, "alpha": alpha}
            for k in ("A", "B", "C"):
                arr = np.asarray(scores[k], dtype=float)
                arr = arr[np.isfinite(arr)]
                store[(mode, alpha, k)] = arr
            rows.append(out)
            print(f"[02] {mode} alpha={alpha:.1f} done, n={len(store[(mode, alpha, 'A')])}", flush=True)

    # 阈值：uniform/alpha=1.0 时各检测器分数的 99 分位（单尾；A 为负向异常，取 1 分位）
    thr = {}
    for k in ("A", "B", "C"):
        arr = store[("uniform", 1.0, k)]
        thr[k] = float(np.percentile(arr, 1 if k == "A" else 99)) if arr.size else np.nan
    print("[02] thresholds:", {k: round(v, 4) for k, v in thr.items()}, flush=True)

    for out in rows:
        mode, alpha = out["mode"], out["alpha"]
        for k in ("A", "B", "C"):
            arr = store[(mode, alpha, k)]
            if arr.size == 0:
                out[f"{k}_far"] = np.nan
                continue
            out[f"{k}_far"] = float((arr < thr[k]).mean() if k == "A" else (arr > thr[k]).mean())
        out["n"] = int(store[(mode, alpha, "C")].size)
    res = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_csv, index=False)
    print(res.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
