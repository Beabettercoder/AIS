#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 02c: 观测退化耦合版 Fig.2 —— 双面板（纯群体减速 / 群体减速+邻船跳变）。

三种检测器：
  A 单船绝对速度 z-score（相对自身历史中位速度）
  B 未加权 kNN 均值残差（8 nm、同向、留一）——参照系会被跳变邻船污染
  C 本文 8 nm 留一法加权局部线性场（w 由运动学相容残差给出）——自动屏蔽被污染邻船

面板左：只注入群体减速（alpha 1.0->0.5）
面板右：群体减速 + 15% 邻船单点退化（0.5–2 km 位置跳变 + 3–5 倍速度尖峰 + 随机航向）

w 的构造：对每条船，e_kin 由"报告速度 vs 真实（群体缩放后）运动"的差与位置跳变量/典型 Δt 合成；
w = exp(-relu(e_kin - tau)^2 / (2 sigma^2))，与主流程的 w_heur 同一公式。
"""
from __future__ import annotations

import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NM = 1852.0
RADIUS_M = 8.0 * NM
MIN_NBRS = 4
MAX_DCOG = np.radians(45.0)
COND_MAX = 1e3
SPREAD_MIN = 0.15
ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
TAU_KIN, SIGMA_KIN = 2.5, 2.0
DT_TYPICAL = 70.0
CORRUPT_FRAC = 0.15


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def field(px, py, vx, vy, cog, w, target_idx, weighted):
    """返回 targets 的 (res, valid)。weighted=False 时忽略 w（未加权均值残差）。"""
    res = np.full(len(target_idx), np.nan)
    valid = np.zeros(len(target_idx), dtype=bool)
    if len(target_idx) == 0:
        return res, valid
    tree = cKDTree(np.column_stack([px, py]))
    balls = tree.query_ball_point(np.column_stack([px[target_idx], py[target_idx]]), r=RADIUS_M)
    for pos, i in enumerate(target_idx):
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
        keep = w[nbr] > 0.05
        if keep.sum() < MIN_NBRS:
            continue
        nbr = nbr[keep]
        dx = (px[nbr] - px[i]) / RADIUS_M
        dy = (py[nbr] - py[i]) / RADIUS_M
        if np.sqrt(dx.var() + dy.var()) <= SPREAD_MIN:
            continue
        sw = np.sqrt(w[nbr])[:, None]
        X = np.column_stack([np.ones_like(dx), dx, dy]) * sw
        if np.linalg.cond(X) >= COND_MAX:
            continue
        Y = np.column_stack([vx[nbr], vy[nbr]]) * sw
        theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res[pos] = np.hypot(vx[i] - theta[0, 0], vy[i] - theta[0, 1])
        valid[pos] = True
    return res, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--centerline", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    mod = SourceFileLoader("scan", "/mnt/sda/WangZijia/AIS/scripts/01c_corridor.py").load_module()
    poly = mod.load_centerline(args.centerline)

    cols = ["mmsi", "timestamp", "base_datetime", "sog", "cog", "vx_rep", "vy_rep", "x_m", "y_m", "kin_ok"]
    df = pd.read_parquet(args.parquet, columns=cols)
    t0 = df.base_datetime.min()
    df = df[(df.base_datetime < t0 + pd.Timedelta(days=args.days)) & df.kin_ok & df.sog.ge(2.0)]
    df["time_bin"] = (df.timestamp // 600) * 600
    snaps = df.sort_values("timestamp").groupby(["time_bin", "mmsi"], as_index=False).tail(1)

    s, d, _, _ = mod.corridor_coords(snaps.x_m.to_numpy() / NM, snaps.y_m.to_numpy() / NM, poly)
    snaps["s_nm"] = s
    snaps["d_nm"] = d
    snaps = snaps[snaps.d_nm.abs() <= 1.0].reset_index(drop=True)
    base = snaps.groupby("mmsi")["sog"].agg(["median", "count"])
    med = base.loc[base["count"] >= 20, "median"]
    snaps = snaps[snaps.mmsi.isin(med.index)].reset_index(drop=True)
    print(f"[02c] corridor snapshots {len(snaps):,}, bins {snaps.time_bin.nunique()}", flush=True)

    rng = np.random.default_rng(args.seed)
    rows, store = [], {}
    for degrade in (False, True):
        for alpha in ALPHAS:
            scores = {k: [] for k in ("A", "B", "C")}
            for _, grp in snaps.groupby("time_bin"):
                if len(grp) < 12:
                    continue
                n = len(grp)
                v0x, v0y = grp.vx_rep.to_numpy(copy=True), grp.vy_rep.to_numpy(copy=True)
                px, py = grp.x_m.to_numpy(copy=True), grp.y_m.to_numpy(copy=True)

                corrupted = np.zeros(n, dtype=bool)
                if degrade:
                    corrupted = rng.random(n) < CORRUPT_FRAC

                vx = v0x * alpha
                vy = v0y * alpha
                w = np.ones(n)

                if degrade and corrupted.any():
                    spike = rng.uniform(3.0, 5.0, corrupted.sum())
                    ang = rng.uniform(-np.pi, np.pi, corrupted.sum())
                    speed = np.hypot(v0x[corrupted], v0y[corrupted]) * spike
                    vx[corrupted] = speed * np.sin(ang)
                    vy[corrupted] = speed * np.cos(ang)
                    jump = rng.uniform(500.0, 2000.0, corrupted.sum())
                    jang = rng.uniform(-np.pi, np.pi, corrupted.sum())
                    px[corrupted] += jump * np.sin(jang)
                    py[corrupted] += jump * np.cos(jang)

                    e_vel = np.hypot(vx[corrupted] - v0x[corrupted] * alpha,
                                     vy[corrupted] - v0y[corrupted] * alpha)
                    e_pos = jump / DT_TYPICAL
                    e_kin = np.hypot(e_vel, e_pos)
                    w[corrupted] = np.exp(-np.maximum(0.0, e_kin - TAU_KIN) ** 2 / (2 * SIGMA_KIN**2))

                cog = np.arctan2(vx, vy)
                tgt = np.flatnonzero(~corrupted)

                ref = grp.mmsi.map(med).to_numpy()
                sog_new = np.hypot(vx, vy) / 0.514444
                scores["A"].extend((sog_new[tgt] / ref[tgt] - 1.0).tolist())

                r_b, ok_b = field(px, py, vx, vy, cog, w, tgt, weighted=False)
                r_c, ok_c = field(px, py, vx, vy, cog, w, tgt, weighted=True)
                scores["B"].extend(r_b[ok_b].tolist())
                scores["C"].extend(r_c[ok_c].tolist())

            for k in ("A", "B", "C"):
                store[(degrade, alpha, k)] = np.asarray(scores[k], dtype=float)
            print(f"[02c] degrade={degrade} alpha={alpha:.1f} n={len(scores['A'])}", flush=True)

    thr = {k: float(np.percentile(store[(False, 1.0, k)], 1 if k == "A" else 99)) for k in ("A", "B", "C")}
    print("[02c] thresholds", {k: round(v, 3) for k, v in thr.items()}, flush=True)

    for degrade in (False, True):
        for alpha in ALPHAS:
            row = {"panel": "degraded" if degrade else "clean", "alpha": alpha}
            for k in ("A", "B", "C"):
                arr = store[(degrade, alpha, k)]
                arr = arr[np.isfinite(arr)]
                row[f"{k}_far"] = float((arr < thr[k]).mean() if k == "A" else (arr > thr[k]).mean())
                row[f"{k}_n"] = int(arr.size)
            rows.append(row)

    res = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_csv, index=False)
    print(res.round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
