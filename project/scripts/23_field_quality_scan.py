#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 23: 质量导向的场参数标定（无标签）。

对每个 (radius, max_dcog) 组合，在目标域正常流量上拟合留一法局部线性场，统计：
  coverage   : 有 >=4 同向邻船的目标占比
  med_|r|    : 残差中位数（越小越好）
  iqr_|r|    : 残差四分位距（越小越稳）
  rho_d      : |r| 与"离中心线距离 d"的 Spearman 相关（越接近 0 越好）
选择准则：coverage >= 0.80 前提下，最小化 iqr_|r|。
"""
from __future__ import annotations

import argparse
import json
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

NM = 1852.0
RADII = [4, 6, 8, 10]
ANGLES = [30, 45, 60, 75]
MIN_NBRS = 4


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--centerline", required=True)
    ap.add_argument("--zone", default="b")
    ap.add_argument("--s_cut", type=float, default=16.0)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--n_bins", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corr = SourceFileLoader("scan", "/mnt/sda/WangZijia/AIS/scripts/01c_corridor.py").load_module()
    poly = corr.load_centerline(args.centerline)

    df = pd.read_parquet(args.parquet, columns=["mmsi", "timestamp", "x_m", "y_m", "sog", "cog",
                                                "vx_rep", "vy_rep", "kin_ok"])
    t0 = df.timestamp.min()
    df = df[(df.timestamp < t0 + args.days * 86400) & df.kin_ok & df.sog.ge(2.0)]
    df["bin"] = (df.timestamp // 600) * 600
    snaps = df.sort_values("timestamp").groupby(["bin", "mmsi"], as_index=False).tail(1)
    s, d, _, _ = corr.corridor_coords(snaps.x_m.to_numpy() / NM, snaps.y_m.to_numpy() / NM, poly)
    snaps["s_nm"], snaps["d_nm"] = s, d
    zone = snaps[(snaps.d_nm.abs() <= 1.0) & ((snaps.s_nm > args.s_cut) if args.zone == "b" else (snaps.s_nm <= args.s_cut))]
    bins = sorted(zone.bin.unique())
    rng = np.random.default_rng(0)
    if len(bins) > args.n_bins:
        bins = rng.choice(bins, size=args.n_bins, replace=False)
    zone = zone[zone.bin.isin(bins)]
    print(f"zone {args.zone}: {len(zone):,} snapshots over {len(bins)} bins", flush=True)

    stats = {(r, a): {"res": [], "valid": 0, "total": 0, "d": []} for r in RADII for a in ANGLES}
    for _, g in zone.groupby("bin"):
        if len(g) < MIN_NBRS + 1:
            continue
        px, py = g.x_m.to_numpy(), g.y_m.to_numpy()
        vx, vy = g.vx_rep.to_numpy(), g.vy_rep.to_numpy()
        cog = np.arctan2(vx, vy)
        tree = cKDTree(np.column_stack([px, py]))
        maxr = max(RADII) * NM
        balls = tree.query_ball_point(np.column_stack([px, py]), r=maxr)
        for (r, a), acc in stats.items():
            rr, lim = r * NM, np.radians(a)
            for i, cand in enumerate(balls):
                acc["total"] += 1
                c = np.asarray(cand)
                c = c[c != i]
                if len(c) < MIN_NBRS:
                    continue
                dd = np.hypot(px[c] - px[i], py[c] - py[i])
                keep = c[(dd <= rr) & (np.abs(wrap(cog[c] - cog[i])) <= lim)]
                if len(keep) < MIN_NBRS:
                    continue
                acc["valid"] += 1
                dx = (px[keep] - px[i]) / rr
                dy = (py[keep] - py[i]) / rr
                X = np.column_stack([np.ones_like(dx), dx, dy])
                try:
                    th, *_ = np.linalg.lstsq(X, np.column_stack([vx[keep], vy[keep]]), rcond=None)
                except np.linalg.LinAlgError:
                    continue
                acc["res"].append(np.hypot(vx[i] - th[0, 0], vy[i] - th[0, 1]))
                acc["d"].append(abs(g.d_nm.iloc[i]))

    table = []
    for (r, a), acc in stats.items():
        res = np.asarray(acc["res"])
        cov = acc["valid"] / max(acc["total"], 1)
        if len(res) < 50:
            continue
        rho = spearmanr(res, np.asarray(acc["d"]))[0]
        table.append({"radius_nm": r, "max_dcog_deg": a, "coverage": round(cov, 4),
                      "n": int(len(res)), "med_abs_r": round(float(np.median(res)), 4),
                      "iqr_abs_r": round(float(np.percentile(res, 75) - np.percentile(res, 25)), 4),
                      "rho_d": round(float(rho), 4)})
    table.sort(key=lambda x: (x["coverage"] < 0.80, x["iqr_abs_r"]))
    best = table[0]
    print(json.dumps({"zone": args.zone, "best": best, "table": table}, indent=2))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"zone": args.zone, "best": best, "table": table}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
