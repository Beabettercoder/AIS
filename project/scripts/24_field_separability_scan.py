#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 24: 可分离性导向的场参数标定（用合成机动做无标签校准）。

对每个 (R, angle)：
  normal_med = 正常目标船的 |r_field| 中位数
  anom_med   = 把目标船速度矢量旋转 75°（模拟个体机动）后的 |r_field| 中位数
  ratio      = anom_med / normal_med    （越大越能把个体机动从群体流里分出来）
选择准则：coverage >= 0.80 前提下最大化 ratio。
"""
from __future__ import annotations

import argparse
import json
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--n_bins", type=int, default=80)
    ap.add_argument("--rot_deg", type=float, default=75.0)
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
    print(f"zone {args.zone}: {len(zone):,} snapshots / {len(bins)} bins", flush=True)

    rot = np.radians(args.rot_deg)
    acc = {(r, a): {"n": [], "a": [], "valid": 0, "total": 0} for r in RADII for a in ANGLES}

    for _, g in zone.groupby("bin"):
        if len(g) < MIN_NBRS + 1:
            continue
        px, py = g.x_m.to_numpy(), g.y_m.to_numpy()
        vx, vy = g.vx_rep.to_numpy(), g.vy_rep.to_numpy()
        cog = np.arctan2(vx, vy)
        tree = cKDTree(np.column_stack([px, py]))
        balls = tree.query_ball_point(np.column_stack([px, py]), r=max(RADII) * NM)
        # 旋转后的速度（模拟个体机动）
        vx2 = vx * np.cos(rot) - vy * np.sin(rot)
        vy2 = vx * np.sin(rot) + vy * np.cos(rot)

        for (r, a), st in acc.items():
            rr, lim = r * NM, np.radians(a)
            for i, cand in enumerate(balls):
                st["total"] += 1
                c = np.asarray(cand); c = c[c != i]
                if len(c) < MIN_NBRS:
                    continue
                dd = np.hypot(px[c] - px[i], py[c] - py[i])
                keep = c[(dd <= rr) & (np.abs(wrap(cog[c] - cog[i])) <= lim)]
                if len(keep) < MIN_NBRS:
                    continue
                st["valid"] += 1
                dx = (px[keep] - px[i]) / rr; dy = (py[keep] - py[i]) / rr
                X = np.column_stack([np.ones_like(dx), dx, dy])
                try:
                    th, *_ = np.linalg.lstsq(X, np.column_stack([vx[keep], vy[keep]]), rcond=None)
                except np.linalg.LinAlgError:
                    continue
                st["n"].append(np.hypot(vx[i] - th[0, 0], vy[i] - th[0, 1]))
                st["a"].append(np.hypot(vx2[i] - th[0, 0], vy2[i] - th[0, 1]))

    table = []
    for (r, a), st in acc.items():
        n, an = np.asarray(st["n"]), np.asarray(st["a"])
        cov = st["valid"] / max(st["total"], 1)
        if len(n) < 50:
            continue
        table.append({"radius_nm": r, "max_dcog_deg": a, "coverage": round(cov, 4),
                      "normal_med": round(float(np.median(n)), 4),
                      "anom_med": round(float(np.median(an)), 4),
                      "ratio": round(float(np.median(an) / max(np.median(n), 1e-6)), 3)})
    table.sort(key=lambda x: (x["coverage"] < 0.80, -x["ratio"]))
    print(json.dumps({"best": table[0], "table": table}, indent=2))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"best": table[0], "table": table}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
