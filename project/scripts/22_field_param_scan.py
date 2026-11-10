#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 22: 目标域场参数自适应扫描（无标签）。

对给定 zone（默认 b），扫描 (radius, max_dcog) 组合，统计"场有效率"
（同向邻船 >= 4 的目标船占比）。选有效率最高且半径不过大的参数，
用于目标域物理场重算。
"""
from __future__ import annotations

import argparse
import json
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NM = 1852.0
RADII = [4, 6, 8, 10, 12]
ANGLES = [30, 45, 60, 75]
MIN_NBRS = 4


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--centerline", required=True)
    ap.add_argument("--zone", default="b", choices=["a", "b"])
    ap.add_argument("--s_cut", type=float, default=16.0)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corr = SourceFileLoader("scan", "/mnt/sda/WangZijia/AIS/scripts/01c_corridor.py").load_module()
    poly = corr.load_centerline(args.centerline)

    df = pd.read_parquet(args.parquet, columns=["mmsi", "timestamp", "x_m", "y_m", "sog", "cog", "kin_ok"])
    t0 = df.timestamp.min()
    df = df[(df.timestamp < t0 + args.days * 86400) & df.kin_ok & df.sog.ge(2.0)]
    df["bin"] = (df.timestamp // 600) * 600
    snaps = df.sort_values("timestamp").groupby(["bin", "mmsi"], as_index=False).tail(1)
    s, d, _, _ = corr.corridor_coords(snaps.x_m.to_numpy() / NM, snaps.y_m.to_numpy() / NM, poly)
    snaps["s_nm"], snaps["d_nm"] = s, d
    snaps = snaps[snaps.d_nm.abs() <= 1.0]
    zone = snaps[(snaps.s_nm > args.s_cut)] if args.zone == "b" else snaps[(snaps.s_nm <= args.s_cut)]
    print(f"zone {args.zone}: {len(zone):,} snapshots in {zone.bin.nunique()} bins", flush=True)

    max_r = max(RADII) * NM
    res = {f"r{r}_a{a}": [0, 0] for r in RADII for a in ANGLES}   # [valid, total]
    for _, g in zone.groupby("bin"):
        if len(g) < MIN_NBRS + 1:
            continue
        pos = np.column_stack([g.x_m, g.y_m])
        cog = np.radians(g.cog.to_numpy())
        tree = cKDTree(pos)
        balls = tree.query_ball_point(pos, r=max_r)
        for r in RADII:
            rr = r * NM
            for a in ANGLES:
                lim = np.radians(a)
                ok = 0
                for i, cand in enumerate(balls):
                    if len(cand) < MIN_NBRS + 1:
                        continue
                    c = np.asarray(cand)
                    c = c[c != i]
                    if len(c) < MIN_NBRS:
                        continue
                    dd = np.hypot(pos[c, 0] - pos[i, 0], pos[c, 1] - pos[i, 1])
                    keep = c[dd <= rr]
                    if len(keep) < MIN_NBRS:
                        continue
                    if (np.abs(wrap(cog[keep] - cog[i])) <= lim).sum() >= MIN_NBRS:
                        ok += 1
                key = f"r{r}_a{a}"
                res[key][0] += ok
                res[key][1] += len(g)

    table = [{"radius_nm": r, "max_dcog_deg": a,
              "valid_ratio": round(res[f"r{r}_a{a}"][0] / max(res[f"r{r}_a{a}"][1], 1), 4)}
             for r in RADII for a in ANGLES]
    best = max(table, key=lambda x: (x["valid_ratio"], -x["radius_nm"]))
    print(json.dumps({"zone": args.zone, "table": table, "best": best}, indent=2))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"zone": args.zone, "table": table, "best": best}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
