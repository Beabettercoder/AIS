#!/usr/bin/env python
"""D0 诊断第二步：邻船密度的半径/角度敏感性扫描。

回答：5 nm / ±45° 不够时，改半径还是改角度能救回来？
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARQUET = "/mnt/sda/WangZijia/houston_galveston_2024_07/data/interim/ais_motion.parquet"
OUT = "/mnt/sda/WangZijia/AIS/diagnostics"
DAYS = ["2024-07-08", "2024-07-12", "2024-07-16"]
RADII_NM = [3.0, 5.0, 8.0, 10.0, 15.0]
ANGLES = [30.0, 45.0, 60.0, 90.0]
MIN_SPEED_KN = 2.0
BIN = "5min"
NM = 1852.0


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    frames = []
    for day in DAYS:
        t0 = pd.Timestamp(day, tz="UTC")
        tbl = pq.read_table(
            PARQUET,
            columns=["mmsi", "base_datetime", "x_m", "y_m", "inferred_speed_knots"],
            filters=[("base_datetime", ">=", t0), ("base_datetime", "<", t0 + pd.Timedelta(days=1))],
        )
        frames.append(tbl.to_pandas())
    df = pd.concat(frames, ignore_index=True).sort_values(["mmsi", "base_datetime"], kind="mergesort")

    g = df.groupby("mmsi", sort=False)
    dx = (g["x_m"].shift(-1) - df["x_m"]).to_numpy()
    dy = (g["y_m"].shift(-1) - df["y_m"]).to_numpy()
    step = np.hypot(dx, dy)
    course = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    course[~(step > 20.0)] = np.nan
    df["course"] = course

    moving = df[(df.inferred_speed_knots >= MIN_SPEED_KN) & df.course.notna()].copy()
    moving["bin"] = moving.base_datetime.dt.floor(BIN)
    snap = (
        moving.groupby(["bin", "mmsi"], sort=False)
        .agg(x=("x_m", "mean"), y=("y_m", "mean"), course=("course", "median"),
             speed=("inferred_speed_knots", "mean"))
        .reset_index()
    )

    results = {(r, a): {"counts": [], "bins": 0} for r in RADII_NM for a in ANGLES}
    max_r = max(RADII_NM) * NM
    cell = max_r

    for ts, part in snap.groupby("bin", sort=True):
        xs = part.x.to_numpy(); ys = part.y.to_numpy(); cs = part.course.to_numpy()
        n = len(xs)
        if n < 2:
            continue
        ix = np.floor(xs / cell).astype(np.int64); iy = np.floor(ys / cell).astype(np.int64)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for k in range(n):
            buckets[(ix[k], iy[k])].append(k)

        pi, pj = [], []
        for k in range(n):
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    for j in buckets.get((ix[k] + ox, iy[k] + oy), ()):
                        if j != k:
                            pi.append(k); pj.append(j)
        if not pi:
            continue
        pi = np.asarray(pi); pj = np.asarray(pj)
        d = np.hypot(xs[pj] - xs[pi], ys[pj] - ys[pi])
        dtheta = np.abs((cs[pj] - cs[pi] + 180.0) % 360.0 - 180.0)

        for r in RADII_NM:
            near = d <= r * NM
            for a in ANGLES:
                mask = near & (dtheta <= a)
                cnt = np.bincount(pi[mask], minlength=n)
                results[(r, a)]["counts"].append(cnt)
                results[(r, a)]["bins"] += 1

    rows = []
    for (r, a), v in sorted(results.items()):
        arr = np.concatenate(v["counts"]) if v["counts"] else np.array([0])
        qs = np.percentile(arr, [25, 50, 75, 90]).round(2).tolist()
        rows.append({
            "radius_nm": r, "angle_deg": a, "vessel_snapshots": int(arr.size),
            "median": qs[1], "p25": qs[0], "p75": qs[2], "p90": qs[3],
            "mean": round(float(arr.mean()), 2),
            "frac_lt4": round(float((arr < 4).mean()), 3),
            "frac_lt6": round(float((arr < 6).mean()), 3),
            "frac_zero": round(float((arr == 0).mean()), 3),
        })

    table = pd.DataFrame(rows)
    table.to_csv(f"{OUT}/sweep_radius_angle.csv", index=False)
    print(table.to_string(index=False), flush=True)
    with open(f"{OUT}/sweep_summary.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
