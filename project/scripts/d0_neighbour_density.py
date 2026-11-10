#!/usr/bin/env python
"""D0 前置诊断：在 Houston/Galveston 真实数据上跑三张分布图（PDF §12-1）。

输入：maritime_case 产出的 ais_motion.parquet（已投影、已算 inferred_speed）。
输出：JSON 摘要 + 两个 CSV（邻船数分布、Δt 分布）。

不做任何训练，只回答一个问题：5 nm 内同向 ±45° 的邻船够不够（中位数 < 4 则背景场方案不成立）。
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARQUET = "/mnt/sda/WangZijia/houston_galveston_2024_07/data/interim/ais_motion.parquet"
OUT = "/mnt/sda/WangZijia/AIS/diagnostics"
DAYS = ["2024-07-08", "2024-07-12", "2024-07-16"]
RADIUS_M = 5 * 1852.0          # 5 nm
ANGLE_DEG = 45.0
MIN_SPEED_KN = 2.0
BIN = "5min"


def wrap_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def bearing_deg(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def main() -> int:
    import os

    os.makedirs(OUT, exist_ok=True)
    frames = []
    for day in DAYS:
        t0 = pd.Timestamp(day, tz="UTC")
        t1 = t0 + pd.Timedelta(days=1)
        tbl = pq.read_table(
            PARQUET,
            columns=["mmsi", "base_datetime", "x_m", "y_m", "delta_seconds", "inferred_speed_knots"],
            filters=[("base_datetime", ">=", t0), ("base_datetime", "<", t1)],
        )
        df = tbl.to_pandas()
        print(f"{day}: {len(df):,} rows, {df.mmsi.nunique():,} vessels", flush=True)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["mmsi", "base_datetime"], kind="mergesort")

    # ---- 航向：相邻两点方位角 ----
    g = df.groupby("mmsi", sort=False)
    dx = g["x_m"].shift(-1) - df["x_m"]
    dy = g["y_m"].shift(-1) - df["y_m"]
    step = np.hypot(dx.to_numpy(), dy.to_numpy())
    course = (np.degrees(np.arctan2(dx.to_numpy(), dy.to_numpy())) + 360.0) % 360.0
    course[~(step > 20.0)] = np.nan          # 位移太小时方位角没有意义
    df["course"] = course

    # ---- Δt 分布 ----
    dt = df["delta_seconds"].dropna()
    dt = dt[(dt > 0) & (dt < 7200)]
    q = np.percentile(dt, [5, 25, 50, 75, 95]).round(1).tolist()
    dt_hist = np.histogram(dt, bins=[0, 10, 30, 60, 120, 300, 600, 1200, 7200])[0].tolist()
    print("dt percentiles p5/p25/p50/p75/p95:", q, flush=True)

    # ---- 轨迹长度（每船每天点数） ----
    per_day = df.assign(day=df.base_datetime.dt.date).groupby(["mmsi", "day"]).size()
    len_q = np.percentile(per_day, [10, 50, 90]).round(0).tolist()
    print("points per vessel-day p10/p50/p90:", len_q, flush=True)

    # ---- 5 分钟快照 ----
    moving = df[(df.inferred_speed_knots >= MIN_SPEED_KN) & df.course.notna()].copy()
    moving["bin"] = moving.base_datetime.dt.floor(BIN)
    snap = (
        moving.groupby(["bin", "mmsi"], sort=False)
        .agg(x=("x_m", "mean"), y=("y_m", "mean"),
             course=("course", "median"), speed=("inferred_speed_knots", "mean"))
        .reset_index()
    )
    print("snapshot rows:", len(snap), "bins:", snap.bin.nunique(), flush=True)

    counts_all: list[int] = []
    per_bin: list[dict] = []
    cell = RADIUS_M
    for ts, part in snap.groupby("bin", sort=True):
        xs = part.x.to_numpy(); ys = part.y.to_numpy(); cs = part.course.to_numpy()
        n = len(xs)
        if n < 2:
            continue
        ix = np.floor(xs / cell).astype(np.int64)
        iy = np.floor(ys / cell).astype(np.int64)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for k in range(n):
            buckets[(ix[k], iy[k])].append(k)

        neighbours = np.zeros(n, dtype=np.int32)
        for k in range(n):
            cnt = 0
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    for j in buckets.get((ix[k] + ox, iy[k] + oy), ()):
                        if j == k:
                            continue
                        d = math.hypot(xs[j] - xs[k], ys[j] - ys[k])
                        if d <= RADIUS_M and abs(wrap_deg(cs[j] - cs[k])) <= ANGLE_DEG:
                            cnt += 1
            neighbours[k] = cnt
        counts_all.extend(neighbours.tolist())
        per_bin.append({
            "bin": str(ts), "vessels": int(n),
            "median": float(np.median(neighbours)),
            "frac_lt4": float((neighbours < 4).mean()),
        })

    arr = np.asarray(counts_all)
    summary = {
        "days": DAYS,
        "radius_nm": 5.0,
        "angle_deg": ANGLE_DEG,
        "min_speed_kn": MIN_SPEED_KN,
        "snapshots": len(per_bin),
        "vessel_snapshots": int(arr.size),
        "neighbour_quantiles": dict(zip(
            ["p05", "p10", "p25", "p50", "p75", "p90", "p95"],
            np.percentile(arr, [5, 10, 25, 50, 75, 90, 95]).round(2).tolist(),
        )),
        "mean": round(float(arr.mean()), 2),
        "frac_lt4": round(float((arr < 4).mean()), 3),
        "frac_lt6": round(float((arr < 6).mean()), 3),
        "frac_zero": round(float((arr == 0).mean()), 3),
        "dt_seconds_percentiles": dict(zip(["p05", "p25", "p50", "p75", "p95"], q)),
        "dt_histogram_edges": [0, 10, 30, 60, 120, 300, 600, 1200, 7200],
        "dt_histogram": dt_hist,
        "points_per_vessel_day": dict(zip(["p10", "p50", "p90"], len_q)),
    }

    pd.DataFrame({"per_vessel_neighbours": arr}).to_csv(f"{OUT}/neighbour_counts.csv", index=False)
    pd.DataFrame(per_bin).to_csv(f"{OUT}/per_bin.csv", index=False)
    with open(f"{OUT}/d0_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
