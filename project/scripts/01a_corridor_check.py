#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 01a: 定义 Houston Ship Channel 走廊，并做覆盖度校验。

走廊 = 到中心线折线的横向距离 <= width_nm；同时给出沿航道坐标 s 与横向坐标 d。
输出覆盖统计 + 密度热图，供人工确认折线是否落在真实航道上。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EARTH_R = 6371000.0
LAT0, LON0 = 29.50, -94.85
NM = 1852.0

# Houston Ship Channel 中心线（Bolivar Roads -> 休斯顿调头区），自南向北
WAYPOINTS = np.array([
    (29.330, -94.715),
    (29.355, -94.760),
    (29.380, -94.820),
    (29.420, -94.880),
    (29.470, -94.920),
    (29.520, -94.950),
    (29.580, -94.975),
    (29.640, -94.995),
    (29.690, -95.010),
    (29.720, -95.040),
    (29.735, -95.075),
    (29.745, -95.100),
])


def to_xy(lat, lon):
    x = EARTH_R * np.radians(lon - LON0) * np.cos(np.radians(LAT0))
    y = EARTH_R * np.radians(lat - LAT0)
    return x, y


def corridor_coords(px, py, poly_xy):
    """返回每个点到折线的 (s 沿程距离, d 横向有符号距离, 最近段切线角)。"""
    n = len(px)
    best_d2 = np.full(n, np.inf)
    s_best = np.zeros(n)
    d_best = np.zeros(n)
    tan_best = np.zeros(n)

    seg_len = np.hypot(np.diff(poly_xy[:, 0]), np.diff(poly_xy[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    for k in range(len(poly_xy) - 1):
        x1, y1 = poly_xy[k]
        x2, y2 = poly_xy[k + 1]
        vx, vy = x2 - x1, y2 - y1
        L2 = vx * vx + vy * vy
        if L2 <= 0:
            continue
        t = ((px - x1) * vx + (py - y1) * vy) / L2
        t = np.clip(t, 0.0, 1.0)
        cx = x1 + t * vx
        cy = y1 + t * vy
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        better = d2 < best_d2
        if better.any():
            best_d2[better] = d2[better]
            s_best[better] = cum[k] + t[better] * seg_len[k]
            # 横向有符号距离（左侧为正）
            cross = ((px[better] - x1) * vy - (py[better] - y1) * vx) / np.sqrt(L2)
            d_best[better] = cross
            tan_best[better] = np.arctan2(vx, vy)
    return s_best, d_best, np.hypot(px - 0, py - 0) * 0 + np.sqrt(best_d2), tan_best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--day", default="2024-07-15")
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["mmsi", "base_datetime", "lat", "lon", "sog", "x_m", "y_m", "in_core_roi"])
    day = pd.Timestamp(args.day, tz="UTC")
    df = df[(df.base_datetime >= day) & (df.base_datetime < day + pd.Timedelta(days=1))]
    mov = df[df.sog >= 2.0]
    print(f"{args.day}: {len(df):,} pts, moving {len(mov):,}")

    poly_xy = np.column_stack(to_xy(WAYPOINTS[:, 0], WAYPOINTS[:, 1]))
    s, d, dist, tan = corridor_coords(mov.x_m.to_numpy(), mov.y_m.to_numpy(), poly_xy)

    for w in (0.5, 1.0, 2.0):
        inside = np.abs(d) <= w * NM
        print(f"  corridor ±{w} nm: {inside.mean():.1%} of moving points, "
              f"s range {s[inside].min()/NM if inside.any() else 0:.1f}-{s[inside].max()/NM if inside.any() else 0:.1f} nm")

    # 沿程分布（每 2 nm 一段的船数）
    hist, edges = np.histogram(s / NM, bins=np.arange(0, 30, 2))
    print("  along-channel moving points per 2 nm:")
    print("   ", dict(zip([f"{edges[i]:.0f}" for i in range(len(hist))], hist.tolist())))

    # 热图 + 折线
    fig, ax = plt.subplots(figsize=(10, 10))
    hb = ax.hexbin(mov.x_m / NM, mov.y_m / NM, gridsize=180, bins="log", cmap="viridis", mincnt=1)
    ax.plot(poly_xy[:, 0] / NM, poly_xy[:, 1] / NM, "-", color="red", lw=1.5, label="corridor centerline")
    for w, ls in ((1.0, "--"), (2.0, ":")):
        ax.plot(poly_xy[:, 0] / NM, poly_xy[:, 1] / NM, ls, color="orange", lw=0.8, alpha=0.8, label=f"±{w} nm")
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm, east of 94.85W)")
    ax.set_ylabel("y (nm, north of 29.50N)")
    ax.set_title(f"Moving-vessel density {args.day} (log scale) + HSC centerline")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="reports")
    fig.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=120)
    print("saved", args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
