#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 01b: 从数据自身拟合航道中心线（密度脊线），替代手写航路点。

做法：对在航点做 2D 直方图（0.25 nm 网格），逐 y 取密度最大 x 形成脊线，
中值滤波后作为中心线；输出 JSON 供后续走廊扫描复用，并画图校验。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NM = 1852.0
BIN_NM = 0.25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--lat0", type=float, default=29.50)
    ap.add_argument("--lon0", type=float, default=-94.85)
    ap.add_argument("--extent_nm", type=float, default=26.0)
    ap.add_argument("--y_range", type=float, nargs=2, default=(-12.0, 15.0))
    ap.add_argument("--x_search", type=float, nargs=2, default=(-22.0, 12.0))
    ap.add_argument("--track_step", type=float, default=2.5)
    ap.add_argument("--local_nm", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["base_datetime", "x_m", "y_m", "sog"])
    t0 = df.base_datetime.min()
    df = df[(df.base_datetime < t0 + pd.Timedelta(days=args.days)) & (df.sog >= 2.0)]
    x = df.x_m.to_numpy() / NM
    y = df.y_m.to_numpy() / NM
    print(f"moving points: {len(x):,}")

    ext = args.extent_nm
    xb = np.arange(-ext, ext + BIN_NM, BIN_NM)
    yb = np.arange(-ext, ext + BIN_NM, BIN_NM)
    H, xe, ye = np.histogram2d(x, y, bins=[xb, yb])

    yc = 0.5 * (ye[:-1] + ye[1:])
    xc = 0.5 * (xe[:-1] + xe[1:])
    # 连续性约束的脊线跟踪：从航道入口（y 最小）向北逐个 y 搜索，限定在上一位置 ±2.5 nm
    ridge_x, ridge_y = [], []
    prev_x = None
    for j, yv in enumerate(yc):
        if yv < args.y_range[0] or yv > args.y_range[1]:
            continue
        col = H[:, j]
        if col.sum() < 30:
            continue
        if prev_x is None:
            lo = np.searchsorted(xc, -ext)
            hi = np.searchsorted(xc, ext)
        else:
            lo = np.searchsorted(xc, prev_x - args.track_step)
            hi = np.searchsorted(xc, prev_x + args.track_step)
        if hi <= lo:
            continue
        k = lo + int(np.argmax(col[lo:hi]))
        prev_x = xc[k]
        ridge_x.append(xc[k])
        ridge_y.append(yv)

    ridge_x = np.asarray(ridge_x)
    ridge_y = np.asarray(ridge_y)
    from scipy.ndimage import median_filter
    ridge_x = median_filter(ridge_x, size=9, mode="nearest")

    # 转回经纬度
    lat = args.lat0 + np.degrees(ridge_y * NM / 6371000.0)
    lon = args.lon0 + np.degrees(ridge_x * NM / (6371000.0 * np.cos(np.radians(args.lat0))))

    order = np.argsort(ridge_y)
    pts = [{"lat": float(lat[i]), "lon": float(lon[i]), "y_nm": float(ridge_y[i]), "x_nm": float(ridge_x[i])}
           for i in order]
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    if args.local_nm:
        payload = {"lat0": args.lat0, "lon0": args.lon0,
                   "x_nm": [p["x_nm"] for p in pts], "y_nm": [p["y_nm"] for p in pts]}
    else:
        payload = pts
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"centerline points: {len(pts)}  y range {ridge_y.min():.1f}..{ridge_y.max():.1f} nm")

    fig, ax = plt.subplots(figsize=(10, 10))
    hb = ax.hexbin(x, y, gridsize=170, bins="log", cmap="viridis", mincnt=1)
    ax.plot(ridge_x[order], ridge_y[order], "-", color="red", lw=1.6, label="fitted centerline")
    ax.set_aspect("equal"); ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    ax.set_title(f"Fitted HSC centerline from {args.days}-day density")
    ax.legend(loc="upper left"); fig.colorbar(hb, ax=ax, label="reports")
    fig.tight_layout(); fig.savefig(args.out_png, dpi=120)
    print("saved", args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
