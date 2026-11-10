#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 27: 用 Hough 式搜索找区域最密集的直线走廊，作为该区中心线。

适用于多平行航道区域（如 San Pedro Bay）：直接取"承载流量最大的单一直线走廊"，
避免逐 y 脊线在多条航道之间横跳。
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
EARTH_R = 6371000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--lat0", type=float, required=True)
    ap.add_argument("--lon0", type=float, required=True)
    ap.add_argument("--extent_nm", type=float, default=22.0)
    ap.add_argument("--width_nm", type=float, default=1.0)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["mmsi", "timestamp", "x_m", "y_m", "sog", "kin_ok"])
    t0 = df.timestamp.min()
    df = df[(df.timestamp < t0 + args.days * 86400) & df.kin_ok & df.sog.ge(2.0)]
    x = df.x_m.to_numpy() / NM
    y = df.y_m.to_numpy() / NM
    print(f"moving points {len(x):,}", flush=True)

    best = None
    for ang_deg in range(0, 180, 5):
        ang = np.radians(ang_deg)
        nx, ny = -np.sin(ang), np.cos(ang)          # 法向
        proj = x * nx + y * ny                      # 到过原点法线的有符号距离
        hist, edges = np.histogram(proj, bins=np.arange(-40, 41, 1.0))
        k = int(np.argmax(hist))
        if best is None or hist[k] > best[0]:
            best = (int(hist[k]), ang_deg, 0.5 * (edges[k] + edges[k + 1]))
    cnt, ang_deg, offset = best
    ang = np.radians(ang_deg)
    nx, ny = -np.sin(ang), np.cos(ang)
    tx, ty = np.cos(ang), np.sin(ang)
    # 走廊内点：|proj - offset| <= width
    d = x * nx + y * ny - offset
    inside = np.abs(d) <= args.width_nm
    t = x * tx + y * ty
    t_lo, t_hi = np.percentile(t[inside], [2, 98])
    p0 = np.array([offset * nx + t_lo * tx, offset * ny + t_lo * ty])
    p1 = np.array([offset * nx + t_hi * tx, offset * ny + t_hi * ty])
    print(f"best angle={ang_deg}° offset={offset:.2f} nm corridor points={cnt:,} "
          f"length={(t_hi - t_lo):.1f} nm  coverage={inside.mean():.1%}", flush=True)

    ts = np.linspace(0, 1, 60)
    xs = p0[0] + ts * (p1[0] - p0[0])
    ys = p0[1] + ts * (p1[1] - p0[1])
    payload = {"lat0": args.lat0, "lon0": args.lon0,
               "x_nm": xs.tolist(), "y_nm": ys.tolist(),
               "hough": {"angle_deg": ang_deg, "offset_nm": float(offset),
                         "coverage": float(inside.mean())}}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 10))
    hb = ax.hexbin(x, y, gridsize=160, bins="log", cmap="viridis", mincnt=1)
    ax.plot(xs, ys, "-", color="red", lw=2, label=f"Hough corridor ({ang_deg}°)")
    ax.set_aspect("equal"); ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    ax.set_title(f"Dominant straight corridor — coverage {inside.mean():.1%}")
    ax.legend(); fig.colorbar(hb, ax=ax, label="reports")
    fig.tight_layout(); fig.savefig(args.out_png, dpi=120)
    print("saved", args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
