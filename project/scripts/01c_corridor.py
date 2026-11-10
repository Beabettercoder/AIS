#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 01c: 航道走廊子 ROI 的 3.2 扫描（断面通量 + 沿程速度 + 环境对齐）。

走廊定义：到中心线横向距离 |d| <= 1.0 nm；中心线 = 密度脊线拟合结果。
断面：沿程 s0（默认 24 nm）以及 12 nm（Red Fish Bar）、28 nm（Morgan's Point），
统计每 10 min 双向穿越的加权船数（船流率 q）。拥堵的物理基础是 q = rho * v。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NM = 1852.0
EARTH_R = 6371000.0
LAT0, LON0 = 29.50, -94.85


def load_centerline(path: str) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "x_nm" in data:
        # 新格式：直接带局部 nm 坐标，避免跨区域投影中心不一致
        return np.column_stack([np.asarray(data["x_nm"], float), np.asarray(data["y_nm"], float)])
    pts = data
    lat = np.array([p["lat"] for p in pts])
    lon = np.array([p["lon"] for p in pts])
    x = EARTH_R * np.radians(lon - LON0) * np.cos(np.radians(LAT0)) / NM
    y = EARTH_R * np.radians(lat - LAT0) / NM
    return np.column_stack([x, y])


def corridor_coords(px, py, poly):
    n = len(px)
    best = np.full(n, np.inf)
    s_out = np.zeros(n)
    d_out = np.zeros(n)
    tan_out = np.zeros(n)
    seg = np.hypot(np.diff(poly[:, 0]), np.diff(poly[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    for k in range(len(poly) - 1):
        x1, y1 = poly[k]
        x2, y2 = poly[k + 1]
        vx, vy = x2 - x1, y2 - y1
        L2 = vx * vx + vy * vy
        if L2 <= 1e-9:
            continue
        t = np.clip(((px - x1) * vx + (py - y1) * vy) / L2, 0.0, 1.0)
        cx, cy = x1 + t * vx, y1 + t * vy
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        m = d2 < best
        if m.any():
            best[m] = d2[m]
            s_out[m] = cum[k] + t[m] * seg[k]
            d_out[m] = ((px[m] - x1) * vy - (py[m] - y1) * vx) / np.sqrt(L2)
            tan_out[m] = np.arctan2(vx, vy)
    return s_out, d_out, np.sqrt(best), tan_out


def section_crossings(snaps: pd.DataFrame, s0: float) -> pd.DataFrame:
    snaps = snaps.sort_values(["mmsi", "time_bin"])
    prev_s = snaps.groupby("mmsi")["s_nm"].shift(1)
    prev_t = snaps.groupby("mmsi")["time_bin"].shift(1)
    contig = (snaps.time_bin - prev_t) == 600
    crossed = contig & (np.sign(snaps.s_nm - s0) != np.sign(prev_s - s0)) & prev_s.notna()
    up = crossed & (snaps.s_nm > prev_s)
    down = crossed & (snaps.s_nm < prev_s)
    out = snaps.loc[crossed, ["time_bin", "w_heur"]].copy()
    out["dir_up"] = up[crossed].to_numpy()
    out["dir_down"] = down[crossed].to_numpy()
    return out.groupby("time_bin").agg(q_up=("dir_up", "sum"), q_down=("dir_down", "sum")).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--centerline", required=True)
    ap.add_argument("--env", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--width_nm", type=float, default=1.0)
    ap.add_argument("--section_nm", type=float, default=24.0)
    ap.add_argument("--bin_sec", type=int, default=600)
    args = ap.parse_args()

    cols = ["mmsi", "timestamp", "base_datetime", "lat", "lon", "x_m", "y_m", "sog", "cog",
            "vx_rep", "vy_rep", "kin_ok", "w_heur", "in_core_roi"]
    df = pd.read_parquet(args.parquet, columns=cols)
    print(f"[01c] rows {len(df):,}", flush=True)

    df = df.loc[df.kin_ok & df.sog.ge(2.0)]
    df["time_bin"] = (df.timestamp // args.bin_sec) * args.bin_sec
    snaps = df.sort_values("timestamp").groupby(["time_bin", "mmsi"], as_index=False).tail(1)
    print(f"[01c] snapshots {len(snaps):,}", flush=True)

    poly = load_centerline(args.centerline)
    s, d, dist, tan = corridor_coords(snaps.x_m.to_numpy() / NM, snaps.y_m.to_numpy() / NM, poly)
    snaps["s_nm"] = s
    snaps["d_nm"] = d
    snaps["tan_rad"] = tan
    snaps["v_along"] = snaps.vx_rep.to_numpy() * np.sin(tan) + snaps.vy_rep.to_numpy() * np.cos(tan)
    snaps["in_corridor"] = np.abs(d) <= args.width_nm
    print(f"[01c] corridor coverage: {snaps.in_corridor.mean():.1%} of moving snapshots", flush=True)

    cor = snaps[snaps.in_corridor].copy()

    q = section_crossings(snaps, args.section_nm).rename(columns={"q_up": "q_up_w", "q_down": "q_down_w"})
    for s0 in (12.0, 28.0):
        qq = section_crossings(snaps, s0).rename(
            columns={"q_up": f"q_up_w_s{int(s0)}", "q_down": f"q_down_w_s{int(s0)}"})
        q = q.merge(qq, on="time_bin", how="left")

    rows = []
    for t_bin, grp in cor.groupby("time_bin", sort=True):
        rows.append({
            "time_bin": int(t_bin),
            "n_corridor": int(len(grp)),
            "n_upstream": int((grp.v_along > 0).sum()),
            "n_downstream": int((grp.v_along < 0).sum()),
            "median_v_along": float(grp.v_along.median()),
            "p10_v_along": float(grp.v_along.quantile(0.10)),
            "median_sog_kn": float(grp.sog.median()),
            "weighted_n": float(grp.w_heur.sum()),
            "s_median_nm": float(grp.s_nm.median()),
        })
    ts = pd.DataFrame(rows).merge(q, on="time_bin", how="left").sort_values("time_bin").reset_index(drop=True)
    for c in [c for c in ts.columns if c.startswith("q_")]:
        ts[c] = ts[c].fillna(0.0)
    ts["q_total_w"] = ts.q_up_w + ts.q_down_w
    ts["net_q_w"] = ts.q_up_w - ts.q_down_w
    for s0 in (12, 28):
        ts[f"q_total_w_s{s0}"] = ts[f"q_up_w_s{s0}"] + ts[f"q_down_w_s{s0}"]
    ts["datetime_utc"] = pd.to_datetime(ts.time_bin, unit="s", utc=True).astype("datetime64[ns, UTC]")
    ts["density_per_nm"] = ts.n_corridor / (2 * args.width_nm * 33.0)
    ts["q_over_rho"] = ts.q_total_w / ts.density_per_nm.replace(0, np.nan)

    ts["hour_utc"] = ts.datetime_utc.dt.hour
    for c in ["median_v_along", "q_total_w", "density_per_nm", "median_sog_kn"]:
        med = ts.groupby("hour_utc")[c].transform("median")
        iqr = ts.groupby("hour_utc")[c].transform(lambda x: (x.quantile(.75) - x.quantile(.25)) / 1.349)
        ts[f"{c}_hz"] = (ts[c] - med) / iqr.replace(0, np.nan)

    if args.env and Path(args.env).exists():
        env = pd.read_parquet(args.env)
        wide = env.pivot_table(index="observed_at", columns="variable", values="value", aggfunc="mean").sort_index()
        wide = wide.resample("10min").mean().reset_index().rename(columns={"observed_at": "datetime_utc"})
        wide["datetime_utc"] = pd.to_datetime(wide["datetime_utc"], utc=True).astype("datetime64[ns, UTC]")
        ts = pd.merge_asof(ts.sort_values("datetime_utc"), wide.sort_values("datetime_utc"),
                           on="datetime_utc", direction="nearest", tolerance=pd.Timedelta("30min"))
        print(f"[01c] env merged: {[c for c in wide.columns if c != 'datetime_utc'][:8]}", flush=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    ts.sort_values("time_bin").to_csv(args.out_csv, index=False)
    print(json.dumps({
        "bins": int(len(ts)),
        "corridor_vessels_median": float(ts.n_corridor.median()),
        "q24_median": float(ts.q_total_w.median()),
        "q12_median": float(ts.q_total_w_s12.median()),
        "q28_median": float(ts.q_total_w_s28.median()),
        "q12_p95": float(ts.q_total_w_s12.quantile(0.95)),
        "q28_p95": float(ts.q_total_w_s28.quantile(0.95)),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
