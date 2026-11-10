#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 00b: 8 nm 留一法局部线性场 + 3.2 无阈值时间序列扫描（cKDTree 版）。

相对 Gemini 版本的修正：
  * 快照用 groupby().tail(1) 取真正最后一条，避免 .last() 逐列跳过 NaN 造成的跨时刻混拼
  * 场拟合只在 (sog>=2kn & kin_ok & w_heur>0.3) 的邻船池上做，目标船也必须在池内
  * ROI 滞回用显式状态机，不用 ffill（ffill 会把"从未进过 ROI 的船"污染成 1）
  * 净通量用边界穿越直接计数；同时输出 ROI 内加权船数，供人工对照
  * 环境表列名不确定，做防御性探测，失败只告警不中断
  * 所有输出 CSV 保留原始量，不做任何阈值裁剪
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NM = 1852.0
RADIUS_M = 8.0 * NM
SIGMA_D = 4.0 * NM
MIN_SOG_KN = 2.0
MIN_NBRS = 4
MAX_DCOG = np.radians(45.0)
SPREAD_MIN = 0.15
COND_MAX = 1e3

INNER = dict(lat=(29.21, 29.79), lon=(-95.09, -94.61))
OUTER = dict(lat=(29.19, 29.81), lon=(-95.11, -94.59))


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def field_stats(px, py, vx, vy, w, cog, target_mask):
    """留一法加权局部线性场。返回逐目标船的 (res, div, curl, valid)。"""
    n_target = int(target_mask.sum())
    res = np.full(n_target, np.nan)
    div = np.full(n_target, np.nan)
    curl = np.full(n_target, np.nan)

    if len(px) < MIN_NBRS + 1 or n_target == 0:
        return res, div, curl

    tree = cKDTree(np.column_stack([px, py]))
    idx_target = np.flatnonzero(target_mask)
    balls = tree.query_ball_point(np.column_stack([px[idx_target], py[idx_target]]), r=RADIUS_M)

    k = 0
    for pos, i in enumerate(idx_target):
        cand = np.asarray(balls[pos], dtype=np.int64)
        if cand.size < MIN_NBRS + 1:
            k += 1
            continue
        dcog = np.abs(wrap(cog[cand] - cog[i]))
        nbr = cand[(cand != i) & (dcog <= MAX_DCOG)]
        if nbr.size < MIN_NBRS:
            k += 1
            continue

        dx = (px[nbr] - px[i]) / RADIUS_M
        dy = (py[nbr] - py[i]) / RADIUS_M
        if np.sqrt(dx.var() + dy.var()) <= SPREAD_MIN:
            k += 1
            continue

        dist2 = (dx * RADIUS_M) ** 2 + (dy * RADIUS_M) ** 2
        tw = w[nbr] * np.exp(-dist2 / (2.0 * SIGMA_D**2))
        if not np.isfinite(tw).all() or tw.sum() <= 0:
            k += 1
            continue

        sw = np.sqrt(tw)[:, None]
        X = np.column_stack([np.ones_like(dx), dx, dy]) * sw
        if np.linalg.cond(X) >= COND_MAX:
            k += 1
            continue

        Y = np.column_stack([vx[nbr], vy[nbr]]) * sw
        theta, *_ = np.linalg.lstsq(X, Y, rcond=None)

        bg_x, bg_y = theta[0, 0], theta[0, 1]
        dvx_dx, dvy_dx = theta[1, 0] / RADIUS_M, theta[1, 1] / RADIUS_M
        dvx_dy, dvy_dy = theta[2, 0] / RADIUS_M, theta[2, 1] / RADIUS_M

        res[k] = np.hypot(vx[i] - bg_x, vy[i] - bg_y)
        div[k] = dvx_dx + dvy_dy
        curl[k] = dvy_dx - dvx_dy
        k += 1
    return res, div, curl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--env_parquet", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--bin_sec", type=int, default=600)
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()

    cols = ["mmsi", "timestamp", "base_datetime", "lat", "lon", "sog", "cog",
            "x_m", "y_m", "vx_rep", "vy_rep", "heading", "kin_ok", "w_heur", "in_core_roi"]
    df = pd.read_parquet(args.in_parquet, columns=cols)
    if args.days:
        t0 = df["base_datetime"].min()
        df = df[df["base_datetime"] < t0 + pd.Timedelta(days=args.days)]
    print(f"[00b] rows {len(df):,}", flush=True)

    df["time_bin"] = (df["timestamp"] // args.bin_sec) * args.bin_sec
    ok = df["kin_ok"] & df["sog"].ge(MIN_SOG_KN) & df["w_heur"].gt(0.3)
    snaps = df.loc[ok].sort_values("timestamp").groupby(["time_bin", "mmsi"], as_index=False).tail(1)
    print(f"[00b] snapshots (moving & kin_ok) {len(snaps):,} over {snaps.time_bin.nunique()} bins", flush=True)

    # ---- 连续可见 >= 3 bins ----
    snaps = snaps.sort_values(["mmsi", "time_bin"]).reset_index(drop=True)
    gb = snaps.groupby("mmsi")["time_bin"]
    prev = gb.shift(1)
    nxt = gb.shift(-1)
    prev2 = gb.shift(2)
    snaps["persistent3"] = (
        ((snaps.time_bin - prev) == args.bin_sec) & ((nxt - snaps.time_bin) == args.bin_sec)
    ) | (
        ((snaps.time_bin - prev) == args.bin_sec) & ((snaps.time_bin - prev2) == 2 * args.bin_sec)
    )

    # ---- ROI 滞回状态机：1=内框内，0=外框外，NaN=滞回带（沿用上一状态） ----
    lat, lon = snaps.lat.to_numpy(), snaps.lon.to_numpy()
    state = np.full(len(snaps), np.nan)
    state[(lat >= INNER["lat"][0]) & (lat <= INNER["lat"][1]) & (lon >= INNER["lon"][0]) & (lon <= INNER["lon"][1])] = 1.0
    state[(lat < OUTER["lat"][0]) | (lat > OUTER["lat"][1]) | (lon < OUTER["lon"][0]) | (lon > OUTER["lon"][1])] = 0.0
    snaps["roi_state"] = state
    snaps["roi_state"] = snaps.groupby("mmsi")["roi_state"].transform(lambda s: s.ffill().fillna(0.0))

    prev_state = snaps.groupby("mmsi")["roi_state"].shift(1)
    prev_time = snaps.groupby("mmsi")["time_bin"].shift(1)
    contiguous = (snaps.time_bin - prev_time) == args.bin_sec
    entered = contiguous & prev_state.eq(0) & snaps.roi_state.eq(1)
    left = contiguous & prev_state.eq(1) & snaps.roi_state.eq(0)

    rows = []
    for t_bin, grp in snaps.groupby("time_bin", sort=True):
        px = grp.x_m.to_numpy(); py = grp.y_m.to_numpy()
        vx = grp.vx_rep.to_numpy(); vy = grp.vy_rep.to_numpy()
        w = grp.w_heur.to_numpy(); cog = np.radians(grp.cog.to_numpy())
        tgt = grp.in_core_roi.to_numpy()

        res, div, curl = field_stats(px, py, vx, vy, w, cog, tgt)
        valid = np.isfinite(div)

        in_roi = grp.roi_state.eq(1) & grp.persistent3
        rows.append({
            "time_bin": int(t_bin),
            "datetime_utc": pd.to_datetime(t_bin, unit="s", utc=True),
            "n_snapshot": int(len(grp)),
            "n_roi_targets": int(tgt.sum()),
            "n_persistent_roi": int(in_roi.sum()),
            "weighted_roi_vessels": float(grp.loc[in_roi, "w_heur"].sum()),
            "flux_in_w": float(grp.loc[entered.reindex(grp.index), "w_heur"].sum()),
            "flux_out_w": float(grp.loc[left.reindex(grp.index), "w_heur"].sum()),
            "median_sog_kn": float(grp.loc[in_roi, "sog"].median()) if in_roi.any() else np.nan,
            "valid_field_ratio": float(valid.mean()) if valid.size else 0.0,
            "median_div_1_s": float(np.nanmedian(div)) if valid.any() else np.nan,
            "median_curl_1_s": float(np.nanmedian(curl)) if valid.any() else np.nan,
            "median_res_ms": float(np.nanmedian(res)) if valid.any() else np.nan,
            "p90_res_ms": float(np.nanpercentile(res, 90)) if valid.any() else np.nan,
        })

    ts = pd.DataFrame(rows).sort_values("time_bin").reset_index(drop=True)
    ts["net_flux_crossing_w"] = ts.flux_in_w - ts.flux_out_w
    ts["throughput_w"] = 0.5 * (ts.flux_in_w + ts.flux_out_w)
    ts["hour_utc"] = ts.datetime_utc.dt.hour

    for col in ["median_sog_kn", "median_div_1_s", "median_curl_1_s", "throughput_w", "median_res_ms"]:
        med = ts.groupby("hour_utc")[col].transform("median")
        iqr = ts.groupby("hour_utc")[col].transform(lambda x: (x.quantile(0.75) - x.quantile(0.25)) / 1.349)
        ts[f"{col}_hz"] = (ts[col] - med) / iqr.replace(0, np.nan)

    if args.env_parquet and Path(args.env_parquet).exists():
        try:
            env = pd.read_parquet(args.env_parquet)
            tcol = next((c for c in env.columns if "time" in c.lower() or "date" in c.lower()), None)
            if tcol:
                env = env.copy()
                env["datetime_utc"] = pd.to_datetime(env[tcol], utc=True, errors="coerce")
                env = env.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
                env = env.drop(columns=[c for c in env.columns if env[c].dtype == object and c != "datetime_utc"], errors="ignore")
                ts = pd.merge_asof(ts.sort_values("datetime_utc"), env, on="datetime_utc",
                                   direction="nearest", tolerance=pd.Timedelta("30min"))
                print(f"[00b] merged env cols: {[c for c in env.columns if c != 'datetime_utc']}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[00b][warn] env merge skipped: {e}", flush=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    ts.to_csv(args.out_csv, index=False)

    print(json.dumps({
        "bins": int(len(ts)),
        "bins_with_field": int(ts.valid_field_ratio.gt(0).sum()),
        "valid_field_ratio_median": float(ts.valid_field_ratio.median()),
        "median_sog_kn_median": float(ts.median_sog_kn.median()),
        "median_div_1_s_p05": float(ts.median_div_1_s.quantile(0.05)),
        "median_div_1_s_p50": float(ts.median_div_1_s.median()),
        "net_flux_p05": float(ts.net_flux_crossing_w.quantile(0.05)),
        "net_flux_p50": float(ts.net_flux_crossing_w.median()),
    }, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
