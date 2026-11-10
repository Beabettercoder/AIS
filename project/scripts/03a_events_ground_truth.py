#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 03a: 固化真实事件锚点 outputs/phase1/events_ground_truth.json

锚点：
  A 格（干净底图）：2024-07-06、2024-07-12 全天
  B 格（观测覆盖中断）：2024-07-20
  D 格（飓风极端）：2024-07-08

每个锚点记录：精确 UTC 起止、走廊船数、三断面流量、场有效率、最大风速/阵风/浪高。
同时输出 clean_pool_excluded_dates，供后续切分与注入强制剔除。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NM = 1852.0
RADIUS_M = 8.0 * NM
MIN_NBRS = 4
MAX_DCOG = np.radians(45.0)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def field_validity_by_bin(parquet: str, centerline: str, width_nm: float = 1.0) -> pd.DataFrame:
    """逐 10-min bin 计算走廊目标船的"场有效比例"（同向 8 nm 内 >=4 条可信邻船）。"""
    from importlib.machinery import SourceFileLoader
    mod = SourceFileLoader("scan", "/mnt/sda/WangZijia/AIS/scripts/01c_corridor.py").load_module()
    poly = mod.load_centerline(centerline)

    cols = ["mmsi", "timestamp", "sog", "cog", "vx_rep", "vy_rep", "x_m", "y_m", "kin_ok", "w_heur"]
    df = pd.read_parquet(parquet, columns=cols)
    df = df.loc[df.kin_ok & df.sog.ge(2.0) & df.w_heur.gt(0.3)]
    df["time_bin"] = (df.timestamp // 600) * 600
    snaps = df.sort_values("timestamp").groupby(["time_bin", "mmsi"], as_index=False).tail(1)
    s, d, _, _ = mod.corridor_coords(snaps.x_m.to_numpy() / NM, snaps.y_m.to_numpy() / NM, poly)
    snaps["s_nm"] = s
    snaps["d_nm"] = d
    cor = snaps[snaps.d_nm.abs() <= width_nm].reset_index(drop=True)

    out = []
    for t_bin, grp in cor.groupby("time_bin"):
        px, py = grp.x_m.to_numpy(), grp.y_m.to_numpy()
        cog = np.radians(grp.cog.to_numpy())
        tree = cKDTree(np.column_stack([px, py]))
        balls = tree.query_ball_point(np.column_stack([px, py]), r=RADIUS_M)
        ok = 0
        for i, cand in enumerate(balls):
            cand = np.asarray(cand, dtype=np.int64)
            if cand.size < MIN_NBRS + 1:
                continue
            dcog = np.abs(wrap(cog[cand] - cog[i]))
            nbr = cand[(cand != i) & (dcog <= MAX_DCOG)]
            if nbr.size >= MIN_NBRS:
                ok += 1
        out.append({"time_bin": int(t_bin), "n_targets": len(grp), "field_valid_ratio": ok / max(len(grp), 1)})
    return pd.DataFrame(out)


def window_stats(ts: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    w = ts[(ts.datetime_utc >= start) & (ts.datetime_utc <= end)]
    env_max = {}
    for c in ["wind_speed", "wind_gust", "wspd", "significant_wave_height", "wvht"]:
        if c in w.columns:
            env_max[c] = round(float(np.nanmax(w[c])), 2) if w[c].notna().any() else None
    return {
        "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_hours": round((end - start).total_seconds() / 3600, 2),
        "bins": int(len(w)),
        "corridor_vessels_median": round(float(w.n_corridor.median()), 1),
        "corridor_vessels_min": int(w.n_corridor.min()),
        "corridor_vessels_p10": round(float(w.n_corridor.quantile(0.10)), 1),
        "q12_sum": float(w.q_total_w_s12.sum()),
        "q24_sum": float(w.q_total_w.sum()),
        "q28_sum": float(w.q_total_w_s28.sum()),
        "q12_median": float(w.q_total_w_s12.median()),
        "q24_median": float(w.q_total_w.median()),
        "q28_median": float(w.q_total_w_s28.median()),
        "field_valid_ratio_median": round(float(w.field_valid_ratio.median()), 3) if "field_valid_ratio" in w else None,
        "field_valid_ratio_p10": round(float(w.field_valid_ratio.quantile(0.10)), 3) if "field_valid_ratio" in w else None,
        "env_max": env_max,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridor_csv", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--centerline", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ts = pd.read_csv(args.corridor_csv, parse_dates=["datetime_utc"])
    ts["datetime_utc"] = pd.to_datetime(ts["datetime_utc"], utc=True)
    fv = field_validity_by_bin(args.parquet, args.centerline)
    fv["datetime_utc"] = pd.to_datetime(fv.time_bin, unit="s", utc=True)
    ts = ts.merge(fv[["datetime_utc", "n_targets", "field_valid_ratio"]], on="datetime_utc", how="left")
    print(f"[03a] merged field validity, median={ts.field_valid_ratio.median():.3f}", flush=True)

    # 事件窗口（人工确认过的 UTC 边界）
    windows = {
        "beryl_d_cell": ("2024-07-07T18:00:00Z", "2024-07-10T12:00:00Z"),
        "coverage_outage_b_cell": ("2024-07-20T00:00:00Z", "2024-07-21T00:00:00Z"),
        "baseline_a_0706": ("2024-07-06T00:00:00Z", "2024-07-06T23:50:00Z"),
        "baseline_a_0712": ("2024-07-12T00:00:00Z", "2024-07-12T23:50:00Z"),
    }

    events = {}
    for name, (a, b) in windows.items():
        s = pd.Timestamp(a)
        e = pd.Timestamp(b)
        events[name] = window_stats(ts, s, e)

    # 7/20 的逐小时细节，便于论文描述
    d20 = ts[(ts.datetime_utc >= "2024-07-20") & (ts.datetime_utc < "2024-07-21")].copy()
    d20["h"] = d20.datetime_utc.dt.floor("h")
    events["coverage_outage_b_cell"]["hourly_n_corridor"] = [
        int(v) for v in d20.groupby("h").n_corridor.median().tolist()
    ]

    payload = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "houston_corridor_timeseries.csv + ais_houston_202407_enriched.parquet",
        "definitions": {
            "corridor": "Houston Ship Channel, |d| <= 1.0 nm from fitted centerline, s in [0, 34] nm",
            "field_valid": "same-direction (<=45 deg) neighbours within 8 nm, w_heur>0.3, n>=4",
            "q": "weighted vessel crossings per 10-min bin at a cross-section (s=12/24/28 nm)",
        },
        "cells": {
            "A": ["baseline_a_0706", "baseline_a_0712"],
            "B": ["coverage_outage_b_cell"],
            "D": ["beryl_d_cell"],
        },
        "events": events,
        "clean_pool_excluded_dates": ["2024-07-07", "2024-07-08", "2024-07-09", "2024-07-10", "2024-07-20"],
        "notes": [
            "7/20 风速/浪高均正常（<=5.3 m/s, <=0.40 m），但走廊船数与场有效率腰斩，判定为观测覆盖中断（B 格）。",
            "7/8 为 Beryl 登陆日，风速 20.1 m/s、浪高 3.75 m，三断面流量同时归零（D 格）。",
            "任何切分/注入脚本必须先读本文件，并从干净底图池中剔除 clean_pool_excluded_dates。",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in
                          ("start_utc", "end_utc", "corridor_vessels_median", "q24_sum",
                           "field_valid_ratio_median", "env_max")}
                      for k, v in events.items()}, indent=2, ensure_ascii=False))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
