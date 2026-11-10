#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 00a: 从 NOAA 日文件 zip 抽取 Houston/Galveston 富字段 Parquet（方案 b）。

与 Gemini 版本的差异（都是为了避免跑挂或跑歪）：
  * usecols + 显式 dtype，只解析需要的 9 列（原始 17 列，速度差一倍）
  * BaseDateTime 用固定 format 解析（推断格式慢一个数量级）
  * SOG/COG/Heading 的 AIS 非法值显式过滤（SOG 102.3、COG 360、Heading 511）
  * 无效差分不再回退成报告速度赋 w=1，而是 kin_ok=False / w_heur=NaN
  * 分日写 parquet 再合并，避免 18 天一次性 concat 的峰值内存

用法：
  python scripts/00a_extract_noaa_raw_to_parquet.py \
      --raw_dir .../data/raw/noaa_ais \
      --out .../data/interim/ais_houston_202407_enriched.parquet
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

KNOTS_TO_MS = 0.514444
EARTH_R = 6371000.0

ROI_LAT = (29.20, 29.80)
ROI_LON = (-95.10, -94.60)
BUFFER_DEG = 0.15
LAT0, LON0 = 29.50, -94.85

USE_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselType", "Status"]
DTYPES = {
    "MMSI": "int64", "BaseDateTime": "string", "LAT": "float64", "LON": "float64",
    "SOG": "float32", "COG": "float32", "Heading": "float32",
    "VesselType": "float32", "Status": "float32",
}


def extract_zip(zip_path: Path) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as fh:
                for chunk in pd.read_csv(
                    fh, usecols=[c for c in USE_COLS], dtype=DTYPES,
                    chunksize=1_000_000, low_memory=False,
                ):
                    m = (
                        (chunk["LAT"] >= ROI_LAT[0] - BUFFER_DEG) & (chunk["LAT"] <= ROI_LAT[1] + BUFFER_DEG)
                        & (chunk["LON"] >= ROI_LON[0] - BUFFER_DEG) & (chunk["LON"] <= ROI_LON[1] + BUFFER_DEG)
                    )
                    sub = chunk.loc[m]
                    if len(sub):
                        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def enrich(df: pd.DataFrame, tau_kin: float = 2.5, sigma_kin: float = 2.0, min_gap: float = 5.0) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    df["base_datetime"] = pd.to_datetime(df["basedatetime"], format="%Y-%m-%dT%H:%M:%S", utc=True, errors="coerce")
    df = df.dropna(subset=["mmsi", "base_datetime", "lat", "lon", "sog", "cog"]).copy()

    # AIS 非法值
    df = df[(df["sog"] >= 0.0) & (df["sog"] < 102.3)]
    df = df[(df["cog"] >= 0.0) & (df["cog"] < 360.0)]
    df["heading"] = df["heading"].where((df["heading"] >= 0.0) & (df["heading"] < 360.0), np.nan)
    df["heading_valid"] = df["heading"].notna()

    # 分辨率无关的秒级时间戳（pandas 3 可能推断为 s/us/ns，直接 astype(int64) 会错位）
    df["timestamp"] = ((df["base_datetime"] - pd.Timestamp("1970-01-01", tz="UTC"))
                       // pd.Timedelta(seconds=1)).astype("int64")

    # (mmsi, ts) 去重，优先保留 heading 有效的
    df = df.sort_values(["mmsi", "timestamp", "heading_valid"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["mmsi", "timestamp"], keep="first")
    df = df.sort_values(["mmsi", "timestamp"], kind="mergesort").reset_index(drop=True)

    # 多接收站重复抑制：按 thin_sec 秒时间桶，每船每桶只保留第一条
    if min_gap and min_gap > 0:
        bucket = (df["timestamp"] // int(min_gap)).astype("int64")
        before = len(df)
        keep = ~pd.DataFrame({"m": df["mmsi"], "b": bucket}).duplicated(keep="first")
        df = df[keep.to_numpy()].reset_index(drop=True)
        print(f"[00a] bucket thinning: {before:,} -> {len(df):,} (每 {int(min_gap)} s 每船一条)", flush=True)

    # 固定中心局部切平面
    dlat = np.radians(df["lat"].to_numpy() - LAT0)
    dlon = np.radians(df["lon"].to_numpy() - LON0)
    df["x_m"] = (EARTH_R * dlon * np.cos(np.radians(LAT0))).astype("float32")
    df["y_m"] = (EARTH_R * dlat).astype("float32")

    df["in_core_roi"] = (
        (df["lat"] >= ROI_LAT[0]) & (df["lat"] <= ROI_LAT[1])
        & (df["lon"] >= ROI_LON[0]) & (df["lon"] <= ROI_LON[1])
    )

    sog_ms = df["sog"].to_numpy() * KNOTS_TO_MS
    cog_rad = np.radians(df["cog"].to_numpy())
    df["vx_rep"] = (sog_ms * np.sin(cog_rad)).astype("float32")
    df["vy_rep"] = (sog_ms * np.cos(cog_rad)).astype("float32")

    g = df.groupby("mmsi", sort=False)
    dt_prev = g["timestamp"].diff(1).to_numpy(dtype="float64")
    dt_next = (-g["timestamp"].diff(-1)).to_numpy(dtype="float64")
    dx_c = (g["x_m"].shift(-1) - g["x_m"].shift(1)).to_numpy(dtype="float64")
    dy_c = (g["y_m"].shift(-1) - g["y_m"].shift(1)).to_numpy(dtype="float64")
    dt_c = dt_prev + dt_next

    df["delta_seconds"] = np.where(dt_prev > 0, dt_prev, np.nan).astype("float32")

    valid = (
        (dt_prev >= 5.0) & (dt_prev <= 600.0)
        & (dt_next >= 5.0) & (dt_next <= 600.0)
    )
    df["kin_ok"] = valid

    with np.errstate(invalid="ignore", divide="ignore"):
        vx_d = np.where(valid, dx_c / dt_c, np.nan)
        vy_d = np.where(valid, dy_c / dt_c, np.nan)
        e_kin = np.hypot(vx_d - df["vx_rep"].to_numpy(), vy_d - df["vy_rep"].to_numpy())

    df["e_kin"] = e_kin.astype("float32")
    excess = np.maximum(0.0, np.nan_to_num(e_kin, nan=np.inf) - tau_kin)
    df["w_heur"] = np.where(valid, np.exp(-(excess**2) / (2.0 * sigma_kin**2)), np.nan).astype("float32")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=0, help="0 = 全部；用于先跑 1 天做烟测")
    ap.add_argument("--region", default="houston", choices=["houston", "la", "ny", "puget", "miami", "honolulu", "sanjuan", "mississippi", "gom_open", "atl_open", "pac_open", "guam", "goldengate", "custom"])
    ap.add_argument("--lat_min", type=float, default=0.0)
    ap.add_argument("--lat_max", type=float, default=0.0)
    ap.add_argument("--lon_min", type=float, default=0.0)
    ap.add_argument("--lon_max", type=float, default=0.0)
    ap.add_argument("--min_gap", type=float, default=5.0, help="同一船相邻报文最小间隔（秒），抑制多站重复")
    args = ap.parse_args()

    global ROI_LAT, ROI_LON, LAT0, LON0, BUFFER_DEG
    REGIONS = {"houston": (29.20, 29.80, -95.10, -94.60),
               "la": (33.60, 34.05, -118.55, -117.85),
               "ny": (40.35, 40.95, -74.35, -73.65),
               "puget": (47.50, 48.50, -124.00, -122.30),
               "miami": (25.40, 26.20, -80.30, -79.80),
               "honolulu": (21.10, 21.50, -158.10, -157.70),
               "sanjuan": (18.30, 18.60, -66.30, -65.90),
               "mississippi": (28.80, 29.50, -89.60, -89.00),
               "gom_open": (26.00, 27.80, -94.00, -91.50),
               "atl_open": (36.50, 38.80, -73.50, -70.50),
               "pac_open": (31.50, 33.20, -121.00, -118.80),
               "guam": (13.20, 13.80, 144.40, 145.00),
               "goldengate": (37.70, 37.90, -122.60, -122.40)}
    if args.region in REGIONS:
        la0, la1, lo0, lo1 = REGIONS[args.region]
    else:
        la0, la1, lo0, lo1 = args.lat_min, args.lat_max, args.lon_min, args.lon_max
    ROI_LAT, ROI_LON = (la0, la1), (lo0, lo1)
    LAT0, LON0 = (la0 + la1) / 2, (lo0 + lo1) / 2
    BUFFER_DEG = 0.15

    zips = sorted(Path(args.raw_dir).glob("AIS_*.zip"))
    if args.days:
        zips = zips[: args.days]
    if not zips:
        raise SystemExit(f"no zip found in {args.raw_dir}")
    print(f"[00a] {len(zips)} zips", flush=True)

    t0 = time.time()
    stats = []
    for i, zp in enumerate(zips, 1):
        t = time.time()
        part = extract_zip(zp)
        stats.append({"file": zp.name, "rows_in_bbox": len(part), "sec": round(time.time() - t, 1)})
        print(f"[00a] {i}/{len(zips)} {zp.name}: {len(part):,} rows, {stats[-1]['sec']}s", flush=True)
        part.to_parquet(Path(args.out).parent / f"_part_{zp.stem}.parquet", index=False)
        del part

    parts = sorted(Path(args.out).parent.glob("_part_*.parquet"))
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    print(f"[00a] merged {len(df):,} rows in {time.time()-t0:.0f}s, enriching...", flush=True)

    df = enrich(df, min_gap=args.min_gap)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")
    for p in parts:
        p.unlink()

    roi = df[df["in_core_roi"]]
    dtv = roi.loc[roi["delta_seconds"] > 0, "delta_seconds"]
    summary = {
        "zips": len(zips),
        "rows_with_buffer": int(len(df)),
        "rows_core_roi": int(len(roi)),
        "unique_mmsi_core_roi": int(roi["mmsi"].nunique()),
        "dt_median_sec": float(dtv.median()),
        "dt_p90_sec": float(dtv.quantile(0.90)),
        "dt_p95_sec": float(dtv.quantile(0.95)),
        "heading_invalid_ratio": float(roi["heading"].isna().mean()),
        "sog_below_2kn_ratio": float((roi["sog"] < 2.0).mean()),
        "kin_ok_ratio": float(roi["kin_ok"].mean()),
        "e_kin_p50": float(roi["e_kin"].median()),
        "e_kin_p90": float(roi["e_kin"].quantile(0.90)),
        "e_kin_p99": float(roi["e_kin"].quantile(0.99)),
        "per_file": stats,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    sp = Path(args.out).with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_file"}, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
