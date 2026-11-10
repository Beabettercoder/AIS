#!/usr/bin/env python3
"""检查环境表 schema，并在 31 天序列上找候选群体减速时段（不设死阈值，只排序）。"""
import pandas as pd
import numpy as np

ENV = "/mnt/sda/WangZijia/houston_galveston_2024_07/data/interim/environment_observations.parquet"
try:
    env = pd.read_parquet(ENV)
    print("=== env schema ===")
    print(env.dtypes)
    print(env.head(3).to_string()[:1200])
    print("rows", len(env))
except Exception as e:
    print("env read failed:", e)

print("\n=== top candidates ===")
ts = pd.read_csv("/mnt/sda/WangZijia/AIS/outputs/phase1/houston_3_2_full.csv", parse_dates=["datetime_utc"])
ts = ts.sort_values("datetime_utc").reset_index(drop=True)
print("bins", len(ts), "range", ts.datetime_utc.min(), "->", ts.datetime_utc.max())

# 组合评分：散度越负、航速越低、净流入越大 → 越像群体减速/积压
ts["score"] = (-ts.median_div_1_s_hz.fillna(0)).clip(lower=0) * 1.0 \
            + (-ts.median_sog_kn_hz.fillna(0)).clip(lower=0) * 1.0 \
            + (ts.net_flux_crossing_w.fillna(0)).clip(lower=0) * 0.3
top = ts.nlargest(12, "score")[
    ["datetime_utc", "score", "median_div_1_s_hz", "median_sog_kn_hz", "net_flux_crossing_w",
     "throughput_w", "median_sog_kn", "valid_field_ratio"]]
print(top.to_string(index=False))

print("\n=== 分位数（用于定阈值） ===")
for c in ["median_div_1_s", "median_div_1_s_hz", "median_sog_kn", "median_sog_kn_hz", "net_flux_crossing_w", "throughput_w", "valid_field_ratio"]:
    q = ts[c].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).round(6).tolist()
    print(f"{c:26s} p1={q[0]} p5={q[1]} p50={q[2]} p95={q[3]} p99={q[4]}")

print("\n=== 按小时均值（看昼夜结构） ===")
print(ts.groupby("hour_utc")[["median_sog_kn", "median_div_1_s", "throughput_w"]].median().round(5).to_string())
