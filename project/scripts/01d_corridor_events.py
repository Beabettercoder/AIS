#!/usr/bin/env python3
"""走廊版 3.2：分布检查 + 排队事件挖掘（密度升、速度降、断面流量降）。"""
import pandas as pd
import numpy as np

P = "/mnt/sda/WangZijia/AIS/outputs/phase1/houston_corridor_timeseries.csv"
ts = pd.read_csv(P, parse_dates=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)

print("bins", len(ts), ts.datetime_utc.min(), "->", ts.datetime_utc.max())
print("\n=== 分布 ===")
for c in ["n_corridor", "q_total_w", "q_up_w", "q_down_w", "median_sog_kn", "density_per_nm", "median_v_along"]:
    q = ts[c].quantile([0.05, 0.5, 0.95]).round(3).tolist()
    print(f"{c:18s} p5={q[0]:>8} p50={q[1]:>8} p95={q[2]:>8}  mean={ts[c].mean():.3f}")

print("\n=== 按小时（日周期） ===")
print(ts.groupby("hour_utc")[["n_corridor", "q_total_w", "median_sog_kn"]].median().round(2).to_string())

# 排队判据：密度 z 高、速度 z 低、断面流量 z 低
for c in ["n_corridor", "median_sog_kn", "q_total_w"]:
    ts[f"{c}_z"] = (ts[c] - ts[c].median()) / ((ts[c].quantile(.75) - ts[c].quantile(.25)) / 1.349)

ts["queue_score"] = ts.n_corridor_z.clip(lower=0) - ts.median_sog_kn_z.clip(upper=0) - ts.q_total_w_z.clip(upper=0)
top = ts.nlargest(15, "queue_score")[["datetime_utc", "queue_score", "n_corridor", "median_sog_kn",
                                      "q_total_w", "n_corridor_z", "median_sog_kn_z", "q_total_w_z"]]
print("\n=== 排队候选 Top15 ===")
print(top.round(2).to_string(index=False))

print("\n=== Beryl 窗口对照 ===")
w = ts[(ts.datetime_utc >= "2024-07-06") & (ts.datetime_utc <= "2024-07-12")]
print(w.groupby(w.datetime_utc.dt.date)[["n_corridor", "q_total_w", "median_sog_kn", "density_per_nm"]].median().round(2).to_string())

ts.to_csv(P, index=False)
print("\n[01d] 已回写 z-score 列")
