#!/usr/bin/env python3
"""断面流量 q12 / q24 / q28 的统计与互相关。"""
import pandas as pd
import numpy as np

ts = pd.read_csv("/mnt/sda/WangZijia/AIS/outputs/phase1/houston_corridor_timeseries.csv", parse_dates=["datetime_utc"])
for s in (12, 24, 28):
    c = f"q_total_w_s{s}" if s != 24 else "q_total_w"
    v = ts[c]
    print(f"q{s:>2}: median={v.median():.2f} mean={v.mean():.2f} p90={v.quantile(.9):.0f} p95={v.quantile(.95):.0f} "
          f"p99={v.quantile(.99):.0f} max={v.max():.0f} frac_zero={(v==0).mean():.2f}")

print("\n按小时（中位数）:")
print(ts.groupby("hour_utc")[["q_total_w_s12", "q_total_w", "q_total_w_s28", "n_corridor"]].median().round(2).to_string())

a, b, c = ts.q_total_w_s12, ts.q_total_w, ts.q_total_w_s28
print("\n相关系数:")
print(f"q12~q24 {a.corr(b):.3f}  q24~q28 {b.corr(c):.3f}  q12~q28 {a.corr(c):.3f}")

print("\nBeryl 窗口 (7/7-7/10) 逐日:")
w = ts[(ts.datetime_utc >= "2024-07-06") & (ts.datetime_utc <= "2024-07-12")]
print(w.groupby(w.datetime_utc.dt.date)[["q_total_w_s12", "q_total_w", "q_total_w_s28"]].sum().to_string())
