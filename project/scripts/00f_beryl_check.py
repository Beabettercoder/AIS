#!/usr/bin/env python3
"""看 7/6-7/11（Beryl 登陆前后）与全月的对比：船数、航速、吞吐是否异常。"""
import pandas as pd

ts = pd.read_csv("/mnt/sda/WangZijia/AIS/outputs/phase1/houston_3_2_full.csv", parse_dates=["datetime_utc"])
ts = ts.sort_values("datetime_utc")
ts["date"] = ts.datetime_utc.dt.date

print("=== 全月按天 ===")
daily = ts.groupby("date").agg(
    bins=("time_bin", "size"),
    n_snap_mean=("n_snapshot", "mean"),
    n_roi_mean=("n_roi_targets", "mean"),
    sog_mean=("median_sog_kn", "mean"),
    div_med=("median_div_1_s", "median"),
    thr_mean=("throughput_w", "mean"),
    valid_mean=("valid_field_ratio", "mean"),
).round(3)
print(daily.to_string())

print("\n=== 7/7 18:00 - 7/10 06:00 逐小时 ===")
w = ts[(ts.datetime_utc >= "2024-07-07 18:00+00:00") & (ts.datetime_utc <= "2024-07-10 06:00+00:00")].copy()
w["h"] = w.datetime_utc.dt.floor("h")
print(w.groupby("h").agg(n_snap=("n_snapshot", "mean"), n_roi=("n_roi_targets", "mean"),
                         sog=("median_sog_kn", "mean"), thr=("throughput_w", "mean"),
                         flux=("net_flux_crossing_w", "sum"), valid=("valid_field_ratio", "mean")).round(2).to_string())
