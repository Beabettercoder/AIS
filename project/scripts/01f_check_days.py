#!/usr/bin/env python3
import pandas as pd
ts = pd.read_csv("/mnt/sda/WangZijia/AIS/outputs/phase1/houston_corridor_timeseries.csv", parse_dates=["datetime_utc"])
ts["d"] = ts.datetime_utc.dt.date
g = ts.groupby("d").agg(bins=("time_bin", "size"), n_cor=("n_corridor", "mean"),
                        q24=("q_total_w", "sum"), q12=("q_total_w_s12", "sum"),
                        q28=("q_total_w_s28", "sum")).round(2)
print(g.to_string())
