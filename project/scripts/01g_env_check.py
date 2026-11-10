import datetime as dt
import pandas as pd

ts = pd.read_csv("/mnt/sda/WangZijia/AIS/outputs/phase1/houston_corridor_timeseries.csv", parse_dates=["datetime_utc"])
ts["d"] = ts.datetime_utc.dt.date
cols = [c for c in ["wind_speed", "wind_gust", "wspd", "significant_wave_height", "wvht", "air_pressure"] if c in ts.columns]
g = ts.groupby("d").agg({**{c: "max" for c in cols}, "n_corridor": "mean"}).round(2)
mask = (g.index >= dt.date(2024, 7, 5)) & (g.index <= dt.date(2024, 7, 22))
print(g[mask].to_string())
