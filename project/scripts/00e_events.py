#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 00e: 在无阈值序列上抽取候选群体减速事件，并与环境观测对齐。

规则（不写死物理阈值，只做统计筛选，供人工复核）：
  1) 只用 valid_field_ratio >= 0.60 且 n_roi_targets >= 30 的 bin（否则散度是小样本噪声）
  2) 散度 z <= -1.5 且持续 >= 3 个连续 bin(>=30 min) 视为候选
  3) 报告事件的时长、散度、航速 z、净通量、吞吐量，以及当时的风速/能见度/气压
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TS = "/mnt/sda/WangZijia/AIS/outputs/phase1/houston_3_2_full.csv"
ENV = "/mnt/sda/WangZijia/houston_galveston_2024_07/data/interim/environment_observations.parquet"


def load_env() -> pd.DataFrame:
    env = pd.read_parquet(ENV)
    wide = (env.pivot_table(index="observed_at", columns="variable", values="value", aggfunc="mean")
               .sort_index())
    wide = wide.resample("10min").mean()
    return wide


def extract_events(ts: pd.DataFrame, z_thr: float = -1.5, min_bins: int = 3) -> pd.DataFrame:
    ok = (ts.valid_field_ratio >= 0.60) & (ts.n_roi_targets >= 30)
    flag = ok & (ts.median_div_1_s_hz <= z_thr)

    events, start = [], None
    for i, f in enumerate(flag.to_numpy()):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_bins:
                events.append((start, i - 1))
            start = None
    if start is not None and len(ts) - start >= min_bins:
        events.append((start, len(ts) - 1))

    rows = []
    for a, b in events:
        seg = ts.iloc[a:b + 1]
        rows.append({
            "start": seg.datetime_utc.iloc[0],
            "end": seg.datetime_utc.iloc[-1],
            "minutes": int((seg.datetime_utc.iloc[-1] - seg.datetime_utc.iloc[0]).total_seconds() / 60) + 10,
            "div_z_mean": round(float(seg.median_div_1_s_hz.mean()), 2),
            "div_mean_1_s": float(seg.median_div_1_s.mean()),
            "sog_z_mean": round(float(seg.median_sog_kn_hz.mean()), 2),
            "sog_kn_mean": round(float(seg.median_sog_kn.mean()), 2),
            "net_flux_mean": round(float(seg.net_flux_crossing_w.mean()), 2),
            "throughput_mean": round(float(seg.throughput_w.mean()), 2),
            "valid_ratio_mean": round(float(seg.valid_field_ratio.mean()), 2),
            "n_targets_mean": round(float(seg.n_roi_targets.mean()), 0),
        })
    return pd.DataFrame(rows).sort_values("minutes", ascending=False)


def main() -> None:
    ts = pd.read_csv(TS, parse_dates=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
    ev = extract_events(ts)
    print(f"[00e] 有效 bin: {((ts.valid_field_ratio>=0.60)&(ts.n_roi_targets>=30)).sum()} / {len(ts)}")
    print(f"[00e] 候选事件: {len(ev)}")
    if len(ev):
        print(ev.head(15).to_string(index=False))

    env = load_env()
    print("\n[00e] 环境变量:", list(env.columns))
    if "wind_speed" in env.columns or "visibility" in env.columns:
        base = env.median()
        print("[00e] 全月环境中位数:\n", base.round(2).to_string())

        for _, r in ev.head(8).iterrows():
            seg = env.loc[(env.index >= r.start - pd.Timedelta("30min")) & (env.index <= r.end + pd.Timedelta("30min"))]
            if len(seg) == 0:
                continue
            cols = [c for c in ["wind_speed", "visibility", "air_pressure", "air_temperature", "water_level"] if c in env.columns]
            print(f"\n{r.start} ~ {r.end} ({r.minutes}min, div_z={r.div_z_mean}, sog={r.sog_kn_mean}kn)")
            print("   ", {c: round(float(seg[c].mean()), 2) for c in cols})

    ev.to_csv("/mnt/sda/WangZijia/AIS/outputs/phase1/candidate_events.csv", index=False)
    print("\n[00e] saved candidate_events.csv")


if __name__ == "__main__":
    main()
