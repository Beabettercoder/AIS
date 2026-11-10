#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 39: 三大场景区域物理特征（Ports / Straits / Open Seas）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NM = 1852.0
RADIUS_M = 8 * NM

REGIONS = [
    # (label, file, scenario, lat, lon)
    ("Houston Ship Channel", "ais_houston_202407_enriched.parquet", "Ports", 29.50, -94.85),
    ("New York / NJ", "ais_ny_enriched.parquet", "Ports", 40.65, -74.05),
    ("Los Angeles / LB", "ais_la_enriched.parquet", "Ports", 33.74, -118.24),
    ("Mississippi Passes", "ais_mississippi_enriched.parquet", "Ports", 29.15, -89.25),
    ("San Juan", "ais_sanjuan_enriched.parquet", "Ports", 18.45, -66.10),
    ("Honolulu", "ais_honolulu_enriched.parquet", "Ports", 21.30, -157.90),
    ("Juan de Fuca", "ais_puget_enriched.parquet", "Straits", 48.25, -123.50),
    ("Straits of Florida", "ais_miami_enriched.parquet", "Straits", 25.80, -80.05),
    ("Outer Gulf of Mexico", "ais_gom_open_enriched.parquet", "Open Seas", 26.90, -92.75),
    ("NW Atlantic Offshore", "ais_atl_open_enriched.parquet", "Open Seas", 37.65, -72.00),
    ("NE Pacific Pelagic", "ais_pac_open_enriched.parquet", "Open Seas", 32.35, -119.90),
]


def stats(path: Path) -> dict:
    df = pd.read_parquet(path, columns=["mmsi", "timestamp", "x_m", "y_m", "sog", "cog", "kin_ok"])
    mov = df[df.kin_ok & df.sog.ge(2.0)]
    if not len(mov):
        return {}
    g = mov.sort_values(["mmsi", "timestamp"])
    dchi = np.abs((np.diff(g.cog.to_numpy(), prepend=g.cog.iloc[0]) + 180) % 360 - 180)
    dt = np.diff(g.timestamp.to_numpy(), prepend=g.timestamp.iloc[0]).astype(float)
    ok = (dt > 0) & (dt < 600)
    turn = np.deg2rad(dchi[ok]) / dt[ok]

    mov = mov.assign(bin=(mov.timestamp // 600) * 600)
    bins = mov.bin.drop_duplicates()
    if len(bins) > 300:
        bins = bins.sample(300, random_state=0)
    valid = total = 0
    spar = 0
    for b in bins:
        sub = mov[mov.bin == b]
        if len(sub) < 3:
            continue
        tree = cKDTree(np.column_stack([sub.x_m, sub.y_m]))
        cnt = tree.query_ball_point(np.column_stack([sub.x_m, sub.y_m]), r=RADIUS_M, return_length=True)
        total += len(sub)
        valid += int((cnt >= 5).sum())
        spar += int((cnt < 5).sum())
    return {
        "vessels": int(mov.mmsi.nunique()),
        "reports": int(len(mov)),
        "sog_med": float(mov.sog.median()),
        "turn_med": float(np.median(turn)) if len(turn) else np.nan,
        "field_valid": valid / max(total, 1),
        "fallback_rate": spar / max(total, 1),
    }


def main() -> None:
    base = Path("/mnt/sda/WangZijia/AIS/data/interim")
    rows = []
    for label, fname, scen, lat, lon in REGIONS:
        p = base / fname
        if not p.exists():
            print("missing", fname)
            continue
        s = stats(p)
        if not s:
            continue
        s.update({"region": label, "scenario": scen, "lat": lat, "lon": lon})
        rows.append(s)
        print(label, s, flush=True)
    df = pd.DataFrame(rows)[["region", "scenario", "lat", "lon", "vessels", "reports",
                             "sog_med", "turn_med", "field_valid", "fallback_rate"]]
    out = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
    df.to_csv(out / "table_3scenarios_characteristics.csv", index=False)
    lines = ["# Three-scenario benchmark characteristics", "",
             "| Region | Scenario | Vessels | Reports | SOG (kn) | Turn rate (rad/s) | Field valid | Fallback rate |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.region} | {r.scenario} | {r.vessels} | {r.reports:,} | {r.sog_med:.1f} | "
                     f"{r.turn_med:.5f} | {r.field_valid:.3f} | {r.fallback_rate:.3f} |")
    for scen in ("Ports", "Straits", "Open Seas"):
        sub = df[df.scenario == scen]
        if len(sub):
            lines.append(f"| **{scen} mean** | | {sub.vessels.mean():.0f} | {sub.reports.mean():,.0f} | "
                         f"{sub.sog_med.mean():.1f} | {sub.turn_med.mean():.5f} | "
                         f"{sub.field_valid.mean():.3f} | {sub.fallback_rate.mean():.3f} |")
    txt = "\n".join(lines)
    (out / "table_3scenarios_characteristics.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
