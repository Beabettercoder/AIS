#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 29: 各港湾物理特征汇总（供 Table global ports characteristics）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NM = 1852.0
RADIUS_M = 8 * NM
MIN_NBRS = 4
MAX_DCOG = np.pi / 4

REGIONS = {
    "houston": ("data/interim/ais_houston_202407_enriched.parquet", "Type I/II (channel + bay)", "Gulf of Mexico"),
    "la": ("data/interim/ais_la_enriched.parquet", "Type III (open bay, multi-lane TSS)", "North Pacific"),
    "ny": ("data/interim/ais_ny_enriched.parquet", "Type II (estuary confluence)", "North Atlantic"),
    "puget": ("data/interim/ais_puget_enriched.parquet", "Type IV (narrow international strait)", "North Pacific"),
    "miami": ("data/interim/ais_miami_enriched.parquet", "Type IV (strong-current strait)", "Atlantic / Caribbean"),
    "honolulu": ("data/interim/ais_honolulu_enriched.parquet", "Type V (oceanic island hub)", "Central Pacific"),
    "sanjuan": ("data/interim/ais_sanjuan_enriched.parquet", "Type V (semi-enclosed caribbean bay)", "Caribbean"),
    "mississippi": ("data/interim/ais_mississippi_enriched.parquet", "Type I (river bend passes)", "Gulf of Mexico"),
}


def region_stats(path: Path) -> dict:
    df = pd.read_parquet(path, columns=["mmsi", "timestamp", "x_m", "y_m", "sog", "cog", "kin_ok"])
    mov = df[df.kin_ok & df.sog.ge(2.0)]
    if len(mov) == 0:
        return {}
    # 转向率
    g = mov.sort_values(["mmsi", "timestamp"])
    dchi = np.abs((np.diff(g.cog.to_numpy(), prepend=g.cog.iloc[0]) + 180) % 360 - 180)
    dt = np.diff(g.timestamp.to_numpy(), prepend=g.timestamp.iloc[0]).astype(float)
    ok = (dt > 0) & (dt < 600)
    turn = np.deg2rad(dchi[ok]) / dt[ok]
    # 场有效率 + 散度/旋度方差（抽样 200 个 10 分钟快照）
    mov = mov.assign(bin=(mov.timestamp // 600) * 600)
    bins = mov.bin.drop_duplicates().sample(min(200, mov.bin.nunique()), random_state=0) if mov.bin.nunique() else []
    valid_n = total_n = 0
    divs, curls = [], []
    for b in bins:
        sub = mov[mov.bin == b]
        if len(sub) < 6:
            continue
        pos = np.column_stack([sub.x_m, sub.y_m])
        cog = np.radians(sub.cog.to_numpy())
        vx = sub.sog.to_numpy() * 0.514444 * np.sin(cog)
        vy = sub.sog.to_numpy() * 0.514444 * np.cos(cog)
        tree = cKDTree(pos)
        balls = tree.query_ball_point(pos, r=RADIUS_M)
        for i, cand in enumerate(balls):
            total_n += 1
            c = np.asarray(cand); c = c[c != i]
            if len(c) < MIN_NBRS:
                continue
            dd = np.hypot(pos[c, 0] - pos[i, 0], pos[c, 1] - pos[i, 1])
            keep = c[(dd <= RADIUS_M) & (np.abs((cog[c] - cog[i] + np.pi) % (2 * np.pi) - np.pi) <= MAX_DCOG)]
            if len(keep) < MIN_NBRS:
                continue
            valid_n += 1
            dx = (pos[keep, 0] - pos[i, 0]) / RADIUS_M
            dy = (pos[keep, 1] - pos[i, 1]) / RADIUS_M
            X = np.column_stack([np.ones_like(dx), dx, dy])
            try:
                th, *_ = np.linalg.lstsq(X, np.column_stack([vx[keep], vy[keep]]), rcond=None)
            except np.linalg.LinAlgError:
                continue
            divs.append((th[1, 0] + th[2, 1]) / RADIUS_M)
            curls.append((th[1, 1] - th[2, 0]) / RADIUS_M)
    return {
        "reports_moving": int(len(mov)),
        "vessels": int(mov.mmsi.nunique()),
        "dt_median_s": float(np.median(dt[ok])) if ok.any() else np.nan,
        "sog_median_kn": float(mov.sog.median()),
        "turn_rate_median_rad_s": float(np.median(turn)) if len(turn) else np.nan,
        "field_valid_ratio": valid_n / max(total_n, 1),
        "div_std": float(np.std(divs)) if divs else np.nan,
        "curl_std": float(np.std(curls)) if curls else np.nan,
    }


def main() -> None:
    base = Path("/mnt/sda/WangZijia/AIS")
    rows = []
    for name, (rel, htype, ocean) in REGIONS.items():
        p = base / rel
        if not p.exists():
            print(f"skip {name} (missing {rel})")
            continue
        s = region_stats(p)
        if not s:
            continue
        s.update({"region": name, "harbour_type": htype, "ocean": ocean})
        rows.append(s)
        print(name, s, flush=True)
    df = pd.DataFrame(rows)[["region", "ocean", "harbour_type", "vessels", "reports_moving",
                             "dt_median_s", "sog_median_kn", "turn_rate_median_rad_s",
                             "field_valid_ratio", "div_std", "curl_std"]]
    out = base / "outputs/phase1/table_global_ports_characteristics.csv"
    df.to_csv(out, index=False)
    lines = ["# Global harbour physical characteristics", "",
             "| Region | Ocean | Harbour type | Vessels | Reports | Δt (s) | SOG (kn) | Turn rate (rad/s) | Field valid | σ(div) | σ(curl) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.region} | {r.ocean} | {r.harbour_type} | {r.vessels} | {r.reports_moving:,} | "
                     f"{r.dt_median_s:.0f} | {r.sog_median_kn:.1f} | {r.turn_rate_median_rad_s:.5f} | "
                     f"{r.field_valid_ratio:.3f} | {r.div_std:.2e} | {r.curl_std:.2e} |")
    txt = "\n".join(lines)
    (base / "outputs/phase1/table_global_ports_characteristics.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
