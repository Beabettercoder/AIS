#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 30: 全球多港湾基准全景图（世界地图 + 8 个区域密度/中心线子图）。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

BASE = Path("/mnt/sda/WangZijia/AIS")
GEO = BASE.parent / "AIS" / "data" / "geo"

REGIONS = [
    ("Houston A", "data/interim/ais_houston_202407_enriched.parquet", -95.05, 29.35, "Type III", "Gulf of Mexico", "open bay / straight transit"),
    ("Houston B", "data/interim/ais_houston_202407_enriched.parquet", -94.95, 29.72, "Type I", "Gulf of Mexico", "bent river channel"),
    ("New York", "data/interim/ais_ny_enriched.parquet", -74.05, 40.65, "Type II", "North Atlantic", "estuary confluence"),
    ("Los Angeles", "data/interim/ais_la_enriched.parquet", -118.24, 33.74, "Type III", "North Pacific", "multi-lane TSS bay"),
    ("Puget Sound", "data/interim/ais_puget_enriched.parquet", -122.90, 47.95, "Type IV", "North Pacific", "narrow intl. strait"),
    ("Miami", "data/interim/ais_miami_enriched.parquet", -80.05, 25.80, "Type IV", "Atlantic/Caribbean", "strong-current strait"),
    ("Honolulu", "data/interim/ais_honolulu_enriched.parquet", -157.90, 21.30, "Type V", "Central Pacific", "oceanic island hub"),
    ("San Juan", "data/interim/ais_sanjuan_enriched.parquet", -66.10, 18.45, "Type V", "Caribbean", "semi-enclosed bay"),
]


def load_xy(rel: str, lat_c: float, lon_c: float, n: int = 120000):
    p = BASE / rel
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["x_m", "y_m", "sog", "lat", "lon"])
    if len(df) > n:
        df = df.sample(n, random_state=0)
    return df


def draw_basemap(ax):
    land = GEO / "ne_110m_land.geojson"
    if not land.exists():
        return
    gj = json.loads(land.read_text())
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                arr = np.asarray(ring)
                ax.fill(arr[:, 0], arr[:, 1], color="#e8e8e8", edgecolor="#b0b0b0", linewidth=0.4, zorder=0)


def main() -> None:
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 1, 1], hspace=0.32, wspace=0.18)

    ax_map = fig.add_subplot(gs[0, :])
    draw_basemap(ax_map)
    ax_map.set_xlim(-180, 30)
    ax_map.set_ylim(-30, 62)
    ax_map.set_title("Global multi-harbour AIS benchmark — 8 regions, 5 hydro-geometric harbour types", fontsize=13)
    ax_map.set_xlabel("Longitude"); ax_map.set_ylabel("Latitude")
    ax_map.grid(alpha=0.25)

    colors = {"Type I": "#8e44ad", "Type II": "#c0392b", "Type III": "#1e8449", "Type IV": "#2471a3", "Type V": "#d68910"}
    for name, rel, lon, lat, htype, ocean, desc in REGIONS:
        ax_map.scatter(lon, lat, s=140, color=colors[htype], edgecolor="k", zorder=5)
        dy = -16 if name == "Houston B" else 6
        ax_map.annotate(f"{name}\n({htype})", (lon, lat), textcoords="offset points", xytext=(7, dy), fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in colors.items()]
    ax_map.legend(handles=handles, loc="lower left", fontsize=8, title="Harbour type")

    for i, (name, rel, lon, lat, htype, ocean, desc) in enumerate(REGIONS):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        df = load_xy(rel, lat, lon)
        if df is None:
            ax.set_axis_off(); continue
        # 以本区中心为原点的局部 nm 坐标
        x = (df.lon - lon) * 60 * np.cos(np.radians(lat))
        y = (df.lat - lat) * 60
        ax.hexbin(x, y, gridsize=90, bins="log", cmap="viridis", mincnt=1)
        ax.set_aspect("equal")
        ax.set_xlim(-14, 14); ax.set_ylim(-14, 14)
        ax.set_title(f"{name} — {htype}\n{desc}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[htype]); spine.set_linewidth(1.6)

    out = BASE / "outputs/phase1/fig1_global_ports_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
