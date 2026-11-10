#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 40: 三大场景全球基准世界地图（英文，Times New Roman，300 dpi）。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

BASE = Path("/mnt/sda/WangZijia/AIS")
GEO = BASE / "data" / "geo"

PORTS = [("P1 Houston", 29.50, -94.85, "ais_houston_202407_enriched.parquet"),
         ("P2 New York", 40.65, -74.05, "ais_ny_enriched.parquet"),
         ("P3 Los Angeles", 33.74, -118.24, "ais_la_enriched.parquet"),
         ("P4 Mississippi", 29.15, -89.25, "ais_mississippi_enriched.parquet"),
         ("P5 San Juan", 18.45, -66.10, "ais_sanjuan_enriched.parquet"),
         ("P6 Honolulu", 21.30, -157.90, "ais_honolulu_enriched.parquet"),
         ("P7 Guam", 13.50, 144.70, "ais_guam_enriched.parquet"),
         ("P8 Golden Gate", 37.80, -122.50, "ais_goldengate_enriched.parquet")]
STRAITS = [("S1 Juan de Fuca", 48.25, -123.50, "ais_puget_enriched.parquet"),
           ("S2 Florida Str.", 25.80, -80.05, "ais_miami_enriched.parquet")]
OPEN = [("O1 Outer GoM", 26.90, -92.75, "ais_gom_open_enriched.parquet"),
        ("O2 NW Atlantic", 37.65, -72.00, "ais_atl_open_enriched.parquet"),
        ("O3 NE Pacific", 32.35, -119.90, "ais_pac_open_enriched.parquet")]
PENDING = []

COL = {"Ports": "#1f77b4", "Straits": "#d62728", "Open Seas": "#2ca02c"}
MARK = {"Ports": "o", "Straits": "^", "Open Seas": "s"}


def basemap(ax):
    gj = json.loads((GEO / "ne_110m_land.geojson").read_text())
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                a = np.asarray(ring)
                ax.fill(a[:, 0], a[:, 1], color="#ececec", edgecolor="#9a9a9a", linewidth=0.35, zorder=0)


def inset(ax, fname, lat, lon, title, color):
    p = BASE / "data/interim" / fname
    if not p.exists():
        ax.set_axis_off(); return
    df = pd.read_parquet(p, columns=["lat", "lon"]).sample(min(80000, len(pd.read_parquet(p, columns=["lat"]))), random_state=0)
    x = (df.lon - lon) * 60 * np.cos(np.radians(lat))
    y = (df.lat - lat) * 60
    ax.hexbin(x, y, gridsize=80, bins="log", cmap="viridis", mincnt=1)
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_aspect("equal")
    ax.set_title(title, fontsize=7.5, color=color, fontweight="bold")
    ax.tick_params(labelsize=5.5, length=2)
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(1.4)


def main() -> None:
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.5, 1, 1], hspace=0.30, wspace=0.16)
    ax = fig.add_subplot(gs[0, :])
    basemap(ax)
    ax.set_xlim(-172, 152); ax.set_ylim(-45, 75)
    ax.set_xlabel("Longitude ($^\\circ$)", fontsize=10)
    ax.set_ylabel("Latitude ($^\\circ$)", fontsize=10)
    ax.set_title("Global AIS benchmark across three maritime scenarios: ports, chokepoint straits and open seas",
                 fontsize=12.5, pad=10)
    ax.grid(alpha=0.22, linestyle=":")
    ax.tick_params(labelsize=8)

    for name, lat, lon, _ in PORTS:
        ax.plot(lon, lat, MARK["Ports"], color=COL["Ports"], ms=8, mec="k", mew=0.5, zorder=5)
    for name, lat, lon, _ in STRAITS:
        ax.plot(lon, lat, MARK["Straits"], color=COL["Straits"], ms=9, mec="k", mew=0.5, zorder=5)
    for name, lat, lon, _ in OPEN:
        ax.plot(lon, lat, MARK["Open Seas"], color=COL["Open Seas"], ms=8, mec="k", mew=0.5, zorder=5)
    for name, lat, lon in PENDING:
        ax.plot(lon, lat, "x", color="0.45", ms=7, mew=1.2, zorder=4, alpha=0.8)

    offsets = {"P1 Houston": (-70, 8), "P2 New York": (10, 9), "P3 Los Angeles": (-82, 10),
               "P4 Mississippi": (12, -26), "P5 San Juan": (8, 2), "P6 Honolulu": (8, -2),
               "S1 Juan de Fuca": (-58, 12), "S2 Florida Str.": (-92, -18),
               "O1 Outer GoM": (10, -30), "O2 NW Atlantic": (12, -12), "O3 NE Pacific": (-72, -14)}
    for name, lat, lon, _ in PORTS + STRAITS + OPEN:
        ax.annotate(name, (lon, lat), textcoords="offset points", xytext=offsets.get(name, (5, 5)), fontsize=7)
    for name, lat, lon in PENDING:
        ax.annotate(name, (lon, lat), textcoords="offset points", xytext=(5, 5), fontsize=6.5, color="0.45")

    handles = [Line2D([], [], marker=MARK[k], ls="", color=COL[k], mec="k", label=f"{k} (n={sum(1 for x in v if x[3])})")
               for k, v in (("Ports", PORTS), ("Straits", STRAITS), ("Open Seas", OPEN))]
    
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.95, ncol=2)

    picks = [PORTS[0], PORTS[1], PORTS[2], PORTS[4], STRAITS[0], STRAITS[1], OPEN[0], OPEN[1]]
    for i, (name, lat, lon, fname) in enumerate(picks):
        sub = fig.add_subplot(gs[1 + i // 4, i % 4])
        scen = "Ports" if name.startswith("P") else ("Straits" if name.startswith("S") else "Open Seas")
        inset(sub, fname, lat, lon, name, COL[scen])

    out = BASE / "outputs/phase1/fig1_global_world_map_3scenarios.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
