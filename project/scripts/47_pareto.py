#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 47: Fig. 4(d) — 13-model Pareto bubble (x: sigma R@5, y: mean R@5, size: params)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")

# (name, mean R@5, std R@5, params)
DATA = [
    ("Transformer", 0.607, 0.014, 68358),
    ("PG-DRGN (Ours)", 0.646, 0.008, 69000),
    ("AISClean+Trans.", 0.614, 0.020, 68358),
    ("Ours (GRU back.)", 0.620, 0.016, 52000),
    ("Interval-GNN", 0.622, 0.020, 68000),
    ("GRU", 0.594, 0.047, 30342),
    ("PhyLoss-TCN", 0.634, 0.023, 44806),
    ("STGVAD", 0.611, 0.021, 68000),
    ("ST-GAT", 0.625, 0.024, 68000),
    ("Mamba-lite", 0.616, 0.021, 23367),
    ("TCN", 0.622, 0.019, 44806),
    ("Neural-CDE", 0.500, 0.043, 15878),
    ("iTransformer", 0.461, 0.023, 69062),
]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for name, mu, sd, prm in DATA:
    ours = name.startswith("PG-DRGN")
    color = "#D62728" if ours else "#4C72B0"
    marker = "*" if ours else "o"
    size = 120 + prm / 120.0
    ax.scatter(sd, mu, s=size, color=color, marker=marker, alpha=0.85 if not ours else 1.0,
               edgecolor="k", linewidth=0.6, zorder=5 if ours else 3)
    ax.annotate(name, (sd, mu), textcoords="offset points", xytext=(7, 5), fontsize=7)

ax.set_xlabel(r"Standard deviation of Recall@5%  $\sigma$", fontsize=10)
ax.set_ylabel("Mean Recall@5%", fontsize=10)
ax.set_title("Accuracy–stability Pareto frontier across 13 models (10 seeds)", fontsize=11)
ax.grid(alpha=0.3, linestyle=":")
lo = min(d[1] for d in DATA) - 0.02
ax.set_ylim(lo, 0.67)
ax.axhline(0.646, color="#D62728", lw=0.8, ls="--", alpha=0.5)
ax.axvline(0.008, color="#D62728", lw=0.8, ls="--", alpha=0.5)
ax.annotate("Pareto corner:\nhighest accuracy + lowest variance", (0.008, 0.646),
            textcoords="offset points", xytext=(28, -34), fontsize=7.5, color="#D62728")
fig.tight_layout()
out = BASE / "fig4d_pareto_bubble.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
