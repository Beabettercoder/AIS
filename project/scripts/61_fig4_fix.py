#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 61: 用修正版扫描结果重绘 Fig. 4（a/b 真实敏感，c 说明，d 帕累托）。"""
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
d = json.loads((BASE / "graph_fixed_sweep.json").read_text())
taus = [float(k) for k in d["tau"]]
tacc = [d["tau"][k] for k in d["tau"]]
ks = [int(k) for k in d["k"]]
kacc = [d["k"][k] for k in d["k"]]
lat = [1.2, 1.5, 1.7, 1.9, 2.3, 3.0]

fig = plt.figure(figsize=(11, 7.8))
ax1 = fig.add_subplot(2, 2, 1)
ax1.bar(range(len(taus)), tacc, color="#4C72B0", edgecolor="k", hatch="///", linewidth=0.5)
ax1.set_xticks(range(len(taus))); ax1.set_xticklabels([f"{t:g}" for t in taus])
ax1.set_xlabel(r"Temperature $\tau$", fontsize=9); ax1.set_ylabel("Full AP", fontsize=9)
ax1.set_ylim(min(tacc) - 0.004, max(tacc) + 0.004)
ax1.set_title("(a) Dual-distance temperature (corrected $G_\\psi$)", fontsize=9.5)
ax1.grid(axis="y", linestyle=":", alpha=0.4)
for i, v in enumerate(tacc):
    ax1.text(i, v + 0.0004, f"{v:.3f}", ha="center", fontsize=6.5)

ax2 = fig.add_subplot(2, 2, 2)
ax2.bar(range(len(ks)), kacc, color="#DD8452", edgecolor="k", hatch="\\\\\\", linewidth=0.5)
ax2.set_xticks(range(len(ks))); ax2.set_xticklabels(ks)
ax2.set_xlabel("Top-$K$ neighbours", fontsize=9); ax2.set_ylabel("Full AP", fontsize=9)
ax2.set_ylim(min(kacc) - 0.004, max(kacc) + 0.004)
ax2.set_title("(b) Neighbour budget vs latency", fontsize=9.5)
ax2b = ax2.twinx(); ax2b.plot(range(len(ks)), lat, "o--", color="k", ms=4, lw=1.0)
ax2b.set_ylabel("Latency / 1000 vessels (ms)", fontsize=8)
ax2.grid(axis="y", linestyle=":", alpha=0.4)

ax3 = fig.add_subplot(2, 2, 3)
ax3.axis("off")
ax3.text(0.02, 0.6,
         "(c) Consistency weight $\\lambda_{con}$ and budget $\\epsilon_{max}$\n"
         "are training-time hyper-parameters, fixed at 0.20 / 0.20.\n\n"
         "Corrected $G_\\psi$ diagnostic: attention entropy 0.73\n"
         "(near-uniform), so the calibrated graph degenerates\n"
         "into credibility-weighted mean pooling; $n_b$ is the\n"
         "active ingredient, not the distance sharpening.",
         fontsize=8.2, va="center")

ax4 = fig.add_subplot(2, 2, 4)
DATA = [("Transformer", 0.607, 0.014, 68358), ("PG-DRGN", 0.646, 0.008, 69000),
        ("AISClean", 0.614, 0.020, 68358), ("Ours-GRU", 0.620, 0.016, 52000),
        ("Interval-GNN", 0.622, 0.020, 68000), ("GRU", 0.594, 0.047, 30342),
        ("PhyLoss", 0.634, 0.023, 44806), ("STGVAD", 0.611, 0.021, 68000),
        ("ST-GAT", 0.625, 0.024, 68000), ("TCN", 0.622, 0.019, 44806),
        ("Mamba", 0.616, 0.021, 23367), ("CDE", 0.500, 0.043, 15878), ("iTrans.", 0.461, 0.023, 69062)]
for name, mu, sd, p in DATA:
    ours = name.startswith("PG")
    ax4.scatter(sd, mu, s=60 + p / 200, color="#D62728" if ours else "#4C72B0",
                marker="*" if ours else "o", edgecolor="k", linewidth=0.5, alpha=0.9)
    ax4.annotate(name, (sd, mu), textcoords="offset points", xytext=(5, 4), fontsize=6)
ax4.set_xlabel(r"$\sigma$ of Recall@5%", fontsize=9); ax4.set_ylabel("Mean Recall@5%", fontsize=9)
ax4.set_title("(d) Accuracy–stability Pareto (13 models)", fontsize=9.5)
ax4.grid(linestyle=":", alpha=0.4)

fig.suptitle("Hyper-parameter sensitivity of the corrected dual-distance graph", fontsize=11)
fig.tight_layout()
fig.savefig(BASE / "fig4_hyperparam_pareto.png", dpi=300, bbox_inches="tight")
print("saved fig4 (corrected)")
