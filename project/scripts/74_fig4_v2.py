#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 74: Fig.4 精修（放大 y 轴 / 三级异常分类 / 帕累托去重叠）。"""
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
lat = [0.9, 1.1, 1.3, 1.6, 2.1, 2.9]

fig = plt.figure(figsize=(12, 8.2))
ax1 = fig.add_subplot(2, 2, 1)
ax1.bar(range(len(taus)), tacc, color="#4C72B0", edgecolor="k", hatch="///", linewidth=0.6)
ax1.set_xticks(range(len(taus))); ax1.set_xticklabels([f"{t:g}" for t in taus])
ax1.set_xlabel(r"Temperature $\tau$", fontsize=9.5); ax1.set_ylabel("Full AP", fontsize=9.5)
ax1.set_ylim(0.818, 0.830)
ax1.set_title("(a) Dual-distance temperature", fontsize=10)
ax1.grid(axis="y", linestyle=":", alpha=0.4)
for i, v in enumerate(tacc):
    ax1.text(i, v + 0.0003, f"{v:.3f}", ha="center", fontsize=7)
ax1.annotate("peak", xy=(1.5, max(tacc)), xytext=(2.4, max(tacc) + 0.0015),
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.7), fontsize=7.5, color="#D62728")

ax2 = fig.add_subplot(2, 2, 2)
ax2.bar(range(len(ks)), kacc, color="#DD8452", edgecolor="k", hatch="\\\\\\", linewidth=0.6)
ax2.set_xticks(range(len(ks))); ax2.set_xticklabels(ks)
ax2.set_xlabel("Top-$K$ neighbours", fontsize=9.5); ax2.set_ylabel("Full AP", fontsize=9.5)
ax2.set_ylim(0.800, 0.826)
ax2.set_title("(b) Neighbour budget vs inference latency", fontsize=10)
for i, v in enumerate(kacc):
    ax2.text(i, v + 0.0003, f"{v:.3f}", ha="center", fontsize=7)
ax2b = ax2.twinx()
ax2b.plot(range(len(ks)), lat, "o--", color="k", ms=4, lw=1.1)
ax2b.set_ylabel("Latency per 1000 vessels (ms)", fontsize=8.5)
ax2b.set_ylim(0.6, 3.4)
ax2.grid(axis="y", linestyle=":", alpha=0.4)

ax3 = fig.add_subplot(2, 2, 3)
groups = ["Tier I\nSelf-referential", "Tier II\nDeparture (900w)", "Tier II\nDeparture (4500w)",
          "Tier III\nGroup (tight)", "Tier III\nGroup (all)"]
single = [0.839, 0.471, 0.775, 0.501, 0.501]
ours = [0.837, 0.579, 0.795, 0.989, 0.972]
x = np.arange(len(groups)); w = 0.36
ax3.bar(x - w / 2, single, w, label="Single-vessel backbone", color="#9ecae1", edgecolor="k", hatch="///", linewidth=0.5)
ax3.bar(x + w / 2, ours, w, label="Ours w/ calibrated graph", color="#fc9272", edgecolor="k", hatch="\\\\\\", linewidth=0.5)
ax3.set_xticks(x); ax3.set_xticklabels(groups, fontsize=6.8)
ax3.set_ylim(0, 1.12); ax3.set_ylabel("AP", fontsize=9.5)
ax3.set_title("(c) Three-tier anomaly taxonomy", fontsize=10)
ax3.legend(fontsize=7, loc="upper right", framealpha=0.95)
ax3.grid(axis="y", linestyle=":", alpha=0.4)
for i in range(len(groups)):
    ax3.text(i - w / 2, single[i] + 0.02, f"{single[i]:.3f}", ha="center", fontsize=6)
    ax3.text(i + w / 2, ours[i] + 0.02, f"{ours[i]:.3f}", ha="center", fontsize=6)

ax4 = fig.add_subplot(2, 2, 4)
DATA = [("Transformer", 0.607, 0.014, 68358, (-54, -24)), ("PG-DRGN (Ours)", 0.646, 0.008, 69000, (12, 8)),
        ("AISClean", 0.614, 0.020, 68358, (-56, 20)), ("Ours (GRU)", 0.620, 0.016, 52000, (6, 30)),
        ("Interval-GNN", 0.622, 0.020, 68000, (34, -30)), ("GRU", 0.594, 0.047, 30342, (-30, 8)),
        ("PhyLoss", 0.634, 0.023, 44806, (24, 14)), ("STGVAD", 0.611, 0.021, 68000, (-62, -30)),
        ("ST-GAT", 0.625, 0.024, 68000, (40, -16)), ("TCN", 0.622, 0.019, 44806, (-30, -44)),
        ("Mamba", 0.616, 0.021, 23367, (20, -44)), ("CDE", 0.500, 0.043, 15878, (-30, -14)),
        ("iTrans.", 0.461, 0.023, 69062, (-40, -16))]
for name, mu, sd, p, off in DATA:
    o = name.startswith("PG")
    ax4.scatter(sd, mu, s=60 + p / 200, color="#D62728" if o else "#4C72B0",
                marker="*" if o else "o", edgecolor="k", linewidth=0.5, alpha=0.9, zorder=5 if o else 3)
    ax4.annotate(name, (sd, mu), textcoords="offset points", xytext=off, fontsize=6,
                 arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, shrinkA=0, shrinkB=2))
ax4.set_xlabel(r"$\sigma$ of Recall@5%", fontsize=9.5); ax4.set_ylabel("Mean Recall@5%", fontsize=9.5)
ax4.set_title("(d) Accuracy–stability Pareto (13 models)", fontsize=10)
ax4.set_ylim(0.44, 0.695)
ax4.grid(linestyle=":", alpha=0.4)

fig.suptitle("Hyper-parameter sensitivity, anomaly taxonomy and model frontier", fontsize=11.5)
fig.tight_layout()
fig.savefig(BASE / "fig4_hyperparam_pareto.png", dpi=300, bbox_inches="tight")
print("saved fig4")
