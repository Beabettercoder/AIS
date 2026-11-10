#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 64: 解析场 vs 学习图 vs 两者并用——参数量与精度的对照。"""
from __future__ import annotations

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
names = ["Analytic field\n(3.3 LOO, 0 graph params)", "Learned graph\n(no field)", "Graph + field\n(current Ours)"]
acc = [0.839, 0.829, 0.837]
params = [44806, 78000, 66825]

fig, ax = plt.subplots(figsize=(7.4, 4.0))
x = np.arange(len(names))
bars = ax.bar(x, acc, 0.5, color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="k",
              hatch=["///", "\\\\\\", "xxx"], linewidth=0.6)
for i, (b, p) in enumerate(zip(bars, params)):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0015,
            f"{acc[i]:.3f}\n{p/1000:.0f}k params", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylim(0.80, 0.855); ax.set_ylabel("Full AP", fontsize=10)
ax.set_title("Group context: analytic field vs learned graph aggregation", fontsize=10.5)
ax.grid(axis="y", linestyle=":", alpha=0.4)
fig.tight_layout()
fig.savefig(BASE / "fig8_field_vs_graph.png", dpi=300, bbox_inches="tight")
print("saved fig8")
