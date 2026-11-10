#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 44: Fig. 2 — strict zero-shot profile across three scenarios (TGSA style)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
rows = json.loads((BASE / "table2_strict_zeroshot.json").read_text())

order = ["Unseen Ports", "Unseen Straits", "Unseen Open Seas"]
colors = {"Unseen Ports": "#dbe9f6", "Unseen Straits": "#fde2e2", "Unseen Open Seas": "#e2f2e2"}
labels = [f"{r['region']}" for r in rows]

fig, ax = plt.subplots(figsize=(9, 4.4))
x = np.arange(len(rows))
ax.set_xlim(-0.6, len(rows) - 0.4)

start = 0
for scen in order:
    idx = [i for i, r in enumerate(rows) if r["scenario"] == scen]
    if not idx:
        continue
    lo, hi = min(idx) - 0.5, max(idx) + 0.5
    ax.add_patch(Rectangle((lo, 0.55), hi - lo, 0.5, color=colors[scen], zorder=0))
    ax.text((lo + hi) / 2, 0.972, scen, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=6)
    start = hi

s2 = [r["S2_full"] for r in rows]
s2p = [r["S2p_full"] for r in rows]
ax.plot(x, s2, "--o", color="#7f7f7f", mfc="white", ms=6, lw=1.3, label="S2 strict zero-shot")
ax.plot(x, s2p, "-*", color="#D62728", ms=13, lw=1.6, label="S2+ unsupervised recalibration (Ours)")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Full AP", fontsize=10)
ax.set_ylim(0.55, 1.02)
ax.set_title("Strict zero-shot transfer from ports to unseen ports, straits and open seas", fontsize=11, pad=14)
ax.grid(axis="y", linestyle=":", alpha=0.45)
ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

fig.tight_layout()
out = BASE / "fig2_strict_zeroshot_profile.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
