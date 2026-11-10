#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 121: 多源轮换汇总表 + 图。"""
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
d = json.loads((BASE / "multisource_zeroshot.json").read_text())
SRC = ["ny", "la", "puget", "miami", "gom_open", "atl_open"]
MODELS = ["Raw-TCN", "GRU", "Ours-TCN"]

rows = []
for s in SRC:
    r = {"source": s}
    for m in MODELS:
        r[m] = float(np.mean([v["ap"] for v in d[m][s].values()]))
    r["n"] = len(d["Raw-TCN"][s])
    rows.append(r)

lines = ["# Multi-source rotation zero-shot (each region used once as source)", "",
         "| Source domain | Domains | Raw-TCN | GRU | **Ours-TCN** | Δ (Ours − best baseline) |",
         "|---|---|---|---|---|---|"]
for r in rows:
    best_base = max(r["Raw-TCN"], r["GRU"])
    lines.append(f"| {r['source']} | {r['n']} | {r['Raw-TCN']:.3f} | {r['GRU']:.3f} | "
                 f"**{r['Ours-TCN']:.3f}** | {r['Ours-TCN']-best_base:+.3f} |")
mean = {m: np.mean([r[m] for r in rows]) for m in MODELS}
lines.append(f"| **Mean** | | {mean['Raw-TCN']:.3f} | {mean['GRU']:.3f} | **{mean['Ours-TCN']:.3f}** | "
             f"{mean['Ours-TCN']-max(mean['Raw-TCN'], mean['GRU']):+.3f} |")
lines += ["", "Houston-as-source reference (Table II): Raw-TCN 0.780 / GRU 0.797 / Ours-TCN 0.805."]
txt = "\n".join(lines)
(BASE / "table2_multisource.md").write_text(txt, encoding="utf-8")
print(txt)

fig, ax = plt.subplots(figsize=(9.2, 4.2))
x = np.arange(len(SRC)); w = 0.26
colors = {"Raw-TCN": "#9a9a9a", "GRU": "#4C72B0", "Ours-TCN": "#D62728"}
for i, m in enumerate(MODELS):
    vals = [r[m] for r in rows]
    ax.bar(x + (i - 1) * w, vals, w, color=colors[m], edgecolor="k", linewidth=0.4,
           hatch="//" if m == "Ours-TCN" else None, label=m)
for i, r in enumerate(rows):
    ax.text(i + w, r["Ours-TCN"] + 0.012, f"{r['Ours-TCN']:.3f}", ha="center", fontsize=7, color="#B01F1F")
ax.axhline(0.78, color="0.4", ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels([s.replace("_", " ") for s in SRC], fontsize=8, rotation=12)
ax.set_ylabel("Mean zero-shot Full AP", fontsize=9); ax.set_ylim(0, 0.9)
ax.set_title("Zero-shot transfer when each region serves once as the source domain", fontsize=10)
ax.legend(fontsize=7, frameon=False); ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(BASE / "fig2_multisource_rotation.png", dpi=300, bbox_inches="tight")
print("saved fig2_multisource_rotation.png")
