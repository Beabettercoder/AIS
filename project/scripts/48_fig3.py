#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 48: Table 3 lower block + Fig. 3 hatched ablation bars."""
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
rows = json.loads((BASE / "ablation_full.json").read_text())

ORDER = ["Baseline (Raw TCN)", "+ KG (3.1 Gate)", "+ KG + LF (3.3 Field)",
         "+ KG + LF + PSA", "PG-DRGN (Ours, Full)"]
SHORT = ["Baseline", "+KG", "+KG+LF", "+KG+LF\n+PSA", "PG-DRGN\n(Ours)"]
METRICS = ["clean_AP", "full_AP", "recall_1pct", "recall_5pct"]

agg = {}
for r in rows:
    agg.setdefault(r["config"], {}).setdefault(r["seed"], r)
stats = {}
for cfg in ORDER:
    d = agg[cfg]
    stats[cfg] = {m: (float(np.mean([d[s][m] for s in d])), float(np.std([d[s][m] for s in d], ddof=1)))
                  for m in METRICS}

lines = ["# Table 3 (lower block) — progressive component ablation (5 seeds)", "",
         "| Configuration | KG | LF | PSA | PG+CDG | Clean AP | Full AP | Recall@1% | Recall@5% |",
         "|---|---|---|---|---|---|---|---|---|"]
flags = {"Baseline (Raw TCN)": "✗ ✗ ✗ ✗", "+ KG (3.1 Gate)": "✓ ✗ ✗ ✗",
         "+ KG + LF (3.3 Field)": "✓ ✓ ✗ ✗", "+ KG + LF + PSA": "✓ ✓ ✓ ✗",
         "PG-DRGN (Ours, Full)": "✓ ✓ ✓ ✓"}
for cfg in ORDER:
    f = flags[cfg].split()
    s = stats[cfg]
    lines.append(f"| {cfg} | {f[0]} | {f[1]} | {f[2]} | {f[3]} | "
                 f"{s['clean_AP'][0]:.3f} ± {s['clean_AP'][1]:.3f} | {s['full_AP'][0]:.3f} ± {s['full_AP'][1]:.3f} | "
                 f"{s['recall_1pct'][0]:.3f} ± {s['recall_1pct'][1]:.3f} | {s['recall_5pct'][0]:.3f} ± {s['recall_5pct'][1]:.3f} |")
txt = "\n".join(lines)
(BASE / "table3_ablation_full.md").write_text(txt, encoding="utf-8")
print(txt)

fig, ax = plt.subplots(figsize=(8.6, 4.4))
x = np.arange(len(ORDER))
width = 0.2
hatches = ["///", "\\\\\\", "xxx", "..."]
labels = ["Clean AP", "Full AP", "Recall@1%", "Recall@5%"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for i, (m, lab) in enumerate(zip(METRICS, labels)):
    vals = [stats[c][m][0] for c in ORDER]
    errs = [stats[c][m][1] for c in ORDER]
    ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=2.5,
           label=lab, color=colors[i], edgecolor="k", linewidth=0.5, hatch=hatches[i], alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(SHORT, fontsize=8.5)
ax.set_ylabel("Score", fontsize=10)
ax.set_ylim(0.30, 0.92)
ax.set_title("Progressive component ablation: from raw TCN to full PG-DRGN", fontsize=11)
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
ax.grid(axis="y", linestyle=":", alpha=0.4)
fig.tight_layout()
out = BASE / "fig3_ablation_hatched_bars.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
