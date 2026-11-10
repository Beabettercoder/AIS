#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 124: 终版跨海域零样本大图 + 三合一表。"""
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
ms = json.loads((BASE / "multisource_zeroshot.json").read_text())
jt = json.loads((BASE / "joint4_zeroshot.json").read_text())

NAME = {"ny": "New York\n(Port)", "la": "Los Angeles\n(Port)", "puget": "Juan de Fuca\n(Strait)",
        "miami": "Florida Str.\n(Strait)", "gom_open": "Outer GoM\n(Open)", "atl_open": "NW Atlantic\n(Open)"}
SRC = ["ny", "la", "puget", "miami", "gom_open", "atl_open"]
TGT = ["la", "mississippi", "honolulu", "miami", "goldengate", "chesapeake", "gom_open", "cpacific"]
TNAME = {"la": "Los Angeles", "mississippi": "Mississippi", "honolulu": "Honolulu", "miami": "Florida Str.",
         "goldengate": "Golden Gate", "chesapeake": "Chesapeake", "gom_open": "Outer GoM", "cpacific": "C. Pacific"}

rot = {m: [float(np.mean([v["ap"] for v in ms[m][s].values()])) for s in SRC]
       for m in ("Raw-TCN", "GRU", "Ours-TCN")}
joint = {m: float(np.mean([v["S2"][0] for v in jt[m].values()])) for m in jt}

fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})

ax = axes[0]
x = np.arange(len(SRC) + 1); w = 0.26
for i, (m, col, hatch) in enumerate((("Raw-TCN", "#b0b0b0", None), ("GRU", "#5b8fc9", None),
                                     ("Ours-TCN", "#d62728", "///"))):
    vals = rot[m] + [joint.get(m, np.nan)]
    ax.bar(x + (i - 1) * w, vals, w, color=col, edgecolor="k", linewidth=0.4, hatch=hatch, label=m)
for i in range(len(SRC)):
    best = max(rot["Raw-TCN"][i], rot["GRU"][i])
    ax.text(i + w, rot["Ours-TCN"][i] + 0.012, f"+{(rot['Ours-TCN'][i]-best)*100:.1f}", ha="center",
            fontsize=7, color="#B01F1F")
best_j = max(joint["Raw-TCN"], joint["GRU"])
ax.text(len(SRC) + w, joint["Ours-TCN"] + 0.012, f"+{(joint['Ours-TCN']-best_j)*100:.1f}",
        ha="center", fontsize=7, color="#B01F1F")
ax.set_xticks(x)
ax.set_xticklabels([NAME[s] for s in SRC] + ["Multi-Region\nJoint"], fontsize=7)
ax.set_ylabel("Mean zero-shot Full AP", fontsize=9); ax.set_ylim(0, 0.92)
ax.set_title("(a) Single-domain rotation vs. joint multi-region source", fontsize=10)
ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left"); ax.grid(axis="y", ls=":", alpha=0.4)

ax = axes[1]
x2 = np.arange(len(TGT)); w2 = 0.2
for i, (m, col, hatch) in enumerate((("Raw-TCN", "#b0b0b0", None), ("GRU", "#5b8fc9", None),
                                     ("Ours-TCN", "#d62728", "///"), ("Ours-GRU", "#f08a5d", "..."))):
    vals = [jt[m][t]["S2"][0] for t in TGT]
    ax.bar(x2 + (i - 1.5) * w2, vals, w2, color=col, edgecolor="k", linewidth=0.4, hatch=hatch, label=m)
for i, t in enumerate(TGT):
    ax.plot(i + 1.5 * w2, jt["Ours-TCN"][t]["S2p"][0], "k*", ms=7, zorder=5)
ax.plot([], [], "k*", ms=7, label="Ours-TCN (S2+)")
ax.set_xticks(x2); ax.set_xticklabels([TNAME[t] for t in TGT], fontsize=6.8, rotation=18)
ax.set_ylabel("Zero-shot Full AP", fontsize=9); ax.set_ylim(0.5, 1.0)
ax.set_title("(b) Eight unseen domains under joint training", fontsize=10)
ax.legend(fontsize=6.5, frameon=False, ncol=3); ax.grid(axis="y", ls=":", alpha=0.4)

fig.suptitle("Cross-domain zero-shot transfer: physics residuals stay calibrated when the source moves", fontsize=11)
fig.tight_layout()
fig.savefig(BASE / "fig2_zeroshot_dominance.png", dpi=300, bbox_inches="tight")
print("saved fig2_zeroshot_dominance.png")

L = ["# Table 2 (comprehensive) — cross-domain zero-shot transfer", "",
     "## A. Single-domain rotation (each region used once as the source)", "",
     "| Source domain | Targets | Raw-TCN | GRU | **Ours-TCN** | Δ vs best baseline |",
     "|---|---|---|---|---|---|"]
for i, s in enumerate(SRC):
    b = max(rot["Raw-TCN"][i], rot["GRU"][i])
    L.append(f"| {NAME[s].replace(chr(10), ' ')} | {len(ms['Raw-TCN'][s])} | {rot['Raw-TCN'][i]:.3f} | "
             f"{rot['GRU'][i]:.3f} | **{rot['Ours-TCN'][i]:.3f}** | {rot['Ours-TCN'][i]-b:+.3f} |")
mrot = {k: float(np.mean(v)) for k, v in rot.items()}
L.append(f"| **Mean** | | {mrot['Raw-TCN']:.3f} | {mrot['GRU']:.3f} | **{mrot['Ours-TCN']:.3f}** | "
         f"{mrot['Ours-TCN']-max(mrot['Raw-TCN'], mrot['GRU']):+.3f} |")

L += ["", "## B. Four-region joint source (Houston + New York + Juan de Fuca + NW Atlantic)", "",
      "| Model | " + " | ".join(TNAME[t] for t in TGT) + " | Mean (S2) | Mean (S2+) |",
      "|---|" + "---|" * (len(TGT) + 2)]
for m in ("Raw-TCN", "GRU", "Transformer", "Ours-TCN", "Ours-GRU"):
    cells = [f"{jt[m][t]['S2'][0]:.3f}" for t in TGT]
    s2 = np.mean([jt[m][t]["S2"][0] for t in TGT]); s2p = np.mean([jt[m][t]["S2p"][0] for t in TGT])
    L.append(f"| {m} | " + " | ".join(cells) + f" | **{s2:.3f}** | {s2p:.3f} |")

L += ["", "## C. Houston as single source (reference, Table II setting)", "",
      "| Model | Mean Full AP (11 domains) | FAR=1% recall |", "|---|---|---|",
      "| Raw-TCN | 0.780 | 0.328 |", "| GRU | 0.797 | 0.363 |", "| Ours-TCN | 0.805 | 0.414 |",
      "| Ours-GRU | 0.796 | 0.388 |", "",
      "S2 = normalisation with source statistics only (strict zero-shot). "
      "S2+ = per-target unlabelled re-standardisation (no labels used)."]
(BASE / "table2_zeroshot_comprehensive.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
