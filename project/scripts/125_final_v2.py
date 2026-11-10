#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 125: 终版双面板（Panel b = S2 / S2+ AP + Recall@FAR）。"""
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
MODELS = ["Raw-TCN", "GRU", "Transformer", "Ours-TCN", "Ours-GRU"]

rot = {m: [float(np.mean([v["ap"] for v in ms[m][s].values()])) for s in SRC] for m in ("Raw-TCN", "GRU", "Ours-TCN")}
joint = {m: float(np.mean([v["S2"][0] for v in jt[m].values()])) for m in MODELS}

agg = {}
for m in MODELS:
    agg[m] = {
        "S2_AP": np.mean([jt[m][t]["S2"][0] for t in TGT]),
        "S2p_AP": np.mean([jt[m][t]["S2p"][0] for t in TGT]),
        "R1": np.mean([jt[m][t]["S2"][1] for t in TGT]),
        "R05": np.mean([jt[m][t]["S2"][2] for t in TGT]),
    }

fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.0), gridspec_kw={"width_ratios": [1.05, 1.1]})

# ---------------- panel (a)
ax = axes[0]
x = np.arange(len(SRC) + 1); w = 0.26
for i, (m, col, hatch) in enumerate((("Raw-TCN", "#b0b0b0", None), ("GRU", "#5b8fc9", None),
                                     ("Ours-TCN", "#d62728", "///"))):
    vals = rot[m] + [joint.get(m, np.nan)]
    ax.bar(x + (i - 1) * w, vals, w, color=col, edgecolor="k", linewidth=0.4, hatch=hatch, label=m)
for i in range(len(SRC)):
    best = max(rot["Raw-TCN"][i], rot["GRU"][i])
    ax.text(i + w, rot["Ours-TCN"][i] + 0.012, f"+{(rot['Ours-TCN'][i]-best)*100:.1f}", ha="center",
            fontsize=7.5, color="#B01F1F", fontweight="bold")
best_j = max(joint["Raw-TCN"], joint["GRU"])
ax.text(len(SRC) + w, joint["Ours-TCN"] + 0.012, f"+{(joint['Ours-TCN']-best_j)*100:.1f}",
        ha="center", fontsize=7.5, color="#B01F1F", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([NAME[s] for s in SRC] + ["Multi-Region\nJoint"], fontsize=7)
ax.set_ylabel("Mean zero-shot Full AP", fontsize=9); ax.set_ylim(0, 0.92)
ax.set_title("(a) Single-domain rotation vs. joint multi-region source", fontsize=10)
ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left"); ax.grid(axis="y", ls=":", alpha=0.4)

# ---------------- panel (b)
ax = axes[1]
x2 = np.arange(len(MODELS)); w2 = 0.26
series = [("S2_AP", "Full AP (S2)", "#4C72B0", None),
          ("S2p_AP", "Full AP (S2+)", "#9ecae1", "///"),
          ("R1", "Recall @ FAR=1%", "#D62728", "\\\\\\")]
for i, (key, lab, col, hatch) in enumerate(series):
    vals = [agg[m][key] for m in MODELS]
    bars = ax.bar(x2 + (i - 1) * w2, vals, w2, color=col, edgecolor="k", linewidth=0.4, hatch=hatch, label=lab)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.3, rotation=90)
ax.set_xticks(x2); ax.set_xticklabels(MODELS, fontsize=7.6, rotation=12)
ax.set_ylabel("Score over 8 unseen domains", fontsize=9); ax.set_ylim(0, 0.95)
ax.set_title("(b) Joint training: calibration stability and strict-FAR recall", fontsize=10)
ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")

drop = {m: agg[m]["S2p_AP"] - agg[m]["S2_AP"] for m in MODELS}
txt = (f"Ours-TCN vs GRU (S2+): +{(agg['Ours-TCN']['S2p_AP']-agg['GRU']['S2p_AP'])*100:.1f} pp\n"
       f"Ours-TCN vs Transformer (S2+): +{(agg['Ours-TCN']['S2p_AP']-agg['Transformer']['S2p_AP'])*100:.1f} pp\n"
       f"Recalibration drop: ours {drop['Ours-TCN']*100:+.1f} pp vs Transformer {drop['Transformer']*100:+.1f} pp\n"
       f"Recall@1%: ours {agg['Ours-TCN']['R1']:.3f} vs GRU {agg['GRU']['R1']:.3f}")
ax.text(0.02, 0.04, txt, transform=ax.transAxes, fontsize=6.6,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))

fig.suptitle("Cross-domain zero-shot transfer under scarce and abundant source data", fontsize=11)
fig.tight_layout()
fig.savefig(BASE / "fig2_zeroshot_dominance_v2.png", dpi=300, bbox_inches="tight")
print("saved fig2_zeroshot_dominance_v2.png")

L = ["# Table 2 (final) — cross-domain zero-shot transfer", "",
     "## A. Single-domain rotation (each region used once as source, 7-8 unseen targets)", "",
     "| Source domain | Raw-TCN | GRU | **Ours-TCN** | Δ vs best baseline |",
     "|---|---|---|---|---|"]
for i, s in enumerate(SRC):
    b = max(rot["Raw-TCN"][i], rot["GRU"][i])
    L.append(f"| {NAME[s].replace(chr(10), ' ')} | {rot['Raw-TCN'][i]:.3f} | {rot['GRU'][i]:.3f} | "
             f"**{rot['Ours-TCN'][i]:.3f}** | {rot['Ours-TCN'][i]-b:+.3f} |")
mrot = {k: float(np.mean(v)) for k, v in rot.items()}
L.append(f"| **Mean** | {mrot['Raw-TCN']:.3f} | {mrot['GRU']:.3f} | **{mrot['Ours-TCN']:.3f}** | "
         f"{mrot['Ours-TCN']-max(mrot['Raw-TCN'], mrot['GRU']):+.3f} |")

L += ["", "## B. Four-region joint source — Full AP, recalibration and strict-FAR recall", "",
      "| Model | Full AP (S2) | Full AP (S2+) | Δ recalibration | Recall@1% | Recall@0.5% |",
      "|---|---|---|---|---|---|"]
for m in MODELS:
    a = agg[m]
    L.append(f"| {m} | **{a['S2_AP']:.3f}** | {a['S2p_AP']:.3f} | {a['S2p_AP']-a['S2_AP']:+.3f} | "
             f"{a['R1']:.3f} | {a['R05']:.3f} |")

L += ["", "## C. Per-domain Full AP under the joint source", "",
      "| Model | " + " | ".join(TGT) + " |", "|---|" + "---|" * len(TGT)]
for m in MODELS:
    L.append(f"| {m} | " + " | ".join(f"{jt[m][t]['S2'][0]:.3f}" for t in TGT) + " |")

L += ["", "## D. Houston as single source (reference)", "",
      "| Model | Mean Full AP | Recall@1% |", "|---|---|---|",
      "| Raw-TCN | 0.780 | 0.328 |", "| GRU | 0.797 | 0.363 |",
      "| Ours-TCN | 0.805 | 0.414 |", "| Ours-GRU | 0.796 | 0.388 |", "",
      "S2: source statistics only. S2+: per-target unlabelled re-standardisation."]
(BASE / "table2_zeroshot_comprehensive.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
