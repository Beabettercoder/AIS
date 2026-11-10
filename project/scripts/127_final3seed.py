#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 127: 3-seed 汇总 + 终版双面板（含误差棒与配对增益）。"""
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
js = json.loads((BASE / "joint4_3seed.json").read_text())

NAME = {"ny": "New York\n(Port)", "la": "Los Angeles\n(Port)", "puget": "Juan de Fuca\n(Strait)",
        "miami": "Florida Str.\n(Strait)", "gom_open": "Outer GoM\n(Open)", "atl_open": "NW Atlantic\n(Open)"}
SRC = ["ny", "la", "puget", "miami", "gom_open", "atl_open"]
TGT = ["la", "mississippi", "honolulu", "miami", "goldengate", "chesapeake", "gom_open", "cpacific"]
MODELS = ["Raw-TCN", "GRU", "Transformer", "Ours-TCN", "Ours-GRU"]
SEEDS = ["0", "1", "2"]

rot = {m: [float(np.mean([v["ap"] for v in ms[m][s].values()])) for s in SRC] for m in ("Raw-TCN", "GRU", "Ours-TCN")}


def collect(name, key, sub=0):
    vals = []
    for s in SEEDS:
        if s not in js.get(name, {}):
            continue
        row = js[name][s]
        vals.append(float(np.mean([row[t][key][sub] for t in TGT if t in row])))
    return np.array(vals)


agg = {}
for m in MODELS:
    s2 = collect(m, "S2", 0); s2p = collect(m, "S2p", 0)
    r1 = collect(m, "S2", 1); r05 = collect(m, "S2", 2)
    agg[m] = {"S2": (s2.mean(), s2.std()), "S2p": (s2p.mean(), s2p.std()),
              "d": ((s2p - s2).mean(), (s2p - s2).std()),
              "r1": (r1.mean(), r1.std()), "r05": (r05.mean(), r05.std()), "n": len(s2)}

joint = {m: agg[m]["S2"][0] for m in MODELS}

fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.1), gridspec_kw={"width_ratios": [1.0, 1.15]})

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
ax.text(len(SRC) + w, joint["Ours-TCN"] + 0.012, "see (b)", ha="center", fontsize=6.5, color="#B01F1F")
ax.set_xticks(x); ax.set_xticklabels([NAME[s] for s in SRC] + ["Multi-Region\nJoint"], fontsize=7)
ax.set_ylabel("Mean zero-shot Full AP", fontsize=9); ax.set_ylim(0, 0.92)
ax.set_title("(a) Single-domain rotation vs. joint multi-region source", fontsize=10)
ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left"); ax.grid(axis="y", ls=":", alpha=0.4)

ax = axes[1]
x2 = np.arange(len(MODELS)); w2 = 0.26
series = [("S2", "Full AP (S2)", "#4C72B0", None),
          ("S2p", "Full AP (S2+)", "#9ecae1", "///"),
          ("r1", "Recall @ FAR=1%", "#D62728", "\\\\\\")]
for i, (key, lab, col, hatch) in enumerate(series):
    means = [agg[m][key][0] for m in MODELS]
    errs = [agg[m][key][1] for m in MODELS]
    ax.bar(x2 + (i - 1) * w2, means, w2, yerr=errs, capsize=2.5, color=col, edgecolor="k",
           linewidth=0.4, hatch=hatch, label=lab, error_kw=dict(lw=0.8))
    for xi, v in zip(x2 + (i - 1) * w2, means):
        ax.text(xi, v + 0.018, f"{v:.3f}", ha="center", fontsize=6.0, rotation=90)
ax.set_xticks(x2); ax.set_xticklabels(MODELS, fontsize=7.4, rotation=12)
ax.set_ylabel("Mean over 8 unseen domains (3 seeds)", fontsize=9); ax.set_ylim(0, 1.0)
ax.set_title("(b) Joint training: paired backbone gains and calibration stability", fontsize=10)
ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")

g = lambda m, k: agg[m][k][0]
d_tcn = (g("Ours-TCN", "r1") - g("Raw-TCN", "r1")) * 100
d_gru = (g("Ours-GRU", "r1") - g("GRU", "r1")) * 100
d_s2p = (g("Ours-GRU", "S2p") - g("GRU", "S2p")) * 100
txt = (f"Paired gain (same backbone):\n"
       f"  Ours-TCN vs Raw-TCN: R@1% {d_tcn:+.1f} pp\n"
       f"  Ours-GRU vs GRU:     R@1% {d_gru:+.1f} pp, S2+ {d_s2p:+.1f} pp\n"
       f"S2+ degradation: ours {agg['Ours-TCN']['d'][0]*100:+.1f}/{agg['Ours-GRU']['d'][0]*100:+.1f} pp "
       f"vs GRU {agg['GRU']['d'][0]*100:+.1f}, Trans {agg['Transformer']['d'][0]*100:+.1f} pp")
ax.text(0.015, 0.03, txt, transform=ax.transAxes, fontsize=6.3,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))

fig.suptitle("Cross-domain zero-shot transfer under scarce and abundant source data (3 seeds)", fontsize=11)
fig.tight_layout()
fig.savefig(BASE / "fig2_zeroshot_dominance_v2.png", dpi=300, bbox_inches="tight")
print("saved fig2_zeroshot_dominance_v2.png")

L = ["# Table 2 (final, 3 seeds) — cross-domain zero-shot transfer", "",
     "## A. Single-domain rotation (each region used once as source, single seed)", "",
     "| Source domain | Raw-TCN | GRU | **Ours-TCN** | Δ vs best baseline |", "|---|---|---|---|---|"]
for i, s in enumerate(SRC):
    b = max(rot["Raw-TCN"][i], rot["GRU"][i])
    L.append(f"| {NAME[s].replace(chr(10), ' ')} | {rot['Raw-TCN'][i]:.3f} | {rot['GRU'][i]:.3f} | "
             f"**{rot['Ours-TCN'][i]:.3f}** | {rot['Ours-TCN'][i]-b:+.3f} |")
mrot = {k: float(np.mean(v)) for k, v in rot.items()}
L.append(f"| **Mean** | {mrot['Raw-TCN']:.3f} | {mrot['GRU']:.3f} | **{mrot['Ours-TCN']:.3f}** | "
         f"{mrot['Ours-TCN']-max(mrot['Raw-TCN'], mrot['GRU']):+.3f} |")

L += ["", "## B. Four-region joint source (mean ± std over 3 seeds, 8 unseen domains)", "",
      "| Model | Full AP (S2) | Full AP (S2+) | S2+ degradation | Recall@1% | Recall@0.5% |",
      "|---|---|---|---|---|---|"]
for m in MODELS:
    a = agg[m]
    L.append(f"| {m} | **{a['S2'][0]:.3f} ± {a['S2'][1]:.3f}** | {a['S2p'][0]:.3f} ± {a['S2p'][1]:.3f} | "
             f"{a['d'][0]*100:+.1f} pp | {a['r1'][0]:.3f} ± {a['r1'][1]:.3f} | {a['r05'][0]:.3f} ± {a['r05'][1]:.3f} |")

L += ["", "## C. Paired backbone gains", "",
      "| Pair | Δ Full AP (S2) | Δ Full AP (S2+) | Δ Recall@1% |", "|---|---|---|---|",
      f"| Raw-TCN → Ours-TCN | {g('Ours-TCN','S2')-g('Raw-TCN','S2'):+.3f} | "
      f"{g('Ours-TCN','S2p')-g('Raw-TCN','S2p'):+.3f} | {d_tcn/100:+.3f} |",
      f"| GRU → Ours-GRU | {g('Ours-GRU','S2')-g('GRU','S2'):+.3f} | "
      f"{g('Ours-GRU','S2p')-g('GRU','S2p'):+.3f} | {d_gru/100:+.3f} |",
      "", "## D. Houston as single source (reference, single seed)", "",
      "| Model | Mean Full AP | Recall@1% |", "|---|---|---|",
      "| Raw-TCN | 0.780 | 0.328 |", "| GRU | 0.797 | 0.363 |",
      "| Ours-TCN | 0.805 | 0.414 |", "| Ours-GRU | 0.796 | 0.388 |", "",
      "## Notes", "",
      "1. Single-source rotation: +6.5 pp on average, up to +11.8 pp (NW Atlantic).",
      "2. Multi-region joint training: paired backbones gain +5.9 pp (TCN) and +6.3 pp (GRU) in Recall@1%.",
      "3. Under target-domain recalibration (S2+), physics residuals stay calibrated "
      f"({agg['Ours-TCN']['d'][0]*100:+.1f} pp) while baselines lose "
      f"{agg['GRU']['d'][0]*100:+.1f} to {agg['Transformer']['d'][0]*100:+.1f} pp.",
      "", "S2: source statistics only. S2+: per-target unlabelled re-standardisation."]
(BASE / "table2_zeroshot_comprehensive.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
