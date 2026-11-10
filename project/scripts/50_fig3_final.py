#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 50: Table 3 final (incl. stress + cross-scenario) and Fig. 3."""
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
main = json.loads((BASE / "ablation_full.json").read_text())
stress = json.loads((BASE / "ablation_stress.json").read_text())

ORDER = ["Baseline (Raw TCN)", "+ KG (3.1 Gate)", "+ KG + LF (3.3 Field)",
         "+ KG + LF + PSA", "PG-DRGN (Ours, Full)"]
ALIAS = {"Baseline (Raw TCN)": "Baseline", "+ KG (3.1 Gate)": "+KG",
         "+ KG + LF (3.3 Field)": "+KG+LF", "+ KG + LF + PSA": "+KG+LF+PSA",
         "PG-DRGN (Ours, Full)": "PG-DRGN"}
SHORT = ["Baseline", "+KG", "+KG+LF", "+KG+LF\n+PSA", "PG-DRGN\n(Ours)"]

main_agg, stress_agg = {}, {}
for r in main:
    main_agg.setdefault(r["config"], []).append(r)
for r in stress:
    stress_agg.setdefault(ALIAS.get(r["config"], r["config"]), []).append(r)

rows = []
for cfg in ORDER:
    m = main_agg[cfg]
    s = stress_agg[ALIAS[cfg]]
    rows.append({
        "config": cfg,
        "clean": np.mean([x["clean_AP"] for x in m]), "clean_sd": np.std([x["clean_AP"] for x in m], ddof=1),
        "full": np.mean([x["full_AP"] for x in m]), "full_sd": np.std([x["full_AP"] for x in m], ddof=1),
        "r1": np.mean([x["recall_1pct"] for x in m]), "r1_sd": np.std([x["recall_1pct"] for x in m], ddof=1),
        "r5": np.mean([x["recall_5pct"] for x in m]), "r5_sd": np.std([x["recall_5pct"] for x in m], ddof=1),
        "stress": np.mean([x["stress_full"] for x in s]), "stress_sd": np.std([x["stress_full"] for x in s], ddof=1),
        "zs": np.mean([x["zs_mean"] for x in s]), "zs_sd": np.std([x["zs_mean"] for x in s], ddof=1),
    })

lines = ["# Table 3 — progressive ablation with stress-test and cross-scenario columns", "",
         "| Configuration | KG | LF | PSA | CDG | Clean AP | Full AP | Recall@1% | Recall@5% | Stress-Test Full AP | Cross-Scenario ZS AP |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
flags = {"Baseline (Raw TCN)": "✗ ✗ ✗ ✗", "+ KG (3.1 Gate)": "✓ ✗ ✗ ✗",
         "+ KG + LF (3.3 Field)": "✓ ✓ ✗ ✗", "+ KG + LF + PSA": "✓ ✓ ✓ ✗",
         "PG-DRGN (Ours, Full)": "✓ ✓ ✓ ✓"}
for r in rows:
    f = flags[r["config"]].split()
    lines.append(f"| {r['config']} | {f[0]} | {f[1]} | {f[2]} | {f[3]} | "
                 f"{r['clean']:.3f} ± {r['clean_sd']:.3f} | {r['full']:.3f} ± {r['full_sd']:.3f} | "
                 f"{r['r1']:.3f} ± {r['r1_sd']:.3f} | {r['r5']:.3f} ± {r['r5_sd']:.3f} | "
                 f"{r['stress']:.3f} ± {r['stress_sd']:.3f} | {r['zs']:.3f} ± {r['zs_sd']:.3f} |")
txt = "\n".join(lines)
(BASE / "table3_ablation_full.md").write_text(txt, encoding="utf-8")
print(txt)

fig, ax = plt.subplots(figsize=(9.4, 4.6))
x = np.arange(len(rows)); w = 0.2
series = [("clean", "Clean AP", "#4C72B0", "///"), ("full", "Full AP", "#DD8452", "\\\\\\"),
          ("stress", "Stress-Test Full AP", "#55A868", "xxx"), ("zs", "Cross-Scenario ZS AP", "#8172B3", "...")]
for i, (k, lab, col, h) in enumerate(series):
    vals = [r[k] for r in rows]; errs = [r[k + "_sd"] for r in rows]
    ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, capsize=2.5, label=lab,
           color=col, edgecolor="k", linewidth=0.5, hatch=h, alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=8.5)
ax.set_ylabel("Score", fontsize=10); ax.set_ylim(0.45, 0.92)
ax.set_title("Progressive component ablation under clean, stressed and cross-scenario evaluation", fontsize=10.5)
ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
ax.grid(axis="y", linestyle=":", alpha=0.4)
fig.tight_layout()
out = BASE / "fig3_ablation_hatched_bars.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
