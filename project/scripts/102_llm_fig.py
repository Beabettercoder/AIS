#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 102: Fig. LLM scaling & family benchmark + table."""
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
d = json.loads((BASE / "llm_multi_benchmark.json").read_text())
d = {k: v for k, v in d.items() if "L1" in v}

ORDER = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "DeepSeek-R1-Distill-Qwen-1.5B",
         "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
         "gemma-2-9b-it", "Qwen2.5-14B-Instruct"]
SIZE = {"Qwen2.5-0.5B-Instruct": 0.5, "Qwen2.5-1.5B-Instruct": 1.5, "DeepSeek-R1-Distill-Qwen-1.5B": 1.5,
        "Qwen2.5-3B-Instruct": 3, "Qwen2.5-7B-Instruct": 7, "Meta-Llama-3.1-8B-Instruct": 8,
        "gemma-2-9b-it": 9, "Qwen2.5-14B-Instruct": 14}
SHORT = {"Qwen2.5-0.5B-Instruct": "Qwen0.5B", "Qwen2.5-1.5B-Instruct": "Qwen1.5B",
         "DeepSeek-R1-Distill-Qwen-1.5B": "DS-1.5B", "Qwen2.5-3B-Instruct": "Qwen3B",
         "Qwen2.5-7B-Instruct": "Qwen7B", "Meta-Llama-3.1-8B-Instruct": "Llama8B",
         "gemma-2-9b-it": "Gemma9B", "Qwen2.5-14B-Instruct": "Qwen14B"}

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
ax = axes[0]
x = np.arange(len(ORDER)); w = 0.36
ax.bar(x - w / 2, [d[k]["L0"] for k in ORDER], w, color="#c9c9c9", edgecolor="k", linewidth=0.5, label="L0 raw trajectory")
ax.bar(x + w / 2, [d[k]["L1"] for k in ORDER], w, color="#D62728", edgecolor="k", hatch="///", linewidth=0.5,
       label="L1 physics-enriched")
ax.axhline(0.47, color="k", ls="--", lw=0.9)
ax.text(len(ORDER) - 0.4, 0.475, r"$\pi$=0.47", fontsize=7, ha="right")
for i, k in enumerate(ORDER):
    ax.text(i + w / 2, max(d[k]["L0"], d[k]["L1"]) + 0.015, f"{d[k]['L1']-d[k]['L0']:+.2f}",
            ha="center", fontsize=6, color="#D62728" if d[k]["L1"] > d[k]["L0"] else "0.35")
ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in ORDER], fontsize=6.8, rotation=20)
ax.set_ylim(0, 0.75); ax.set_ylabel("Full AP", fontsize=9.5)
ax.set_title("(a) Physics prompting across families and sizes", fontsize=10)
ax.legend(fontsize=7, loc="upper left", frameon=False); ax.grid(axis="y", ls=":", alpha=0.4)

ax = axes[1]
for k in ORDER:
    c = "#D62728" if "Qwen2.5" in k else ("#1f77b4" if "Llama" in k else ("#2ca02c" if "gemma" in k else "#9467bd"))
    ax.scatter(SIZE[k], d[k]["L0"], s=45, color=c, edgecolor="k", zorder=3)
    ax.annotate(SHORT[k], (SIZE[k], d[k]["L0"]), textcoords="offset points", xytext=(5, 4), fontsize=6)
ax.scatter(6.68e-5, 0.826, marker="*", s=260, color="#D62728", edgecolor="k", zorder=5, label="PG-DRGN (66.8 K)")
ax.axhline(0.47, color="k", ls="--", lw=0.9)
ax.set_xscale("log"); ax.set_xlim(3e-5, 40); ax.set_ylim(0.35, 0.92)
ax.set_xlabel("Parameters (log scale)", fontsize=9.5); ax.set_ylabel("Full AP (L0)", fontsize=9.5)
ax.set_title("(b) Accuracy vs. model size", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left", frameon=False); ax.grid(ls=":", alpha=0.4)

fig.suptitle("General-purpose LLMs vs a 66.8 K physics-guided detector on MMSI-disjoint AIS test windows", fontsize=10.5)
fig.tight_layout()
fig.savefig(BASE / "fig_llm_scaling_benchmark.png", dpi=300, bbox_inches="tight")
print("saved fig_llm_scaling_benchmark")

lines = ["# Multi-family / multi-scale LLM benchmark (200 MMSI-disjoint windows, π = 0.47)", "",
         "| Model | Params | L0 raw text | L1 physics-enriched | Δ(L1−L0) | sec / 200 |",
         "|---|---|---|---|---|---|"]
for k in ORDER:
    lines.append(f"| {k} | {SIZE[k]} B | {d[k]['L0']:.3f} | {d[k]['L1']:.3f} | "
                 f"{d[k]['L1']-d[k]['L0']:+.3f} | {d[k]['L0_sec']:.0f} / {d[k]['L1_sec']:.0f} |")
lines += ["", f"| **PG-DRGN (ours)** | **66.8 K** | **0.826** | — | — | <1 |", "",
          "All LLM scores sit between 0.41 and 0.62, i.e. around the no-information level (π = 0.47). "
          "Injecting the physics channels helps the two smallest Qwen models (+0.08 / +0.09) but *hurts* "
          "every model at 3 B and above (−0.10 to −0.14): the extra columns add attention load without "
          "adding discriminative text."]
(BASE / "table7_multi_llm_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
