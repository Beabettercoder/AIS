#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 104: 全协议（L0/L1/3-shot/LoRA）家族×尺寸对比图与表。"""
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
d = json.loads((BASE / "llm_full_protocol_fixed.json").read_text())
d = {k: v for k, v in d.items() if "L0" in v}
SIZE = {"Qwen2.5-0.5B-Instruct": 0.5, "Qwen2.5-1.5B-Instruct": 1.5, "DeepSeek-R1-Distill-Qwen-1.5B": 1.5,
        "Qwen2.5-3B-Instruct": 3, "Qwen2.5-7B-Instruct": 7, "Meta-Llama-3.1-8B-Instruct": 8,
        "gemma-2-9b-it": 9, "Mistral-7B-Instruct-v0.2": 7, "Qwen2.5-14B-Instruct": 14, "Qwen3.8-27B": 27}
SHORT = {"Qwen2.5-0.5B-Instruct": "Qwen0.5B", "Qwen2.5-1.5B-Instruct": "Qwen1.5B",
         "DeepSeek-R1-Distill-Qwen-1.5B": "DS1.5B", "Qwen2.5-3B-Instruct": "Qwen3B",
         "Qwen2.5-7B-Instruct": "Qwen7B", "Meta-Llama-3.1-8B-Instruct": "Llama8B",
         "gemma-2-9b-it": "Gemma9B", "Mistral-7B-Instruct-v0.2": "Mistral7B", "Qwen2.5-14B-Instruct": "Qwen14B", "Qwen3.8-27B": "Qwen3.8-27B"}
ORDER = [k for k in ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "DeepSeek-R1-Distill-Qwen-1.5B",
                     "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.2",
                     "Meta-Llama-3.1-8B-Instruct", "gemma-2-9b-it", "Qwen2.5-14B-Instruct", "Qwen3.8-27B"] if k in d]

fig, ax = plt.subplots(figsize=(11.5, 4.6))
x = np.arange(len(ORDER)); w = 0.2
series = [("L0", "L0 zero-shot raw", "#c9c9c9", ""), ("L1", "L1 physics prompt", "#ff9f9f", "///"),
          ("3shot", "3-shot ICL", "#8fbcd4", "xxx"), ("L2_lora", "L2 LoRA (same labels)", "#D62728", "\\\\\\\\")]
for i, (key, lab, col, h) in enumerate(series):
    vals = [d[k].get(key) if d[k].get(key) is not None else np.nan for k in ORDER]
    ax.bar(x + (i - 1.5) * w, vals, w, color=col, edgecolor="k", linewidth=0.4, hatch=h, label=lab)
ax.axhline(0.47, color="k", ls="--", lw=0.9)
ax.text(len(ORDER) - 0.5, 0.478, r"$\pi$ = 0.47", fontsize=7, ha="right")
ax.axhline(0.826, color="#D62728", ls="-", lw=1.2, alpha=0.6)
ax.text(len(ORDER) - 0.5, 0.835, "PG-DRGN 66.8 K = 0.826", fontsize=7.5, ha="right", color="#D62728")
ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in ORDER], fontsize=7.5, rotation=18)
ax.set_ylim(0, 0.92); ax.set_ylabel("Full AP (MMSI-disjoint 200 windows)", fontsize=9)
ax.set_title("Four protocols across LLM families and sizes", fontsize=10.5)
ax.legend(fontsize=7, ncol=4, loc="upper center", frameon=False)
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(BASE / "fig_llm_full_protocol.png", dpi=300, bbox_inches="tight")
print("saved fig_llm_full_protocol")

lines = ["# Full protocol across LLM families (200 MMSI-disjoint windows, π = 0.47)", "",
         "| Model | Params | L0 | L1 | 3-shot | L2 LoRA | best |", "|---|---|---|---|---|---|---|"]
for k in ORDER:
    r = d[k]
    vals = {kk: r.get(kk) for kk in ("L0", "L1", "3shot", "L2_lora")}
    best = max((v for v in vals.values() if v is not None), default=float("nan"))
    fmt = lambda v: "–" if v is None else f"{v:.3f}"
    lines.append(f"| {k} | {SIZE[k]} B | {fmt(vals['L0'])} | {fmt(vals['L1'])} | {fmt(vals['3shot'])} | "
                 f"{fmt(vals['L2_lora'])} | **{best:.3f}** |")
lines += ["", "| **PG-DRGN (ours)** | **66.8 K** | **0.826** | — | — | — | **0.826** |", "",
          "- Token ids are resolved per model (` Yes`/` No` last sub-word), so families are comparable.",
          "- 3-shot is no longer degenerate (0.42-0.60) once left-truncation keeps the test window intact.",
          "- Physics prompting (L1) helps Qwen-7B (+10.8), Mistral (+7.9) and Gemma (+6.7), "
          "but hurts Llama (-7.2); it is not a universal gain.",
          "- Best LLM is Qwen2.5-7B with physics prompting (0.688) and Gemma-2-9B (0.684).",
          "- The 66.8 K physics-guided detector (0.826) is 0.14 AP above the best LLM result."]
(BASE / "table7_full_protocol.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
