#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 95: 通用外挂汇总 + 配对增益图。"""
from __future__ import annotations

import collections
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
rows = json.loads((BASE / "plugplay.json").read_text())
agg = collections.defaultdict(lambda: {"van": [], "ours": []})
for r in rows:
    agg[r["backbone"]]["ours" if r["ours"] else "van"].append(r)

f = lambda k, arr: float(np.mean([x[k] for x in arr]))
order = ["TCN", "GRU", "Transformer", "iTransformer", "PatchTST", "Mamba-lite", "Neural-CDE", "ST-GAT"]

lines = ["# Plug-and-play gain across 8 modern backbones (3 seeds)", "",
         "| Backbone | Vanilla Full AP | +Ours Full AP | Δ | Vanilla Clean AP | +Ours Clean AP | Δ | Vanilla R@1% | +Ours R@1% | Δ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
summary = []
for b in order:
    v, o = agg[b]["van"], agg[b]["ours"]
    row = dict(backbone=b,
               v_full=f("full_AP", v), o_full=f("full_AP", o),
               v_clean=f("clean_AP", v), o_clean=f("clean_AP", o),
               v_r1=f("r1", v), o_r1=f("r1", o),
               v_r5=f("r5", v), o_r5=f("r5", o))
    summary.append(row)
    lines.append(f"| {b} | {row['v_full']:.3f} | {row['o_full']:.3f} | {row['o_full']-row['v_full']:+.3f} | "
                 f"{row['v_clean']:.3f} | {row['o_clean']:.3f} | {row['o_clean']-row['v_clean']:+.3f} | "
                 f"{row['v_r1']:.3f} | {row['o_r1']:.3f} | {row['o_r1']-row['v_r1']:+.3f} |")
txt = "\n".join(lines)
(BASE / "table_plug_and_play.md").write_text(txt, encoding="utf-8")
print(txt)

x = np.arange(len(order)); w = 0.2
fig, ax = plt.subplots(figsize=(10.5, 4.6))
for i, (key, lab, col, h) in enumerate([("full", "Full AP", "#4C72B0", "///"),
                                       ("clean", "Clean AP", "#DD8452", "\\\\\\"),
                                       ("r1", "Recall@1%", "#55A868", "xxx")]):
    vv = [r[f"v_{key}"] for r in summary]
    oo = [r[f"o_{key}"] for r in summary]
    ax.bar(x + (i - 1) * 2 * w - w / 2, vv, w, color="#cccccc", edgecolor="k", linewidth=0.4, label=f"Vanilla {lab}" if i == 0 else None)
    ax.bar(x + (i - 1) * 2 * w + w / 2, oo, w, color=col, edgecolor="k", hatch=h, linewidth=0.4, label=f"+Ours {lab}")
for i, r in enumerate(summary):
    d = (r["o_full"] - r["v_full"]) / max(r["v_full"], 1e-6) * 100
    ax.text(i, max(r["v_full"], r["o_full"]) + 0.02, f"{d:+.1f}%", ha="center", fontsize=7, color="#D62728")
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7.5, rotation=15)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score", fontsize=9.5)
ax.set_title("Plug-and-play gain of KG + LF + 4D-PSA across 8 backbones", fontsize=10.5)
ax.legend(fontsize=6.5, ncol=3, loc="upper center", frameon=False)
ax.grid(axis="y", linestyle=":", alpha=0.4)
fig.tight_layout()
fig.savefig(BASE / "fig_plug_and_play_gain.png", dpi=300, bbox_inches="tight")
print("saved fig_plug_and_play_gain.png")
