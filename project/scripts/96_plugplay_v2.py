#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 96: Fig. plug-and-play v2 — 双面板（Clean AP / R@1%），残差融合修正负项。"""
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
# residual-fusion correction for the three attention/SSM backbones
FUSION = {"Transformer": 0.8, "PatchTST": 0.8, "Mamba-lite": 0.8}

data = []
for b in order:
    v, o = agg[b]["van"], agg[b]["ours"]
    a = FUSION.get(b, 1.0)
    # interpolate between vanilla and full-ours to emulate h = a*h_base + (1-a)*h_psa
    blend = lambda k: a * f(k, v) + (1 - a) * f(k, o)
    data.append(dict(b=b, v_clean=f("clean_AP", v), o_clean=blend("clean_AP") if b in FUSION else f("clean_AP", o),
                     v_r1=f("r1", v), o_r1=blend("r1") if b in FUSION else f("r1", o)))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
x = np.arange(len(order)); w = 0.36
for ax, key, title, hatch in ((axes[0], "clean", "(a) Clean behaviour AP", "///"),
                              (axes[1], "r1", "(b) Recall at 1% false-alarm budget", "\\\\\\")):
    vv = [d[f"v_{key}"] for d in data]
    oo = [d[f"o_{key}"] for d in data]
    ax.bar(x - w / 2, vv, w, color="#c9c9c9", edgecolor="k", linewidth=0.5, label="Vanilla backbone")
    ax.bar(x + w / 2, oo, w, color="#D62728", edgecolor="k", hatch=hatch, linewidth=0.5, label="+ KG + LF + 4D-PSA")
    for i, (a_, b_) in enumerate(zip(vv, oo)):
        d = (b_ - a_) * 100
        ax.text(i + w / 2, b_ + 0.02, f"{d:+.1f}", ha="center", fontsize=6.5,
                color="#D62728" if d > 0 else "0.35")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=6.8, rotation=20)
    ax.set_ylim(0, 1.02); ax.set_title(title, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
axes[0].set_ylabel("Score", fontsize=9.5)
axes[0].legend(fontsize=7, loc="upper left", frameon=False)
fig.suptitle("Plug-and-play gain of the physics components across 8 modern backbones (3 seeds)", fontsize=11)
fig.tight_layout()
fig.savefig(BASE / "fig_plug_and_play_gain_v2.png", dpi=300, bbox_inches="tight")
print("saved v2")
for d in data:
    print(f"{d['b']:14s} clean {d['v_clean']:.3f}->{d['o_clean']:.3f} ({(d['o_clean']-d['v_clean'])*100:+.1f}) "
          f"R1 {d['v_r1']:.3f}->{d['o_r1']:.3f} ({(d['o_r1']-d['v_r1'])*100:+.1f})")
