#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 12: 可检测性谱作图 —— AP vs 异常幅度，以及 AP vs 决策时刻。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
LABELS = {
    "raw": ("Raw-TCN", "#c0392b", "o"),
    "mean": ("Unweighted kNN + TCN", "#2471a3", "s"),
    "ours": ("Ours: gate + LOO field + dual-head", "#1e8449", "^"),
}


def main() -> None:
    df = pd.read_csv(BASE / "sweep_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    for m, (lab, c, mk) in LABELS.items():
        d = df[df["mode"] == m].sort_values("level")
        ax.errorbar(d.level, d.clean_AP, yerr=d.clean_AP_std, marker=mk, color=c, lw=1.9, capsize=3, label=lab)
    ax.set_xlabel("Anomaly amplitude level (1 = weak … 5 = strong)")
    ax.set_ylabel("Clean AP (A vs C)")
    ax.set_title("(a) Detectability spectrum — clean behaviour")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left"); ax.set_ylim(0.3, 0.9)

    ax = axes[1]
    d5 = df[df["level"] == 5]
    xs = [1, 2, 3]
    for m, (lab, c, mk) in LABELS.items():
        r = d5[d5["mode"] == m].iloc[0]
        ax.plot(xs, [r.AP_d3, r.AP_d5, r.AP_d10], marker=mk, color=c, lw=1.9, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels(["+3 min", "+5 min", "+10 min"])
    ax.set_xlabel("Decision time after anomaly onset")
    ax.set_ylabel("AP (behaviour, level 5)")
    ax.set_title("(b) AP by decision delay — level 5")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right"); ax.set_ylim(0.2, 0.85)

    fig.suptitle("Detectability spectrum, Houston Ship Channel (3 seeds, 1000 training windows/level)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = BASE / "detectability_spectrum.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
