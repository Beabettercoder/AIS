#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 02d: 终极版 Fig.2（双面板）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STYLE = {
    "A": ("Single-vessel speed z-score", "#c0392b", "o"),
    "B": ("Unweighted kNN residual (no credibility weighting)", "#2471a3", "s"),
    "C": ("Ours: LOO local-linear field + w-gate", "#1e8449", "^"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    titles = {
        "clean": "(a) Group slowdown only — no observation degradation",
        "degraded": "(b) Group slowdown + 15% neighbour jumps/spikes",
    }
    for ax, panel in zip(axes, ["clean", "degraded"]):
        part = df[df.panel == panel].sort_values("alpha", ascending=False)
        for k, (label, color, marker) in STYLE.items():
            ax.plot(part.alpha, part[f"{k}_far"] * 100, marker=marker, ms=5,
                    color=color, lw=1.9, label=label)
        ax.set_xlabel("Group speed factor alpha  (1.0 = normal, 0.5 = half speed)")
        ax.set_title(titles[panel], fontsize=10)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
        ax.set_ylim(0, 17)
    axes[0].set_ylabel("False alarm rate (%)  @ 1% clean operating point")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig. 2 — FAR of single-vessel vs field-based detectors under group deceleration,\n"
                 "Houston Ship Channel corridor (2024-07, 44k target-vessel snapshots)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
