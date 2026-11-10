#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 02b: Fig.2 绘图——FAR vs 群体减速强度（uniform / gradient 两个 panel）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "A": ("Single-vessel speed z-score", "#c0392b", "o"),
    "B": ("Unweighted kNN residual (8 nm)", "#2980b9", "s"),
    "C": ("Ours: LOO local-linear field (8 nm)", "#27ae60", "^"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, mode, title in zip(axes, ["uniform", "gradient"],
                               ["Uniform group slowdown", "Spatial queue (gradient at s=24 nm)"]):
        part = df[df["mode"] == mode].sort_values("alpha", ascending=False)
        for k, (lab, color, marker) in LABELS.items():
            ax.plot(part.alpha, part[f"{k}_far"] * 100, marker=marker, color=color, lw=1.8, label=lab)
        ax.set_xlabel("Group speed factor alpha (1.0 = normal)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
    axes[0].set_ylabel("False alarm rate (%)  @ 1% operating point")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig. 2 prototype — FAR under group deceleration, Houston Ship Channel corridor", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
