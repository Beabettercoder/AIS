#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 00c: 无阈值时间序列四联图（英文标签，服务器无中文字体）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = [
    ("median_div_1_s_hz", "Divergence z-score (neg = converging flow)", "#c0392b"),
    ("net_flux_crossing_w", "Net inbound flux (weighted vessels / 10 min)", "#2980b9"),
    ("median_sog_kn", "Median SOG in ROI (kn)", "#27ae60"),
    ("throughput_w", "Throughput (in+out)/2, weighted", "#8e44ad"),
]


def plot(df: pd.DataFrame, out: Path, title: str) -> None:
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(16, 2.8 * len(PANELS)), sharex=True)
    for ax, (col, label, color) in zip(axes, PANELS):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(df.datetime_utc, df[col], lw=0.8, color=color)
        if col.endswith("_hz"):
            ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("UTC")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_day", action="store_true")
    ap.add_argument("--zoom", default="", help="e.g. 2024-07-06,2024-07-12")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["datetime_utc"]).sort_values("datetime_utc")
    plot(df, Path(args.out), "Houston 3.2 no-threshold series (R=8 nm, 10-min bins)")

    if args.zoom:
        a, b = [s.strip() for s in args.zoom.split(",")]
        seg = df[(df.datetime_utc >= a) & (df.datetime_utc <= b)]
        plot(seg, Path(args.out).with_name(f"{Path(args.out).stem}_zoom.png"),
             f"Zoom {a} ~ {b} (Hurricane Beryl window)")

    if args.per_day:
        for day, part in df.groupby(df.datetime_utc.dt.date):
            plot(part, Path(args.out).with_name(f"{Path(args.out).stem}_{day}.png"), str(day))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
