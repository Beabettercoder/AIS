#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 09: 汇总三个特征组的 5-seed 结果，生成 Table 1 / Table 3（markdown + csv）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROWS = [
    ("clean_AP", "Clean AP (A vs C)"),
    ("full_AP", "Full AP (A+B vs C+D)"),
    ("recall_at_1pctFAR", "Recall @1% FAR"),
    ("AP_turn_back", "AP turn_back"),
    ("AP_deviation", "AP deviation"),
    ("AP_stall", "AP stall"),
    ("AP_sinuous", "AP sinuous"),
    ("AP_speed_drop", "AP speed_drop"),
    ("recall_delay_3min", "Recall @t0+3min"),
    ("recall_delay_5min", "Recall @t0+5min"),
    ("recall_delay_10min", "Recall @t0+10min"),
]
NAMES = {"raw": "Raw-TCN", "mean": "Unweighted kNN + TCN", "ours": "Ours (gate + field + dual-head)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    data = {}
    for p in args.inputs:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        data[d["mode"]] = d

    rows = []
    for key, label in ROWS:
        row = {"metric": label}
        for mode in ("raw", "mean", "ours"):
            if mode not in data:
                row[NAMES[mode]] = "-"
                continue
            a = data[mode]["agg"].get(key)
            row[NAMES[mode]] = f"{a['mean']:.3f} ± {a['std']:.3f}" if a and a["mean"] == a["mean"] else "-"
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = list(df.columns)
    lines = ["# Table 1 / Table 3 (5 seeds, mean ± std)", "", "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md = ["\n".join(lines), ""]
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    df.to_csv(Path(args.out_md).with_suffix(".csv"), index=False)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
