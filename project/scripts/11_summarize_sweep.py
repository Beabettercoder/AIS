#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 11: 汇总可检测性谱（5 档幅度 × 3 特征组 × 3 seeds）。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
LEVELS = [1, 2, 3, 4, 5]
MODES = ["raw", "mean", "ours"]
COLS = ["clean_AP", "full_AP", "recall_1pct", "recall_5pct", "AP_d3", "AP_d5", "AP_d10"]


def main() -> None:
    rows = []
    for L in LEVELS:
        for M in MODES:
            p = BASE / f"eval_v2_l{L}" / f"res_{M}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            row = {"level": L, "mode": M}
            for c in COLS:
                a = d["agg"].get(c)
                row[c] = round(a["mean"], 3) if a else None
                row[c + "_std"] = round(a["std"], 3) if a else None
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "sweep_summary.csv", index=False)

    piv = df.pivot(index="level", columns="mode", values="clean_AP")
    pivot_full = df.pivot(index="level", columns="mode", values="full_AP")
    lines = ["# Detectability spectrum (AP vs anomaly amplitude)\n",
             "## Clean AP (A vs C)\n", piv.to_string(), "",
             "## Full AP (A+B vs C+D)\n", pivot_full.to_string(), ""]
    det = df[df.level == 5]
    lines.append("## Level-5 (strongest) detail\n")
    lines.append(det.to_string(index=False))
    txt = "\n".join(lines)
    (BASE / "sweep_summary.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
