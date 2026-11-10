#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 37: 最终 13 模型表（含 R@5%）：日志(8 基线) + JSON(5 图/Ours)。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path("/mnt/sda/WangZijia/AIS")
PAT = re.compile(r"^(\S+)\s+s(\d+): clean=([\d.]+) full=([\d.]+) R1=([\d.]+) R5=([\d.]+)")

rows = {}
log = BASE / "logs/57_fair8.log"
for line in log.read_text(errors="ignore").splitlines():
    m = PAT.match(line.strip())
    if m:
        rows.setdefault(m.group(1), {})[int(m.group(2))] = tuple(float(m.group(i)) for i in (3, 4, 5, 6))

for r in json.loads((BASE / "outputs/phase1/graph_ours_5seed.json").read_text()):
    rows.setdefault(r["model"], {})[r["seed"]] = (r["clean_AP"], r["full_AP"], r["recall_1pct"], r["recall_5pct"])

table = []
for name, d in rows.items():
    seeds = sorted(d)
    arr = np.array([d[s] for s in seeds])
    table.append({"model": name, "clean": arr[:, 0].mean(), "clean_std": arr[:, 0].std(),
                  "full": arr[:, 1].mean(), "full_std": arr[:, 1].std(),
                  "r1": arr[:, 2].mean(), "r1_std": arr[:, 2].std(),
                  "r5": arr[:, 3].mean(), "r5_std": arr[:, 3].std(), "seeds": d, "n": len(seeds)})
table.sort(key=lambda x: -x["full"])

ours = next(t for t in table if t["model"] == "Ours-Full-TCN")
base_f = max((t for t in table if "Ours" not in t["model"]), key=lambda t: t["full"])
base_c = max((t for t in table if "Ours" not in t["model"]), key=lambda t: t["clean"])
p_full = stats.ttest_rel([ours["seeds"][s][1] for s in sorted(ours["seeds"])],
                         [base_f["seeds"][s][1] for s in sorted(base_f["seeds"])])[1]
p_clean = stats.ttest_rel([ours["seeds"][s][0] for s in sorted(ours["seeds"])],
                          [base_c["seeds"][s][0] for s in sorted(base_c["seeds"])])[1]
p_r1 = stats.ttest_rel([ours["seeds"][s][2] for s in sorted(ours["seeds"])],
                       [base_f["seeds"][s][2] for s in sorted(base_f["seeds"])])[1]

lines = ["# Table 1 — 13-model fair benchmark (5 seeds, 7-channel protocol)", "",
         "| # | Model | Clean AP | Full AP | Recall@1% | Recall@5% |", "|---|---|---|---|---|---|"]
for i, t in enumerate(table, 1):
    lines.append(f"| {i} | {t['model']} | {t['clean']:.3f} ± {t['clean_std']:.3f} | {t['full']:.3f} ± {t['full_std']:.3f} | "
                 f"{t['r1']:.3f} ± {t['r1_std']:.3f} | {t['r5']:.3f} ± {t['r5_std']:.3f} |")
lines += ["", f"**Paired t-test (Ours-Full-TCN)** — Full AP vs {base_f['model']}: p = {p_full:.4f}; "
              f"Clean AP vs {base_c['model']}: p = {p_clean:.4f}; Recall@1% vs {base_f['model']}: p = {p_r1:.4f}", ""]
txt = "\n".join(lines)
(BASE / "outputs/phase1/table1_fair13.md").write_text(txt, encoding="utf-8")
print(txt)
