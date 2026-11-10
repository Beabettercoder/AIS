#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 36: 汇总 13 模型公平对比大表（mean ± std + paired t-test）。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path("/mnt/sda/WangZijia/AIS")
PAT = re.compile(r"^([A-Za-z][\w\-\+\.]*)\s+s(\d+): clean=([\d.]+) full=([\d.]+) R1=([\d.]+)")

rows = {}
for log in ("logs/53_fair13.log", "logs/54_fair13b.log", "logs/55_fair13c.log"):
    p = BASE / log
    if not p.exists():
        continue
    for line in p.read_text(errors="ignore").splitlines():
        m = PAT.match(line.strip())
        if not m:
            continue
        name, seed, c, f, r1 = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
        rows.setdefault(name, {})[seed] = (c, f, r1)

# 图模型/ours 的 5-seed JSON 覆盖（优先）
j = BASE / "outputs/phase1/graph_ours_5seed.json"
if j.exists():
    for r in json.loads(j.read_text()):
        rows.setdefault(r["model"], {})[r["seed"]] = (r["clean_AP"], r["full_AP"], r["recall_1pct"])

table = []
for name, d in rows.items():
    seeds = sorted(d)
    if len(seeds) < 5:
        continue
    c = np.array([d[s][0] for s in seeds]); f = np.array([d[s][1] for s in seeds]); r = np.array([d[s][2] for s in seeds])
    table.append({"model": name, "clean": c.mean(), "clean_std": c.std(), "full": f.mean(), "full_std": f.std(),
                  "r1": r.mean(), "r1_std": r.std(), "n": len(seeds), "seeds": d})
table.sort(key=lambda x: -x["full"])

base = max((t for t in table if "Ours" not in t["model"]), key=lambda t: t["full"])
ours = next(t for t in table if t["model"] == "Ours-Full-TCN")
p_full = stats.ttest_rel([ours["seeds"][s][1] for s in sorted(ours["seeds"])],
                         [base["seeds"][s][1] for s in sorted(base["seeds"])])[1]
base_c = max((t for t in table if "Ours" not in t["model"]), key=lambda t: t["clean"])
p_clean = stats.ttest_rel([ours["seeds"][s][0] for s in sorted(ours["seeds"])],
                          [base_c["seeds"][s][0] for s in sorted(base_c["seeds"])])[1]

lines = ["# Table 1 — 13-model fair benchmark (5 seeds)", "",
         "| # | Model | Clean AP | Full AP | Recall@1% | Recall@5% |", "|---|---|---|---|---|---|"]
for i, t in enumerate(table, 1):
    lines.append(f"| {i} | {t['model']} | {t['clean']:.3f} ± {t['clean_std']:.3f} | {t['full']:.3f} ± {t['full_std']:.3f} | "
                 f"{t['r1']:.3f} ± {t['r1_std']:.3f} | - |")
lines += ["", f"**Paired t-test (Ours-Full-TCN)**: vs {base['model']} on Full AP → p = {p_full:.4f}; "
              f"vs {base_c['model']} on Clean AP → p = {p_clean:.4f}", ""]
txt = "\n".join(lines)
(BASE / "outputs/phase1/table1_fair13.md").write_text(txt, encoding="utf-8")
print(txt)
