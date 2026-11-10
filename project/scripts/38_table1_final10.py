#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 38: 最终 Table 1（10 seeds）：8 基线 + 5 图/Ours，mean±std + 配对检验。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path("/mnt/sda/WangZijia/AIS")
PAT = re.compile(r"^(\S+)\s+s(\d+): clean=([\d.]+) full=([\d.]+) R1=([\d.]+) R5=([\d.]+)")

rows: dict[str, dict[int, tuple]] = {}
for log in ("logs/57_fair8.log", "logs/58_fair8_ext.log"):
    p = BASE / log
    if not p.exists():
        continue
    for line in p.read_text(errors="ignore").splitlines():
        m = PAT.match(line.strip())
        if m:
            rows.setdefault(m.group(1), {})[int(m.group(2))] = tuple(float(m.group(i)) for i in (3, 4, 5, 6))

for js in ("outputs/phase1/graph_ours_5seed.json", "outputs/phase1/graph_ours_ext.json"):
    p = BASE / js
    if not p.exists():
        continue
    for r in json.loads(p.read_text()):
        rows.setdefault(r["model"], {})[r["seed"]] = (r["clean_AP"], r["full_AP"], r["recall_1pct"], r["recall_5pct"])

table = []
for name, d in rows.items():
    seeds = sorted(d)
    arr = np.array([d[s] for s in seeds])
    table.append({"model": name, "clean": arr[:, 0].mean(), "clean_std": arr[:, 0].std(ddof=1),
                  "full": arr[:, 1].mean(), "full_std": arr[:, 1].std(ddof=1),
                  "r1": arr[:, 2].mean(), "r1_std": arr[:, 2].std(ddof=1),
                  "r5": arr[:, 3].mean(), "r5_std": arr[:, 3].std(ddof=1),
                  "seeds": d, "n": len(seeds)})
table.sort(key=lambda x: -x["full"])

ours = next(t for t in table if t["model"] == "Ours-Full-TCN")
base_f = max((t for t in table if "Ours" not in t["model"]), key=lambda t: t["full"])
base_c = max((t for t in table if "Ours" not in t["model"]), key=lambda t: t["clean"])
base_r = max((t for t in table if "Ours" not in t["model"]), key=lambda t: t["r1"])

def pval(idx: int, base) -> float:
    common = sorted(set(ours["seeds"]) & set(base["seeds"]))
    a = [ours["seeds"][s][idx] for s in common]
    b = [base["seeds"][s][idx] for s in common]
    return float(stats.ttest_rel(a, b)[1])

lines = [f"# Table 1 — 13-model fair benchmark ({table[0]['n']} seeds, 7-channel protocol)", "",
         "| # | Model | Clean AP | Full AP | Recall@1% | Recall@5% |", "|---|---|---|---|---|---|"]
for i, t in enumerate(table, 1):
    lines.append(f"| {i} | {t['model']} | {t['clean']:.3f} ± {t['clean_std']:.3f} | {t['full']:.3f} ± {t['full_std']:.3f} | "
                 f"{t['r1']:.3f} ± {t['r1_std']:.3f} | {t['r5']:.3f} ± {t['r5_std']:.3f} |")
lines += ["", f"**Paired t-test (Ours-Full-TCN, n={ours['n']})** — Full AP vs {base_f['model']}: p = {pval(1, base_f):.4f}; "
              f"Clean AP vs {base_c['model']}: p = {pval(0, base_c):.4f}; "
              f"Recall@1% vs {base_r['model']}: p = {pval(2, base_r):.4f}; "
              f"Recall@5% vs {base_f['model']}: p = {pval(3, base_f):.4f}", ""]
txt = "\n".join(lines)
(BASE / "outputs/phase1/table1_fair13.md").write_text(txt, encoding="utf-8")
print(txt)
