#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 83: Table 7 修正版（正确子词 + 同集对比 + 无信息基线）。"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
z = json.loads((BASE / "llm27b_fixed_result.json").read_text())
s3 = json.loads((BASE / "llm27b_3shot_result.json").read_text())
ours = json.loads((BASE / "ours_on_llm_subset.json").read_text())
q = {r["tag"]: r for r in json.loads((BASE / "quantization_report.json").read_text())}

lines = ["# Table 7 — deployment cost vs detection quality (same 200-window subset)", "",
         f"Positive rate on the subset: **π = {ours['pos_rate']:.2f}** (the no-information AP level).", "",
         "| Model / setting | Params | Weight size | Latency | AP (200 windows) |",
         "|---|---|---|---|---|",
         f"| PG-DRGN, FP32 | 66.8 K | 261 KB | 2.73 ms / 1000 | **{ours['AP']:.3f}** |",
         f"| PG-DRGN, FP16 | 66.8 K | 130.5 KB | 1.32 ms / 1000 | **{q['FP16']['AP']:.3f}** (full test set) |",
         f"| PG-DRGN, BF16 | 66.8 K | 130.5 KB | 0.88 ms / 1000 | **{q['BF16']['AP']:.3f}** (full test set) |",
         f"| PG-DRGN, INT8 (dynamic, CPU) | 66.8 K | 224.5 KB | CPU real-time | **{q['INT8 dynamic (CPU)']['AP']:.3f}** |",
         f"| Qwen3.8-27B, zero-shot | 27 B | 52 GB | {z['elapsed_s']/z['n']*1000:.0f} ms / window | {z['AP_logitdiff']:.3f} |",
         f"| Qwen3.8-27B, 3-shot ICL | 27 B | 52 GB | {s3['elapsed_s']/s3['n']*1000:.0f} ms / window | {s3['AP']:.3f} |",
         "",
         "**Correction note.** An earlier version of this table reported 0.392 / 0.437 for the LLM. "
         "Those numbers were produced with the wrong sub-word ids: the scorer compared `Yes`/`No` (9175/2665) "
         "while the model emits ` Yes`/` No` (7179/2233) after the chat template. Re-scoring with the correct "
         "ids and with thinking mode explicitly disabled (`enable_thinking=False`) gives the values above. "
         "The LLM's hard answers are also heavily skewed (it answers *Yes* for only 18.5% of windows while "
         f"π = {ours['pos_rate']:.2f}), so its score is close to the no-information level.",
         "",
         "**Timing scope.** 2.73 / 1.32 / 0.88 ms are *network-only* GPU forward latencies for 1000 windows; "
         "the full end-to-end pipeline including CPU cKDTree retrieval and the per-window least-squares local "
         "linear field solve is **11.4 ms per 1000 vessels** (Table 6). LLM latencies are wall-clock per single "
         "window and include tokenisation, prompt assembly and the decoding head.",
         "",
         "**What this table does and does not show.** It compares a *supervised, physics-augmented* detector "
         "with an *off-the-shelf, zero/few-shot* general model. It does not establish that a 66.8K model "
         "outperforms a 27B model under equal supervision or equal input information.",
         ]
txt = "\n".join(lines)
(BASE / "table7_llm_quantization.md").write_text(txt, encoding="utf-8")
print(txt)
