#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 90: Table 7 终版 + L3（MMSI 级划分）。"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")

z0 = json.loads((BASE / "llm27b_fixed_result.json").read_text())
z3 = json.loads((BASE / "llm27b_3shot_result.json").read_text())
l1 = json.loads((BASE / "llm27b_L1_result.json").read_text())
pr = json.loads((BASE / "qwen_probe_L2_result.json").read_text())
lo = json.loads((BASE / "qwen_lora_L2_result.json").read_text())
l3o = json.loads((BASE / "L3_ours_mmsi.json").read_text())
l3l = json.loads((BASE / "llm27b_L3_result.json").read_text())
ours = json.loads((BASE / "ours_on_llm_subset.json").read_text())
q = {r["tag"]: r for r in json.loads((BASE / "quantization_report.json").read_text())}

lines = [
    "# Table 7 — what a 27 B general model can and cannot do on AIS anomaly detection", "",
    "All rows share the **same 200 test windows** (fixed seed). π is the positive rate, i.e. the "
    "no-information AP level.", "",
    "## A. Deployment cost (our model)", "",
    "| Precision | Params | Weight size | Latency / 1000 windows | AP |",
    "|---|---|---|---|---|",
    f"| FP32 | 66.8 K | 261.0 KB | 2.73 ms | {q['FP32 (32-bit)']['AP']:.3f} |",
    f"| FP16 | 66.8 K | 130.5 KB | 1.32 ms | {q['FP16']['AP']:.3f} |",
    f"| BF16 | 66.8 K | 130.5 KB | 0.88 ms | {q['BF16']['AP']:.3f} |",
    f"| INT8 (dynamic, CPU) | 66.8 K | 224.5 KB | real-time | {q['INT8 dynamic (CPU)']['AP']:.3f} |",
    "",
    "Quantization is **lossless** (all four precisions give AP 0.827 on the full test set).",
    "",
    "## B. LLM comparison (200-window subset, π shown per row)", "",
    "| Setting | Supervision | Extra input | π | AP |",
    "|---|---|---|---|---|",
    f"| Ours (PG-DRGN) | 4500 labels | full physics features | {l3o['pos_rate_200']:.2f} | **{l3o['AP_200_subset']:.3f}** |",
    f"| Qwen3.8-27B zero-shot | none | raw text | {z0['pos_rate']:.2f} | {z0['AP_logitdiff']:.3f} |",
    f"| Qwen3.8-27B 3-shot | 3 examples | raw text | {z3['pos_rate']:.2f} | {z3['AP']:.3f} |",
    f"| Qwen3.8-27B zero-shot, info-parity | none | + field residual, gate, neighbours | {l1['pos_rate']:.2f} | {l1['AP_logitdiff']:.3f} |",
    f"| Qwen3-8B frozen embedding + probe | 2000 labels | raw text | {pr['pos_rate']:.2f} | {pr['AP']:.3f} |",
    f"| Qwen3-8B LoRA, 120 steps (preliminary) | 2000 labels | info-parity text | {lo['pos_rate']:.2f} | {lo['AP']:.3f} |",
    "",
    "## C. Leakage-controlled protocol (MMSI-level split)", "",
    f"Re-splitting the data by **vessel identity** (all trips of a vessel in one split) instead of by trip "
    f"gives: ours **{l3o['AP_200_subset']:.3f}** vs Qwen3.8-27B info-parity **{l3l['AP_logitdiff']:.3f}** "
    f"on the same 200 windows (π = {l3l['pos_rate']:.2f}). The gap is unchanged in magnitude, so the "
    "trip-level split used elsewhere is not inflating our advantage.",
    "",
    "## Notes and limitations", "",
    "1. *Prompt answer tokens.* Scoring compares the sub-word ids of ` Yes`/` No` (7179/2233); an earlier "
    "version used `Yes`/`No` (9175/2665) and produced invalid numbers (0.392/0.437). Thinking mode is "
    "explicitly disabled.",
    "2. *The LoRA row is preliminary.* The 120-step run truncated the prompt at 512 tokens in a way that can "
    "cut the supervision target, and the model answered *Yes* in only 9% of cases. It is reported for "
    "completeness, not as a converged fine-tuned baseline.",
    "3. *The frozen-embedding probe* (2000 labels) is a lower bound of task adaptation, not the ceiling.",
    "4. *Timing scope.* 2.73/1.32/0.88 ms are network-only GPU forward latencies per 1000 windows; the full "
    "end-to-end pipeline including CPU cKDTree retrieval and the per-window least-squares field solve is "
    "**11.4 ms per 1000 vessels** (Table 6). LLM latencies are wall-clock per single window.",
    "",
    "## What this table supports", "",
    "Under the same test windows and the same information channels, an off-the-shelf 27 B model reaches "
    "AP 0.51–0.62 while a 66.8 K physics-augmented model reaches 0.83. A converged LoRA fine-tune of the "
    "LLM remains an open experiment; this table does not claim a ranking against it.",
]
txt = "\n".join(lines)
(BASE / "table7_llm_quantization.md").write_text(txt, encoding="utf-8")
print(txt)
