#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 87: Table 7 终版（零样本 / 3-shot / L1 信息对等 / L2 探针 / 我们的模型）。"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
z0 = json.loads((BASE / "llm27b_fixed_result.json").read_text())
z3 = json.loads((BASE / "llm27b_3shot_result.json").read_text())
l1 = json.loads((BASE / "llm27b_L1_result.json").read_text())
pr = json.loads((BASE / "qwen_probe_L2_result.json").read_text())
ours = json.loads((BASE / "ours_on_llm_subset.json").read_text())
q = {r["tag"]: r for r in json.loads((BASE / "quantization_report.json").read_text())}

lines = [
    "# Table 7 — deployment cost vs detection quality (identical 200-window test subset)", "",
    f"No-information level (positive rate): **π = {ours['pos_rate']:.2f}**.", "",
    "| Setting | Model | Labels used | Weight size | Latency | AP |",
    "|---|---|---|---|---|---|",
    f"| Supervised (ours) | PG-DRGN FP32 | 4500 | 261 KB | 2.73 ms / 1000 | **{ours['AP']:.3f}** |",
    f"| Supervised (ours) | PG-DRGN FP16 | 4500 | 130.5 KB | 1.32 ms / 1000 | {q['FP16']['AP']:.3f} (full test) |",
    f"| Supervised (ours) | PG-DRGN INT8 | 4500 | 224.5 KB | CPU real-time | {q['INT8 dynamic (CPU)']['AP']:.3f} |",
    f"| Zero-shot | Qwen3.8-27B | 0 | 52 GB | {z0['elapsed_s']/z0['n']*1000:.0f} ms / window | {z0['AP_logitdiff']:.3f} |",
    f"| 3-shot ICL | Qwen3.8-27B | 3 | 52 GB | {z3['elapsed_s']/z3['n']*1000:.0f} ms / window | {z3['AP']:.3f} |",
    f"| **L1 info-parity zero-shot** | Qwen3.8-27B | 0 | 52 GB | {l1['elapsed_s']/l1['n']*1000:.0f} ms / window | **{l1['AP_logitdiff']:.3f}** |",
    f"| **L2 frozen embedding + probe** | Qwen3-8B | {pr['n_train']} | 16 GB | 1.1 s / window (encode) | **{pr['AP']:.3f}** |",
    "",
    "**Protocol notes.**",
    "",
    "1. *Same test windows for every row*: the 200 windows are drawn once with a fixed seed and shared by all models; "
    "π = 0.42, so AP ≈ 0.42 is the no-information level (not 0.5).",
    f"2. *L1 (information parity)*: the prompt additionally provides per-step field residual, kinematic gate weight, "
    f"neighbour count and neighbour median speed — i.e. the same information channels as our model, in text form. "
    f"AP rises from {z0['AP_logitdiff']:.3f} (raw text) to {l1['AP_logitdiff']:.3f}, still far below 0.804.",
    f"3. *L2 (supervision parity, probe)*: frozen Qwen3-8B last-token embeddings + logistic regression trained on "
    f"{pr['n_train']} of the same labelled windows gives AP {pr['AP']:.3f} ≈ π, i.e. the frozen language "
    f"representation is not linearly separable for this task.",
    "4. *LoRA fine-tuning was attempted but not completed*: all eight GPUs were occupied by other users' jobs "
    "(64–81 GB held per card) and the runs hit CUDA OOM. A task-adapted LoRA result therefore remains an open item; "
    "the L2 probe row should be read as a lower bound of adaptation, not as the ceiling.",
    "",
    "**Timing scope.** 2.73 / 1.32 / 0.88 ms are *network-only* GPU forward latencies per 1000 windows; the full "
    "end-to-end pipeline including CPU cKDTree retrieval and the per-window least-squares local-linear field solve "
    "is **11.4 ms per 1000 vessels** (Table 6). LLM latencies are wall-clock per single window.",
    "",
    "**What this table supports.** Under the same supervision and the same input information, a 66.8 K "
    "physics-augmented model reaches AP 0.804 while an off-the-shelf LLM stays at 0.51–0.59 zero/few-shot and 0.39 "
    "with a frozen-embedding probe. It does **not** yet establish the ranking against a LoRA-tuned LLM.",
]
txt = "\n".join(lines)
(BASE / "table7_llm_quantization.md").write_text(txt, encoding="utf-8")
print(txt)
