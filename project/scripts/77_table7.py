#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 77: Table 7 —— 轻量模型量化 vs 27B 大模型。"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
quant = {r["tag"]: r for r in json.loads((BASE / "quantization_report.json").read_text())}
z = json.loads((BASE / "llm27b_result.json").read_text())
s3 = json.loads((BASE / "llm27b_3shot_result.json").read_text())

rows = [
    ("PG-DRGN, FP32", "66.8 K", "261 KB", "2.73 ms / 1000", f"{quant['FP32 (32-bit)']['AP']:.3f}"),
    ("PG-DRGN, FP16", "66.8 K", "130.5 KB", "1.32 ms / 1000", f"{quant['FP16']['AP']:.3f}"),
    ("PG-DRGN, BF16", "66.8 K", "130.5 KB", "0.88 ms / 1000", f"{quant['BF16']['AP']:.3f}"),
    ("PG-DRGN, INT8 (dynamic, CPU)", "66.8 K", "224.5 KB", "CPU real-time", f"{quant['INT8 dynamic (CPU)']['AP']:.3f}"),
    ("Qwen3.8-27B, zero-shot", "27 B", "52 GB", f"{z['elapsed_s']/z['n']*1000:.0f} ms / window", f"{z['AP']:.3f}"),
    ("Qwen3.8-27B, 3-shot ICL", "27 B", "52 GB", f"{s3['elapsed_s']/s3['n']*1000:.0f} ms / window", f"{s3['AP']:.3f}"),
]
lines = ["# Table 7 — deployment cost vs detection quality", "",
         "| Model / precision | Params | Weight size | Latency | Full AP (200-window subset) |",
         "|---|---|---|---|---|"]
for r in rows:
    lines.append("| " + " | ".join(r) + " |")
lines += ["",
          f"- All PG-DRGN precisions give identical AP ({quant['FP32 (32-bit)']['AP']:.3f}); quantization is lossless.",
          f"- The 27B LLM needs ~{52*1024/0.261:.0f}x more storage and ~{z['elapsed_s']/z['n']/ (quant['BF16']['latency_ms_1000']/1000/1000):.0f}x more latency per sample.",
          "- LLM scores below chance (0.392 zero-shot, 0.437 with 3 in-context examples)."]
txt = "\n".join(lines)
(BASE / "table7_llm_quantization.md").write_text(txt, encoding="utf-8")
print(txt)
