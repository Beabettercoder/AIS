#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 79: Table 7 补充计时口径脚注。"""
from pathlib import Path

p = Path("/mnt/sda/WangZijia/AIS/outputs/phase1/table7_llm_quantization.md")
t = p.read_text(encoding="utf-8")
note = ("\n**Timing scope.** The 2.73 ms (FP32) / 1.32 ms (FP16) / 0.88 ms (BF16) entries are "
        "*network-only* GPU forward latencies for 1000 windows. The full end-to-end pipeline — "
        "including CPU cKDTree neighbour retrieval and the per-window least-squares local-linear field "
        "solve — is **11.4 ms per 1000 vessels** (see Table 6). The LLM latencies are wall-clock "
        "per single window and therefore include tokenisation, prompt assembly and autoregressive head pass.\n")
if "Timing scope" not in t:
    t = t.rstrip() + "\n" + note
p.write_text(t, encoding="utf-8")
print(t[-600:])
