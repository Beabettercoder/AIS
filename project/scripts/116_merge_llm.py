#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 116: 合并修正后的 L0/L1/3-shot，保留原 LoRA 结果。"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
main = json.loads((BASE / "llm_full_protocol.json").read_text())
fixed = {}
for p in BASE.glob("llm_fix_*.json"):
    name = p.stem.replace("llm_fix_", "")
    fixed[name] = json.loads(p.read_text())

for name, res in fixed.items():
    old = main.get(name, {})
    lora = old.get("L2_lora")
    new = {"L0": res["L0"], "L1": res["L1"], "3shot": res["3shot"],
           "L2_lora": lora,
           "L0_sec": res.get("L0_sec"), "L1_sec": res.get("L1_sec"),
           "3shot_sec": res.get("3shot_sec"), "L2_sec": old.get("L2_sec")}
    main[name] = new

(BASE / "llm_full_protocol_fixed.json").write_text(json.dumps(main, indent=2))
print(f"merged {len(fixed)} models")
for k, v in sorted(main.items()):
    print(f"{k:32s} L0={v.get('L0')} L1={v.get('L1')} 3s={v.get('3shot')} LoRA={v.get('L2_lora')}")
