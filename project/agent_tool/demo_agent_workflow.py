#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end agent demo: user question -> tool call -> local LLM briefing -> markdown log."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maritime_situational_tool import TOOL_SCHEMA, analyze_maritime_situation  # noqa: E402

import os
BASE = Path(os.environ.get("PGDRGN_OUT", Path(__file__).resolve().parent))
LLM = os.environ.get("PGDRGN_LLM", "/mnt/sda/WangZijia/models/Qwen2.5-7B-Instruct")
QUESTION = "请分析当前海域交通流态势，排查是否存在非正常的船舶机动或信号干扰？"

PROMPT = """You are a maritime situational-awareness agent. A tool returned the following JSON.

{payload}

Write a concise brief (Chinese first, then an English one-paragraph summary) with:
1) the macro flow state,
2) how many vessels had corrupt signals intercepted by the kinematic gate (and why that matters),
3) the highest-risk vessels with their tier and behaviour intent,
4) one recommended action for the watch officer.
Keep it under 220 Chinese characters + 90 English words. Do not invent numbers."""


def main() -> None:
    src = BASE / "sample" / "sample_ais_windows.npz"
    if not src.exists():
        src = Path("/mnt/sda/WangZijia/AIS/outputs/phase1/eval_v2_graph2/test.npz")
    z = np.load(src)
    lab = z["label_class"]
    idx = np.flatnonzero(lab > 0)[:3]
    if len(lab) > 120:
        idx = np.concatenate([np.arange(30), idx + 100]).astype(int)
    else:
        idx = np.arange(len(lab))
    data = {"raw": z["raw"][idx], "w": z["w_self"][idx], "r_field": z["r_field"][idx],
            "nb_xy": z["nb_xy"][idx], "nb_v": z["nb_v"][idx], "nb_mask": z["nb_mask"][idx]}

    t0 = time.perf_counter()
    result = analyze_maritime_situation(data, region_name="Houston Ship Channel", far_budget=0.01)
    tool_ms = (time.perf_counter() - t0) * 1000
    print(f"[tool] {tool_ms:.1f} ms, {len(result['vessel_anomaly_alerts']['alerts'])} alerts")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    t1 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
    msgs = [{"role": "user", "content": PROMPT.format(payload=json.dumps(result, ensure_ascii=False))}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, truncation=True, max_length=3072, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=320, do_sample=False)
    brief = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    llm_s = time.perf_counter() - t1

    md = [f"# 海事智能体端到端演示报告", "",
          f"- 用户提问：**{QUESTION}**",
          f"- 工具：`analyze_maritime_situation`（PG-DRGN, 66.8 K 参数）",
          f"- 工具耗时：**{tool_ms:.1f} ms**（{len(idx)} 个窗口）",
          f"- 本地模型：`{LLM}`，生成耗时 {llm_s:.1f} s",
          f"- Function-Calling Schema：`{TOOL_SCHEMA['function']['name']}`", "",
          "## 1. 工具返回的结构化 JSON", "", "```json",
          json.dumps(result, ensure_ascii=False, indent=2), "```", "",
          "## 2. 智能体生成简报", "", brief, "",
          "## 3. 复现", "", "```bash", "python agent_tool/demo_agent_workflow.py", "```"]
    (BASE / "agent_tool_demo_report.md").write_text("\n".join(md), encoding="utf-8")
    print(brief[:600])
    print("\nsaved", BASE / "agent_tool_demo_report.md")


if __name__ == "__main__":
    main()
