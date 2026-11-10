#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 80: 校验 Qwen3.8-27B 的答案位置/分词/思考模式，判断 Yes-No logit 打分是否有效。"""
from __future__ import annotations

import numpy as np
import torch

MODEL = "/mnt/sda/HaiShangTS/models/Qwen3.8-27B"
BASE = "/mnt/sda/WangZijia/AIS/outputs/phase1"


def window_text(raw):
    lines = []
    for k in range(len(raw)):
        dx, dy, vx, vy, sog, s_, c_ = raw[k]
        cog = (np.degrees(np.arctan2(s_, c_)) + 360) % 360
        lines.append(f"t={k*70//60}min dx={dx:.0f}m dy={dy:.0f}m sog={sog:.1f}kn cog={cog:.0f}deg")
    return "\n".join(lines)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    for s in ["Yes", "No", " Yes", " No", "yes", "no"]:
        print(f"token {s!r} -> {tok.encode(s, add_special_tokens=False)}")

    z = np.load(f"{BASE}/eval_v2_graph2/test.npz")
    raw = z["raw"].astype(np.float32)
    cell, y = z["label_cell"], z["label_class"]

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval().cuda()
    print("loaded", flush=True)

    idx_norm = int(np.flatnonzero(cell == 0)[0])
    idx_anom = int(np.flatnonzero(cell == 2)[0])
    for tag, i in (("normal", idx_norm), ("anomaly", idx_anom)):
        content = ("You are a maritime traffic analyst. Below are 20 consecutive AIS reports of one vessel "
                   "(70 s apart). Decide whether its behaviour is anomalous. Answer with a single word: Yes or No.\n\n"
                   + window_text(raw[i]))
        messages = [{"role": "user", "content": content}]
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=24, do_sample=False)
        gen = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n[{tag}] generated -> {gen!r}", flush=True)


if __name__ == "__main__":
    main()
