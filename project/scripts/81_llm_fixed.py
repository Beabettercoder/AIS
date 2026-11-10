#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 81: 修正版 LLM 打分（正确的 Yes/No 子词 + 关闭思考），重新计算 AP。"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

MODEL = "/mnt/sda/HaiShangTS/models/Qwen3.8-27B"
BASE = "/mnt/sda/WangZijia/AIS/outputs/phase1"
N = 200


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def window_text(raw):
    return "\n".join(
        f"t={k*70//60}min dx={raw[k,0]:.0f}m dy={raw[k,1]:.0f}m sog={raw[k,4]:.1f}kn "
        f"cog={(np.degrees(np.arctan2(raw[k,5], raw[k,6]))+360)%360:.0f}deg" for k in range(len(raw)))


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    z = np.load(f"{BASE}/eval_v2_graph2/test.npz")
    raw = z["raw"].astype(np.float32)
    y = z["label_class"]
    idx = np.random.default_rng(0).choice(len(raw), size=min(N, len(raw)), replace=False)
    labels = (y[idx] > 0).astype(int)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    yes_ids = tok.encode(" Yes", add_special_tokens=False) + tok.encode("Yes", add_special_tokens=False)
    no_ids = tok.encode(" No", add_special_tokens=False) + tok.encode("No", add_special_tokens=False)
    print("yes ids", yes_ids, "no ids", no_ids, flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval().cuda()

    scores, answers = [], []
    t0 = time.time()
    for n, i in enumerate(idx):
        content = ("You are a maritime traffic analyst. Below are 20 consecutive AIS reports of one vessel "
                   "(70 s apart). Decide whether its behaviour is anomalous. Answer with a single word: Yes or No.\n\n"
                   + window_text(raw[i]))
        try:
            text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inp)
        lg = out.logits[0, -1]
        py = max(float(lg[j]) for j in yes_ids)
        pn = max(float(lg[j]) for j in no_ids)
        scores.append(py - pn)
        answers.append(1 if py > pn else 0)
        if (n + 1) % 50 == 0:
            print(f"{n+1}/{len(idx)} {time.time()-t0:.0f}s", flush=True)

    scores = np.array(scores); answers = np.array(answers)
    out = {"n": len(idx), "AP_logitdiff": round(ap_score(scores, labels), 3),
           "pos_rate": float(labels.mean()), "yes_rate": float(answers.mean()),
           "elapsed_s": round(time.time() - t0, 1)}
    print(out)
    with_yes = ap_score(answers.astype(float), labels)
    out["AP_hard_yes"] = round(with_yes, 3)
    (__import__("pathlib").Path(BASE) / "llm27b_fixed_result.json").write_text(json.dumps(out, indent=2))
    print("hard-answer AP:", round(with_yes, 3))


if __name__ == "__main__":
    main()
