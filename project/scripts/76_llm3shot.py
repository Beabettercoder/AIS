#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 76: Qwen3.8-27B 3-shot 上下文学习异常判别（同一 200 窗）。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
MODEL = "/mnt/sda/HaiShangTS/models/Qwen3.8-27B"
N = 200


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def window_text(raw):
    lines = []
    for k in range(len(raw)):
        dx, dy, vx, vy, sog, s_, c_ = raw[k]
        cog = (np.degrees(np.arctan2(s_, c_)) + 360) % 360
        lines.append(f"t={k*70//60}min dx={dx:.0f}m dy={dy:.0f}m sog={sog:.1f}kn cog={cog:.0f}deg")
    return "\n".join(lines)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    z = np.load(BASE / "eval_v2_graph2" / "test.npz")
    raw = z["raw"].astype(np.float32)
    cell, y = z["label_cell"], z["label_class"]
    idx = np.random.default_rng(0).choice(len(raw), size=min(N, len(raw)), replace=False)
    labels = (y[idx] > 0).astype(int)

    ex_norm = window_text(raw[np.flatnonzero(cell == 0)[0]])
    ex_turn = window_text(raw[np.flatnonzero((cell == 2) & (y == 1))[0]])
    ex_jump = window_text(raw[np.flatnonzero(cell == 1)[0]])

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval().cuda()
    print("loaded in %.1fs" % (time.time() - t0), flush=True)

    yes_ids = tok.encode(" Yes", add_special_tokens=False) + tok.encode("Yes", add_special_tokens=False)
    no_ids = tok.encode(" No", add_special_tokens=False) + tok.encode("No", add_special_tokens=False)

    scores = []
    t1 = time.time()
    for n, i in enumerate(idx):
        body = window_text(raw[i])
        content = ("You are a maritime traffic analyst. Below are 20 consecutive AIS reports of one vessel "
                   "(70 s apart). Decide whether its behaviour is anomalous.\n\n"
                   "Example 1 (normal, straight transit):\n" + ex_norm + "\nAnswer: No\n\n"
                   "Example 2 (anomalous, sharp turn):\n" + ex_turn + "\nAnswer: Yes\n\n"
                   "Example 3 (anomalous, degraded observations):\n" + ex_jump + "\nAnswer: Yes\n\n"
                   "Now classify this vessel, answer Yes or No:\n" + body)
        text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1]
        py = max(float(logits[j]) for j in yes_ids)
        pn = max(float(logits[j]) for j in no_ids)
        m = max(py, pn)
        scores.append(float(np.exp(py - m) / (np.exp(py - m) + np.exp(pn - m))))
        if (n + 1) % 50 == 0:
            print(f"{n+1}/{len(idx)} {time.time()-t1:.0f}s", flush=True)

    scores = np.array(scores)
    out = {"n": len(idx), "AP": round(ap_score(scores, labels), 3),
           "pos_rate": float(labels.mean()), "elapsed_s": round(time.time() - t1, 1)}
    (BASE / "llm27b_3shot_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("3-shot AP =", out["AP"])


if __name__ == "__main__":
    main()
