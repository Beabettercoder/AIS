#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 73: 27B LLM 零样本异常判别（同一测试窗、同一 AP 口径）。"""
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


def window_text(raw, rf, t):
    """把 20 步窗口序列化为文本。"""
    lines = []
    for k in range(len(raw)):
        dx, dy, vx, vy, sog, s_, c_ = raw[k]
        cog = (np.degrees(np.arctan2(s_, c_)) + 360) % 360
        lines.append(f"t={k*70//60}min dx={dx:.0f}m dy={dy:.0f}m sog={sog:.1f}kn cog={cog:.0f}deg")
    return "\n".join(lines)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    z = np.load(BASE / "eval_v2_graph2" / "test.npz")
    raw, rf = z["raw"].astype(np.float32), z["r_field"].astype(np.float32)
    cell, y = z["label_cell"], z["label_class"]
    idx = np.random.default_rng(0).choice(len(raw), size=min(N, len(raw)), replace=False)
    labels = (y[idx] > 0).astype(int)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval().cuda()
    print("model loaded in %.1fs" % (time.time() - t0), flush=True)

    yes_ids = tok.encode("Yes", add_special_tokens=False)[:1]
    no_ids = tok.encode("No", add_special_tokens=False)[:1]
    print("yes/no ids:", yes_ids, no_ids, flush=True)

    scores = []
    t1 = time.time()
    for n, i in enumerate(idx):
        body = window_text(raw[i], rf[i], None)
        msg = [{"role": "user", "content":
                "You are a maritime traffic analyst. Below are 20 consecutive AIS reports of one vessel "
                "(70 s apart). Decide whether this vessel's behaviour is anomalous (an abnormal manoeuvre "
                "or a degraded observation). Answer with a single word: Yes or No.\n\n" + body}]
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1]
        py = max(float(logits[j]) for j in yes_ids)
        pn = max(float(logits[j]) for j in no_ids)
        m = max(py, pn)
        scores.append(float(np.exp(py - m) / (np.exp(py - m) + np.exp(pn - m))))
        if (n + 1) % 25 == 0:
            print(f"{n+1}/{len(idx)} elapsed {time.time()-t1:.0f}s", flush=True)

    scores = np.array(scores)
    (BASE / "llm27b_result.json").write_text(json.dumps(
        {"n": len(idx), "AP": round(ap_score(scores, labels), 3),
         "labels_pos_rate": float(labels.mean()),
         "elapsed_s": round(time.time() - t1, 1)}, indent=2), encoding="utf-8")
    print("LLM AP =", round(ap_score(scores, labels), 3))


if __name__ == "__main__":
    main()
