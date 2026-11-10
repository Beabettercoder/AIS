#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 84: L1 信息对等评测——把 r_field / w_t / 邻船统计也喂给 LLM。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

MODEL = "/mnt/sda/HaiShangTS/models/Qwen3.8-27B"
BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
N = 200


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def window_text(raw, rf, w, nb_mask, nb_v):
    nb_speed = np.linalg.norm(nb_v, axis=-1)
    valid = int(nb_mask.sum())
    grp = float(np.median(nb_speed[nb_mask])) if valid else -1.0
    lines = []
    for k in range(len(raw)):
        dx, dy, vx, vy, sog, s_, c_ = raw[k]
        cog = (np.degrees(np.arctan2(s_, c_)) + 360) % 360
        lines.append(f"t={k} dx={dx:.0f}m dy={dy:.0f}m sog={sog:.1f}kn cog={cog:.0f}deg "
                     f"field_residual={rf[k]:.2f}m/s gate_w={w[k]:.2f} n_neigh={valid} "
                     f"group_sog={grp:.1f}kn")
    return "\n".join(lines)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    z = np.load(BASE / "eval_v2_mmsi" / "test.npz")
    raw = z["raw"].astype(np.float32)
    rf = z["r_field"].astype(np.float32)
    w = z["w_self"].astype(np.float32)
    nb_m = z["nb_mask"].astype(bool)
    nb_v = z["nb_v"].astype(np.float32)
    y = z["label_class"]
    idx = np.load(BASE / "L3_subset_idx.npy")
    labels = (y[idx] > 0).astype(int)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    yes_ids = tok.encode(" Yes", add_special_tokens=False) + tok.encode("Yes", add_special_tokens=False)
    no_ids = tok.encode(" No", add_special_tokens=False) + tok.encode("No", add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval().cuda()

    scores, answers = [], []
    t0 = time.time()
    for n, i in enumerate(idx):
        body = window_text(raw[i], rf[i], w[i], nb_m[i], nb_v[i])
        content = (
            "You are a maritime traffic analyst. Each line is one AIS report of a target vessel "
            "(70 s apart). Fields: dx/dy = displacement since the previous report; sog = speed over ground; "
            "cog = course over ground; field_residual = deviation from the local traffic-flow field estimated "
            "from neighbouring vessels (larger = more unusual relative to the surrounding traffic); "
            "gate_w = kinematic credibility of that report (1 = trustworthy, 0 = position jump / dropout); "
            "n_neigh = number of neighbouring vessels used for the field; group_sog = median speed of those "
            "neighbours. Decide whether the target vessel's behaviour is anomalous. Answer Yes or No.\n\n"
            + body)
        text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inp)
        lg = out.logits[0, -1]
        py = max(float(lg[j]) for j in yes_ids)
        pn = max(float(lg[j]) for j in no_ids)
        scores.append(py - pn); answers.append(int(py > pn))
        if (n + 1) % 50 == 0:
            print(f"{n+1}/{len(idx)} {time.time()-t0:.0f}s", flush=True)

    scores = np.array(scores); answers = np.array(answers)
    res = {"n": len(idx), "AP_logitdiff": round(ap_score(scores, labels), 3),
           "AP_hard": round(ap_score(answers.astype(float), labels), 3),
           "yes_rate": float(answers.mean()), "pos_rate": float(labels.mean()),
           "elapsed_s": round(time.time() - t0, 1), "setting": "L3 MMSI-disjoint, information-parity prompt"}
    print(res)
    (BASE / "llm27b_L3_result.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
