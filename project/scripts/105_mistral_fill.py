#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 105: 补 Mistral 的 3-shot 并合并进 JSON。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
M = Path("/mnt/sda/WangZijia/models/Mistral-7B-Instruct-v0.2")
JS = BASE / "llm_full_protocol.json"


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def body(z, i):
    raw = z["raw"][i]; rf = z["r_field"][i]; w = z["w_self"][i]
    m = z["nb_mask"][i]; nv = z["nb_v"][i]
    sp = np.linalg.norm(nv, axis=-1); valid = int(m.sum())
    grp = float(np.median(sp[m])) if valid else -1.0
    return "\n".join(
        f"t={k} dx={raw[k,0]:.0f}m dy={raw[k,1]:.0f}m sog={raw[k,4]:.1f}kn "
        f"cog={(np.degrees(np.arctan2(raw[k,5], raw[k,6]))+360)%360:.0f}deg "
        f"field_residual={rf[k]:.2f} gate_w={w[k]:.2f} n_neigh={valid} group_sog={grp:.1f}kn"
        for k in range(len(raw)))


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(M), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(M), dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
    z = np.load(BASE / "eval_v2_mmsi" / "test.npz")
    idx = np.load(BASE / "L3_subset_idx.npy")
    labels = (z["label_class"][idx] > 0).astype(int)
    ztr = np.load(BASE / "eval_v2_graph2" / "train.npz")
    pos = int(np.flatnonzero(ztr["label_class"] > 0)[0]); neg = int(np.flatnonzero(ztr["label_class"] == 0)[0])
    ex = ("Example 1 (normal):\n" + body(ztr, neg) + "\nAnswer: No\n\n"
          "Example 2 (anomalous):\n" + body(ztr, pos) + "\nAnswer: Yes\n\n")
    instr = ("You are a maritime traffic analyst. Each line is one AIS report of one vessel (70 s apart). "
             "field_residual = deviation from the local traffic-flow field; gate_w = kinematic credibility; "
             "n_neigh = neighbours used; group_sog = median neighbour speed. "
             "Decide whether the behaviour is anomalous. Answer Yes or No.\n\n")
    yes_ids = tok.encode(" Yes", add_special_tokens=False) + tok.encode("Yes", add_special_tokens=False)
    no_ids = tok.encode(" No", add_special_tokens=False) + tok.encode("No", add_special_tokens=False)
    sc = []
    for i in idx:
        content = instr + ex + body(z, int(i))
        try:
            text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True)
        inp = tok(text, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inp)
        lg = out.logits[0, -1]
        sc.append(max(float(lg[j]) for j in yes_ids) - max(float(lg[j]) for j in no_ids))
    three = round(ap_score(np.array(sc), labels), 3)
    d = json.loads(JS.read_text())
    d["Mistral-7B-Instruct-v0.2"] = {"L0": 0.497, "L1": 0.485, "3shot": three, "L2_lora": 0.610,
                                     "L0_sec": 19, "L1_sec": 26, "3shot_sec": 30, "L2_sec": 699}
    JS.write_text(json.dumps(d, indent=2))
    print("mistral 3-shot =", three, "| models:", len(d))


if __name__ == "__main__":
    main()
