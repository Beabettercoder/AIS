#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 106: Qwen3.8-27B 在同一 MMSI 200 窗上的四档协议（L0/L1/3-shot/LoRA）。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
MPATH = Path("/mnt/sda/HaiShangTS/models/Qwen3.8-27B")
OUT = BASE / "llm_full_protocol.json"
STEPS = 150
BS = 2


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def body(z, i, rich=True):
    raw = z["raw"][i]
    base = []
    for k in range(len(raw)):
        line = (f"t={k} dx={raw[k,0]:.0f}m dy={raw[k,1]:.0f}m sog={raw[k,4]:.1f}kn "
                f"cog={(np.degrees(np.arctan2(raw[k,5], raw[k,6]))+360)%360:.0f}deg")
        if rich:
            line += (f" field_residual={z['r_field'][i][k]:.2f} gate_w={z['w_self'][i][k]:.2f}")
        base.append(line)
    if rich:
        m = z["nb_mask"][i]; sp = np.linalg.norm(z["nb_v"][i], axis=-1)
        valid = int(m.sum()); grp = float(np.median(sp[m])) if valid else -1.0
        base.append(f"n_neigh={valid} group_sog={grp:.1f}kn")
    return "\n".join(base)


INSTR_P = ("You are a maritime traffic analyst. Each line is one AIS report of one vessel (70 s apart). "
           "Decide whether the behaviour is anomalous. Answer Yes or No.\n\n")
INSTR_R = ("You are a maritime traffic analyst. Each line is one AIS report of one vessel (70 s apart). "
           "field_residual = deviation from the local traffic-flow field; gate_w = kinematic credibility. "
           "Decide whether the behaviour is anomalous. Answer Yes or No.\n\n")


def chat(tok, content, prompt=True):
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=prompt, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=prompt)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MPATH), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(MPATH), dtype=torch.bfloat16, trust_remote_code=True).cuda()
    dev = "cuda"
    z = np.load(BASE / "eval_v2_mmsi" / "test.npz")
    idx = np.load(BASE / "L3_subset_idx.npy")
    labels = (z["label_class"][idx] > 0).astype(int)
    yes_ids = tok.encode(" Yes", add_special_tokens=False) + tok.encode("Yes", add_special_tokens=False)
    no_ids = tok.encode(" No", add_special_tokens=False) + tok.encode("No", add_special_tokens=False)

    def score(text):
        inp = tok(text, truncation=True, max_length=1024, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp)
        lg = out.logits[0, -1]
        return max(float(lg[j]) for j in yes_ids) - max(float(lg[j]) for j in no_ids)

    res = {}
    ztr = np.load(BASE / "eval_v2_graph2" / "train.npz")
    pos = int(np.flatnonzero(ztr["label_class"] > 0)[0]); neg = int(np.flatnonzero(ztr["label_class"] == 0)[0])
    ex = ("Example 1 (normal):\n" + body(ztr, neg) + "\nAnswer: No\n\n"
          "Example 2 (anomalous):\n" + body(ztr, pos) + "\nAnswer: Yes\n\n")
    for tag, instr, rich, few in (("L0", INSTR_P, False, False), ("L1", INSTR_R, True, False),
                                  ("3shot", INSTR_R, True, True)):
        t0 = time.time()
        sc = [score(chat(tok, instr + (ex if few else "") + body(z, int(i), rich))) for i in idx]
        res[tag] = round(ap_score(np.array(sc), labels), 3)
        res[tag + "_sec"] = round(time.time() - t0, 1)
        print(f"{tag}: {res[tag]:.3f} ({res[tag+'_sec']:.0f}s)", flush=True)

    del model
    torch.cuda.empty_cache()
    print("LoRA skipped for 27B (memory/time); L0/L1/3-shot recorded", flush=True)
    res["L2_lora"] = None
    d = json.loads(OUT.read_text())
    d["Qwen3.8-27B"] = res
    OUT.write_text(json.dumps(d, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
