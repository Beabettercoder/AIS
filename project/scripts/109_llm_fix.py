#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 109: 修正版 LLM 评测——按模型取子词 id，左侧截断，3-shot 用短示例。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
MODELS_DIR = Path("/mnt/sda/WangZijia/models")
MODEL_27B = Path("/mnt/sda/HaiShangTS/models/Qwen3.8-27B")
MAXLEN = 2048


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def body(z, i, rich, steps=None):
    raw = z["raw"][i]
    rng = range(len(raw)) if steps is None else range(max(0, len(raw) - steps), len(raw))
    out = []
    for k in rng:
        line = (f"t={k} dx={raw[k,0]:.0f}m dy={raw[k,1]:.0f}m sog={raw[k,4]:.1f}kn "
                f"cog={(np.degrees(np.arctan2(raw[k,5], raw[k,6]))+360)%360:.0f}deg")
        if rich:
            line += f" field_residual={z['r_field'][i][k]:.2f} gate_w={z['w_self'][i][k]:.2f}"
        out.append(line)
    return "\n".join(out)


INSTR_P = ("You are a maritime traffic analyst. Each line is one AIS report of one vessel (70 s apart). "
           "Decide whether the behaviour is anomalous. Answer Yes or No.\n\n")
INSTR_R = ("You are a maritime traffic analyst. Each line is one AIS report of one vessel (70 s apart). "
           "field_residual = deviation from the local traffic-flow field; gate_w = kinematic credibility. "
           "Decide whether the behaviour is anomalous. Answer Yes or No.\n\n")


def main() -> None:
    names = sys.argv[1].split(",")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    z = np.load(BASE / "eval_v2_mmsi" / "test.npz")
    idx = np.load(BASE / "L3_subset_idx.npy")
    labels = (z["label_class"][idx] > 0).astype(int)
    ztr = np.load(BASE / "eval_v2_graph2" / "train.npz")
    pos = int(np.flatnonzero(ztr["label_class"] > 0)[0]); neg = int(np.flatnonzero(ztr["label_class"] == 0)[0])

    for name in names:
        d = MODEL_27B if name == "Qwen3.8-27B" else MODELS_DIR / name
        if not d.exists():
            print("skip", name, flush=True); continue
        print("###", name, flush=True)
        tok = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.bfloat16,
                                                     trust_remote_code=True).eval().cuda()
        yes_id = tok.encode(" Yes", add_special_tokens=False)[-1]
        no_id = tok.encode(" No", add_special_tokens=False)[-1]
        print(f"  ids: yes={yes_id} no={no_id}", flush=True)

        def score(text):
            inp = tok(text, truncation=True, max_length=MAXLEN, truncation_side="left",
                      return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**inp)
            lg = out.logits[0, -1]
            return float(lg[yes_id] - lg[no_id])

        def chat(content):
            msgs = [{"role": "user", "content": content}]
            try:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        ex = ("Example 1 (normal):\n" + body(ztr, neg, True, 5) + "\nAnswer: No\n\n"
              "Example 2 (anomalous):\n" + body(ztr, pos, True, 5) + "\nAnswer: Yes\n\n")
        res = {}
        for tag, instr, rich, few in (("L0", INSTR_P, False, False), ("L1", INSTR_R, True, False),
                                      ("3shot", INSTR_R, True, True)):
            t0 = time.time()
            sc = [score(chat(instr + (ex if few else "") + body(z, int(i), rich))) for i in idx]
            res[tag] = round(ap_score(np.array(sc), labels), 3)
            res[tag + "_sec"] = round(time.time() - t0, 1)
            print(f"  {tag}: {res[tag]:.3f} ({res[tag+'_sec']:.0f}s)", flush=True)
        (BASE / f"llm_fix_{name}.json").write_text(json.dumps(res, indent=2))
        del model
        torch.cuda.empty_cache()
    print("WORKER DONE")


if __name__ == "__main__":
    main()
