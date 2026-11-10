#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 101: 多家族多尺寸 LLM 横评（L0 原始文本 vs L1 物理增强），同一 200 窗。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
MODELS_DIR = Path("/mnt/sda/WangZijia/models")
N = 200

CANDIDATES = [
    "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct",
    "DeepSeek-R1-Distill-Qwen-1.5B", "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.2", "gemma-2-9b-it",
]


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def body(raw, rf, w, nbm, nbv, rich):
    if not rich:
        return "\n".join(f"t={k} dx={raw[k,0]:.0f}m dy={raw[k,1]:.0f}m sog={raw[k,4]:.1f}kn "
                         f"cog={(np.degrees(np.arctan2(raw[k,5], raw[k,6]))+360)%360:.0f}deg"
                         for k in range(len(raw)))
    nb_speed = np.linalg.norm(nbv, axis=-1)
    valid = int(nbm.sum())
    grp = float(np.median(nb_speed[nbm])) if valid else -1.0
    return "\n".join(
        f"t={k} dx={raw[k,0]:.0f}m dy={raw[k,1]:.0f}m sog={raw[k,4]:.1f}kn "
        f"cog={(np.degrees(np.arctan2(raw[k,5], raw[k,6]))+360)%360:.0f}deg "
        f"field_residual={rf[k]:.2f} gate_w={w[k]:.2f} n_neigh={valid} group_sog={grp:.1f}kn"
        for k in range(len(raw)))


INSTR_PLAIN = ("You are a maritime traffic analyst. Each line is one AIS report of a target vessel "
               "(70 s apart). Decide whether its behaviour is anomalous. Answer Yes or No.\n\n")
INSTR_RICH = ("You are a maritime traffic analyst. Each line is one AIS report of a target vessel (70 s apart). "
              "field_residual = deviation from the local traffic-flow field; gate_w = kinematic credibility; "
              "n_neigh = neighbours used; group_sog = median neighbour speed. Decide whether the target's "
              "behaviour is anomalous. Answer Yes or No.\n\n")


def run(model_dir: Path, z, idx, labels, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval().cuda()
    yes_ids = tok.encode(" Yes", add_special_tokens=False) + tok.encode("Yes", add_special_tokens=False)
    no_ids = tok.encode(" No", add_special_tokens=False) + tok.encode("No", add_special_tokens=False)
    res = {}
    for setting, rich in (("L0", False), ("L1", True)):
        scores = []
        t0 = time.time()
        for i in idx:
            content = (INSTR_RICH if rich else INSTR_PLAIN) + body(
                z["raw"][i], z["r_field"][i], z["w_self"][i], z["nb_mask"][i], z["nb_v"][i], rich)
            msgs = [{"role": "user", "content": content}]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(text, truncation=True, max_length=1024, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inp)
            lg = out.logits[0, -1]
            scores.append(max(float(lg[j]) for j in yes_ids) - max(float(lg[j]) for j in no_ids))
        res[setting] = round(ap_score(np.array(scores), labels), 3)
        res[setting + "_sec"] = round(time.time() - t0, 1)
        print(f"  {setting} AP={res[setting]:.3f} ({res[setting+'_sec']:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main() -> None:
    z = np.load(BASE / "eval_v2_mmsi" / "test.npz")
    idx = np.load(BASE / "L3_subset_idx.npy")
    labels = (z["label_class"][idx] > 0).astype(int)
    out_path = BASE / "llm_multi_benchmark.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    dev = "cuda"
    for name in CANDIDATES:
        d = MODELS_DIR / name
        if not d.exists() or not any(d.glob("*.safetensors")):
            print(f"skip {name} (not downloaded)", flush=True)
            continue
        if name in done and "L1" in done[name]:
            print(f"already done {name}", flush=True)
            continue
        print(f"### {name} ###", flush=True)
        try:
            done[name] = run(d, z, idx, labels, dev)
        except Exception as e:
            done[name] = {"error": str(e)[:200]}
            print("  failed:", str(e)[:160], flush=True)
        out_path.write_text(json.dumps(done, indent=2))
    print("ALL DONE")


if __name__ == "__main__":
    main()
