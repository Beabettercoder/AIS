#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 91: LoRA v2 — 只对答案 token 计损失 + 动态 padding，收敛版。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

MODEL = "/mnt/sda/byc/Qwen3-8B"
BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
STEPS = 300
BS = 4


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def body(z, i):
    raw, rf, w, m, nv = z["raw"][i], z["r_field"][i], z["w_self"][i], z["nb_mask"][i], z["nb_v"][i]
    nb_speed = np.linalg.norm(nv, axis=-1)
    valid = int(m.sum())
    grp = float(np.median(nb_speed[m])) if valid else -1.0
    return "\n".join(
        f"t={k} dx={raw[k,0]:.0f}m dy={raw[k,1]:.0f}m sog={raw[k,4]:.1f}kn "
        f"cog={(np.degrees(np.arctan2(raw[k,5], raw[k,6]))+360)%360:.0f}deg "
        f"field_residual={rf[k]:.2f} gate_w={w[k]:.2f} n_neigh={valid} group_sog={grp:.1f}kn"
        for k in range(len(raw)))


INSTR = ("You are a maritime traffic analyst. Each line is one AIS report of a target vessel (70 s apart). "
         "field_residual = deviation from the local traffic-flow field; gate_w = kinematic credibility "
         "(1 = trustworthy); n_neigh = neighbours used; group_sog = median neighbour speed. "
         "Decide whether the target's behaviour is anomalous. Answer Yes or No.\n\n")


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True).cuda()
    model.gradient_checkpointing_enable()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    ztr = np.load(BASE / "eval_v2_graph2" / "train.npz")
    zte = np.load(BASE / "eval_v2_mmsi" / "test.npz")
    idx_sub = np.load(BASE / "L3_subset_idx.npy")
    order = np.random.default_rng(0).permutation(len(ztr["raw"]))

    def make(i, label):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": INSTR + body(ztr, i)}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok("Yes" if label else "No", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = p_ids + a_ids
        lab = [-100] * len(p_ids) + a_ids
        return ids, lab

    t0 = time.time()
    model.train()
    step = 0
    while step < STEPS:
        batch = [int(i) for i in order[(step * BS):(step * BS + BS)]]
        seqs = [make(i, int(ztr["label_class"][i] > 0)) for i in batch]
        ml = max(len(s[0]) for s in seqs)
        ids = torch.full((len(seqs), ml), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(seqs), ml), -100, dtype=torch.long)
        for r, (a, b) in enumerate(seqs):
            ids[r, :len(a)] = torch.tensor(a); lab[r, :len(b)] = torch.tensor(b)
        out = model(input_ids=ids.cuda(), labels=lab.cuda())
        (out.loss).backward()
        opt.step(); opt.zero_grad()
        if step % 25 == 0:
            print(f"step {step} loss {out.loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        step += 1

    model.eval()
    yes_ids = tok.encode(" Yes", add_special_tokens=False) + tok.encode("Yes", add_special_tokens=False)
    no_ids = tok.encode(" No", add_special_tokens=False) + tok.encode("No", add_special_tokens=False)
    labels, scores, ans = [], [], []
    for k in idx_sub:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": INSTR + body(zte, int(k))}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model(**inp)
        lg = o.logits[0, -1]
        py = max(float(lg[j]) for j in yes_ids); pn = max(float(lg[j]) for j in no_ids)
        scores.append(py - pn); ans.append(int(py > pn))
        labels.append(int(zte["label_class"][k] > 0))
    res = {"setting": "L2/L3 LoRA v2 (answer-only loss)", "steps": STEPS, "train_windows_seen": STEPS * BS,
           "AP": round(ap_score(np.array(scores), np.array(labels)), 3),
           "yes_rate": float(np.mean(ans)), "pos_rate": float(np.mean(labels)),
           "train_seconds": round(time.time() - t0, 1)}
    print(res)
    (BASE / "qwen_lora_v2_result.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
