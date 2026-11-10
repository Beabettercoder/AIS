#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 85: L2 同监督——Qwen3-8B LoRA 微调（同一 4500 训练窗、同一 200 测试窗）。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

MODEL = "/mnt/sda/byc/Qwen3-8B"
BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
MAXLEN = 512


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def body_text(raw, rf, w, nb_mask, nb_v):
    nb_speed = np.linalg.norm(nb_v, axis=-1)
    valid = int(nb_mask.sum())
    grp = float(np.median(nb_speed[nb_mask])) if valid else -1.0
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

    ztr = np.load(BASE / "eval_v2_graph2" / "train.npz")
    zte = np.load(BASE / "eval_v2_graph2" / "test.npz")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True).cuda()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def encode(raw, rf, w, nbm, nbv, label):
        text = tok.apply_chat_template(
            [{"role": "user", "content": INSTR + body_text(raw, rf, w, nbm, nbv)},
             {"role": "assistant", "content": "Yes" if label else "No"}],
            tokenize=False, add_generation_prompt=False)
        enc = tok(text, truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"][0]
        labels = ids.clone()
        labels[enc["attention_mask"][0] == 0] = -100
        return ids, labels, enc["attention_mask"][0]

    n_tr = 2000
    rng = np.random.default_rng(0)
    order = rng.permutation(len(ztr["raw"]))[:n_tr]
    t0 = time.time()
    model.train()
    for step, s in enumerate(range(0, n_tr, 2)):
        batch = order[s:s + 2]
        xs, ys, ms = [], [], []
        for i in batch:
            a, b, c = encode(ztr["raw"][i], ztr["r_field"][i], ztr["w_self"][i],
                             ztr["nb_mask"][i], ztr["nb_v"][i], int(ztr["label_class"][i] > 0))
            xs.append(a); ys.append(b); ms.append(c)
        ids = torch.stack(xs).cuda(); lab = torch.stack(ys).cuda(); am = torch.stack(ms).cuda()
        out = model(input_ids=ids, attention_mask=am, labels=lab)
        (out.loss / 1).backward()
        opt.step(); opt.zero_grad()
        if step % 20 == 0:
            print(f"step {step} loss {out.loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if step >= 120:
            break

    model.eval()
    idx = np.random.default_rng(0).choice(len(zte["raw"]), size=200, replace=False)
    labels = (zte["label_class"][idx] > 0).astype(int)
    yes_ids = tok.encode(" Yes", add_special_tokens=False) + tok.encode("Yes", add_special_tokens=False)
    no_ids = tok.encode(" No", add_special_tokens=False) + tok.encode("No", add_special_tokens=False)
    scores, ans = [], []
    for i in idx:
        content = INSTR + body_text(zte["raw"][i], zte["r_field"][i], zte["w_self"][i],
                                    zte["nb_mask"][i], zte["nb_v"][i])
        text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        inp = tok(text, truncation=True, max_length=MAXLEN, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model(**inp)
        lg = o.logits[0, -1]
        py = max(float(lg[j]) for j in yes_ids); pn = max(float(lg[j]) for j in no_ids)
        scores.append(py - pn); ans.append(int(py > pn))
    scores = np.array(scores)
    res = {"setting": "L2 LoRA fine-tuned Qwen3-8B", "train_windows": n_tr,
           "AP": round(ap_score(scores, labels), 3), "yes_rate": float(np.mean(ans)),
           "pos_rate": float(labels.mean()), "train_seconds": round(time.time() - t0, 1)}
    print(res)
    (BASE / "qwen_lora_L2_result.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
