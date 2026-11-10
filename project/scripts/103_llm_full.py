#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 103: 每个模型的完整协议 —— L0 / L1 / 3-shot / LoRA(L2)。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
MODELS_DIR = Path("/mnt/sda/WangZijia/models")
OUT = BASE / "llm_full_protocol.json"
N_STEPS = 300          # LoRA steps
BS = 4


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def body(z, i, rich):
    raw = z["raw"][i]
    if not rich:
        return "\n".join(f"t={k} dx={raw[k,0]:.0f}m dy={raw[k,1]:.0f}m sog={raw[k,4]:.1f}kn "
                         f"cog={(np.degrees(np.arctan2(raw[k,5], raw[k,6]))+360)%360:.0f}deg" for k in range(len(raw)))
    rf, w, m, nv = z["r_field"][i], z["w_self"][i], z["nb_mask"][i], z["nb_v"][i]
    sp = np.linalg.norm(nv, axis=-1)
    valid = int(m.sum())
    grp = float(np.median(sp[m])) if valid else -1.0
    return "\n".join(
        f"t={k} dx={raw[k,0]:.0f}m dy={raw[k,1]:.0f}m sog={raw[k,4]:.1f}kn "
        f"cog={(np.degrees(np.arctan2(raw[k,5], raw[k,6]))+360)%360:.0f}deg "
        f"field_residual={rf[k]:.2f} gate_w={w[k]:.2f} n_neigh={valid} group_sog={grp:.1f}kn"
        for k in range(len(raw)))


INSTR_P = ("You are a maritime traffic analyst. Each line is one AIS report of one vessel (70 s apart). "
           "Decide whether the behaviour is anomalous. Answer Yes or No.\n\n")
INSTR_R = ("You are a maritime traffic analyst. Each line is one AIS report of one vessel (70 s apart). "
           "field_residual = deviation from the local traffic-flow field; gate_w = kinematic credibility; "
           "n_neigh = neighbours used; group_sog = median neighbour speed. "
           "Decide whether the behaviour is anomalous. Answer Yes or No.\n\n")


def chat(tok, content, prompt=True):
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=prompt, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=prompt)


def score_yesno(model, tok, text, dev, maxlen=1024):
    yes_ids = tok.encode(" Yes", add_special_tokens=False) + tok.encode("Yes", add_special_tokens=False)
    no_ids = tok.encode(" No", add_special_tokens=False) + tok.encode("No", add_special_tokens=False)
    inp = tok(text, truncation=True, max_length=maxlen, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inp)
    lg = out.logits[0, -1]
    return max(float(lg[j]) for j in yes_ids) - max(float(lg[j]) for j in no_ids)


def run_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = MODELS_DIR / name
    tok = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.bfloat16, trust_remote_code=True).cuda()
    dev = "cuda"
    z = np.load(BASE / "eval_v2_mmsi" / "test.npz")
    idx = np.load(BASE / "L3_subset_idx.npy")
    labels = (z["label_class"][idx] > 0).astype(int)
    res = {}

    # L0 / L1 / 3-shot
    for tag, rich, few in (("L0", False, False), ("L1", True, False), ("3shot", True, True)):
        ex = ""
        if few:
            ztr = np.load(BASE / "eval_v2_graph2" / "train.npz")
            pos = int(np.flatnonzero(ztr["label_class"] > 0)[0]); neg = int(np.flatnonzero(ztr["label_class"] == 0)[0])
            ex = ("Example 1 (normal):\n" + body(ztr, neg, True) + "\nAnswer: No\n\n"
                  "Example 2 (anomalous):\n" + body(ztr, pos, True) + "\nAnswer: Yes\n\n")
        t0 = time.time()
        sc = [score_yesno(model, tok, chat(tok, (INSTR_R if rich else INSTR_P) + ex + body(z, int(i), rich)), dev) for i in idx]
        res[tag] = round(ap_score(np.array(sc), labels), 3)
        res[tag + "_sec"] = round(time.time() - t0, 1)
        print(f"  {tag}: {res[tag]:.3f} ({res[tag+'_sec']:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()

    # LoRA (L2)
    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.bfloat16, trust_remote_code=True).cuda()
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    ztr = np.load(BASE / "eval_v2_graph2" / "train.npz")
    order = np.random.default_rng(0).permutation(len(ztr["raw"]))
    t0 = time.time()
    model.train()
    for step in range(N_STEPS):
        batch = [int(i) for i in order[step * BS:(step + 1) * BS]]
        seqs = []
        for i in batch:
            prompt = chat(tok, INSTR_R + body(ztr, i, True))
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            ans = "Yes" if ztr["label_class"][i] > 0 else "No"
            a_ids = tok(ans, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            seqs.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))
        ml = max(len(s[0]) for s in seqs)
        ids = torch.full((len(seqs), ml), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(seqs), ml), -100, dtype=torch.long)
        for r, (a, b) in enumerate(seqs):
            ids[r, :len(a)] = torch.tensor(a); lab[r, :len(b)] = torch.tensor(b)
        out = model(input_ids=ids.cuda(), labels=lab.cuda())
        out.loss.backward(); opt.step(); opt.zero_grad()
        if step % 100 == 0:
            print(f"  lora step {step} loss {out.loss.item():.3f}", flush=True)
    model.eval()
    sc = [score_yesno(model, tok, chat(tok, INSTR_R + body(z, int(i), True)), dev) for i in idx]
    res["L2_lora"] = round(ap_score(np.array(sc), labels), 3)
    res["L2_sec"] = round(time.time() - t0, 1)
    print(f"  L2 LoRA: {res['L2_lora']:.3f} ({res['L2_sec']:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main() -> None:
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "DeepSeek-R1-Distill-Qwen-1.5B",
                 "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
                 "gemma-2-9b-it", "Mistral-7B-Instruct-v0.2", "Qwen2.5-14B-Instruct"]:
        d = MODELS_DIR / name
        if not d.exists() or not (any(d.glob("*.safetensors")) or any(d.glob("*.bin"))):
            print("skip", name, flush=True); continue
        if name in done and "L2_lora" in done[name]:
            print("done", name, flush=True); continue
        print("###", name, flush=True)
        try:
            done[name] = run_model(name)
        except Exception as e:
            done[name] = {"error": str(e)[:300]}
            print("  FAILED", str(e)[:200], flush=True)
        OUT.write_text(json.dumps(done, indent=2))
    print("FULL PROTOCOL DONE")


if __name__ == "__main__":
    main()
