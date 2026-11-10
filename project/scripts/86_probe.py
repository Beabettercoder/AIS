#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 86: L2 同监督（可行版）——冻结 LLM 表征 + 线性探针，用同一批标签训练。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

MODEL = "/mnt/sda/byc/Qwen3-8B"
BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def body(raw, rf, w, nb_mask, nb_v):
    nb_speed = np.linalg.norm(nb_v, axis=-1)
    valid = int(nb_mask.sum())
    grp = float(np.median(nb_speed[nb_mask])) if valid else -1.0
    return "\n".join(
        f"t={k} dx={raw[k,0]:.0f}m dy={raw[k,1]:.0f}m sog={raw[k,4]:.1f}kn "
        f"cog={(np.degrees(np.arctan2(raw[k,5], raw[k,6]))+360)%360:.0f}deg "
        f"field_residual={rf[k]:.2f} gate_w={w[k]:.2f} n_neigh={valid} group_sog={grp:.1f}kn"
        for k in range(len(raw)))


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval().cuda()
    dev = next(model.parameters()).device

    def embed(z, idx):
        feats = []
        for i in idx:
            content = "AIS sequence of a vessel (70 s apart).\n" + body(
                z["raw"][i], z["r_field"][i], z["w_self"][i], z["nb_mask"][i], z["nb_v"][i])
            inp = tok(content, truncation=True, max_length=384, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            feats.append(out.hidden_states[-1][0, -1].float().cpu().numpy())
        return np.asarray(feats)

    ztr = np.load(BASE / "eval_v2_graph2" / "train.npz")
    zte = np.load(BASE / "eval_v2_graph2" / "test.npz")
    rng = np.random.default_rng(0)
    itr = rng.permutation(len(ztr["raw"]))[:2000]
    ite = rng.permutation(len(zte["raw"]))[:200]
    t0 = time.time()
    Xtr = embed(ztr, itr); ytr = (ztr["label_class"][itr] > 0).astype(int)
    Xte = embed(zte, ite); yte = (zte["label_class"][ite] > 0).astype(int)
    print("encoded", Xtr.shape, Xte.shape, f"{time.time()-t0:.0f}s", flush=True)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(Xtr, ytr)
    p = clf.decision_function(Xte)
    res = {"setting": "L2 frozen Qwen3-8B embeddings + logistic probe (same labels)",
           "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "AP": round(ap_score(p, yte), 3), "pos_rate": float(yte.mean()),
           "elapsed_s": round(time.time() - t0, 1)}
    print(res)
    (BASE / "qwen_probe_L2_result.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
