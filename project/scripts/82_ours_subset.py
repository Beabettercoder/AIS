#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 82: 在同一 200 窗子集上计算我们模型的 AP，保证与 LLM 完全同集可比。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(BASE / "eval_v2_graph2" / "test.npz")
    idx = np.random.default_rng(0).choice(len(z["raw"]), size=200, replace=False)
    labels = (z["label_class"][idx] > 0).astype(int)
    te = sw.load(BASE / "eval_v2_graph2", "test")
    ck = torch.load(W / "ablation_4_full_cdg.pt", map_location="cpu", weights_only=False)
    m = sw.Net(ck["kind"], ck["nf"], use_graph=True).to(dev)
    m.load_state_dict(ck["sd"]); m.eval()
    mu, sd = ck["stats"]
    X = (np.concatenate([te["gated"], te["rf"][..., None], te["w"][..., None]], -1) - mu) / sd
    sub = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[0] == len(X) else v) for k, v in te.items()}
    with torch.no_grad():
        pb, _, _ = m(torch.tensor(X[idx], dtype=torch.float32, device=dev),
                     torch.tensor(sub["nb_seq"], device=dev), torch.tensor(sub["nb_xy"], device=dev),
                     torch.tensor(sub["nb_mask"], device=dev), torch.tensor(sub["nb_w"], device=dev),
                     torch.tensor(sub["rf"][:, -1], device=dev))
        p = torch.sigmoid(pb).cpu().numpy()
    out = {"n": 200, "AP": round(ap_score(p, labels), 3), "pos_rate": float(labels.mean())}
    print(out)
    (BASE / "ours_on_llm_subset.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
