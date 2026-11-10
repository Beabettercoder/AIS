#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 118: 核查零样本评测是否偷用了目标域统计量（源域 vs 目标域归一化对照）。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)

DOMAINS = ["ny", "goldengate", "gom_open", "nepacific", "cpacific"]
MODELS = [("Ours-TCN", "ablation_4_full_cdg"), ("Raw-TCN", "ablation_0_raw_tcn")]


def ap_score(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for label, f in MODELS:
        ck = torch.load(W / f"{f}.pt", map_location="cpu", weights_only=False)
        model = sw.Net(ck["kind"], ck["nf"], use_graph=ck.get("use_graph", False)).to(dev)
        model.load_state_dict(ck["sd"]); model.eval()
        mu_s, sd_s = ck["stats"]
        cfg = ck["cfg"]
        print(f"### {label}")
        for dname in DOMAINS:
            p = BASE / f"{dname}_dataset"
            if not (p / "test.npz").exists():
                print(f"  {dname}: no dataset"); continue
            d = sw.load(p, "test")
            X = sw.build(d, cfg["use_kg"], cfg["use_lf"])
            res = {}
            for tag, (mu, sd) in (("src", (mu_s, sd_s)),
                                  ("tgt", (X[d["mask"]].mean(0, keepdims=True),
                                           X[d["mask"]].std(0, keepdims=True) + 1e-6))):
                with torch.no_grad():
                    xb = torch.tensor(((X - mu) / sd).astype(np.float32), device=dev)
                    if cfg.get("use_graph"):
                        pb, _, _ = model(xb, torch.tensor(d["nb_seq"], device=dev), torch.tensor(d["nb_xy"], device=dev),
                                         torch.tensor(d["nb_mask"], device=dev), torch.tensor(d["nb_w"], device=dev),
                                         torch.tensor(d["rf"][:, -1], device=dev))
                    else:
                        pb, _, _ = model(xb)
                    pr = torch.sigmoid(pb).cpu().numpy()
                res[tag] = ap_score(pr, d["y"] > 0)
            print(f"  {dname:12s} src-norm {res['src']:.3f}   tgt-norm {res['tgt']:.3f}   Δ {res['tgt']-res['src']:+.3f}")


if __name__ == "__main__":
    main()
