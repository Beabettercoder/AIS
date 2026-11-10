#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 57: 低误报工作点扫描（FAR 0.1%–5%）+ 参数/延迟，检验方法真正的优势区间。"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)

FARS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
MODELS = [("Baseline TCN", "ablation_0_raw_tcn", "#7f7f7f", "o"),
          ("+KG", "ablation_1_kg", "#1f77b4", "s"),
          ("+KG+LF", "ablation_2_kglf", "#2ca02c", "^"),
          ("+KG+LF+PSA", "ablation_3_psa", "#9467bd", "D"),
          ("PG-DRGN", "ablation_4_full_cdg", "#D62728", "*"),
          ("GRU", "base_gru", "#8c564b", "v"),
          ("Transformer", "base_transformer", "#e377c2", "P")]


def load_model(name, dev):
    ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
    m = sw.Net(ck["kind"], ck["nf"], use_graph=ck["use_graph"]).to(dev)
    m.load_state_dict(ck["sd"]); m.eval()
    return m, ck["stats"], ck["cfg"], ck["use_graph"]


def fwd(model, d, cfg, stats, graph, dev):
    mu, sd = stats
    X = (sw.build(d, cfg["use_kg"], cfg["use_lf"]) - mu) / sd
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=dev)
        if graph:
            pb, _, _ = model(xb, torch.tensor(d["nb_seq"], device=dev), torch.tensor(d["nb_xy"], device=dev),
                             torch.tensor(d["nb_mask"], device=dev), torch.tensor(d["nb_w"], device=dev),
                             torch.tensor(d["rf"][:, -1], device=dev))
        else:
            pb, _, _ = model(xb)
        return torch.sigmoid(pb).cpu().numpy()


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    cell, y = te["cell"], te["y"]
    clean = np.isin(cell, [0, 2])
    yb = (y > 0).astype(int)
    rows = {}
    for label, f, c, mk in MODELS:
        m, stats, cfg, graph = load_model(f, dev)
        p = fwd(m, te, cfg, stats, graph, dev)
        neg, pos = p[cell == 0], p[cell == 2]
        curve = []
        for far in FARS:
            thr = np.quantile(neg, 1 - far / 100)
            curve.append(float((pos > thr).mean()))
        n_par = sum(q.numel() for q in m.parameters())
        rows[label] = {"curve": curve, "params": int(n_par),
                       "mean_R": float(np.mean(curve)),
                       "auc_lowfar": float(np.mean(curve[:3]))}
        print(label, [round(v, 3) for v in curve], n_par, flush=True)

    (BASE / "far_sweep.json").write_text(json.dumps({"fars": FARS, "rows": rows}, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for label, f, c, mk in MODELS:
        ax.plot(FARS, rows[label]["curve"], "-" + mk, color=c, ms=6 if mk != "*" else 12,
                lw=1.3 if mk != "*" else 2.0, label=f"{label}  (mean {rows[label]['mean_R']:.3f})")
    ax.set_xscale("log")
    ax.set_xticks(FARS); ax.set_xticklabels([f"{f:g}%" for f in FARS])
    ax.set_xlabel("False alarm rate budget", fontsize=10)
    ax.set_ylabel("Recall (detection rate)", fontsize=10)
    ax.set_title("Operating-point profile at strict false-alarm budgets", fontsize=10.5)
    ax.grid(alpha=0.35, linestyle=":")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(BASE / "fig7_far_profile.png", dpi=300, bbox_inches="tight")
    print("saved fig7")


if __name__ == "__main__":
    main()
