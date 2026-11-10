#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 58: Ours-Full with GRU backbone -> FAR profile（验证方法可与强骨干叠加）。"""
from __future__ import annotations

import importlib.util
import json
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


def load_model(name, dev):
    ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
    m = sw.Net(ck["kind"], ck["nf"], use_graph=ck["use_graph"]).to(dev)
    m.load_state_dict(ck["sd"]); m.eval()
    return m, ck["stats"], ck["cfg"], ck["use_graph"]


def forward(model, d, cfg, stats, graph, dev):
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
    tr = sw.load(BASE / "eval_v2_graph2", "train")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    p = W / "ablation_5_gru_full.pt"
    if p.exists():
        m, stats, cfg, graph = load_model("ablation_5_gru_full", dev)
    else:
        model, mu, sd, nf = sw.train("gru", tr, dev, use_kg=True, use_lf=True, psa=True, graph=True)
        torch.save({"sd": model.state_dict(), "stats": (mu, sd), "kind": "gru", "use_graph": True,
                    "nf": int(nf), "cfg": dict(use_kg=True, use_lf=True, psa=True, graph=True)}, p)
        m, stats, cfg, graph = load_model("ablation_5_gru_full", dev)
    prob = forward(m, te, cfg, stats, graph, dev)
    cell = te["cell"]
    neg, pos = prob[cell == 0], prob[cell == 2]
    curve = [float((pos > np.quantile(neg, 1 - f / 100)).mean()) for f in FARS]
    print("Ours-GRU curve", [round(v, 3) for v in curve], "mean", round(float(np.mean(curve)), 3))

    prev = json.loads((BASE / "far_sweep.json").read_text())
    rows = prev["rows"]
    rows["Ours (GRU backbone)"] = {"curve": curve, "params": int(sum(q.numel() for q in m.parameters())),
                                   "mean_R": float(np.mean(curve)), "auc_lowfar": float(np.mean(curve[:3]))}
    (BASE / "far_sweep.json").write_text(json.dumps({"fars": FARS, "rows": rows}, indent=2), encoding="utf-8")

    style = {"Baseline TCN": ("#7f7f7f", "o"), "+KG": ("#1f77b4", "s"), "+KG+LF": ("#2ca02c", "^"),
             "+KG+LF+PSA": ("#9467bd", "D"), "PG-DRGN": ("#D62728", "*"), "GRU": ("#8c564b", "v"),
             "Transformer": ("#e377c2", "P"), "Ours (GRU backbone)": ("#ff7f0e", "X")}
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for k, v in rows.items():
        c, mk = style.get(k, ("k", "."))
        ax.plot(FARS, v["curve"], "-" + mk, color=c, ms=6 if mk != "*" else 12,
                lw=1.7 if k in ("GRU", "Ours (GRU backbone)") else 1.1,
                label=f"{k}  (mean {v['mean_R']:.3f})")
    ax.set_xscale("log"); ax.set_xticks(FARS); ax.set_xticklabels([f"{f:g}%" for f in FARS])
    ax.set_xlabel("False alarm rate budget", fontsize=10)
    ax.set_ylabel("Recall (detection rate)", fontsize=10)
    ax.set_title("Strict-FAR operating profile: physics components stack on any backbone", fontsize=10.5)
    ax.grid(alpha=0.35, linestyle=":")
    ax.legend(fontsize=6.8, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(BASE / "fig7_far_profile.png", dpi=300, bbox_inches="tight")
    print("saved fig7")


if __name__ == "__main__":
    main()
