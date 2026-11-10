#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 100: 12 区域零样本（4 港口 + 4 海峡 + 4 开放海域）× 7 模型。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)

TARGETS = [("ny", "New York", "Ports"), ("la", "Los Angeles", "Ports"),
           ("mississippi", "Mississippi", "Ports"), ("honolulu", "Honolulu", "Ports"),
           ("puget", "Juan de Fuca", "Straits"), ("miami", "Florida Str.", "Straits"),
           ("goldengate", "Golden Gate", "Straits"), ("chesapeake", "Chesapeake", "Straits"),
           ("gom_open", "Outer GoM", "Open Seas"), ("atl_open", "NW Atlantic", "Open Seas"),
           ("pac_open", "NE Pacific", "Open Seas"), ("cpacific", "C. Pacific", "Open Seas")]
MODELS = [("Raw-TCN", "ablation_0_raw_tcn", "#7f7f7f", "o", "--"),
          ("GRU", "base_gru", "#1f77b4", "s", "--"),
          ("Transformer", "base_transformer", "#2ca02c", "^", "--"),
          ("AISClean", "base_aisclean", "#9467bd", "D", "--"),
          ("STGVAD", "base_stgv", "#8c564b", "v", "--"),
          ("Ours (TCN)", "ablation_4_full_cdg", "#D62728", "*", "-"),
          ("Ours (GRU)", "ablation_5_gru_full", "#ff7f0e", "X", "-")]
COL = {"Ports": "#dbe9f6", "Straits": "#fde2e2", "Open Seas": "#e2f2e2"}


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def load_model(name, dev):
    ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
    m = sw.Net(ck["kind"], ck["nf"], use_graph=ck.get("use_graph", False)).to(dev)
    m.load_state_dict(ck["sd"]); m.eval()
    return m, ck["stats"], ck["cfg"], ck.get("use_graph", False)


def fwd(m, d, cfg, stats, graph, dev):
    mu, sd = stats
    X = (sw.build(d, cfg["use_kg"], cfg["use_lf"]) - mu) / sd
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=dev)
        if graph:
            pb, _, _ = m(xb, torch.tensor(d["nb_seq"], device=dev), torch.tensor(d["nb_xy"], device=dev),
                         torch.tensor(d["nb_mask"], device=dev), torch.tensor(d["nb_w"], device=dev),
                         torch.tensor(d["rf"][:, -1], device=dev))
        else:
            pb, _, _ = m(xb)
        return torch.sigmoid(pb).cpu().numpy()


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {}
    for t, _, _ in TARGETS:
        p = BASE / f"{t}_dataset"
        if (p / "test.npz").exists():
            data[t] = sw.load(p, "test")
    print("loaded", list(data), flush=True)
    res = {}
    for label, f, c, mk, ls in MODELS:
        m, stats, cfg, graph = load_model(f, dev)
        r1, ap = [], []
        for t, _, _ in TARGETS:
            if t not in data:
                r1.append(np.nan); ap.append(np.nan); continue
            d = data[t]
            p = fwd(m, d, cfg, stats, graph, dev)
            cell = d["cell"]
            thr = np.quantile(p[cell == 0], 0.99)
            r1.append(float((p[cell == 2] > thr).mean()))
            ap.append(ap_score(p, d["y"] > 0))
        res[label] = {"r1": r1, "ap": ap}
        print(f"{label:12s} R@1% {np.nanmean(r1):.3f}  AP {np.nanmean(ap):.3f}", flush=True)
    (BASE / "fig2_12regions_data.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.4), sharex=True)
    x = np.arange(len(TARGETS))
    for ax, key, ylabel, title, ylim in ((axes[0], "r1", "Recall at FAR = 1%", "(a) Strict false-alarm operating point", (0, 0.95)),
                                         (axes[1], "ap", "Full AP", "(b) Threshold-free ranking quality", (0.55, 1.0))):
        for scen in ["Ports", "Straits", "Open Seas"]:
            idx = [i for i, t in enumerate(TARGETS) if t[2] == scen]
            ax.add_patch(Rectangle((min(idx) - 0.5, 0), len(idx), 2, color=COL[scen], zorder=0))
            ax.text(np.mean(idx), 0.96, scen, ha="center", va="top", fontsize=8,
                    transform=ax.get_xaxis_transform(), fontweight="bold")
        for label, f, c, mk, ls in MODELS:
            ax.plot(x, res[label][key], ls, marker=mk, color=c, ms=10 if mk == "*" else 5,
                    lw=1.9 if "Ours" in label else 1.0, label=label if key == "r1" else None)
        ax.set_ylabel(ylabel, fontsize=9.5); ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim); ax.grid(axis="y", linestyle=":", alpha=0.4)
        if key == "r1":
            ax.legend(fontsize=6.8, ncol=4, loc="lower right", framealpha=0.95)
    axes[1].set_xticks(x); axes[1].set_xticklabels([t[1] for t in TARGETS], fontsize=7.5, rotation=12)
    axes[1].set_xlim(-0.6, len(TARGETS) - 0.4)
    fig.suptitle("Strict zero-shot transfer across 12 unseen maritime domains", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(BASE / "fig2_12regions_zeroshot.png", dpi=300, bbox_inches="tight")
    print("saved fig2_12regions")

    lines = ["# Strict zero-shot transfer across 12 unseen domains (4 ports / 4 straits / 4 open seas)", "",
             "| Region | Scenario | " + " | ".join(m[0] for m in MODELS) + " |",
             "|---|---|" + "---|" * len(MODELS)]
    for i, (t, name, scen) in enumerate(TARGETS):
        cells = []
        for label, *_ in MODELS:
            v = res[label]["ap"][i]
            cells.append("–" if not np.isfinite(v) else f"{v:.3f}")
        lines.append(f"| {name} | {scen} | " + " | ".join(cells) + " |")
    lines += ["", "| Mean | | " + " | ".join(f"{np.nanmean(res[m[0]]['ap']):.3f}" for m in MODELS) + " |", "",
              "Recall@FAR=1% means: " + ", ".join(f"{m[0]} {np.nanmean(res[m[0]]['r1']):.3f}" for m in MODELS)]
    txt = "\n".join(lines)
    (BASE / "table2_12regions_zeroshot.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
