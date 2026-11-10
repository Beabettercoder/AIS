#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 56: Fig. 2 (baselines + ours), Fig. 4 (a,b,d), Fig. 5 (binary t-SNE)."""
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
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)

TARGETS = [("ny", "NY", "Unseen Ports"), ("la", "LA", "Unseen Ports"),
           ("puget", "Juan de Fuca", "Unseen Straits"), ("miami", "Florida", "Unseen Straits"),
           ("gom_open", "Outer GoM", "Unseen Open Seas"), ("atl_open", "NW Atlantic", "Unseen Open Seas")]


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def load_model(name, dev):
    ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
    m = sw.Net(ck["kind"], ck["nf"], use_graph=ck["use_graph"]).to(dev)
    m.load_state_dict(ck["sd"]); m.eval()
    return m, ck["stats"], ck["cfg"], ck["use_graph"]


def fwd(model, d, cfg, stats, graph, dev, k=None, tau=1.0):
    mu, sd = stats
    X = (sw.build(d, cfg["use_kg"], cfg["use_lf"]) - mu) / sd
    mask = d["nb_mask"].copy()
    if k is not None:
        mask[:, k:] = False
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=dev)
        if graph:
            if tau != 1.0:
                model.alpha.data /= tau; model.beta.data /= tau
            pb, _, _ = model(xb, torch.tensor(d["nb_seq"], device=dev), torch.tensor(d["nb_xy"], device=dev),
                             torch.tensor(mask, device=dev), torch.tensor(d["nb_w"], device=dev),
                             torch.tensor(d["rf"][:, -1], device=dev))
            if tau != 1.0:
                model.alpha.data *= tau; model.beta.data *= tau
        else:
            pb, _, _ = model(xb)
        return torch.sigmoid(pb).cpu().numpy()


def fig2(dev):
    rows = json.loads((BASE / "table2_strict_zeroshot.json").read_text())
    ours_s2 = [r["S2_full"] for r in rows]
    ours_s2p = [r["S2p_full"] for r in rows]
    base_names = [("Raw-TCN", "ablation_0_raw_tcn"), ("GRU", "base_gru"),
                  ("Transformer", "base_transformer"), ("AISClean", "base_aisclean"), ("STGVAD", "base_stgv")]
    curves = {}
    for label, f in base_names:
        m, stats, cfg, graph = load_model(f, dev)
        vals = []
        for t, _, _ in TARGETS:
            d = sw.load(BASE / f"{t}_dataset", "test")
            vals.append(ap_score(fwd(m, d, cfg, stats, graph, dev), d["y"] > 0))
        curves[label] = vals

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(TARGETS))
    colors = {"Unseen Ports": "#dbe9f6", "Unseen Straits": "#fde2e2", "Unseen Open Seas": "#e2f2e2"}
    pos = 0
    for scen in ["Unseen Ports", "Unseen Straits", "Unseen Open Seas"]:
        idx = [i for i, t in enumerate(TARGETS) if t[2] == scen]
        ax.add_patch(Rectangle((min(idx) - 0.5, 0.45), len(idx), 0.55, color=colors[scen], zorder=0))
        ax.text(np.mean(idx), 0.985, scen, ha="center", va="top", fontsize=8.5, fontweight="bold")
    styles = {"Raw-TCN": ("#7f7f7f", "o"), "GRU": ("#1f77b4", "s"), "Transformer": ("#2ca02c", "^"),
              "AISClean": ("#9467bd", "D"), "STGVAD": ("#8c564b", "v")}
    for label, vals in curves.items():
        c, mk = styles[label]
        ax.plot(x, vals, "--", marker=mk, ms=4.5, color=c, lw=1.1, mfc="white", label=f"{label} (S2)")
    ax.plot(x, ours_s2, "--*", ms=12, color="#D62728", lw=1.4, mfc="white", label="PG-DRGN (S2, zero-shot)")
    ax.plot(x, ours_s2p, "-*", ms=14, color="#D62728", lw=1.8, label="PG-DRGN (S2+, ours)")
    ax.set_xticks(x); ax.set_xticklabels([t[1] for t in TARGETS], fontsize=8)
    ax.set_ylabel("Full AP", fontsize=10); ax.set_ylim(0.55, 1.0)
    ax.set_title("Strict zero-shot transfer: ports → unseen ports, straits and open seas", fontsize=10.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(BASE / "fig2_strict_zeroshot_profile.png", dpi=300, bbox_inches="tight")
    print("fig2 done", {k: round(float(np.mean(v)), 3) for k, v in curves.items()})


def fig4(dev):
    te = sw.load(BASE / "eval_v2_graph2", "test")
    m, stats, cfg, graph = load_model("ablation_4_full_cdg", dev)
    taus = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    acc = [ap_score(fwd(m, te, cfg, stats, graph, dev, tau=t), te["y"] > 0) for t in taus]
    ks = [4, 8, 12, 16, 24, 32]
    acc_k, lat = [], []
    for k in ks:
        acc_k.append(ap_score(fwd(m, te, cfg, stats, graph, dev, k=k), te["y"] > 0))
        x = torch.randn(1000, 20, 9, device=dev)
        ns = torch.randn(1000, k, 5, 4, device=dev); nxy = torch.randn(1000, k, 2, device=dev)
        nm = torch.ones(1000, k, dtype=torch.bool, device=dev); nw = torch.ones(1000, k, device=dev)
        ri = torch.rand(1000, device=dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            m(x, ns, nxy, nm, nw, ri)
        lat.append((time.perf_counter() - t0) * 1000)

    fig = plt.figure(figsize=(11, 7.6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(range(len(taus)), acc, color="#4C72B0", edgecolor="k", hatch="///", linewidth=0.5)
    ax1.set_xticks(range(len(taus))); ax1.set_xticklabels(taus)
    ax1.set_xlabel(r"Temperature $\tau$", fontsize=9); ax1.set_ylabel("Full AP", fontsize=9)
    ax1.set_title("(a) Dual-distance temperature", fontsize=9.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(range(len(ks)), acc_k, color="#DD8452", edgecolor="k", hatch="\\\\\\", linewidth=0.5)
    ax2.set_xticks(range(len(ks))); ax2.set_xticklabels(ks)
    ax2.set_xlabel("Top-$K$ neighbours", fontsize=9); ax2.set_ylabel("Full AP", fontsize=9)
    ax2.set_title("(b) Neighbour budget vs latency", fontsize=9.5)
    ax2b = ax2.twinx()
    ax2b.plot(range(len(ks)), lat, "o--", color="k", ms=4, lw=1.0)
    ax2b.set_ylabel("Latency per 1000 vessels (ms)", fontsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis("off")
    ax3.text(0.02, 0.55, "(c) Consistency weight $\\lambda_{con}$ and budget $\\epsilon_{max}$\n"
                         "are training-time hyper-parameters;\n"
                         "they were fixed at 0.20 / 0.20 for all reported models\n"
                         "and are therefore not swept at inference.",
             fontsize=8.5, va="center")

    ax4 = fig.add_subplot(2, 2, 4)
    DATA = [("Transformer", 0.607, 0.014, 68358), ("PG-DRGN (Ours)", 0.646, 0.008, 69000),
            ("AISClean", 0.614, 0.020, 68358), ("Ours (GRU)", 0.620, 0.016, 52000),
            ("Interval-GNN", 0.622, 0.020, 68000), ("GRU", 0.594, 0.047, 30342),
            ("PhyLoss", 0.634, 0.023, 44806), ("STGVAD", 0.611, 0.021, 68000),
            ("ST-GAT", 0.625, 0.024, 68000), ("TCN", 0.622, 0.019, 44806),
            ("Mamba", 0.616, 0.021, 23367), ("CDE", 0.500, 0.043, 15878), ("iTrans.", 0.461, 0.023, 69062)]
    for name, mu, sd, p in DATA:
        ours = name.startswith("PG")
        ax4.scatter(sd, mu, s=60 + p / 200, color="#D62728" if ours else "#4C72B0",
                    marker="*" if ours else "o", edgecolor="k", linewidth=0.5, alpha=0.9)
        ax4.annotate(name, (sd, mu), textcoords="offset points", xytext=(5, 4), fontsize=6)
    ax4.set_xlabel(r"$\sigma$ of Recall@5%", fontsize=9); ax4.set_ylabel("Mean Recall@5%", fontsize=9)
    ax4.set_title("(d) Accuracy–stability Pareto (13 models)", fontsize=9.5)
    ax4.grid(linestyle=":", alpha=0.4)
    fig.suptitle("Hyper-parameter sensitivity and model Pareto frontier", fontsize=11)
    fig.tight_layout()
    fig.savefig(BASE / "fig4_hyperparam_pareto.png", dpi=300, bbox_inches="tight")
    print("fig4 done", [round(a, 3) for a in acc], [round(a, 3) for a in acc_k], [round(l, 1) for l in lat])


def fig5(dev):
    te = sw.load(BASE / "eval_v2_graph2", "test")
    ny = sw.load(BASE / "ny_dataset", "test")
    mB, sB, cB, gB = load_model("ablation_0_raw_tcn", dev)
    mO, sO, cO, gO = load_model("ablation_4_full_cdg", dev)

    def feats(m, d, c, s, g):
        mu, sd = s
        X = (sw.build(d, c["use_kg"], c["use_lf"]) - mu) / sd
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=dev)
            if g:
                _, _, h = m(xb, torch.tensor(d["nb_seq"], device=dev), torch.tensor(d["nb_xy"], device=dev),
                            torch.tensor(d["nb_mask"], device=dev), torch.tensor(d["nb_w"], device=dev),
                            torch.tensor(d["rf"][:, -1], device=dev))
            else:
                _, _, h = m(xb)
        return h.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2))
    for col, (d, title) in enumerate([(te, "Source domain (degraded)"), (ny, "Houston→NY zero-shot")]):
        for row, (m, s, c, g, tag) in enumerate([(mB, sB, cB, gB, "Raw-TCN"), (mO, sO, cO, gO, "PG-DRGN")]):
            F = feats(m, d, c, s, g)
            idx = np.random.default_rng(0).choice(len(F), size=min(600, len(F)), replace=False)
            Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(F[idx])
            y = (d["y"][idx] > 0).astype(int)
            ax = axes[row, col]
            ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=7, color="#1f77b4", alpha=0.7, label="normal" if (row + col == 0) else None)
            ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=9, color="#D62728", marker="*", alpha=0.85,
                       label="anomaly" if (row + col == 0) else None)
            sil = silhouette_score(F[idx], y)
            ax.set_title(f"{tag} — {title}\nSilhouette = {sil:.3f}", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("t-SNE of penultimate features (normal vs anomalous)", fontsize=11)
    fig.tight_layout()
    fig.savefig(BASE / "fig5_tsne_clusters.png", dpi=300, bbox_inches="tight")
    print("fig5 done")


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig2(dev)
    fig4(dev)
    fig5(dev)
