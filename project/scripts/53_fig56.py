#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 53: Fig. 5 t-SNE (baseline vs ours) and Fig. 6 trajectory/saliency heatmaps."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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


def load_model(name, dev):
    ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
    m = sw.Net(ck["kind"], ck["nf"], use_graph=ck["use_graph"]).to(dev)
    m.load_state_dict(ck["sd"]); m.eval()
    return m, ck["stats"], ck["cfg"], ck["use_graph"]


def feats(model, d, cfg, stats, graph, dev):
    mu, sd = stats
    X = (sw.build(d, cfg["use_kg"], cfg["use_lf"]) - mu) / sd
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=dev)
        if graph:
            _, _, h = model(xb, torch.tensor(d["nb_seq"], device=dev), torch.tensor(d["nb_xy"], device=dev),
                            torch.tensor(d["nb_mask"], device=dev), torch.tensor(d["nb_w"], device=dev),
                            torch.tensor(d["rf"][:, -1], device=dev))
        else:
            _, _, h = model(xb)
    return h.cpu().numpy(), X


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    ny = sw.load(BASE / "ny_dataset", "test")
    mB, sB, cB, gB = load_model("ablation_0_raw_tcn", dev)
    mO, sO, cO, gO = load_model("ablation_4_full_cdg", dev)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4))
    cmap = plt.get_cmap("tab10")
    names = ["A normal", "B degraded", "C behaviour", "D both"]
    for col, (d, title) in enumerate([(te, "Source domain (degraded test)"), (ny, "Houston→NY zero-shot")]):
        for row, (m, s, c, g, tag) in enumerate([(mB, sB, cB, gB, "Raw-TCN"), (mO, sO, cO, gO, "PG-DRGN (Ours)")]):
            F, _ = feats(m, d, c, s, g, dev)
            idx = np.random.default_rng(0).choice(len(F), size=min(600, len(F)), replace=False)
            Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(F[idx])
            y = d["y"][idx]
            ax = axes[row, col]
            for k in range(5):
                msk = y == k
                if msk.sum():
                    ax.scatter(Z[msk, 0], Z[msk, 1], s=6, color=cmap(k), alpha=0.75,
                               label=f"class {k+1}" if (row == 0 and col == 0) else None)
            sil = silhouette_score(F[idx], y) if len(set(y)) > 1 else float("nan")
            ax.set_title(f"{tag} — {title}\nSilhouette = {sil:.3f}", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(fontsize=6.5, markerscale=1.6, loc="best")
    fig.suptitle("t-SNE of penultimate features: Raw-TCN vs PG-DRGN", fontsize=11)
    fig.tight_layout()
    fig.savefig(BASE / "fig5_tsne_clusters.png", dpi=300, bbox_inches="tight")
    print("saved fig5")

    # ---- Fig. 6 ----
    _z = np.load(BASE / "eval_v2_graph2" / "test.npz")
    if "r_vec" in _z.files:
        te["r_vec"] = _z["r_vec"].astype(np.float32)
    cell, y = te["cell"], te["y"]
    picks = []
    for want in [("D", 3), ("C", 1), ("C", 3), ("C", 2)]:
        c_, k_ = want
        idx = np.flatnonzero((cell == (3 if c_ == "D" else 2)) & (y == k_))
        if len(idx):
            picks.append(int(idx[0]))
    while len(picks) < 4:
        picks.append(int(np.flatnonzero(cell == 2)[0]))
    titles = ["Group slowdown / degraded", "GPS jump degradation", "Early turn-back", "Channel deviation"]

    fig, axes = plt.subplots(3, 4, figsize=(13, 7.6), gridspec_kw={"height_ratios": [1.25, 1, 1]})
    for j, si in enumerate(picks):
        d1 = {k: (v[si:si + 1] if isinstance(v, np.ndarray) and v.shape[0] == len(cell) else v) for k, v in te.items()}
        mB.train(); mO.train()
        mu, sd = sB
        Xb = (sw.build(d1, cB["use_kg"], cB["use_lf"]) - mu) / sd
        xb = torch.tensor(Xb, dtype=torch.float32, device=dev, requires_grad=True)
        pb, _, _ = mB(xb) if not gB else mB(xb, torch.tensor(d1["nb_seq"], device=dev), torch.tensor(d1["nb_xy"], device=dev),
                                          torch.tensor(d1["nb_mask"], device=dev), torch.tensor(d1["nb_w"], device=dev),
                                          torch.tensor(d1["rf"][:, -1], device=dev))
        pb.sum().backward()
        sal_b = xb.grad.abs().sum(-1)[0].detach().cpu().numpy()

        mu2, sd2 = sO
        Xo = (sw.build(d1, cO["use_kg"], cO["use_lf"]) - mu2) / sd2
        xo = torch.tensor(Xo, dtype=torch.float32, device=dev, requires_grad=True)
        pb2, _, _ = mO(xo, torch.tensor(d1["nb_seq"], device=dev), torch.tensor(d1["nb_xy"], device=dev),
                       torch.tensor(d1["nb_mask"], device=dev), torch.tensor(d1["nb_w"], device=dev),
                       torch.tensor(d1["rf"][:, -1], device=dev))
        pb2.sum().backward()
        sal_o = xo.grad.abs().sum(-1)[0].detach().cpu().numpy()

        raw = te["raw"][si]
        pos = np.cumsum(raw[:, 0:2], axis=0)
        ax = axes[0, j]
        ax.plot(pos[:, 0], pos[:, 1], "-o", ms=2, lw=1.2, color="#1f77b4")
        rv = np.zeros((len(pos), 2)) if "r_vec" not in te else te["r_vec"][si]
        ax.quiver(pos[::4, 0], pos[::4, 1], rv[::4, 0] * 30, rv[::4, 1] * 30, color="#d62728", width=0.004)
        ax.set_title(titles[j], fontsize=8.5)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)
        ax.set_xlabel("Δx (m)", fontsize=7); ax.set_ylabel("Δy (m)", fontsize=7)

        axes[1, j].imshow(sal_b[None, :], aspect="auto", cmap="Reds")
        axes[1, j].set_yticks([]); axes[1, j].set_title("Raw-TCN saliency", fontsize=7.5)
        axes[1, j].tick_params(labelsize=6)

        axes[2, j].imshow(sal_o[None, :], aspect="auto", cmap="Reds")
        ax2 = axes[2, j].twinx()
        ax2.plot(te["w"][si], color="k", lw=1.0, ls="--")
        ax2.set_ylim(0, 1.05); ax2.tick_params(labelsize=6)
        axes[2, j].set_yticks([])
        axes[2, j].set_title("PG-DRGN saliency + $w_t$", fontsize=7.5)
        axes[2, j].tick_params(labelsize=6)
        axes[2, j].set_xlabel("time step", fontsize=7)
    mB.eval(); mO.eval()
    fig.suptitle("Trajectory context, temporal saliency and kinematic gate", fontsize=11)
    fig.tight_layout()
    fig.savefig(BASE / "fig6_trajectory_saliency_heatmap.png", dpi=300, bbox_inches="tight")
    print("saved fig6")


if __name__ == "__main__":
    main()
