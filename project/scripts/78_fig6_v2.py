#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 78: Fig.6 重绘——原始 vs 门控轨迹 + 时序显著性。"""
from __future__ import annotations

import importlib.util
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


def pick(z):
    cell, y, w = z["label_cell"], z["label_class"], z["w_self"]
    minw, argmin = w.min(1), w.argmin(1)
    jump = np.flatnonzero((np.isin(cell, [1, 3])) & (minw < 0.25) & (argmin >= 6) & (argmin <= 14))
    slow = np.flatnonzero((cell == 2) & (y == 5))
    turn = np.flatnonzero((cell == 2) & (y == 1))
    dev = np.flatnonzero((cell == 2) & (y == 2))
    return [int(slow[0]), int(jump[0]) if len(jump) else int(slow[1]), int(turn[0]), int(dev[0])], jump


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(BASE / "eval_v2_graph2" / "test.npz")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    picks, jumps = pick(z)
    print("picks", picks, "n jump-like", len(jumps))

    def saliency(name, idx):
        ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
        m = sw.Net(ck["kind"], ck["nf"], use_graph=ck.get("use_graph", False)).to(dev)
        m.load_state_dict(ck["sd"]); m.train()
        mu, sd = ck["stats"]
        X = (sw.build(te, ck["cfg"]["use_kg"], ck["cfg"]["use_lf"])[idx:idx + 1] - mu) / sd
        xb = torch.tensor(X, dtype=torch.float32, device=dev, requires_grad=True)
        if ck.get("use_graph", False):
            sub = {k: (v[idx:idx + 1] if isinstance(v, np.ndarray) and len(v) == len(te["raw"]) else v)
                   for k, v in te.items()}
            pb, _, _ = m(xb, torch.tensor(sub["nb_seq"], device=dev), torch.tensor(sub["nb_xy"], device=dev),
                         torch.tensor(sub["nb_mask"], device=dev), torch.tensor(sub["nb_w"], device=dev),
                         torch.tensor(sub["rf"][:, -1], device=dev))
        else:
            pb, _, _ = m(xb)
        pb.sum().backward()
        m.eval()
        return xb.grad.abs().sum(-1)[0].detach().cpu().numpy()

    titles = ["Group slowdown", "GPS jump degradation", "Early turn-back", "Channel deviation"]
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.2), gridspec_kw={"height_ratios": [1.35, 1, 1]})
    for j, si in enumerate(picks):
        raw = te["raw"][si]
        gated = te["gated"][si]
        pos_raw = np.cumsum(raw[:, 0:2], axis=0)
        pos_g = np.cumsum(gated[:, 0:2], axis=0)
        ax = axes[0, j]
        ax.plot(pos_raw[:, 0], pos_raw[:, 1], "--", color="#D62728", lw=1.1, label="raw observation")
        ax.plot(pos_g[:, 0], pos_g[:, 1], "-", color="#1f77b4", lw=1.6, label="gate-corrected")
        if j == 1:
            ax.plot(pos_raw[:, 0], pos_raw[:, 1], "x", color="#D62728", ms=6, mew=1.4)
        ax.set_title(titles[j], fontsize=8.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6, loc="best")

        sb = saliency("ablation_0_raw_tcn", si)
        so = saliency("ablation_4_full_cdg", si)
        sb = sb / (sb.max() + 1e-9); so = so / (so.max() + 1e-9)
        axes[1, j].imshow(sb[None, :], aspect="auto", cmap="Reds", vmin=0, vmax=1)
        axes[1, j].set_yticks([]); axes[1, j].set_title("Raw-TCN saliency", fontsize=7.5)
        axes[1, j].tick_params(labelsize=6)
        axes[2, j].imshow(so[None, :], aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax2 = axes[2, j].twinx()
        ax2.plot(z["w_self"][si], "k--", lw=1.0)
        ax2.set_ylim(0, 1.05); ax2.tick_params(labelsize=5.5)
        axes[2, j].set_yticks([]); axes[2, j].set_title("PG-DRGN saliency + $w_t$", fontsize=7.5)
        axes[2, j].set_xlabel("time step", fontsize=7)
        axes[2, j].tick_params(labelsize=6)
    fig.suptitle("Observation gating, temporal saliency and kinematic credibility", fontsize=11)
    fig.tight_layout()
    fig.savefig(BASE / "fig6_trajectory_saliency_heatmap.png", dpi=300, bbox_inches="tight")
    print("saved fig6")


if __name__ == "__main__":
    main()
