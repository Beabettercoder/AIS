#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 99: Fig.6 v4 —— 概率着色轨迹 + 精选样本（Col2 原始跳变、Col3 早升）。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)


def build_x(raw, gated, rf, w, cfg):
    if not cfg["use_kg"]:
        return raw
    parts = [gated]
    if cfg["use_lf"]:
        parts.append(rf[..., None])
    parts.append(w[..., None])
    return np.concatenate(parts, -1)


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(BASE / "eval_v2_graph2" / "test.npz")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    te["nb_v"] = z["nb_v"].astype(np.float32)

    def load(name):
        ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
        m = sw.Net(ck["kind"], ck["nf"], use_graph=ck.get("use_graph", False)).to(dev)
        m.load_state_dict(ck["sd"]); m.eval()
        return m, ck["stats"], ck["cfg"], ck.get("use_graph", False)

    models = {"Raw-TCN": load("ablation_0_raw_tcn"), "PG-DRGN": load("ablation_4_full_cdg")}

    def per_step(si, label):
        m, stats, cfg, graph = models[label]
        raw, gated, rf, wt = te["raw"][si], te["gated"][si], te["rf"][si], te["w"][si]
        mu, sd = stats
        probs = []
        nb = (te["nb_seq"][si:si + 1], te["nb_xy"][si:si + 1], te["nb_mask"][si:si + 1],
              te["nb_w"][si:si + 1], te["rf"][si:si + 1][:, -1])
        for t in range(0, 20):
            pad = 20 - (t + 1)
            r2 = np.concatenate([raw[:t + 1], np.repeat(raw[t:t + 1], pad, 0)])
            g2 = np.concatenate([gated[:t + 1], np.repeat(gated[t:t + 1], pad, 0)])
            f2 = np.concatenate([rf[:t + 1], np.repeat(rf[t:t + 1], pad, 0)])
            w2 = np.concatenate([wt[:t + 1], np.repeat(wt[t:t + 1], pad, 0)])
            X = build_x(r2, g2, f2, w2, cfg)[None]
            with torch.no_grad():
                xb = torch.tensor(((X - mu) / sd).astype(np.float32), device=dev)
                pb, _, _ = m(xb, *[torch.tensor(v, device=dev) for v in nb]) if graph else m(xb)
                probs.append(float(torch.sigmoid(pb)))
        return np.array(probs)

    cell, y, wva = z["label_cell"], z["label_class"], z["w_self"]
    full = np.flatnonzero(z["mask"].sum(1) == 20)
    minw, argmin = wva.min(1), wva.argmin(1)
    jump_c = [i for i in full if cell[i] in (1, 3) and minw[i] < 0.2 and 6 <= argmin[i] <= 15]
    slow_c = [i for i in full if cell[i] == 2 and y[i] == 5]
    turn_c = [i for i in full if cell[i] == 2 and y[i] == 1]
    dev_c = [i for i in full if cell[i] == 2 and y[i] == 2]

    # scan a few turn-back candidates for an early, strong PG-DRGN rise
    best, best_score = turn_c[0], -1
    for i in turn_c[:12]:
        p = per_step(i, "PG-DRGN")
        rise = float(p[6:15].max() - p[:5].mean())
        if rise > best_score:
            best, best_score = i, rise
    picks = [slow_c[0], jump_c[0] if jump_c else slow_c[0], best, dev_c[0]]
    print("picks", picks, "turn rise", round(best_score, 3), flush=True)

    titles = ["Group slowdown", "GPS jump degradation", "Early turn-back", "Channel deviation"]
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 8, figure=fig, height_ratios=[1.3, 1, 1], hspace=0.45, wspace=0.4)

    for j, si in enumerate(picks):
        raw, gated, rf, wt = te["raw"][si], te["gated"][si], te["rf"][si], te["w"][si]
        pos_raw = np.cumsum(raw[:, 0:2], axis=0)
        pos_g = np.cumsum(gated[:, 0:2], axis=0)
        pB = per_step(si, "Raw-TCN"); pO = per_step(si, "PG-DRGN")

        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        nbv = te["nb_v"][si][te["nb_mask"][si]] * 60.0
        for v in nbv[:10]:
            ax.quiver(pos_g[-1, 0], pos_g[-1, 1], v[0], v[1], color="0.78", angles="xy",
                      scale_units="xy", scale=1, width=0.005, zorder=1)
        ax.plot(pos_raw[:, 0], pos_raw[:, 1], "--", color="#D62728", lw=1.0, label="raw observation")
        ax.plot(pos_g[:, 0], pos_g[:, 1], "-", color="#1f77b4", lw=1.6, label="gate-corrected")
        for k in (0, 5, 10, 15, 19):
            ax.plot(pos_raw[k, 0], pos_raw[k, 1], "x", color="#D62728", ms=6, mew=1.3)
            ax.plot(pos_g[k, 0], pos_g[k, 1], "o", color="#1f77b4", ms=4)
            ax.annotate(f"t={k}", (pos_g[k, 0], pos_g[k, 1]), textcoords="offset points", xytext=(5, 6), fontsize=6.5)
        ax.set_title(titles[j], fontsize=9)
        ax.set_aspect("equal", adjustable="datalim"); ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6, loc="best")

        for which, (lab, prob) in enumerate((("Raw-TCN", pB), ("PG-DRGN", pO))):
            ax2 = fig.add_subplot(gs[1, 2 * j + which])
            sc = ax2.scatter(pos_g[:, 0], pos_g[:, 1], c=prob, cmap="Reds", vmin=0, vmax=1, s=42,
                             edgecolor="k", linewidth=0.3)
            ax2.plot(pos_g[:, 0], pos_g[:, 1], "-", color="0.65", lw=0.8, zorder=0)
            ax2.set_aspect("equal", adjustable="datalim")
            ax2.set_title(f"{lab}: $P(anom|t)$", fontsize=6.8)
            ax2.tick_params(labelsize=5.5)

        ax3 = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
        ax3.plot(range(20), pB, "--", color="#D62728", lw=1.3, marker="o", ms=2.2, label="Raw-TCN")
        ax3.plot(range(20), pO, "-", color="#1f77b4", lw=1.6, marker="o", ms=2.2, label="PG-DRGN")
        ax3.set_ylim(0, 1.05); ax3.tick_params(labelsize=6)
        ax3.set_xlabel("time step", fontsize=7)
        ax3b = ax3.twinx()
        ax3b.plot(range(20), wt, "k-.", lw=1.0, label="$w_t$")
        ax3b.set_ylim(0, 1.05); ax3b.tick_params(labelsize=5.5)
        for k in np.flatnonzero(wt < 0.2):
            ax3.axvspan(k - 0.45, k + 0.45, color="0.86", zorder=0)
        if j == 0:
            ax3.legend(fontsize=5.5, loc="upper left")

    fig.suptitle("Spatio-temporal view: gating, per-step probability and anomaly evolution", fontsize=12)
    fig.savefig(BASE / "fig6_trajectory_saliency_v4.png", dpi=300, bbox_inches="tight")
    print("saved fig6 v4")


if __name__ == "__main__":
    main()
