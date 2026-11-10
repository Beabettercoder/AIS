#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 67: 群体异常强度扫描 + 出图（fig9_group_relative.png）。"""
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
sp = importlib.util.spec_from_file_location("gf", "/mnt/sda/WangZijia/AIS/scripts/60_graph_fix.py")
gf = importlib.util.module_from_spec(sp); sp.loader.exec_module(gf)


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    te["nb_v"] = np.load(BASE / "eval_v2_graph2" / "test.npz")["nb_v"].astype(np.float32)
    X = np.concatenate([te["gated"], te["rf"][..., None], te["w"][..., None]], -1)
    ck = torch.load(W / "group_anomaly_graph.pt", map_location="cpu", weights_only=False)
    mu, sd = ck["stats"]
    X = (X - mu) / sd
    m = gf.Net2("tcn", X.shape[-1]).to(dev); m.load_state_dict(ck["sd"]); m.eval()

    def score(d):
        with torch.no_grad():
            pb, _, _, _ = m(torch.tensor(X, dtype=torch.float32, device=dev),
                            *gf.tensors(d, np.arange(len(X)), dev))
            return torch.sigmoid(pb).cpu().numpy()

    base = score(te)
    factors = [0.95, 0.90, 0.85, 0.80, 0.70, 0.60]
    aps = []
    for f in factors:
        d = dict(te)
        d["nb_v"] = te["nb_v"] * f
        d["nb_seq"] = te["nb_seq"].copy()
        d["nb_seq"][:, :, :, 2:4] *= f
        s = np.r_[base, score(d)]
        y = np.r_[np.zeros(len(base)), np.ones(len(base))]
        aps.append(ap_score(s, y))
        print(f"group factor {f}: AP={aps[-1]:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot([1 - f for f in factors], aps, "-o", color="#D62728", ms=7, lw=1.6, label="PG-DRGN (graph)")
    ax.axhline(0.5, color="0.4", ls="--", lw=1.0, label="single-vessel models (by construction)")
    ax.set_xlabel("Group speed deviation  $1-f$", fontsize=10)
    ax.set_ylabel("Group-anomaly AP", fontsize=10)
    ax.set_title("Detection of group-relative anomalies", fontsize=10.5)
    ax.set_ylim(0.45, 1.02); ax.grid(alpha=0.35, linestyle=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(BASE / "fig9_group_relative.png", dpi=300, bbox_inches="tight")
    print("saved fig9")


if __name__ == "__main__":
    main()
