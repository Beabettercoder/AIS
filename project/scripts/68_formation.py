#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 68: 真实编队验证——按邻船编队紧致度分层，检验"谁变了"的四种情形。"""
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

    # 邻船速度一致性（编队紧致度）
    nb_speed = np.linalg.norm(te["nb_v"], axis=-1)
    nbm = te["nb_mask"]
    disp = np.array([np.std(nb_speed[i][nbm[i]]) if nbm[i].sum() > 2 else np.nan for i in range(len(nb_speed))])
    disp = np.nan_to_num(disp, nan=99.0)
    tight = np.argsort(disp)[: len(disp) // 3]
    loose = np.argsort(disp)[-len(disp) // 3:]
    print("formation dispersion kn: tight median", round(float(np.median(disp[tight])), 3),
          " loose median", round(float(np.median(disp[loose])), 3))

    def score(d, idx):
        with torch.no_grad():
            pb, _, _, _ = m(torch.tensor(X[idx], dtype=torch.float32, device=dev),
                            *gf.tensors(d, idx, dev))
            return torch.sigmoid(pb).cpu().numpy()

    def shift(d, idx, f):
        o = dict(d)
        o["nb_v"] = d["nb_v"].copy(); o["nb_seqv"] = None
        o["nb_v"][idx] *= f
        o["nb_seq"] = d["nb_seq"].copy()
        o["nb_seq"][idx, :, :, 2:4] *= f
        return o

    rows = {}
    for name, idx in (("tight formation", tight), ("loose formation", loose)):
        base = score(te, idx)
        cow = score(shift(te, idx, 0.75), idx)     # 编队整体减速、目标未跟随 = 群体异常
        y = np.r_[np.zeros(len(idx)), np.ones(len(idx))]
        rows[name] = ap_score(np.r_[base, cow], y)
        print(f"{name}: group-anomaly AP = {rows[name]:.3f} (n={len(idx)})")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(list(rows), list(rows.values()), color=["#D62728", "#7f7f7f"], edgecolor="k",
           hatch=["///", "\\\\\\"], width=0.5)
    ax.axhline(0.5, color="k", ls="--", lw=0.9)
    for i, (k, v) in enumerate(rows.items()):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("Group-anomaly AP", fontsize=10)
    ax.set_title("Effect of real formation tightness", fontsize=10.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(BASE / "fig10_formation_tightness.png", dpi=300, bbox_inches="tight")
    print("saved fig10")


if __name__ == "__main__":
    main()
