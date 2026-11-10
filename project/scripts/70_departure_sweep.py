#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 70: 脱离角扫描——证明"目标自身改变"这类异常单船模型本就够用。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
sp = importlib.util.spec_from_file_location("df", "/mnt/sda/WangZijia/AIS/scripts/69_departure.py")
df = importlib.util.module_from_spec(sp); sp.loader.exec_module(df)


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    tr = sw.load(BASE / "eval_v2_graph2", "train")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    for d, s in ((tr, "train"), (te, "test")):
        d["nb_v"] = np.load(BASE / "eval_v2_graph2" / f"{s}.npz")["nb_v"].astype(np.float32)

    def tight(d):
        nb = np.linalg.norm(d["nb_v"], axis=-1); m = d["nb_mask"]
        disp = np.array([np.std(nb[i][m[i]]) if m[i].sum() >= 3 else 99.0 for i in range(len(nb))])
        return np.flatnonzero(disp < 1.5)

    itr, ite = tight(tr), tight(te)

    def pack(d, idx, ang):
        raw = df.depart(d["raw"][idx], rng, turn_deg=ang) if ang > 0 else d["raw"][idx].copy()
        g = d["gated"][idx].copy()
        g[:, :, 0:2] = raw[:, :, 0:2]; g[:, :, 2:7] = raw[:, :, 2:7]
        w = df.recompute_w(g)
        return np.concatenate([g, d["rf"][idx][..., None], w[..., None]], -1)

    Xn = pack(tr, itr, 0); Xn_t = pack(te, ite, 0)
    mu = Xn[tr["mask"][itr]].mean(0, keepdims=True); sd = Xn[tr["mask"][itr]].std(0, keepdims=True) + 1e-6
    norm = lambda X: ((X - mu) / sd).astype(np.float32)

    angles = [2, 4, 6, 9, 12, 18]
    aps = []
    for a in angles:
        Xp = pack(tr, itr, a); Xp_t = pack(te, ite, a)
        torch.manual_seed(0)
        m = sw.Net("tcn", Xn.shape[-1], use_graph=False).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
        xn, xp = norm(Xn), norm(Xp)
        for _ in range(15):
            m.train(); perm = np.random.permutation(len(xn))
            for b in range(0, len(xn), 256):
                ii = perm[b:b + 256]
                pn, _, _ = m(torch.tensor(xn[ii]).to(dev)); pp, _, _ = m(torch.tensor(xp[ii]).to(dev))
                loss = F.binary_cross_entropy_with_logits(pn, torch.zeros_like(pn)) + \
                       F.binary_cross_entropy_with_logits(pp, torch.ones_like(pp))
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pn, _, _ = m(torch.tensor(norm(Xn_t)).to(dev))
            pp, _, _ = m(torch.tensor(norm(Xp_t)).to(dev))
            s = torch.sigmoid(torch.cat([pn, pp])).cpu().numpy()
        y = np.r_[np.zeros(len(pn)), np.ones(len(pp))]
        aps.append(df.ap_score(s, y))
        print(f"turn {a}°: single-vessel AP = {aps[-1]:.3f}", flush=True)

    (BASE / "departure_sweep.json").write_text(json.dumps(dict(zip(map(str, angles), map(float, aps))), indent=2))
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.plot(angles, aps, "-o", color="#1f77b4", ms=6, lw=1.5, label="single-vessel model")
    ax.axhline(0.5, color="0.5", ls="--", lw=0.9, label="chance")
    ax.set_xlabel("Departure angle (deg)", fontsize=10)
    ax.set_ylabel("Departure AP", fontsize=10)
    ax.set_title("Target-initiated departure is detectable by single-vessel models", fontsize=9.5)
    ax.set_ylim(0.45, 1.02); ax.grid(alpha=0.35, linestyle=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(BASE / "fig11_departure_sweep.png", dpi=300, bbox_inches="tight")
    print("saved fig11")


if __name__ == "__main__":
    main()
