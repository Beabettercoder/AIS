#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 98: Fig. injection fidelity —— 平滑 vs 硬注入四联图。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)

zs = np.load(BASE / "eval_v2_graph2" / "test.npz")
zc = np.load(BASE / "eval_crude" / "test.npz")
ts = sw.load(BASE / "eval_v2_graph2", "test")
tc = sw.load(BASE / "eval_crude", "test")


def ekin(raw, tm, sel):
    sub = raw[sel]
    pos = np.cumsum(sub[:, :, 0:2], axis=1)
    v_imp = np.diff(pos, axis=1) / 70.0
    v_rep = sub[:, :-1, 4:5] * 0.514444 * np.stack([np.sin(np.arctan2(sub[:, :-1, 5], sub[:, :-1, 6])),
                                                    np.cos(np.arctan2(sub[:, :-1, 5], sub[:, :-1, 6]))], -1)
    e = np.linalg.norm(v_rep - v_imp, axis=-1)
    return e[tm[sel][:, :-1]]


normal = ekin(ts["raw"], ts["mask"], np.isin(ts["cell"], [0]))
smooth = ekin(ts["raw"], ts["mask"], np.isin(ts["cell"], [2]))
crude = ekin(tc["raw"], tc["mask"], np.isin(tc["cell"], [2]))
mat = json.loads((BASE / "injection_matrix.json").read_text())

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

# (a) COG / SOG profiles
ax = axes[0, 0]
i_s = int(np.flatnonzero(ts["cell"] == 2)[3])
i_c = int(np.flatnonzero(tc["cell"] == 2)[3])
for d, idx, lab, ls in ((ts, i_s, "Ours smooth", "-"), (tc, i_c, "Crude hard", "--")):
    cog = np.degrees(np.arctan2(d["raw"][idx][:, 5], d["raw"][idx][:, 6]))
    ax.plot(np.arange(20), cog, ls, lw=1.5, label=f"{lab} COG")
ax.set_xlabel("time step", fontsize=9); ax.set_ylabel("COG (deg)", fontsize=9)
ax2 = ax.twinx()
for d, idx, lab, ls in ((ts, i_s, "Ours smooth", "-"), (tc, i_c, "Crude hard", "--")):
    ax2.plot(np.arange(20), d["raw"][idx][:, 4], ls, lw=1.2, color="0.6")
ax2.set_ylabel("SOG (kn)", fontsize=9, color="0.5")
ax.set_title("(a) Course and speed profiles", fontsize=10)
ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3, ls=":")

# (b) ekin KDE
ax = axes[0, 1]
bins = np.logspace(-3, 2, 60)
for v, lab, col in ((normal, "Real normal", "#4C72B0"), (smooth, "Ours smooth anomaly", "#55A868"),
                    (crude, "Crude hard anomaly", "#D62728")):
    ax.hist(np.clip(v, 1e-3, 100), bins=bins, density=True, histtype="step", lw=1.6, label=f"{lab} (med {np.median(v):.3g})")
ax.set_xscale("log"); ax.set_xlabel(r"$e_{kin}$ (m/s$^2$)", fontsize=9); ax.set_ylabel("density", fontsize=9)
ax.set_title("(b) Kinematic residual distribution", fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=0.3, ls=":")

# (c) phase space
ax = axes[1, 0]
for d, cells, lab, col, a in ((ts, [2], "Ours smooth", "#55A868", 0.5), (tc, [2], "Crude hard", "#D62728", 0.5),
                              (ts, [0], "Real normal", "#4C72B0", 0.35)):
    m = np.isin(d["cell"], cells)
    sog = d["raw"][m][:, :, 4]
    cog = np.arctan2(d["raw"][m][:, :, 5], d["raw"][m][:, :, 6])
    rot = np.abs(np.degrees(np.diff(cog, axis=1))) / 70.0
    ax.scatter(sog[:, 1:].ravel(), rot.ravel(), s=3, alpha=a, color=col, label=lab)
ax.set_xlabel("SOG (kn)", fontsize=9); ax.set_ylabel(r"|ROT| (deg/s)", fontsize=9)
ax.set_ylim(0, 5); ax.set_xlim(0, 16)
ax.set_title("(c) Phase space: within the manoeuvring envelope?", fontsize=10)
ax.legend(fontsize=7, markerscale=3); ax.grid(alpha=0.3, ls=":")

# (d) cross-generalization bars
ax = axes[1, 1]
labels = ["Smooth→Smooth", "Smooth→Crude", "Crude→Crude", "Crude→Smooth"]
vals = [mat["smooth->smooth"], mat["smooth->crude"], mat["crude->crude"], mat["crude->smooth"]]
cols = ["#55A868", "#55A868", "#D62728", "#D62728"]
bars = ax.bar(labels, vals, color=cols, edgecolor="k", hatch="//", width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylim(0, 1.08); ax.set_ylabel("Full AP", fontsize=9)
ax.set_title("(d) Shortcut exposure via cross-evaluation", fontsize=10)
ax.tick_params(labelsize=7.5, rotation=12); ax.grid(axis="y", ls=":", alpha=0.4)

fig.suptitle("Physically consistent injection vs crude hard-setting injection", fontsize=11.5)
fig.tight_layout()
fig.savefig(BASE / "fig_injection_fidelity.png", dpi=300, bbox_inches="tight")
print("saved fig_injection_fidelity")

lines = ["# Injection fidelity: smooth vs crude hard-setting", "",
         "| Train → Test | Full AP |", "|---|---|",
         f"| Ours smooth → Ours smooth | {mat['smooth->smooth']:.3f} |",
         f"| Ours smooth → Crude hard | {mat['smooth->crude']:.3f} |",
         f"| Crude hard → Crude hard | **{mat['crude->crude']:.3f}** |",
         f"| Crude hard → Ours smooth | **{mat['crude->smooth']:.3f}** |", "",
         "| Distribution | median $e_{kin}$ (m/s^2) | p95 |",
         "|---|---|---|",
         f"| Real normal | {np.median(normal):.3g} | {np.percentile(normal,95):.3g} |",
         f"| Ours smooth anomaly | {np.median(smooth):.3g} | {np.percentile(smooth,95):.3g} |",
         f"| Crude hard anomaly | {np.median(crude):.3g} | {np.percentile(crude,95):.3g} |", "",
         f"- Crude-trained models score {mat['crude->crude']:.3f} on crude anomalies but collapse to "
         f"{mat['crude->smooth']:.3f} on physically consistent ones ({(mat['crude->smooth']-mat['crude->crude'])*100:+.1f} pp).",
         f"- Smooth-trained models keep {mat['smooth->smooth']:.3f} on smooth and {mat['smooth->crude']:.3f} on crude, "
         "i.e. they do not rely on the injection artefact."]
(BASE / "table_injection_fidelity.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
