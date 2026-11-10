#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 108: 修正版注入保真度图——在注入切换步 t0 上计算瞬时物理撕裂。"""
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
DT = 70.0


def switch_metrics(d, cells):
    """在 |dCOG| 最大的那一步（注入切换点）上计算瞬时物理量。"""
    m = np.isin(d["cell"], cells)
    raw = d["raw"][m]
    pos = np.cumsum(raw[:, :, 0:2], axis=1)
    cog = np.arctan2(raw[:, :, 5], raw[:, :, 6])
    dcog = np.abs(np.diff(cog, axis=1))
    t0 = dcog.argmax(axis=1) + 1                       # 切换步
    idx = np.arange(len(raw))
    e_kin = np.degrees(dcog[idx, t0 - 1])                  # single-step course jump at t0
    rot = np.degrees(dcog[idx, t0 - 1]) / DT               # deg/s
    return e_kin, rot, raw[idx, t0, 4]


def main() -> None:
    ts = sw.load(BASE / "eval_v2_graph2", "test")
    tc = sw.load(BASE / "eval_crude", "test")
    e_norm, r_norm, s_norm = switch_metrics(ts, [0])
    e_sm, r_sm, s_sm = switch_metrics(ts, [2])
    e_cr, r_cr, s_cr = switch_metrics(tc, [2])
    mat = json.loads((BASE / "injection_matrix.json").read_text())
    print(f"median e_kin@t0: normal {np.median(e_norm):.3f}  smooth {np.median(e_sm):.3f}  crude {np.median(e_cr):.3f}")
    print(f"median rot@t0 : normal {np.median(r_norm):.3f}  smooth {np.median(r_sm):.3f}  crude {np.median(r_cr):.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))
    ax = axes[0, 0]
    i_s = int(np.flatnonzero(ts["cell"] == 2)[3]); i_c = int(np.flatnonzero(tc["cell"] == 2)[3])
    for d, i, lab, ls in ((ts, i_s, "Ours smooth", "-"), (tc, i_c, "Crude hard", "--")):
        cog = np.degrees(np.arctan2(d["raw"][i][:, 5], d["raw"][i][:, 6]))
        ax.plot(np.arange(20), cog, ls, lw=1.6, label=f"{lab} COG")
    ax.set_xlabel("time step", fontsize=9); ax.set_ylabel("COG (deg)", fontsize=9)
    ax.set_title("(a) Course profile at the injection step", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, ls=":")

    ax = axes[0, 1]
    bins = np.linspace(0, 180, 46)
    for v, lab, c in ((e_norm, "Real normal", "#4C72B0"), (e_sm, "Ours smooth", "#55A868"),
                      (e_cr, "Crude hard", "#D62728")):
        ax.hist(np.clip(v, 0, 180), bins=bins, density=True, histtype="step", lw=1.7,
                color=c, label=f"{lab} (med {np.median(v):.1f} deg)")
    ax.set_xscale("log"); ax.set_xlabel("single-step course jump at t0 (deg)", fontsize=9)
    ax.set_ylabel("density", fontsize=9); ax.legend(fontsize=7)
    ax.set_title("(b) Instantaneous residual at the switch step", fontsize=10)
    ax.grid(alpha=0.3, ls=":")

    ax = axes[1, 0]
    for s, r, lab, c, a in ((s_norm, r_norm, "Real normal", "#4C72B0", 0.35),
                            (s_sm, r_sm, "Ours smooth", "#55A868", 0.5),
                            (s_cr, r_cr, "Crude hard", "#D62728", 0.5)):
        ax.scatter(s, r, s=8, alpha=a, color=c, label=lab)
    ax.set_xlabel("SOG at $t_0$ (kn)", fontsize=9); ax.set_ylabel(r"$|\mathrm{ROT}|$ at $t_0$ (deg/s)", fontsize=9)
    ax.set_ylim(0, 4); ax.set_xlim(0, 20); ax.legend(fontsize=7, markerscale=2)
    ax.set_title("(c) Phase space at the switch step", fontsize=10); ax.grid(alpha=0.3, ls=":")

    ax = axes[1, 1]
    labels = ["Smooth→Smooth", "Smooth→Crude", "Crude→Crude", "Crude→Smooth"]
    vals = [mat["smooth->smooth"], mat["smooth->crude"], mat["crude->crude"], mat["crude->smooth"]]
    bars = ax.bar(labels, vals, color=["#55A868", "#55A868", "#D62728", "#D62728"],
                  edgecolor="k", hatch="//", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Full AP", fontsize=9)
    ax.tick_params(labelsize=7.5, rotation=12)
    ax.set_title("(d) Shortcut exposure via cross-evaluation", fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.4)

    fig.suptitle("Physically consistent injection vs crude hard-setting injection", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(BASE / "fig_injection_fidelity.png", dpi=300, bbox_inches="tight")

    lines = ["# Injection fidelity (instantaneous metrics at the switch step)", "",
             "| Group | median $e_{kin}(t_0)$ m/s | median ROT(t0) deg/s | p95 ROT |",
             "|---|---|---|---|",
             f"| Real normal | {np.median(e_norm):.3f} | {np.median(r_norm):.3f} | {np.percentile(r_norm,95):.3f} |",
             f"| Ours smooth | {np.median(e_sm):.3f} | {np.median(r_sm):.3f} | {np.percentile(r_sm,95):.3f} |",
             f"| Crude hard | {np.median(e_cr):.3f} | {np.median(r_cr):.3f} | {np.percentile(r_cr,95):.3f} |", "",
             "| Train → Test | Full AP |", "|---|---|",
             f"| Smooth → Smooth | {mat['smooth->smooth']:.3f} |",
             f"| Smooth → Crude | {mat['smooth->crude']:.3f} |",
             f"| Crude → Crude | {mat['crude->crude']:.3f} |",
             f"| Crude → Smooth | {mat['crude->smooth']:.3f} |"]
    (BASE / "table_injection_fidelity.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved fig_injection_fidelity + table")


if __name__ == "__main__":
    main()
