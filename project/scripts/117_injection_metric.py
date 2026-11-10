#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 117: 面板 (b) 改为米制瞬时速度相容残差 e_kin(t0)（m/s）。"""
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
    m = np.isin(d["cell"], cells)
    raw = d["raw"][m]
    pos = np.cumsum(raw[:, :, 0:2], axis=1)
    cog = np.arctan2(raw[:, :, 5], raw[:, :, 6])
    dcog = np.abs(np.diff(cog, axis=1))
    t0 = dcog.argmax(axis=1) + 1
    idx = np.arange(len(raw))
    prev = np.clip(t0 - 1, 0, None)
    vx = raw[idx, :, 4] * 0.514444 * np.sin(cog[idx, :])
    vy = raw[idx, :, 4] * 0.514444 * np.cos(cog[idx, :])
    e_kin = np.hypot(vx[idx, t0] - vx[idx, prev], vy[idx, t0] - vy[idx, prev])
    rot = np.degrees(dcog[idx, t0 - 1]) / DT
    return e_kin, rot, raw[idx, t0, 4]


def main() -> None:
    ts = sw.load(BASE / "eval_v2_graph2", "test")
    tc = sw.load(BASE / "eval_crude", "test")
    e_n, r_n, s_n = switch_metrics(ts, [0])
    e_s, r_s, s_s = switch_metrics(ts, [2])
    e_c, r_c, s_c = switch_metrics(tc, [2])
    mat = json.loads((BASE / "injection_matrix.json").read_text())
    print(f"e_kin(t0) median m/s: normal {np.median(e_n):.2f}  smooth {np.median(e_s):.2f}  crude {np.median(e_c):.2f}")

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
    bins = np.linspace(0, 20, 41)
    for v, lab, c in ((e_n, "Real normal", "#4C72B0"), (e_s, "Ours smooth", "#55A868"),
                      (e_c, "Crude hard", "#D62728")):
        ax.hist(np.clip(v, 0, 20), bins=bins, density=True, histtype="step", lw=1.7, color=c,
                label=f"{lab} (med {np.median(v):.2f} m/s)")
    ax.set_xlabel(r"$e_{kin}(t_0)$  (m/s)", fontsize=9); ax.set_ylabel("density", fontsize=9)
    ax.set_title("(b) Instantaneous kinematic residual at the switch step", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, ls=":")

    ax = axes[1, 0]
    for s, r, lab, c, a in ((s_n, r_n, "Real normal", "#4C72B0", 0.35), (s_s, r_s, "Ours smooth", "#55A868", 0.5),
                            (s_c, r_c, "Crude hard", "#D62728", 0.5)):
        ax.scatter(s, r, s=8, alpha=a, color=c, label=lab)
    ax.axhline(1.0, color="k", ls=":", lw=1)
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

    lines = ["# Injection fidelity (instantaneous metrics at the switch step t0)", "",
             "| Group | median $e_{kin}(t_0)$ (m/s) | p95 | median ROT(t0) (deg/s) |",
             "|---|---|---|---|",
             f"| Real normal | {np.median(e_n):.2f} | {np.percentile(e_n,95):.2f} | {np.median(r_n):.2f} |",
             f"| Ours smooth | {np.median(e_s):.2f} | {np.percentile(e_s,95):.2f} | {np.median(r_s):.2f} |",
             f"| Crude hard | {np.median(e_c):.2f} | {np.percentile(e_c,95):.2f} | {np.median(r_c):.2f} |", "",
             "| Train → Test | Full AP |", "|---|---|",
             f"| Smooth → Smooth | {mat['smooth->smooth']:.3f} |",
             f"| Smooth → Crude | {mat['smooth->crude']:.3f} |",
             f"| Crude → Crude | {mat['crude->crude']:.3f} |",
             f"| Crude → Smooth | {mat['crude->smooth']:.3f} |"]
    (BASE / "table_injection_fidelity.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()
