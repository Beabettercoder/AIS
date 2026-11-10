#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 20: Zone A vs Zone B 域偏移诊断图。

统计正常窗口（cell==0）上的：SOG、转向率、局部散度 D、局部旋度 C、留一法场残差 |r|。
两区之间计算 Wasserstein-1 距离，并画成 6 面板图。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


def stats(npz_path: Path):
    z = np.load(npz_path)
    cell = z["label_cell"]
    m = cell == 0                     # 只用正常窗口
    raw = z["raw"][m]
    mask = z["mask"][m]
    t = z["t"][m]
    cog = raw[:, :, 6]                # cos
    sin = raw[:, :, 5]
    chi = np.arctan2(sin, cog)        # 近似复原航向
    dt = np.diff(t, axis=1, prepend=t[:, :1])
    dt = np.where(dt <= 0, 70.0, dt)
    dchi = np.abs(np.diff(np.unwrap(chi, axis=1), axis=1)) / dt[:, 1:]
    ep = z["e_phys"][m]
    return {
        "sog": raw[:, :, 4][mask][np.isfinite(raw[:, :, 4][mask])],
        "turn_rate": dchi[np.isfinite(dchi)],
        "div": ep[:, :, 2][mask],
        "curl": ep[:, :, 3][mask],
        "r_field": z["r_field"][m][mask],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = stats(Path(args.a))
    B = stats(Path(args.b))
    panels = [("sog", "SOG (kn)", (-0.1, 20)),
              ("turn_rate", "|dχ/dt| (rad/s)", (0, 0.02)),
              ("div", "local divergence D (1/s)", (-6e-4, 6e-4)),
              ("curl", "local curl C (1/s)", (-6e-4, 6e-4)),
              ("r_field", "|r_field| (m/s)", (0, 2.5))]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, (key, label, rng) in zip(axes, panels):
        a, b = A[key], B[key]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        # 1–99 分位裁剪（两边用同一组边界），避免离群值主导 W1
        lo, hi = np.percentile(np.concatenate([a, b]), [1, 99])
        a = a[(a >= lo) & (a <= hi)]; b = b[(b >= lo) & (b <= hi)]
        logx = key == "r_field"
        va, vb = (np.log1p(a), np.log1p(b)) if logx else (a, b)
        bins = np.linspace(va.min(), vb.max(), 60)
        ax.hist(va, bins=bins, density=True, alpha=0.55, label="Zone A (open bay)", color="#1e8449")
        ax.hist(vb, bins=bins, density=True, alpha=0.55, label="Zone B (bent channel)", color="#c0392b")
        w = wasserstein_distance(va, vb)
        ax.set_title(f"{label}  (clipped p1–p99, {'log1p' if logx else 'linear'})\n"
                     f"Wasserstein-1 = {w:.4g}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    # 第六格：SOG vs 转向率的联合密度（A/B 叠加）
    ax = axes[5]
    for d, c, lab in ((A, "#1e8449", "A"), (B, "#c0392b", "B")):
        n = min(len(d["sog"]), len(d["turn_rate"]))
        ax.scatter(d["sog"][:n], d["turn_rate"][:n], s=3, alpha=0.25, color=c, label=f"Zone {lab}")
    ax.set_xlabel("SOG (kn)"); ax.set_ylabel("|dχ/dt| (rad/s)")
    ax.set_ylim(0, 0.02); ax.set_xlim(0, 16)
    ax.set_title("Joint: speed vs turn-rate", fontsize=10)
    ax.legend(fontsize=7, markerscale=3); ax.grid(alpha=0.25)

    fig.suptitle("Cross-channel physical domain shift — Zone A (open bay) vs Zone B (bent channel)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print("saved", args.out)
    for key, label, _ in panels:
        a, b = A[key], B[key]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        lo, hi = np.percentile(np.concatenate([a, b]), [1, 99])
        a = a[(a >= lo) & (a <= hi)]; b = b[(b >= lo) & (b <= hi)]
        if key == "r_field":
            w = wasserstein_distance(np.log1p(a), np.log1p(b))
        else:
            w = wasserstein_distance(a, b)
        print(f"{key:10s} W1(clipped)={w:.5g}  median A={np.median(a):.4g} B={np.median(b):.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
