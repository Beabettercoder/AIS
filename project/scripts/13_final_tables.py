#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 13: 定稿版 Table 1 / Recall–FAR 曲线 / Table 4（延迟分层）。

输入：08b 训练时落盘的各 seed 预测 npz
输出：
  final_tables.md / .csv
  recall_far_curve.png
  table4_delay.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["normal", "turn_back", "deviation", "stall", "sinuous", "speed_drop"]
MODES = {"raw": "Raw-TCN", "mean": "kNN + TCN", "ours": "Ours (gate+field+dual-head)"}
FARS = [0.5, 1, 5, 10]


def ap_score(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def load_preds(folder: Path, mode: str, seeds: int):
    out = []
    for s in range(seeds):
        p = folder / f"res_{mode}_s{s}.npz"
        if p.exists():
            z = np.load(p)
            out.append({k: z[k] for k in z.files})
    return out


def recall_far_curve(pb, cell, npts=200):
    """扫阈值得到 Recall–FAR 曲线（正类 = C，负类 = A）。"""
    pos = pb[cell == 2]
    neg = pb[cell == 0]
    thrs = np.quantile(neg, np.linspace(0.0, 0.98, npts))
    rec = [(pos > t).mean() for t in thrs]
    far = [(neg > t).mean() for t in thrs]
    return np.array(far) * 100, np.array(rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", required=True, help="e.g. l4_4500 l5_4500")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary, delay_rows = [], []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for li, level in enumerate(args.levels):
        folder = base / f"eval_v2_{level}"
        curves = {}
        for mode in MODES:
            preds = load_preds(folder, mode, args.seeds)
            if not preds:
                continue
            clean = [ap_score(p["pb"][np.isin(p["cell"], [0, 2])], p["y"][np.isin(p["cell"], [0, 2])] > 0) for p in preds]
            full = [ap_score(p["pb"], p["y"] > 0) for p in preds]
            per = {}
            for k in range(1, 6):
                per[CLASSES[k]] = [ap_score(p["pc"][p["cell"] == 2][:, k - 1], p["y"][p["cell"] == 2] == k) for p in preds]
            row = {"level": level, "mode": mode,
                   "clean_AP_mean": np.nanmean(clean), "clean_AP_std": np.nanstd(clean),
                   "full_AP_mean": np.nanmean(full), "full_AP_std": np.nanstd(full)}
            for k, v in per.items():
                row[f"AP_{k}"] = np.nanmean(v)
                row[f"AP_{k}_std"] = np.nanstd(v)
            # 1%/5% FAR recall
            r1, r5 = [], []
            for p in preds:
                neg = p["pb"][p["cell"] == 0]
                pos = p["pb"][p["cell"] == 2]
                for far, bucket in ((1, r1), (5, r5)):
                    thr = np.quantile(neg, 1 - far / 100)
                    bucket.append((pos > thr).mean())
            row["recall_1pct"] = np.nanmean(r1)
            row["recall_5pct"] = np.nanmean(r5)
            summary.append(row)

            # 曲线（seed 平均）
            cs = [recall_far_curve(p["pb"], p["cell"]) for p in preds]
            far_grid = np.linspace(0, 10, 200)
            acc = []
            for f, r in cs:
                o = np.argsort(f)
                acc.append(np.interp(far_grid, f[o], r[o]))
            curves[mode] = np.nanmean(acc, axis=0)

            # Table 4：延迟分层（固定 FAR + 平衡正负比的 AP）
            for d0, dlab in ((180, "t0+3min"), (300, "t0+5min"), (600, "t0+10min")):
                rr1, rr5, aps = [], [], []
                rng = np.random.default_rng(0)
                for p in preds:
                    neg = p["pb"][p["cell"] == 0]
                    m = (p["cell"] == 2) & (p["delay"] == d0)
                    pos = p["pb"][m]
                    if len(pos) == 0:
                        continue
                    for far, bucket in ((1, rr1), (5, rr5)):
                        thr = np.quantile(neg, 1 - far / 100)
                        bucket.append((pos > thr).mean())
                    # 平衡正负比：每个 seed 抽 2x 正类数量的负类，重复 20 次取均值
                    vals = []
                    for _ in range(20):
                        sub = rng.choice(neg, size=min(len(neg), 2 * len(pos)), replace=False)
                        s = np.concatenate([pos, sub])
                        lab = np.concatenate([np.ones(len(pos)), np.zeros(len(sub))])
                        vals.append(ap_score(s, lab))
                    aps.append(np.nanmean(vals))
                delay_rows.append({"level": level, "mode": mode, "delay": dlab,
                                   "AP_balanced": np.nanmean(aps) if aps else np.nan,
                                   "recall_1pct": np.nanmean(rr1) if rr1 else np.nan,
                                   "recall_5pct": np.nanmean(rr5) if rr5 else np.nan})

        ax = axes[li]
        for mode, lab in MODES.items():
            if mode not in curves:
                continue
            ax.plot(far_grid, curves[mode] * 100, lw=2, label=lab)
        for far in FARS:
            ax.axvline(far, ls=":", color="gray", lw=0.8)
        ax.set_xlim(0, 10); ax.set_ylim(0, 100)
        ax.set_xlabel("False alarm rate (%)")
        ax.set_ylabel("Recall of behaviour anomalies (%)")
        ax.set_title(f"Recall–FAR — {level}")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(outdir / "recall_far_curve.png", dpi=140)
    print("saved", outdir / "recall_far_curve.png")

    df = pd.DataFrame(summary)
    df.to_csv(outdir / "final_tables.csv", index=False)
    d4 = pd.DataFrame(delay_rows)
    d4.to_csv(outdir / "table4_delay.csv", index=False)

    lines = ["# Final tables (4500 train windows, 5 seeds)\n", "## Table 1 / Table 3\n", df.round(3).to_string(index=False),
             "\n## Table 4 — early detection by decision delay\n", d4.round(3).to_string(index=False)]
    (outdir / "final_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
