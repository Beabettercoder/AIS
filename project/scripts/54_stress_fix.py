#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 54: 修正 stress（常量漂移 + 邻船同步 + 速度群缩放）→ Table 3 + Fig. 3。"""
from __future__ import annotations

import importlib.util
import json
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


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def load_model(name, dev):
    ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
    m = sw.Net(ck["kind"], ck["nf"], use_graph=ck["use_graph"]).to(dev)
    m.load_state_dict(ck["sd"]); m.eval()
    return m, ck["stats"], ck["cfg"], ck["use_graph"]


def stress(d, rng, drop=0.35, frac=0.25):
    out = dict(d)
    raw = d["raw"].copy()
    gated = d["gated"].copy()
    nb_seq = d["nb_seq"].copy()
    nb_w = d["nb_w"].copy()
    n, T = raw.shape[0], raw.shape[1]
    mask = d["nb_mask"]
    for i in range(n):
        off = rng.uniform(1000, 3000)
        ang = rng.uniform(-np.pi, np.pi)
        dx, dy = off * np.sin(ang), off * np.cos(ang)
        raw[i, :, 0] += dx; raw[i, :, 1] += dy
        gated[i, :, 0] += dx; gated[i, :, 1] += dy
        raw[i, :, 2:4] *= 0.7; gated[i, :, 2:4] *= 0.7
        k = int(T * drop)
        if k:
            idx = rng.choice(T, size=k, replace=False)
            raw[i, idx] = 0.0; gated[i, idx] = 0.0
        idxn = np.flatnonzero(mask[i])
        if len(idxn):
            bad = rng.choice(idxn, size=max(1, int(len(idxn) * frac)), replace=False)
            nb_seq[i, bad, :, 0] += off
            nb_seq[i, bad, :, 2:4] *= 3.0
            nb_w[i, bad] = 0.05
        nb_seq[i, :, :, 0] += dx * 0.3
        nb_seq[i, :, :, 1] += dy * 0.3
    out.update(raw=raw, gated=gated, nb_seq=nb_seq, nb_w=nb_w)
    return out


def forward(model, d, cfg, stats, graph, dev):
    mu, sd = stats
    X = (sw.build(d, cfg["use_kg"], cfg["use_lf"]) - mu) / sd
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=dev)
        if graph:
            pb, _, _ = model(xb, torch.tensor(d["nb_seq"], device=dev), torch.tensor(d["nb_xy"], device=dev),
                             torch.tensor(d["nb_mask"], device=dev), torch.tensor(d["nb_w"], device=dev),
                             torch.tensor(d["rf"][:, -1], device=dev))
        else:
            pb, _, _ = model(xb)
        return torch.sigmoid(pb).cpu().numpy()


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    names = [("Baseline", "ablation_0_raw_tcn"), ("+KG", "ablation_1_kg"), ("+KG+LF", "ablation_2_kglf"),
             ("+KG+LF+PSA", "ablation_3_psa"), ("PG-DRGN", "ablation_4_full_cdg")]
    rows = []
    for label, f in names:
        model, stats, cfg, graph = load_model(f, dev)
        p_clean = forward(model, te, cfg, stats, graph, dev)
        p_stress = forward(model, stress(te, np.random.default_rng(7)), cfg, stats, graph, dev)
        rows.append({"config": label, "clean": ap_score(p_clean, te["y"] > 0),
                     "stress": ap_score(p_stress, te["y"] > 0)})
        print(rows[-1], flush=True)
    (BASE / "stress_fixed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    prev = json.loads((BASE / "ablation_stress.json").read_text())
    prev_zs = {}
    for r in prev:
        prev_zs.setdefault(r["config"], []).append(r["zs_mean"])
    zs_map = {"Baseline": "Baseline", "+KG": "+KG", "+KG+LF": "+KG+LF",
              "+KG+LF+PSA": "+KG+LF+PSA", "PG-DRGN": "PG-DRGN"}
    table = ["# Table 3 (final) — clean / stress / cross-scenario", "",
             "| Config | Clean AP | Stress Full AP | Cross-Scenario ZS AP |", "|---|---|---|---|"]
    for r in rows:
        z = np.mean(prev_zs.get(zs_map[r["config"]], [np.nan]))
        r["zs"] = float(z)
        table.append(f"| {r['config']} | {r['clean']:.3f} | {r['stress']:.3f} | {z:.3f} |")
    (BASE / "table3_stress_final.md").write_text("\n".join(table), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(rows)); w = 0.26
    ax.bar(x - w, [r["clean"] for r in rows], w, label="Clean AP", color="#4C72B0",
           edgecolor="k", hatch="///", linewidth=0.5)
    ax.bar(x, [r["stress"] for r in rows], w, label="Stress-Test Full AP", color="#55A868",
           edgecolor="k", hatch="xxx", linewidth=0.5)
    ax.bar(x + w, [r["zs"] for r in rows], w, label="Cross-Scenario ZS AP", color="#8172B3",
           edgecolor="k", hatch="...", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([r["config"] for r in rows], fontsize=8)
    ax.set_ylim(0.55, 0.90); ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Ablation under clean, stressed and cross-scenario evaluation", fontsize=10.5)
    ax.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper center")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(BASE / "fig3_ablation_hatched_bars.png", dpi=300, bbox_inches="tight")
    print("\n".join(table))


if __name__ == "__main__":
    main()
