#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 52: 纯前向评估（修正 stress + 6 目标域），权重全部来自 vis_weights/。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
TARGETS = ["ny", "la", "puget", "miami", "gom_open", "atl_open"]
import importlib.util

spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def build_input(d, cfg):
    return sw.build(d, cfg["use_kg"], cfg["use_lf"])


def load_model(name, dev):
    ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
    model = sw.Net(ck["kind"], ck["nf"], use_graph=ck["use_graph"]).to(dev)
    model.load_state_dict(ck["sd"]); model.eval()
    return model, ck["stats"], ck["cfg"], ck["use_graph"]


def corrupt(d, rng, drift=3.0, frac=0.25):
    """共模漂移同时作用于目标与邻船；25% 邻船注入 1–2 km 跳变，nb_w 相应置低。"""
    out = dict(d)
    T = d["raw"].shape[1]
    drift_seq = np.linspace(0, 1, T)[None, :, None] * drift
    raw = d["raw"].copy(); raw[:, :, 0:2] += drift_seq
    gated = d["gated"].copy(); gated[:, :, 0:2] += drift_seq
    nb_seq = d["nb_seq"].copy()
    nb_seq[:, :, :, 0:2] += np.linspace(0, 1, nb_seq.shape[2])[None, None, :, None] * drift
    nb_w = d["nb_w"].copy()
    mask = d["nb_mask"]
    n = len(raw)
    for i in range(n):
        idx = np.flatnonzero(mask[i])
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=max(1, int(len(idx) * frac)), replace=False)
        for j in pick:
            off = rng.uniform(1000, 2000)
            ang = rng.uniform(-np.pi, np.pi)
            nb_seq[i, j, -1, 0] += off * np.sin(ang)
            nb_seq[i, j, -1, 1] += off * np.cos(ang)
            nb_w[i, j] = 0.05
    out.update(raw=raw, gated=gated, nb_seq=nb_seq, nb_w=nb_w)
    return out


def forward(model, d, cfg, stats, graph, dev):
    mu, sd = stats
    X = (build_input(d, cfg) - mu) / sd
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
    tgts = {t: sw.load(BASE / f"{t}_dataset", "test") for t in TARGETS if (BASE / f"{t}_dataset" / "test.npz").exists()}
    names = ["ablation_0_raw_tcn", "ablation_1_kg", "ablation_2_kglf", "ablation_3_psa", "ablation_4_full_cdg"]
    rows = []
    for name in names:
        model, stats, cfg, graph = load_model(name, dev)
        rng = np.random.default_rng(0)
        p_clean = forward(model, te, cfg, stats, graph, dev)
        p_stress = forward(model, corrupt(te, rng), cfg, stats, graph, dev)
        zs = []
        for t, d in tgts.items():
            pz = forward(model, corrupt(d, np.random.default_rng(1)), cfg, stats, graph, dev)
            zs.append(ap_score(pz, d["y"] > 0))
        rows.append({"model": name, "clean_AP": ap_score(p_clean, te["y"] > 0),
                     "stress_full": ap_score(p_stress, te["y"] > 0),
                     "zs_mean": float(np.nanmean(zs))})
        print(rows[-1], flush=True)
    (BASE / "ablation_stress_fixed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # 基线在 6 目标域的零样本（7 通道）
    base_rows = []
    for name in ["base_gru", "base_transformer", "base_aisclean", "base_stgv"]:
        model, stats, cfg, graph = load_model(name, dev)
        zs = []
        for t, d in tgts.items():
            pz = forward(model, d, cfg, stats, graph, dev)
            zs.append(ap_score(pz, d["y"] > 0))
        base_rows.append({"model": name, "zs_mean": float(np.nanmean(zs)), "per_target": zs})
        print(base_rows[-1], flush=True)
    (BASE / "zs_baselines.json").write_text(json.dumps(base_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
