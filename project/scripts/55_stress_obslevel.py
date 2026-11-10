#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 55: 观测层注入污染 → 重算门控 w → 重建 stress 测试集 → 纯前向评测。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
OUTDIR = BASE / "eval_v2_stress"
OUTDIR.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)

DT = 70.0


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def recompute_w(pos):
    """三点中心差分运动学残差 → 可信度 w。"""
    n, T, _ = pos.shape
    e = np.zeros((n, T))
    for k in range(1, T - 1):
        acc = (pos[:, k + 1] - 2 * pos[:, k] + pos[:, k - 1]) / (DT ** 2)
        e[:, k] = np.linalg.norm(acc, axis=-1)
    w = np.exp(-np.maximum(0.0, e - 2.5) ** 2 / 8.0)
    w[:, 0] = w[:, 1] if T > 1 else 1.0
    w[:, -1] = w[:, -2] if T > 1 else 1.0
    return w


def build_stress(rng, drop=0.30, frac_jump=0.15):
    z = dict(np.load(BASE / "eval_v2_graph2" / "test.npz"))
    raw = z["raw"].astype(np.float32)
    n, T, C = raw.shape
    pos = np.cumsum(raw[:, :, 0:2], axis=1)                       # 原始位置
    corrupted = pos.copy()
    for i in range(n):
        # 1) 区域共模漂移（整窗常量平移）
        off = rng.uniform(1000, 3000); ang = rng.uniform(-np.pi, np.pi)
        corrupted[i] += np.array([off * np.sin(ang), off * np.cos(ang)])[None, :]
        # 2) 15% 时间步 GPS 跳变 1–3 km
        idx = rng.choice(T, size=max(1, int(T * frac_jump)), replace=False)
        for k in idx:
            j = rng.uniform(1000, 3000); a = rng.uniform(-np.pi, np.pi)
            corrupted[i, k] += np.array([j * np.sin(a), j * np.cos(a)])
        # 3) 30% 丢点（位置冻结 → 速度归零）
        idx2 = rng.choice(T, size=int(T * drop), replace=False)
        corrupted[i, idx2] = corrupted[i, idx2 - 1] if True else corrupted[i, idx2]
    w_new = recompute_w(corrupted)
    vel = np.gradient(corrupted, axis=1) / DT                         # 速度 m/s
    speed = np.linalg.norm(vel, axis=-1)
    gated = raw.copy()
    dp = np.diff(corrupted, axis=1, prepend=corrupted[:, :1])
    dp = dp * w_new[..., None]
    gated[:, :, 0:2] = dp
    gated[:, :, 2:4] = vel * 0.7                                       # 群体减速 30%
    gated[:, :, 4] = speed * 0.7 * 1.944                               # 节
    ang = np.arctan2(vel[..., 0], vel[..., 1])
    gated[:, :, 5] = np.sin(ang); gated[:, :, 6] = np.cos(ang)
    for ch in (2, 3, 4, 5, 6):
        v = gated[:, :, ch]; low = w_new < 0.2
        for t in range(1, T):
            m = low[:, t]
            v[m, t] = v[m, t - 1]
        gated[:, :, ch] = v
    raw_c = raw.copy()
    raw_c[:, :, 0:2] = np.diff(corrupted, axis=1, prepend=corrupted[:, :1])
    raw_c[:, :, 2:4] = vel * 0.7
    raw_c[:, :, 4] = speed * 0.7 * 1.944
    raw_c[:, :, 5] = np.sin(ang)
    raw_c[:, :, 6] = np.cos(ang)
    z["raw_corrupt"] = raw_c.astype(np.float32)
    z["raw"] = raw
    z["gated_corrupt"] = gated.astype(np.float32)
    z["w_corrupt"] = w_new.astype(np.float32)
    np.savez_compressed(OUTDIR / "test.npz", **z)
    return z


def load_model(name, dev):
    ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
    m = sw.Net(ck["kind"], ck["nf"], use_graph=ck["use_graph"]).to(dev)
    m.load_state_dict(ck["sd"]); m.eval()
    return m, ck["stats"], ck["cfg"], ck["use_graph"]


def forward(model, raw, gated, w, cfg, stats, graph, dev, nb):
    if not cfg["use_kg"]:
        X = raw
    else:
        parts = [gated]
        if cfg["use_lf"]:
            parts.append(nb["rf"][..., None])
        parts.append(w[..., None])
        X = np.concatenate(parts, -1)
    mu, sd = stats
    X = (X - mu) / sd
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=dev)
        if graph:
            pb, _, _ = model(xb, torch.tensor(nb["nb_seq"], device=dev), torch.tensor(nb["nb_xy"], device=dev),
                             torch.tensor(nb["nb_mask"], device=dev), torch.tensor(nb["nb_w"], device=dev),
                             torch.tensor(nb["rf"][:, -1], device=dev))
        else:
            pb, _, _ = model(xb)
        return torch.sigmoid(pb).cpu().numpy()


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = build_stress(np.random.default_rng(3))
    nb = sw.load(BASE / "eval_v2_graph2", "test")
    clean = sw.load(BASE / "eval_v2_graph2", "test")
    rows = []
    for label, f in [("Baseline", "ablation_0_raw_tcn"), ("+KG", "ablation_1_kg"),
                     ("+KG+LF", "ablation_2_kglf"), ("+KG+LF+PSA", "ablation_3_psa"),
                     ("PG-DRGN", "ablation_4_full_cdg")]:
        m, stats, cfg, graph = load_model(f, dev)
        p_clean = forward(m, clean["raw"], clean["gated"], clean["w"], cfg, stats, graph, dev, clean)
        p_st = forward(m, z["raw_corrupt"], z["gated_corrupt"], z["w_corrupt"], cfg, stats, graph, dev, nb)
        rows.append({"config": label, "clean": ap_score(p_clean, clean["y"] > 0),
                     "stress": ap_score(p_st, nb["y"] > 0)})
        print(rows[-1], flush=True)
    (BASE / "stress_obslevel.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Table 3 (observation-level stress, gate recomputed)", "",
             "| Config | Clean AP | Stress Full AP |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['clean']:.3f} | {r['stress']:.3f} |")
    print("\n".join(lines))
    (BASE / "table3_stress_obslevel.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
