#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 65: 群体相对异常受控实验（Group-relative anomaly）。

构造：同一批测试窗，目标船输入完全不变，仅把"邻船群体"同步减速/加速。
   正常样本：邻船保持原速
   异常样本：邻船整体 -30%（目标船未跟随）
→ 任何不使用邻船信息的模型，两版打分完全相同，AP 必然 = 0.5；
   图模型只要能感知邻船速度变化，AP 就会 > 0.5。
这是对"图到底有没有用"最干净的隔离实验。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
sp = importlib.util.spec_from_file_location("gf", "/mnt/sda/WangZijia/AIS/scripts/60_graph_fix.py")
gf = importlib.util.module_from_spec(sp); sp.loader.exec_module(gf)


def ap_score(s, y):
    """y: 0=正常, 1=群体异常；分数越高越异常。"""
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def shift_group(te, factor, rng=None):
    """只改邻船速度（群体协同变速），目标船输入保持不变。"""
    d = dict(te)
    d["nb_v"] = te["nb_v"] * factor
    d["nb_seq"] = te["nb_seq"].copy()
    d["nb_seq"][:, :, :, 2:4] *= factor
    if rng is not None:  # 邻居噪声：20% 个体额外抖动
        n = len(d["nb_v"])
        for i in range(n):
            idx = np.flatnonzero(te["nb_mask"][i])
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, size=max(1, int(len(idx) * 0.2)), replace=False)
            d["nb_seq"][i, pick, -1, 0] += rng.uniform(500, 1500)
    return d


def score_ours(model, d, mu, sd, dev, tau=0.2):
    X = (np.concatenate([d["gated"], d["rf"][..., None], d["w"][..., None]], -1) - mu) / sd
    with torch.no_grad():
        pb, _, _, _ = model(torch.tensor(X, dtype=torch.float32, device=dev),
                            torch.tensor(d["nb_seq"], device=dev), torch.tensor(d["nb_v"], device=dev),
                            torch.tensor(d["nb_xy"], device=dev), torch.tensor(d["nb_mask"], device=dev),
                            torch.tensor(d["nb_w"], device=dev), torch.tensor(d["rf"][:, -1], device=dev),
                            torch.tensor(d["raw"][:, -1, 2:4], device=dev), tau=tau)
        return torch.sigmoid(pb).cpu().numpy()


def score_sw(name, d, te, dev):
    ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
    m = sw.Net(ck["kind"], ck["nf"], use_graph=ck["use_graph"]).to(dev)
    m.load_state_dict(ck["sd"]); m.eval()
    mu, sd = ck["stats"]
    X = (sw.build(d, ck["cfg"]["use_kg"], ck["cfg"]["use_lf"]) - mu) / sd
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=dev)
        if ck["use_graph"]:
            pb, _, _ = m(xb, torch.tensor(d["nb_seq"], device=dev), torch.tensor(d["nb_xy"], device=dev),
                         torch.tensor(d["nb_mask"], device=dev), torch.tensor(d["nb_w"], device=dev),
                         torch.tensor(d["rf"][:, -1], device=dev))
        else:
            pb, _, _ = m(xb)
        return torch.sigmoid(pb).cpu().numpy()


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    te["nb_v"] = np.load(BASE / "eval_v2_graph2" / "test.npz")["nb_v"].astype(np.float32)
    n = len(te["raw"])
    y = np.r_[np.zeros(n), np.ones(n)]

    ck = torch.load(W / "graph_fixed.pt", map_location="cpu", weights_only=False)
    mfix = gf.Net2(ck["kind"], ck["nf"]).to(dev); mfix.load_state_dict(ck["sd"]); mfix.eval()
    mu, sd = ck["stats"]

    results = {}
    for name, fn in [
        ("+KG+LF (no graph)", lambda d: score_sw("ablation_2_kglf", d, te, dev)),
        ("PG-DRGN (old Gpsi)", lambda d: score_sw("ablation_4_full_cdg", d, te, dev)),
        ("PG-DRGN (fixed Gpsi)", lambda d: score_ours(mfix, d, mu, sd, dev)),
        ("PG-DRGN (fixed, tau=0.05)", lambda d: score_ours(mfix, d, mu, sd, dev, tau=0.05)),
        ("STGVAD-style graph", lambda d: score_sw("base_stgv", d, te, dev)),
    ]:
        p_norm = fn(te)
        p_abn = fn(shift_group(te, 0.7))
        s = np.r_[p_norm, p_abn]
        results[name] = round(ap_score(s, y), 3)
        print(f"{name:30s} group-anomaly AP = {results[name]:.3f}", flush=True)

    (BASE / "group_relative_result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
