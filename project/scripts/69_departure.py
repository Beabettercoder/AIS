#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 69: 真实编队 + 物理一致的"单船脱离编队"注入（本工作的方法贡献之一）。

注入规则（全部遵循可观测的运动学约束）：
  * 转向：4 步内累计 18°（在 70 s 采样下相当于 ~0.9°/s，与实测中小型船转向率同量级）
  * 减速：转向期间速度下降 10%（转向诱导阻力）
  * 位置：由新速度积分得到，保证与速度通道自洽
  * 门控：按同一 e_kin 公式重算 w_t
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
sp = importlib.util.spec_from_file_location("gf", "/mnt/sda/WangZijia/AIS/scripts/60_graph_fix.py")
gf = importlib.util.module_from_spec(sp); sp.loader.exec_module(gf)

DT = 70.0


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def recompute_w(gated):
    n, T, _ = gated.shape
    pos = np.cumsum(gated[:, :, 0:2], axis=1)
    e = np.zeros((n, T))
    for k in range(1, T - 1):
        acc = (pos[:, k + 1] - 2 * pos[:, k] + pos[:, k - 1]) / (DT ** 2)
        e[:, k] = np.linalg.norm(acc, axis=-1)
    w = np.exp(-np.maximum(0.0, e - 2.5) ** 2 / 8.0)
    w[:, 0] = w[:, 1]; w[:, -1] = w[:, -2]
    return w


def depart(raw, rng, turn_deg=18.0, steps=4, start=14, speed_loss=0.10):
    """对目标船注入一次物理一致的脱离编队机动。"""
    out = raw.copy()
    n = len(out)
    A = np.deg2rad(turn_deg) * rng.choice([-1.0, 1.0], size=n)
    vx, vy = out[:, :, 2].copy(), out[:, :, 3].copy()
    speed = np.hypot(vx, vy)
    psi = np.arctan2(vx, vy)
    dpsi = np.zeros_like(psi)
    for k in range(start, min(start + steps, psi.shape[1])):
        dpsi[:, k:] += A[:, None] / steps
    psi_new = psi + dpsi
    cut = np.clip((np.arange(psi.shape[1]) - start) / steps, 0, 1)
    spd_new = speed * (1 - speed_loss * cut[None, :])
    vx2 = spd_new * np.sin(psi_new); vy2 = spd_new * np.cos(psi_new)
    out[:, :, 2] = vx2; out[:, :, 3] = vy2
    # keep original observation noise so injected samples are not "too clean"
    res_x = raw[:, :, 0] - vx * DT
    res_y = raw[:, :, 1] - vy * DT
    out[:, :, 0] = vx2 * DT + res_x
    out[:, :, 1] = vy2 * DT + res_y
    out[:, :, 4] = raw[:, :, 4] * (1 - speed_loss * cut[None, :])
    psi_h = np.arctan2(raw[:, :, 5], raw[:, :, 6])
    out[:, :, 5] = np.sin(psi_h + dpsi)
    out[:, :, 6] = np.cos(psi_h + dpsi)
    return out


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    tr = sw.load(BASE / "eval_v2_graph2", "train")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    for d, s in ((tr, "train"), (te, "test")):
        d["nb_v"] = np.load(BASE / "eval_v2_graph2" / f"{s}.npz")["nb_v"].astype(np.float32)

    # 只保留紧致编队（邻船速度离散度 < 1.5 kn 且 >=3 邻居）
    def tight_idx(d):
        nb = np.linalg.norm(d["nb_v"], axis=-1); m = d["nb_mask"]
        disp = np.array([np.std(nb[i][m[i]]) if m[i].sum() >= 3 else 99.0 for i in range(len(nb))])
        return np.flatnonzero(disp < 1.5)

    itr, ite = tight_idx(tr), tight_idx(te)
    print(f"tight-formation windows: train {len(itr)}, test {len(ite)}")

    def make(d, idx):
        raw_p = depart(d["raw"][idx], rng)
        g_p = d["gated"][idx].copy()
        g_p[:, :, 0:2] = np.diff(raw_p[:, :, 0:2], axis=1, prepend=0.0)
        g_p[:, :, 0:2] = raw_p[:, :, 0:2]
        g_p[:, :, 2:7] = raw_p[:, :, 2:7]
        return raw_p, g_p

    def pack(d, idx, pos=False):
        raw = d["raw"][idx].copy(); gated = d["gated"][idx].copy()
        if pos:
            raw, gated = make(d, idx)
        w = recompute_w(gated)
        if pos:
            # apply the same gate weighting as the original pipeline
            gated[:, :, 0:2] = raw[:, :, 0:2] * w[..., None]
        # drop r_field here: it is not recomputed after injection and would leak the original trajectory\n        X = np.concatenate([gated, w[..., None]], -1)
        return X, raw, gated, w

    Xn, _, _, _ = pack(tr, itr, pos=False)
    Xp, _, _, _ = pack(tr, itr, pos=True)
    Xn_t, _, _, _ = pack(te, ite, pos=False)
    Xp_t, _, _, _ = pack(te, ite, pos=True)
    mu = Xn[tr["mask"][itr]].mean(0, keepdims=True); sd = Xn[tr["mask"][itr]].std(0, keepdims=True) + 1e-6
    norm = lambda X: ((X - mu) / sd).astype(np.float32)

    def train_model(graph, epochs=20):
        torch.manual_seed(0)
        m = (gf.Net2("tcn", Xn.shape[-1]) if graph else sw.Net("tcn", Xn.shape[-1], use_graph=False)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
        xn, xp = norm(Xn), norm(Xp)
        for _ in range(epochs):
            m.train()
            perm = np.random.permutation(len(xn))
            for b in range(0, len(xn), 256):
                ii = perm[b:b + 256]
                xbn = torch.tensor(xn[ii]).to(dev); xbp = torch.tensor(xp[ii]).to(dev)
                if graph:
                    pn, _, _, _ = m(xbn, *gf.tensors(tr, itr[ii], dev))
                    pp, _, _, _ = m(xbp, *gf.tensors(tr, itr[ii], dev))
                else:
                    pn, _, _ = m(xbn); pp, _, _ = m(xbp)
                loss = F.binary_cross_entropy_with_logits(pn, torch.zeros_like(pn)) + \
                       F.binary_cross_entropy_with_logits(pp, torch.ones_like(pp))
                opt.zero_grad(); loss.backward(); opt.step()
        return m

    res = {}
    for name, graph in (("single-vessel", False), ("graph (Ours)", True)):
        m = train_model(graph); m.eval()
        xn, xp = norm(Xn_t), norm(Xp_t)
        with torch.no_grad():
            if graph:
                pn, _, _, _ = m(torch.tensor(xn, dtype=torch.float32, device=dev), *gf.tensors(te, ite, dev))
                pp, _, _, _ = m(torch.tensor(xp, dtype=torch.float32, device=dev), *gf.tensors(te, ite, dev))
            else:
                pn, _, _ = m(torch.tensor(xn, dtype=torch.float32, device=dev))
                pp, _, _ = m(torch.tensor(xp, dtype=torch.float32, device=dev))
            s = torch.sigmoid(torch.cat([pn, pp])).cpu().numpy()
        y = np.r_[np.zeros(len(xn)), np.ones(len(xp))]
        res[name] = round(ap_score(s, y), 3)
        print(f"{name}: AP = {res[name]:.3f}", flush=True)
    (BASE / "departure_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
