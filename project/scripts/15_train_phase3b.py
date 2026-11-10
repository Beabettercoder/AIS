#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 15: Phase 3 收官 —— ABL5 v2 三种融合 + ε 边界日志 + 延迟分层表。

融合方式：
  single   主分支 9 通道（raw + r_field + w）直接进 TCN（= ABL4 基准）
  residual h = h_abs + λ ⊙ LayerNorm(h_rel)，λ 逐通道、初始化为 0
  concat   [pool(h_abs), pool(h_rel)] -> Linear(128->64)
相对分支输入 8 通道：[r_x, r_y, |r|, w, D, C, d, dchi]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["normal", "turn_back", "deviation", "stall", "sinuous", "speed_drop"]
EPS_MAX, EPS_MIN = 0.3, 0.01


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


def load(d, split):
    z = np.load(Path(d) / f"{split}.npz")
    raw = z["raw"].astype(np.float32)
    w = z["w_self"].astype(np.float32)
    dp = np.diff(raw[:, :, 0:2], axis=1, prepend=0.0) * w[..., None]
    raw[:, :, 0:2] = np.cumsum(dp, axis=1)
    for ch in (2, 3, 4, 5, 6):
        v = raw[:, :, ch]
        low = w < 0.2
        for t in range(1, v.shape[1]):
            m = low[:, t]
            v[m, t] = v[m, t - 1]
        raw[:, :, ch] = v
    rv = z["r_vec"]; fr = z["frenet3"]; ep = z["e_phys"]
    main = np.concatenate([raw, z["r_field"][..., None], w[..., None]], -1).astype(np.float32)
    rel = np.concatenate([rv, np.linalg.norm(rv, axis=-1, keepdims=True), w[..., None],
                          ep[:, :, 2:4], fr[:, :, 1:3]], -1).astype(np.float32)
    return dict(main=main, rel=rel, ep=ep.astype(np.float32), mask=z["mask"],
                y=z["label_class"], cell=z["label_cell"], delay=z["label_delay"])


class TCN(nn.Module):
    def __init__(self, nf, nh=64):
        super().__init__()
        self.c1 = nn.Conv1d(nf, nh, 5, padding=2)
        self.c2 = nn.Conv1d(nh, nh, 5, padding=4, dilation=2)
        self.c3 = nn.Conv1d(nh, nh, 5, padding=8, dilation=4)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.gelu(self.c1(h)); h = F.gelu(self.c2(h)); return F.gelu(self.c3(h))


class Net(nn.Module):
    def __init__(self, mode, nh=64):
        super().__init__()
        self.mode = mode
        self.main_b = TCN(9, nh)
        self.rel_b = TCN(8, nh) if mode != "single" else None
        if mode == "residual":
            self.ln = nn.LayerNorm(nh)
            self.lam = nn.Parameter(torch.zeros(nh))
        if mode == "concat":
            self.proj = nn.Linear(4 * nh, nh)
        self.ctrl = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 2), nn.Tanh())
        self.bin = nn.Linear(2 * nh, 1)
        self.cls = nn.Linear(2 * nh, 5)

    def forward(self, main, rel, ep, use_psa=False):
        ha = self.main_b(main)
        if self.mode == "single":
            h = ha
        else:
            hr = self.rel_b(rel)
            if self.mode == "residual":
                h = ha + self.lam[None, :, None] * self.ln(hr.transpose(1, 2)).transpose(1, 2)
            else:
                pa = torch.cat([ha.mean(-1), ha.max(-1).values], -1)
                pr = torch.cat([hr.mean(-1), hr.max(-1).values], -1)
                p = F.gelu(self.proj(torch.cat([pa, pr], -1)))
                h = p[:, :, None].expand(-1, -1, ha.shape[-1])
        eps_hi = eps_lo = 0.0
        if use_psa:
            mu = h.mean(2, keepdim=True).detach()
            sd = h.std(2, keepdim=True).detach().clamp_min(1e-5)
            e = self.ctrl(ep.mean(1))
            a = (EPS_MIN + (EPS_MAX - EPS_MIN) * (e[:, 0:1] + 1) / 2)
            b = e[:, 1:2] * EPS_MAX
            h = mu + sd * (((h - mu) / sd) * (1 + a[..., None]) + b[..., None])
            eps_hi = float((a > 0.29).float().mean().detach())
            eps_lo = float((a < 0.02).float().mean().detach())
            eps_val = float(a.mean().detach())
        else:
            eps_val = 0.0
        p = torch.cat([h.mean(-1), h.max(-1).values], -1)
        return self.bin(p).squeeze(-1), self.cls(p), eps_val, eps_hi, eps_lo


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + 1e-8) - torch.log(b + 1e-8))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run(mode, seed, tr, te, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = {}
    for k in ("main", "rel", "ep"):
        m_ = tr[k][tr["mask"]] if k == "main" else tr[k].reshape(-1, tr[k].shape[-1])
        stats[k] = (np.nanmean(m_, 0, keepdims=True), np.nanstd(m_, 0, keepdims=True) + 1e-6)

    def prep(d):
        return (torch.tensor((d["main"] - stats["main"][0]) / stats["main"][1]),
                torch.tensor((d["rel"] - stats["rel"][0]) / stats["rel"][1]),
                torch.tensor(np.nan_to_num((d["ep"] - stats["ep"][0]) / stats["ep"][1], nan=0.0)))

    Mtr, Rtr, Etr = prep(tr); Mte, Rte, Ete = prep(te)
    model = Net(mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    eps_all, hi_all, lo_all = [], [], []

    for _ in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            m_, r_, e_ = Mtr[ii].to(dev), Rtr[ii].to(dev), Etr[ii].to(dev)
            pb, pc, _, _, _ = model(m_, r_, e_, use_psa=False)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            keep = (torch.rand(m_.shape[0], m_.shape[1], device=dev) > 0.25).float()
            m2 = m_ * keep[..., None]
            j = torch.randint(0, m_.shape[1], (m_.shape[0],), device=dev)
            m2[torch.arange(m_.shape[0]), j, 0] += 1500.0
            pb2, pc2, ev, eh, el = model(m2, r_, e_, use_psa=True)
            eps_all.append(ev); hi_all.append(eh); lo_all.append(el)
            p1, p2 = F.softmax(pc, -1), F.softmax(pc2, -1)
            ret = keep.mean(1, keepdim=True)
            loss = loss + 0.3 * (js(p1, p2) * ret).mean() \
                        + 0.5 * F.binary_cross_entropy_with_logits(pb2, torch.tensor(yb[ii]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pb, pc, _, _, _ = model(Mte.to(dev), Rte.to(dev), Ete.to(dev), use_psa=False)
        pb = torch.sigmoid(pb).cpu().numpy(); pc = F.softmax(pc, -1).cpu().numpy()
    cell, y, dl = te["cell"], te["y"], te["delay"]
    out = {"mode": mode, "seed": seed,
           "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(pb, y > 0),
           "eps_mean": float(np.mean(eps_all)), "eps_hi_ratio": float(np.mean(hi_all)),
           "eps_lo_ratio": float(np.mean(lo_all))}
    for k in range(1, 6):
        m = cell == 2
        out[f"AP_{CLASSES[k]}"] = ap_score(pc[m, k - 1], (y[m] == k))
    neg = pb[cell == 0]
    rng = np.random.default_rng(0)
    for far in (1, 5):
        thr = np.quantile(neg, 1 - far / 100)
        out[f"recall_{far}pct"] = float((pb[cell == 2] > thr).mean())
        for d0 in (180, 300, 600):
            m = (cell == 2) & (dl == d0)
            out[f"recall_{far}pct_d{d0//60}"] = float((pb[m] > thr).mean()) if m.any() else np.nan
    for d0 in (180, 300, 600):
        m = (cell == 2) & (dl == d0)
        pos = pb[m]
        vals = []
        for _ in range(20):
            sub = rng.choice(neg, size=min(len(neg), 2 * len(pos)), replace=False)
            vals.append(ap_score(np.concatenate([pos, sub]),
                                 np.concatenate([np.ones(len(pos)), np.zeros(len(sub))])))
        out[f"AP_bal_d{d0//60}"] = float(np.nanmean(vals))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", required=True, choices=["single", "residual", "concat"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tr, te = load(args.dir, "train"), load(args.dir, "test")
    rows = [run(args.mode, s, tr, te, args.epochs) for s in range(args.seeds)]
    keys = [k for k in rows[0] if k not in ("mode", "seed")]
    agg = {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows]))}
           for k in keys}
    Path(args.out).write_text(json.dumps({"mode": args.mode, "runs": rows, "agg": agg}, indent=2), encoding="utf-8")
    print(json.dumps({"mode": args.mode, **{k: round(v["mean"], 3) for k, v in agg.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
