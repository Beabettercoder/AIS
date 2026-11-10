#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 14: Phase 3 —— e_phys 控制器 + PSA + 双分支一致性 + 绝对/相对双表征融合。

消融：
  --abl4 0/1   是否启用 PSA + 退化分支 + 一致性
  --abl5 0/1   0 = 单路（raw+r_field+w 拼成 9 通道）；1 = 绝对/相对双路 + 自适应门控
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
    rel = np.stack([z["r_field"], w], -1).astype(np.float32)
    ep = z["e_phys"].astype(np.float32)
    return dict(raw=raw, rel=rel, ep=ep, w=w, mask=z["mask"],
                y=z["label_class"], cell=z["label_cell"], delay=z["label_delay"])


class Branch(nn.Module):
    def __init__(self, nf, nh=64):
        super().__init__()
        self.c1 = nn.Conv1d(nf, nh, 5, padding=2)
        self.c2 = nn.Conv1d(nh, nh, 5, padding=4, dilation=2)
        self.c3 = nn.Conv1d(nh, nh, 5, padding=8, dilation=4)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.gelu(self.c1(h)); h = F.gelu(self.c2(h)); return F.gelu(self.c3(h))


class Net(nn.Module):
    def __init__(self, dual, nh=64):
        super().__init__()
        self.dual = dual
        self.abs_b = Branch(7, nh)
        self.rel_b = Branch(2, nh) if dual else Branch(9, nh)
        self.gate = nn.Linear(2 * nh, 1) if dual else None
        self.ctrl = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 2), nn.Tanh())
        self.bin = nn.Linear(2 * nh, 1)
        self.cls = nn.Linear(2 * nh, 5)

    def encode(self, raw, rel, ep, use_psa):
        ha = self.abs_b(raw)
        hr = self.rel_b(rel if self.dual else torch.cat([raw, rel], -1))
        if self.dual:
            g = torch.sigmoid(self.gate(torch.cat([ha.mean(-1), hr.mean(-1)], -1)))[..., None]
            h = g * ha + (1 - g) * hr
        else:
            h = hr
        eps_mean = 0.0
        if use_psa:
            mu = h.mean(2, keepdim=True).detach()
            sd = h.std(2, keepdim=True).detach().clamp_min(1e-5)
            eps = self.ctrl(ep.mean(1))
            a = (EPS_MIN + (EPS_MAX - EPS_MIN) * (eps[:, 0:1] + 1) / 2)[..., None]
            b = (eps[:, 1:2] * EPS_MAX)[..., None]
            h = mu + sd * (((h - mu) / sd) * (1 + a) + b)
            eps_mean = float(a.mean().detach())
        p = torch.cat([h.mean(-1), h.max(-1).values], -1)
        return p, eps_mean

    def forward(self, raw, rel, ep, use_psa=False):
        p, e = self.encode(raw, rel, ep, use_psa)
        return self.bin(p).squeeze(-1), self.cls(p), e


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + 1e-8) - torch.log(b + 1e-8))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run(cfg, seed, tr, te, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = tr["raw"][tr["mask"]].mean(0, keepdims=True); sd = tr["raw"][tr["mask"]].std(0, keepdims=True) + 1e-6
    rmu, rsd = tr["rel"][tr["mask"]].mean(0, keepdims=True), tr["rel"][tr["mask"]].std(0, keepdims=True) + 1e-6
    emu = np.nanmean(tr["ep"], axis=(0, 1), keepdims=True); esd = np.nanstd(tr["ep"], axis=(0, 1), keepdims=True) + 1e-6

    def prep(d):
        raw = (d["raw"] - mu) / sd
        rel = (d["rel"] - rmu) / rsd
        ep = np.nan_to_num((d["ep"] - emu) / esd, nan=0.0)
        return torch.tensor(raw), torch.tensor(rel), torch.tensor(ep)

    Rtr, Ltr, Etr = prep(tr); Rte, Lte, Ete = prep(te)
    model = Net(bool(cfg["abl5"])).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    eps_log = []

    for _ in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            raw, rel, ep = Rtr[ii].to(dev), Ltr[ii].to(dev), Etr[ii].to(dev)
            pb, pc, _ = model(raw, rel, ep, use_psa=False)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            if cfg["abl4"]:
                keep = (torch.rand(raw.shape[0], raw.shape[1], device=dev) > 0.25).float()
                raw2 = raw * keep[..., None]
                j = torch.randint(0, raw.shape[1], (raw.shape[0],), device=dev)
                raw2[torch.arange(raw.shape[0]), j, 0] += 1500.0
                pb2, pc2, e = model(raw2, rel, ep, use_psa=True)
                eps_log.append(e)
                p1, p2 = F.softmax(pc, -1), F.softmax(pc2, -1)
                ret = keep.mean(1, keepdim=True)
                loss = loss + 0.3 * (js(p1, p2) * ret).mean() \
                            + 0.5 * F.binary_cross_entropy_with_logits(pb2, torch.tensor(yb[ii]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pb, pc, _ = model(Rte.to(dev), Lte.to(dev), Ete.to(dev), use_psa=False)
        pb = torch.sigmoid(pb).cpu().numpy(); pc = F.softmax(pc, -1).cpu().numpy()
    cell, y, dl = te["cell"], te["y"], te["delay"]
    out = {"abl4": cfg["abl4"], "abl5": cfg["abl5"], "seed": seed,
           "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(pb, y > 0)}
    for k in range(1, 6):
        m = cell == 2
        out[f"AP_{CLASSES[k]}"] = ap_score(pc[m, k - 1], (y[m] == k))
    neg, pos = pb[cell == 0], pb[cell == 2]
    for far in (1, 5):
        thr = np.quantile(neg, 1 - far / 100)
        out[f"recall_{far}pct"] = float((pos > thr).mean())
        for d0 in (180, 300, 600):
            m = (cell == 2) & (dl == d0)
            out[f"recall_{far}pct_d{d0//60}"] = float((pb[m] > thr).mean()) if m.any() else float("nan")
    out["eps_mean"] = float(np.mean(eps_log)) if eps_log else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--abl4", type=int, default=1)
    ap.add_argument("--abl5", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tr, te = load(args.dir, "train"), load(args.dir, "test")
    cfg = {"abl4": bool(args.abl4), "abl5": bool(args.abl5)}
    rows = [run(cfg, s, tr, te, args.epochs) for s in range(args.seeds)]
    keys = [k for k in rows[0] if k not in ("abl4", "abl5", "seed")]
    agg = {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows]))}
           for k in keys}
    Path(args.out).write_text(json.dumps({"cfg": cfg, "runs": rows, "agg": agg}, indent=2), encoding="utf-8")
    print(json.dumps({"abl4": args.abl4, "abl5": args.abl5,
                      **{k: round(v["mean"], 3) for k, v in agg.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
