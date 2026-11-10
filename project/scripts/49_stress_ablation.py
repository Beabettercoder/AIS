#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 49: 消融 + 两个高难列（Stress-Test Full AP / 跨场景零样本均值）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ["ny", "la", "puget", "miami", "gom_open", "atl_open"]
BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")


def ap_score(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def load(d, split):
    z = np.load(Path(d) / f"{split}.npz")
    raw = z["raw"].astype(np.float32)
    w = z["w_self"].astype(np.float32)
    gated = raw.copy()
    dp = np.diff(gated[:, :, 0:2], axis=1, prepend=0.0) * w[..., None]
    gated[:, :, 0:2] = np.cumsum(dp, axis=1)
    for ch in (2, 3, 4, 5, 6):
        v = gated[:, :, ch]; low = w < 0.2
        for t in range(1, v.shape[1]):
            m = low[:, t]; v[m, t] = v[m, t - 1]
        gated[:, :, ch] = v
    n = len(raw)
    nb_seq = z["nb_seq"].astype(np.float32) if "nb_seq" in z.files else np.zeros((n, 32, 5, 4), np.float32)
    nb_xy = z["nb_xy"].astype(np.float32) if "nb_xy" in z.files else np.zeros((n, 32, 2), np.float32)
    nb_w = z["nb_w"].astype(np.float32) if "nb_w" in z.files else np.ones((n, 32), np.float32)
    nb_m = z["nb_mask"].astype(bool) if "nb_mask" in z.files else np.zeros((n, 32), bool)
    return dict(raw=np.nan_to_num(raw), gated=np.nan_to_num(gated), w=w, rf=z["r_field"],
                mask=z["mask"], y=z["label_class"], cell=z["label_cell"],
                nb_seq=nb_seq, nb_xy=nb_xy, nb_w=nb_w, nb_mask=nb_m)


class Net(nn.Module):
    def __init__(self, nf, use_graph):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(nf, 64, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(64, 64, 5, padding=4, dilation=2), nn.GELU(),
                                 nn.Conv1d(64, 64, 5, padding=8, dilation=4), nn.GELU())
        self.use_graph = use_graph
        self.bin = nn.Linear(128, 1); self.cls = nn.Linear(128, 5)
        if use_graph:
            self.rnn = nn.GRU(4, 64, batch_first=True)
            self.gproj = nn.Linear(66, 128)
            self.alpha = nn.Parameter(torch.tensor(1.0)); self.beta = nn.Parameter(torch.tensor(1.0))
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, nseq=None, nxy=None, nm=None, nw=None, ri=None):
        h = self.net(x.transpose(1, 2))
        h = torch.cat([h.mean(-1), h.max(-1).values], -1)
        if self.use_graph:
            b, k, s, f = nseq.shape
            g, _ = self.rnn(nseq.reshape(b * k, s, f))
            g = self.gproj(torch.cat([g[:, -1].reshape(b, k, -1), nxy / 1852.0], -1)) * nm[..., None].float()
            d = torch.norm(g, dim=-1) * 0.1
            sc = -(torch.tanh(self.alpha * d) + torch.tanh(self.beta * torch.abs(ri[:, None] - d)))
            a = torch.softmax(sc, -1) * nm.float() * nw
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
            rho = torch.clamp(nm.float().sum(1) / 4.0, max=1.0)
            h = h + self.gamma * torch.einsum("bk,bkh->bh", a, g) * rho[:, None]
        return self.bin(h).squeeze(-1), self.cls(h), h


def features(d, cfg, stats):
    kg, lf = cfg
    if not kg:
        X = d["raw"]
    else:
        parts = [d["gated"]]
        if lf:
            parts.append(d["rf"][..., None])
        parts.append(d["w"][..., None])
        X = np.concatenate(parts, -1)
    return (X - stats[0]) / stats[1]


def stress(x, rng):
    x = x.copy()
    B, T, C = x.shape
    drop = rng.random((B, T)) < 0.35
    x[drop] = 0.0
    drift = np.linspace(0, 1, T)[None, :, None] * 3.0
    x[:, :, 0:2] += drift
    return x


def eval_set(model, d, cfg, stats, dev, stress_on=False, seed=0):
    X = features(d, cfg, stats)
    if stress_on:
        X = stress(X, np.random.default_rng(seed))
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=dev)
        if model.use_graph:
            pb, pc, _ = model(xb, torch.tensor(d["nb_seq"], device=dev), torch.tensor(d["nb_xy"], device=dev),
                              torch.tensor(d["nb_mask"], device=dev), torch.tensor(d["nb_w"], device=dev),
                              torch.tensor(d["rf"][:, -1], device=dev))
        else:
            pb, pc, _ = model(xb)
        pb = torch.sigmoid(pb).cpu().numpy()
    return pb


def train_one(cfg, seed, tr, dev, epochs=25, use_psa=False, use_graph=False):
    torch.manual_seed(seed); np.random.seed(seed)
    X = features(tr, cfg, (0.0, 1.0))
    mu = X[tr["mask"]].mean(0, keepdims=True); sd = X[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd
    model = Net(X.shape[-1], use_graph).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    for _ in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            xb = torch.tensor(Xn[ii]).to(dev)
            if use_graph:
                ns = torch.tensor(tr["nb_seq"][ii]).to(dev); nx = torch.tensor(tr["nb_xy"][ii]).to(dev)
                nm = torch.tensor(tr["nb_mask"][ii]).to(dev); nw = torch.tensor(tr["nb_w"][ii]).to(dev)
                ri = torch.tensor(tr["rf"][ii][:, -1]).to(dev)
                pb, pc, h = model(xb, ns, nx, nm, nw, ri)
            else:
                pb, pc, h = model(xb)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            if use_psa:
                keep = (torch.rand(xb.shape[0], xb.shape[1], device=dev) > 0.15).float()
                xb2 = xb * keep[..., None]
                pb2, pc2, h2 = (model(xb2, ns, nx, nm, nw, ri) if use_graph else model(xb2))
                m_ = h2.mean(0, keepdim=True).detach(); s_ = h2.std(0, keepdim=True).detach().clamp_min(1e-5)
                h2 = m_ + s_ * (((h2 - m_) / s_) * 1.1 + 0.05)
                pb2b = model.bin(h2).squeeze(-1); pc2b = model.cls(h2)
                p1, p2 = F.softmax(pc, -1), F.softmax(pc2b, -1)
                mm = 0.5 * (p1 + p2)
                kl = lambda a, bb: (a * (torch.log(a + 1e-8) - torch.log(bb + 1e-8))).sum(-1)
                con = (0.5 * kl(p1, mm) + 0.5 * kl(p2, mm)) * keep.mean(1)
                loss = loss + 0.20 * con.mean() + 0.3 * F.binary_cross_entropy_with_logits(pb2b, torch.tensor(yb[ii]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    return model, (mu, sd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = load(BASE / "eval_v2_graph2", "train")
    te = load(BASE / "eval_v2_graph2", "test")
    tgts = {}
    for t in TARGETS:
        p = BASE / f"{t}_dataset"
        if (p / "test.npz").exists():
            tgts[t] = load(p, "test")

    configs = [("Baseline", (False, False), False, False),
               ("+KG", (True, False), False, False),
               ("+KG+LF", (True, True), False, False),
               ("+KG+LF+PSA", (True, True), True, False),
               ("PG-DRGN", (True, True), True, True)]
    rows = []
    for name, cfg, psa, graph in configs:
        for s in range(args.seeds):
            model, stats = train_one(cfg, s, tr, dev, use_psa=psa, use_graph=graph)
            model.eval()
            pb = eval_set(model, te, cfg, stats, dev)
            cell, y = te["cell"], te["y"]
            clean = ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0)
            full = ap_score(pb, y > 0)
            pbs = eval_set(model, te, cfg, stats, dev, stress_on=True, seed=s)
            stress_full = ap_score(pbs, y > 0)
            zs = []
            for t, d in tgts.items():
                pbz = eval_set(model, d, cfg, stats, dev)
                zs.append(ap_score(pbz, d["y"] > 0))
            r = {"config": name, "seed": s, "clean_AP": clean, "full_AP": full,
                 "stress_full": stress_full, "zs_mean": float(np.nanmean(zs))}
            rows.append(r)
            print(f"{name} s{s}: clean={clean:.3f} full={full:.3f} stress={stress_full:.3f} zs={r['zs_mean']:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
