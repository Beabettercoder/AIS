#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 46: ✗/✓ 渐进组件消融（Baseline -> +KG -> +LF -> +PSA -> +PG/CDG）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    return dict(raw=np.nan_to_num(raw), gated=np.nan_to_num(gated), w=w,
                rf=z["r_field"], mask=z["mask"], y=z["label_class"], cell=z["label_cell"],
                nb_seq=z["nb_seq"].astype(np.float32), nb_xy=z["nb_xy"].astype(np.float32),
                nb_w=z["nb_w"].astype(np.float32), nb_mask=z["nb_mask"].astype(bool))


class Net(nn.Module):
    def __init__(self, nf, use_graph):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(nf, 64, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(64, 64, 5, padding=4, dilation=2), nn.GELU(),
                                 nn.Conv1d(64, 64, 5, padding=8, dilation=4), nn.GELU())
        self.use_graph = use_graph
        self.bin = nn.Linear(128, 1)
        self.cls = nn.Linear(128, 5)
        self.ctrl = nn.Linear(4, 2)
        if use_graph:
            self.rnn = nn.GRU(4, 64, batch_first=True)
            self.gproj = nn.Linear(66, 128)
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.beta = nn.Parameter(torch.tensor(1.0))
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, nseq=None, nxy=None, nm=None, nw=None, ri=None):
        h = self.net(x.transpose(1, 2))
        h = torch.cat([h.mean(-1), h.max(-1).values], -1)
        if self.use_graph:
            b, k, s, f = nseq.shape
            g, _ = self.rnn(nseq.reshape(b * k, s, f))
            g = self.gproj(torch.cat([g[:, -1].reshape(b, k, -1), nxy / 1852.0], -1))
            g = g * nm[..., None].float()
            d = torch.norm(g, dim=-1) * 0.1
            score = -(torch.tanh(self.alpha * d) + torch.tanh(self.beta * torch.abs(ri[:, None] - d)))
            a = torch.softmax(score, -1) * nm.float() * nw
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
            rho = torch.clamp(nm.float().sum(1) / 4.0, max=1.0)
            h = h + self.gamma * torch.einsum("bk,bkh->bh", a, g) * rho[:, None]
        return self.bin(h).squeeze(-1), self.cls(h), h


def run(seed, tr, te, use_kg, use_lf, use_psa, use_graph, epochs=25):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 公平协议：Baseline 严格 7 通道原始（无门控、无 w、无 r_field）
    def build(d):
        if not use_kg:
            return d["raw"]
        parts = [d["gated"]]
        if use_lf:
            parts.append(d["rf"][..., None])
        parts.append(d["w"][..., None])
        return np.concatenate(parts, -1)
    Xtr = build(tr)
    Xte = build(te)
    mu = Xtr[tr["mask"]].mean(0, keepdims=True); sd = Xtr[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

    model = Net(Xtr.shape[-1], use_graph).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)

    def T(d, ii):
        return (torch.tensor(d["nb_seq"][ii]).to(dev), torch.tensor(d["nb_xy"][ii]).to(dev),
                torch.tensor(d["nb_mask"][ii]).to(dev), torch.tensor(d["nb_w"][ii]).to(dev),
                torch.tensor(d["rf"][ii][:, -1]).to(dev))

    for _ in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            xb = torch.tensor(Xtr[ii]).to(dev)
            if use_graph:
                ns, nx, nm, nw, ri = T(tr, ii)
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
                jj = torch.randint(0, xb.shape[1], (xb.shape[0],), device=dev)
                xb2[torch.arange(xb.shape[0]), jj, 0] += 2.0
                if use_graph:
                    pb2, pc2, h2 = model(xb2, ns, nx, nm, nw, ri)
                else:
                    pb2, pc2, h2 = model(xb2)
                m_ = h2.mean(0, keepdim=True).detach(); s_ = h2.std(0, keepdim=True).detach().clamp_min(1e-5)
                h2 = m_ + s_ * (((h2 - m_) / s_) * 1.1 + 0.05)
                pb2b = model.bin(h2).squeeze(-1); pc2b = model.cls(h2)
                p1, p2 = F.softmax(pc, -1), F.softmax(pc2b, -1)
                mm = 0.5 * (p1 + p2)
                kl = lambda a, bb: (a * (torch.log(a + 1e-8) - torch.log(bb + 1e-8))).sum(-1)
                con = (0.5 * kl(p1, mm) + 0.5 * kl(p2, mm)) * keep.mean(1)
                loss = loss + 0.20 * con.mean() + 0.3 * F.binary_cross_entropy_with_logits(pb2b, torch.tensor(yb[ii]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte).to(dev)
        if use_graph:
            ns, nx, nm, nw, ri = T(te, np.arange(len(xb)))
            pb, pc, _ = model(xb, ns, nx, nm, nw, ri)
        else:
            pb, pc, _ = model(xb)
        pb = torch.sigmoid(pb).cpu().numpy(); pc = F.softmax(pc, -1).cpu().numpy()
    cell, y = te["cell"], te["y"]
    out = {"seed": seed, "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(pb, y > 0)}
    for far in (1, 5):
        thr = np.quantile(pb[cell == 0], 1 - far / 100)
        out[f"recall_{far}pct"] = float((pb[cell == 2] > thr).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    tr, te = load(args.dir, "train"), load(args.dir, "test")
    configs = [("Baseline (Raw TCN)", False, False, False, False),
               ("+ KG (3.1 Gate)", True, False, False, False),
               ("+ KG + LF (3.3 Field)", True, True, False, False),
               ("+ KG + LF + PSA", True, True, True, False),
               ("PG-DRGN (Ours, Full)", True, True, True, True)]
    rows = []
    for name, kg, lf, psa, gh in configs:
        for s in range(args.seeds):
            r = run(s, tr, te, kg, lf, psa, gh)
            r["config"] = name
            rows.append(r)
            print(f"{name} s{s}: clean={r['clean_AP']:.3f} full={r['full_AP']:.3f} "
                  f"R1={r['recall_1pct']:.3f} R5={r['recall_5pct']:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
