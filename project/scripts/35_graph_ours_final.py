#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 35: 定稿版图模型（7 通道公平图基线 + 9 通道 Ours-Full），含 1-seed 断言。

⑨ ST-GAT / ⑩ Interval-GNN / ⑪ STGVAD：X7 + nb_seq，无 nb_w、无 r_field
⑫ Ours-Full：X9 + nb_w 门控 Gψ + PSA + 双分支一致性（gamma 注册为 Parameter）
"""
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
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def load(d, split):
    z = np.load(Path(d) / f"{split}.npz")
    raw = z["raw"].astype(np.float32)
    w = z["w_self"].astype(np.float32)
    gated = raw.copy()
    dp = np.diff(gated[:, :, 0:2], axis=1, prepend=0.0) * w[..., None]
    gated[:, :, 0:2] = np.cumsum(dp, axis=1)
    for ch in (2, 3, 4, 5, 6):
        v = gated[:, :, ch]
        low = w < 0.2
        for t in range(1, v.shape[1]):
            m = low[:, t]
            v[m, t] = v[m, t - 1]
        gated[:, :, ch] = v
    X9 = np.concatenate([gated, z["r_field"][..., None], w[..., None]], -1)
    return dict(X7=np.nan_to_num(raw.copy()), X9=np.nan_to_num(X9), mask=z["mask"],
                y=z["label_class"], cell=z["label_cell"], rf=z["r_field"],
                nb_seq=z["nb_seq"].astype(np.float32), nb_xy=z["nb_xy"].astype(np.float32),
                nb_v=z["nb_v"].astype(np.float32), nb_w=z["nb_w"].astype(np.float32), nb_mask=z["nb_mask"].astype(bool),
                ep=z["e_phys"].astype(np.float32))


class Enc(nn.Module):
    def __init__(self, kind, nf, nh=64):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(nf, nh, batch_first=True, bidirectional=True)
            self.dim = 4 * nh
        else:
            self.net = nn.Sequential(nn.Conv1d(nf, nh, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(nh, nh, 5, padding=4, dilation=2), nn.GELU(),
                                     nn.Conv1d(nh, nh, 5, padding=8, dilation=4), nn.GELU())
            self.dim = 2 * nh

    def seq(self, x):
        if self.kind == "gru":
            h, _ = self.rnn(x)
            return h.transpose(1, 2)
        return self.net(x.transpose(1, 2))

    def forward(self, x):
        h = self.seq(x)
        return torch.cat([h.mean(-1), h.max(-1).values], -1)


class NbEnc(nn.Module):
    def __init__(self, nh=64):
        super().__init__()
        self.rnn = nn.GRU(4, nh, batch_first=True)
        self.proj = nn.Linear(nh + 2, nh)

    def forward(self, nb_seq, nb_xy):
        b, k, s, f = nb_seq.shape
        h, _ = self.rnn(nb_seq.reshape(b * k, s, f))
        h = h[:, -1].reshape(b, k, -1)
        return self.proj(torch.cat([h, nb_xy / 1852.0], -1))


class Net(nn.Module):
    def __init__(self, kind, nf, mode, nh=64):
        super().__init__()
        self.mode = mode
        self.enc = Enc(kind, nf, nh)
        self.nb = NbEnc(nh)
        self.proj = nn.Linear(nh, self.enc.dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.bin = nn.Linear(self.enc.dim, 1)
        self.cls = nn.Linear(self.enc.dim, 5)
        self.ctrl = nn.Linear(4, 2)
        if mode == "ours":
            self.gamma = nn.Parameter(torch.zeros(1))
            self.rec = nn.Linear(self.enc.dim, 2)
        if mode == "interval":
            self.vmin = nn.Linear(nh, 1)
            self.vmax = nn.Linear(nh, 1)
        if mode == "vad":
            self.rec = nn.Linear(self.enc.dim, 2)

    def graph(self, h, nseq, nxy, nmask, nb_w, r_i):
        g = self.proj(self.nb(nseq, nxy)) * nmask[..., None].float()
        d_abs = torch.norm(g, dim=-1) * 0.1
        if self.mode == "ours" and r_i is not None:
            d_rel = torch.abs(r_i[:, None] - d_abs)
        else:
            d_rel = torch.zeros_like(d_abs)
        s = torch.tanh(self.alpha * d_abs) + torch.tanh(self.beta * d_rel)
        a = torch.softmax(-s, -1) * nmask.float()
        if self.mode == "ours" and nb_w is not None:
            a = a * nb_w
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
        ctx = torch.einsum("bk,bkh->bh", a, g)
        if self.mode == "ours":
            rho = torch.clamp(nmask.float().sum(1) / 4.0, max=1.0)
            return h + self.gamma * ctx * rho[:, None], ctx
        return h + ctx, ctx

    def forward(self, x, nseq, nxy, nmask, nb_w=None, r_i=None):
        h = self.enc(x)
        h, ctx = self.graph(h, nseq, nxy, nmask, nb_w, r_i)
        return self.bin(h).squeeze(-1), self.cls(h), h, ctx


def run(mode, kind, seed, tr, te, epochs=25, use_ours=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xk = "X9" if use_ours else "X7"
    mu = tr[xk][tr["mask"]].mean(0, keepdims=True); sd = tr[xk][tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = (tr[xk] - mu) / sd; Xte = (te[xk] - mu) / sd
    model = Net(kind, Xtr.shape[-1], mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    eps_log, con_log = [], []

    def tens(d, ii):
        return (torch.tensor(d["nb_seq"][ii]).to(dev), torch.tensor(d["nb_xy"][ii]).to(dev),
                torch.tensor(d["nb_mask"][ii]).to(dev), torch.tensor(d["nb_w"][ii]).to(dev),
                torch.tensor(d["rf"][ii][:, -1]).to(dev), torch.tensor(d["ep"][ii]).to(dev),
                torch.tensor(d["nb_v"][ii]).to(dev))

    for _ in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            xb = torch.tensor(Xtr[ii]).to(dev)
            nseq, nxy, nm, nw, ri, ep, nv = tens(tr, ii)
            pb, pc, h, ctx = model(xb, nseq, nxy, nm, nw if use_ours else None, ri if use_ours else None)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            if mode == "interval":
                v = torch.tensor(tr["X9"][ii][:, -1, 4] * sd[0, 4] + mu[0, 4], device=dev)
                vmin = model.vmin(model.nb(nseq, nxy)).squeeze(-1).min(1).values
                vmax = model.vmax(model.nb(nseq, nxy)).squeeze(-1).max(1).values
                loss = loss + 0.3 * (F.relu(vmin - v) + F.relu(v - vmax)).mean()
            if mode == "vad":
                mv = (nv * nm[..., None]).sum(1) / nm.sum(1, keepdim=True).clamp_min(1.0)
                loss = loss + 0.1 * F.mse_loss(model.rec(h), mv)
            if use_ours:
                keep = (torch.rand(xb.shape[0], xb.shape[1], device=dev) > 0.15).float()
                xb2 = xb * keep[..., None]
                jj = torch.randint(0, xb.shape[1], (xb.shape[0],), device=dev)
                xb2[torch.arange(xb.shape[0]), jj, 0] += 2.0
                pb2, pc2, h2, _ = model(xb2, nseq, nxy, nm, nw, ri)
                e = model.ctrl(ep.mean(1))
                eps = 0.05 + 0.2 * (torch.tanh(e[:, 0]) + 1) / 2
                m_ = h2.mean(0, keepdim=True).detach(); s_ = h2.std(0, keepdim=True).detach().clamp_min(1e-5)
                h2p = m_ + s_ * (((h2 - m_) / s_) * (1 + eps[:, None]) + 0.02 * torch.tanh(e[:, 1])[:, None])
                pb2b = model.bin(h2p).squeeze(-1); pc2b = model.cls(h2p)
                p1, p2 = F.softmax(pc, -1), F.softmax(pc2b, -1)
                mm = 0.5 * (p1 + p2)
                kl = lambda a, bb: (a * (torch.log(a + 1e-8) - torch.log(bb + 1e-8))).sum(-1)
                con = (0.5 * kl(p1, mm) + 0.5 * kl(p2, mm)) * keep.mean(1)
                loss = loss + 0.20 * con.mean() + 0.3 * F.binary_cross_entropy_with_logits(pb2b, torch.tensor(yb[ii]).to(dev))
                hs = model.enc.seq(xb)
                if hs.shape[-1] > 3:
                    loss = loss + 0.01 * ((hs[:, :, 2:] - 2 * hs[:, :, 1:-1] + hs[:, :, :-2]) ** 2).mean()
                eps_log.append(float(eps.mean().detach())); con_log.append(float(con.mean().detach()))
            opt.zero_grad(); loss.backward(); opt.step()

    if use_ours:
        eps_m, con_m = float(np.mean(eps_log)), float(np.mean(con_log))
        assert 0.05 <= eps_m <= 0.25, f"eps out of range {eps_m}"
        assert con_m > 0, f"consistency inactive {con_m}"
    else:
        eps_m = con_m = 0.0

    model.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte).to(dev)
        nseq, nxy, nm, nw, ri, ep, nv = tens(te, np.arange(len(xb)))
        pb, pc, _, _ = model(xb, nseq, nxy, nm, nw if use_ours else None, ri if use_ours else None)
        pb = torch.sigmoid(pb).cpu().numpy(); pc = F.softmax(pc, -1).cpu().numpy()
    cell, y = te["cell"], te["y"]
    out = {"seed": seed, "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(pb, y > 0), "eps_mean": eps_m, "con_mean": con_m,
           "params": int(sum(p.numel() for p in model.parameters()))}
    for k, cn in enumerate(["turn_back", "deviation", "stall", "sinuous", "speed_drop"], start=1):
        m = cell == 2
        out[f"AP_{cn}"] = ap_score(pc[m, k - 1], (y[m] == k))
    neg, pos = pb[cell == 0], pb[cell == 2]
    for far in (1, 5):
        thr = np.quantile(neg, 1 - far / 100)
        out[f"recall_{far}pct"] = float((pos > thr).mean())
    return out, pb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed_start", type=int, default=0)
    args = ap.parse_args()
    tr, te = load(args.dir, "train"), load(args.dir, "test")
    specs = [("ST-GAT", "gat", "tcn", False), ("Interval-GNN", "interval", "tcn", False),
             ("STGVAD", "vad", "tcn", False), ("Ours-Full-TCN", "ours", "tcn", True),
             ("Ours-Full-GRU", "ours", "gru", True)]
    rows, preds = [], {}
    for label, mode, kind, ours in specs:
        for s in range(args.seed_start, args.seed_start + args.seeds):
            r, pb = run(mode, kind, s, tr, te, use_ours=ours)
            r["model"] = label
            rows.append(r); preds.setdefault(label, []).append(pb)
            print(f"{label} s{s}: clean={r['clean_AP']:.3f} full={r['full_AP']:.3f} R1={r['recall_1pct']:.3f} eps={r['eps_mean']:.3f} con={r['con_mean']:.4f}", flush=True)
    a, b, c = preds["ST-GAT"][0], preds["Interval-GNN"][0], preds["STGVAD"][0]
    assert not np.allclose(a, b) and not np.allclose(a, c) and not np.allclose(b, c), "graph baselines identical!"
    print("[assert] three graph baselines are pairwise different ✓", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
