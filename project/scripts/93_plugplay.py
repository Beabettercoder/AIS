#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 93: 通用物理外挂实验——8 种骨干 × {Vanilla 7ch, +KG+LF+PSA}（3 seeds）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
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
    return dict(raw=np.nan_to_num(raw), gated=np.nan_to_num(gated), w=w, rf=z["r_field"],
                mask=z["mask"], y=z["label_class"], cell=z["label_cell"],
                nb_v=z["nb_v"].astype(np.float32), nb_xy=z["nb_xy"].astype(np.float32),
                nb_mask=z["nb_mask"].astype(bool), nb_w=z["nb_w"].astype(np.float32),
                nb_seq=z["nb_seq"].astype(np.float32))


class Enc(nn.Module):
    def __init__(self, kind, nf, nh=64):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(nf, nh, batch_first=True, bidirectional=True); self.dim = 4 * nh
        elif kind == "transformer":
            self.proj = nn.Linear(nf, nh)
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(nh, 4, 2 * nh, batch_first=True, dropout=0.1), 2); self.dim = 2 * nh
        elif kind == "itransformer":
            self.proj = nn.Linear(20, nh)
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(nh, 4, 2 * nh, batch_first=True, dropout=0.1), 2); self.dim = 2 * nh
        elif kind == "patchtst":
            self.patch = nn.Conv1d(nf, nh, 5, stride=3, padding=2)
            self.proj = nn.Linear(nh, nh)
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(nh, 4, 2 * nh, batch_first=True, dropout=0.1), 2); self.dim = 2 * nh
        elif kind == "mamba":
            self.inp = nn.Linear(nf, nh); self.dw = nn.Conv1d(nh, nh, 3, padding=1, groups=nh)
            self.delta = nn.Linear(nh, 1); self.B = nn.Linear(nh, nh); self.C = nn.Linear(nh, nh)
            self.A_log = nn.Parameter(torch.zeros(nh)); self.gate = nn.Linear(nh, nh)
            self.out = nn.Linear(nh, 2 * nh); self.dim = 4 * nh
        elif kind == "cde":
            self.f = nn.Sequential(nn.Linear(nh + nf, nh), nn.Tanh())
            self.z0 = nn.Linear(nf, nh); self.dxp = nn.Linear(nf, nh)
            self.out = nn.Linear(nh, 2 * nh); self.dim = 4 * nh
        else:  # tcn / stgat share the TCN trunk
            self.net = nn.Sequential(nn.Conv1d(nf, nh, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(nh, nh, 5, padding=4, dilation=2), nn.GELU(),
                                     nn.Conv1d(nh, nh, 5, padding=8, dilation=4), nn.GELU()); self.dim = 2 * nh

    def forward(self, x):
        if self.kind == "gru":
            h, _ = self.rnn(x); return torch.cat([h.mean(1), h.max(1).values], -1)
        if self.kind in ("transformer", "itransformer"):
            h = self.enc(self.proj(x if self.kind == "transformer" else x.transpose(1, 2)))
            return torch.cat([h.mean(1), h.max(1).values], -1)
        if self.kind == "patchtst":
            h = self.enc(self.proj(self.patch(x.transpose(1, 2)).transpose(1, 2)))
            return torch.cat([h.mean(1), h.max(1).values], -1)
        if self.kind == "mamba":
            u = self.dw(self.inp(x).transpose(1, 2)).transpose(1, 2)
            A = -torch.exp(self.A_log); h = torch.zeros(u.shape[0], u.shape[2], device=u.device); ys = []
            for t in range(u.shape[1]):
                dt = F.softplus(self.delta(u[:, t])) + 1e-3
                h = torch.exp(dt * A) * h + dt * self.B(u[:, t]) * u[:, t]
                ys.append(self.C(u[:, t]) * h)
            y = torch.stack(ys, 1) * F.silu(self.gate(u)); p = self.out(y)
            return torch.cat([p.mean(1), p.max(1).values], -1)
        if self.kind == "cde":
            dx = torch.diff(x, dim=1); z = self.z0(x[:, 0]); zs = []
            for t in range(dx.shape[1]):
                z = z + self.f(torch.cat([z, x[:, t]], -1)) * self.dxp(dx[:, t]); zs.append(z)
            p = self.out(torch.stack(zs, 1))
            return torch.cat([p.mean(1), p.max(1).values], -1)
        h = self.net(x.transpose(1, 2))
        return torch.cat([h.mean(-1), h.max(-1).values], -1)


class Net(nn.Module):
    def __init__(self, kind, nf, graph=False):
        super().__init__()
        self.enc = Enc(kind, nf); self.graph = graph
        self.bin = nn.Linear(self.enc.dim, 1); self.cls = nn.Linear(self.enc.dim, 5)
        if graph:
            self.nbrnn = nn.GRU(4, 32, batch_first=True)
            self.nbproj = nn.Linear(34, self.enc.dim)
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, nbfeat=None):
        h = self.enc(x)
        if self.graph and nbfeat is not None:
            ns, nxy, nm, nw = nbfeat
            b, k, s, f = ns.shape
            g, _ = self.nbrnn(ns.reshape(b * k, s, f))
            g = self.nbproj(torch.cat([g[:, -1].reshape(b, k, -1), nxy / 1852.0], -1)) * nm[..., None].float()
            a = nw / nw.sum(-1, keepdim=True).clamp_min(1e-6)
            h = h + self.gamma * torch.einsum("bk,bkh->bh", a * nm.float(), g)
        return self.bin(h).squeeze(-1), self.cls(h), h


def build_x(d, ours):
    if not ours:
        return d["raw"]
    return np.concatenate([d["gated"], d["rf"][..., None], d["w"][..., None]], -1)


def run(kind, seed, tr, te, ours, graph=False, epochs=20):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Xte = build_x(tr, ours), build_x(te, ours)
    mu = Xtr[tr["mask"]].mean(0, keepdims=True); sd = Xtr[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).astype(np.float32); Xte = ((Xte - mu) / sd).astype(np.float32)
    m = Net(kind, Xtr.shape[-1], graph=graph).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2]); yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)

    def nb(d, ii):
        return (torch.tensor(d["nb_seq"][ii]).to(dev), torch.tensor(d["nb_xy"][ii]).to(dev),
                torch.tensor(d["nb_mask"][ii]).to(dev), torch.tensor(d["nb_w"][ii]).to(dev))

    for _ in range(epochs):
        m.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 256):
            ii = idx[b:b + 256]
            xb = torch.tensor(Xtr[ii]).to(dev)
            nb_ = nb(tr, ii) if graph else None
            pb, pc, h = m(xb, nb_)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            mask = torch.tensor(yb[ii] > 0).to(dev)
            if mask.any():
                yy = torch.tensor(tr["y"][ii][mask.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[mask], yy, weight=cw)
            if ours:  # 4D-PSA + dual-branch consistency
                keep = (torch.rand(xb.shape[0], xb.shape[1], device=dev) > 0.15).float()
                xb2 = xb * keep[..., None]
                pb2, pc2, h2 = m(xb2, nb_)
                mu_ = h2.mean(0, keepdim=True).detach(); sd_ = h2.std(0, keepdim=True).detach().clamp_min(1e-5)
                h2p = mu_ + sd_ * (((h2 - mu_) / sd_) * 1.1 + 0.05)
                p1, p2 = F.softmax(pc, -1), F.softmax(m.cls(h2p), -1)
                mm = 0.5 * (p1 + p2)
                kl = lambda a, bb: (a * (torch.log(a + 1e-8) - torch.log(bb + 1e-8))).sum(-1)
                con = (0.5 * kl(p1, mm) + 0.5 * kl(p2, mm)) * keep.mean(1)
                loss = loss + 0.2 * con.mean()
            opt.zero_grad(); loss.backward(); opt.step()

    m.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte).to(dev)
        nb_ = nb(te, np.arange(len(xb))) if graph else None
        pb, pc, _ = m(xb, nb_)
        p = torch.sigmoid(pb).cpu().numpy(); pq = F.softmax(pc, -1).cpu().numpy()
    y, cell = te["y"], te["cell"]
    out = {"clean_AP": ap_score(p[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(p, y > 0)}
    for far in (1, 5):
        thr = np.quantile(p[cell == 0], 1 - far / 100)
        out[f"r{far}"] = float((p[cell == 2] > thr).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(BASE / "plugplay.json"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    tr = load(BASE / "eval_v2_graph2", "train"); te = load(BASE / "eval_v2_graph2", "test")
    specs = [("TCN", "tcn", False), ("GRU", "gru", False), ("Transformer", "transformer", False),
             ("iTransformer", "itransformer", False), ("PatchTST", "patchtst", False),
             ("Mamba-lite", "mamba", False), ("Neural-CDE", "cde", False), ("ST-GAT", "tcn", True)]
    if args.only:
        keep = set(args.only.split(","))
        specs = [s for s in specs if s[0] in keep]
    rows = []
    for name, kind, graph in specs:
        for ours in (False, True):
            for s in range(args.seeds):
                r = run(kind, s, tr, te, ours, graph=graph)
                r.update({"backbone": name, "ours": ours, "seed": s})
                rows.append(r)
                print(f"{name:14s} {'+Ours' if ours else 'Vanilla'} s{s}: clean={r['clean_AP']:.3f} "
                      f"full={r['full_AP']:.3f} R1={r['r1']:.3f} R5={r['r5']:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
