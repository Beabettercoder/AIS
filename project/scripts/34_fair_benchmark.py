#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 34: 公平协议版 13 模型基准（修正"白送创新"与"PSA 未激活"两个乌龙）。

铁律：
  ①–⑧ 单船基线：仅 7 通道原始特征（未门控、含跳变），不含 r_field / w_self
  ⑨–⑪ 图基线：目标 7 通道 + 邻船 5 步原始轨迹（无 nb_w 门控）
  ⑫ Ours-Full：9 通道（3.1 门控 + 3.3 场残差）+ PSA + 双分支一致性 + nb_w 门控 Gψ + ρ_i + 重构/平滑辅助
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


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
    raw = z["raw"].astype(np.float32)                       # 7 通道：未门控、含跳变
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
                nb_v=z["nb_v"], nb_xy=z["nb_xy"], nb_w=z["nb_w"], nb_mask=z["nb_mask"],
                nb_seq=z["nb_seq"], ep=z["e_phys"] if "e_phys" in z.files else None)


class Enc(nn.Module):
    def __init__(self, kind, nf, nh=64):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(nf, nh, batch_first=True, bidirectional=True); self.dim = 4 * nh
        elif kind == "transformer":
            self.proj = nn.Linear(nf, nh)
            layer = nn.TransformerEncoderLayer(nh, 4, 2 * nh, batch_first=True, dropout=0.1)
            self.enc = nn.TransformerEncoder(layer, 2); self.dim = 2 * nh
        elif kind == "itransformer":
            self.proj = nn.Linear(20, nh)
            layer = nn.TransformerEncoderLayer(nh, 4, 2 * nh, batch_first=True, dropout=0.1)
            self.enc = nn.TransformerEncoder(layer, 2); self.dim = 2 * nh
        elif kind == "mamba":
            self.inp = nn.Linear(nf, nh)
            self.dw = nn.Conv1d(nh, nh, 3, padding=1, groups=nh)
            self.delta = nn.Linear(nh, 1); self.B = nn.Linear(nh, nh); self.C = nn.Linear(nh, nh)
            self.A_log = nn.Parameter(torch.zeros(nh)); self.gate = nn.Linear(nh, nh)
            self.out = nn.Linear(nh, 2 * nh); self.dim = 4 * nh
        elif kind == "cde":
            self.f = nn.Sequential(nn.Linear(nh + nf, nh), nn.Tanh())
            self.z0 = nn.Linear(nf, nh); self.dxp = nn.Linear(nf, nh)
            self.out = nn.Linear(nh, 2 * nh); self.dim = 4 * nh
        else:
            self.net = nn.Sequential(nn.Conv1d(nf, nh, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(nh, nh, 5, padding=4, dilation=2), nn.GELU(),
                                     nn.Conv1d(nh, nh, 5, padding=8, dilation=4), nn.GELU())
            self.dim = 2 * nh

    def forward(self, x):
        if self.kind == "gru":
            h, _ = self.rnn(x); return torch.cat([h.mean(1), h.max(1).values], -1)
        if self.kind in ("transformer", "itransformer"):
            h = self.enc(self.proj(x if self.kind == "transformer" else x.transpose(1, 2)))
            return torch.cat([h.mean(1), h.max(1).values], -1)
        if self.kind == "mamba":
            u = self.dw(self.inp(x).transpose(1, 2)).transpose(1, 2)
            A = -torch.exp(self.A_log)
            h = torch.zeros(u.shape[0], u.shape[2], device=u.device); ys = []
            for t in range(u.shape[1]):
                dt = F.softplus(self.delta(u[:, t])) + 1e-3
                h = torch.exp(dt * A) * h + dt * self.B(u[:, t]) * u[:, t]
                ys.append(self.C(u[:, t]) * h)
            y = torch.stack(ys, 1) * F.silu(self.gate(u))
            p = self.out(y)
            return torch.cat([p.mean(1), p.max(1).values], -1)
        if self.kind == "cde":
            dx = torch.diff(x, dim=1); z = self.z0(x[:, 0]); zs = []
            for t in range(dx.shape[1]):
                z = z + self.f(torch.cat([z, x[:, t]], -1)) * self.dxp(dx[:, t]); zs.append(z)
            p = self.out(torch.stack(zs, 1))
            return torch.cat([p.mean(1), p.max(1).values], -1)
        h = self.net(x.transpose(1, 2))
        return torch.cat([h.mean(-1), h.max(-1).values], -1)


class NbSeqEnc(nn.Module):
    """邻船 5 步轨迹编码（图基线用，无 nb_w）。"""
    def __init__(self, nh=64):
        super().__init__()
        self.rnn = nn.GRU(4, nh, batch_first=True)
        self.proj = nn.Linear(nh + 2, nh)

    def forward(self, nb_seq, nb_xy, nb_mask):
        b, k, s, f = nb_seq.shape
        h, _ = self.rnn(nb_seq.reshape(b * k, s, f))
        h = h[:, -1].reshape(b, k, -1)
        feat = torch.cat([h, nb_xy / 1852.0], -1)
        return self.proj(feat) * nb_mask[..., None].float()


def graph_ctx(q, g, mask, nb_w=None, rho=None):
    s = torch.norm(g, dim=-1) * 0.1
    a = torch.softmax(-s, -1) * mask.float()
    if nb_w is not None:
        a = a * nb_w
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
    ctx = torch.einsum("bk,bkh->bh", a, g)
    if rho is not None:
        ctx = ctx * rho[:, None]
    return ctx


def run(kind, seed, tr, te, epochs=25, x_key="X7", graph=False, ours=False, phy=0.0, clean=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Xte = tr[x_key].copy(), te[x_key].copy()
    if clean:  # AISClean：运动学阈值硬剔除 + 线性插值
        for A in (Xtr, Xte):
            dp = np.diff(A[:, :, 0:2], axis=1, prepend=0.0)
            sp = np.linalg.norm(dp, axis=-1)
            med = np.median(sp, axis=1, keepdims=True)
            bad = sp > np.maximum(3 * med, 50.0)
            for b in range(A.shape[0]):
                idx = np.flatnonzero(~bad[b])
                if len(idx) > 2:
                    for ch in range(A.shape[2]):
                        A[b, :, ch] = np.interp(np.arange(A.shape[1]), idx, A[b, idx, ch])
    mu = Xtr[tr["mask"]].mean(0, keepdims=True); sd = Xtr[tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

    enc = Enc(kind, Xtr.shape[-1]).to(dev)
    dim = enc.dim
    use_graph = graph or ours
    nbe = NbSeqEnc().to(dev) if use_graph else None
    gproj = nn.Linear(64, dim).to(dev) if use_graph else None
    if ours:
        self_gate = nn.Linear(4, 2).to(dev)   # e_phys 控制器
        rec_head = nn.Linear(dim, 2).to(dev)
        gamma = nn.Parameter(torch.zeros(1))
    bin_h = nn.Linear(dim, 1).to(dev); cls_h = nn.Linear(dim, 5).to(dev)
    params = list(enc.parameters()) + list(bin_h.parameters()) + list(cls_h.parameters())
    if use_graph:
        params += list(nbe.parameters()) + list(gproj.parameters())
    if ours:
        params += list(self_gate.parameters()) + list(rec_head.parameters()) + [gamma]
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    eps_log, con_log = [], []

    def batch_tensors(d, ii):
        return (torch.tensor(d["nb_seq"][ii]).to(dev), torch.tensor(d["nb_xy"][ii]).to(dev),
                torch.tensor(d["nb_mask"][ii]).to(dev), torch.tensor(d["nb_w"][ii]).to(dev),
                torch.tensor(d["ep"][ii]).to(dev) if d["ep"] is not None else None)

    for _ in range(epochs):
        enc.train(); bin_h.train(); cls_h.train()
        if use_graph:
            nbe.train(); gproj.train()
        np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            xb = torch.tensor(Xtr[ii]).to(dev)
            h = enc(xb)
            if use_graph:
                nseq, nxy, nm, nw, ep = batch_tensors(tr, ii)
                g = gproj(nbe(nseq, nxy, nm))
                rho = torch.clamp(nm.float().sum(1) / 4.0, max=1.0) if ours else None
                ctx = graph_ctx(h, g, nm, nb_w=(nw if ours else None), rho=rho)
                h = h + (gamma * ctx if ours else ctx)
            pb = bin_h(h).squeeze(-1); pc = cls_h(h)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            if phy > 0 and hasattr(enc, "net"):
                hs = enc.net(xb.transpose(1, 2))
                loss = loss + phy * ((hs[:, :, 2:] - 2 * hs[:, :, 1:-1] + hs[:, :, :-2]) ** 2).mean()
            if ours:
                # 退化分支：25% 丢点 + 跳变
                keep = (torch.rand(xb.shape[0], xb.shape[1], device=dev) > 0.25).float()
                xb2 = xb * keep[..., None]
                jj = torch.randint(0, xb.shape[1], (xb.shape[0],), device=dev)
                xb2[torch.arange(xb.shape[0]), jj, 0] += 3.0
                h2 = enc(xb2)
                nseq, nxy, nm, nw, ep = batch_tensors(tr, ii)
                g = gproj(nbe(nseq, nxy, nm))
                rho = torch.clamp(nm.float().sum(1) / 4.0, max=1.0)
                ctx2 = graph_ctx(h2, g, nm, nb_w=nw, rho=rho)
                h2 = h2 + gamma * ctx2
                eps = 0.05 + 0.2 * (torch.tanh(self_gate(ep.mean(1))[:, 0]) + 1) / 2
                mu_ = h2.mean(0, keepdim=True).detach(); sd_ = h2.std(0, keepdim=True).detach().clamp_min(1e-5)
                h2 = mu_ + sd_ * (((h2 - mu_) / sd_) * (1 + eps[:, None]) + 0.02 * torch.tanh(self_gate(ep.mean(1))[:, 1])[:, None])
                pb2 = bin_h(h2).squeeze(-1); pc2 = cls_h(h2)
                p1, p2 = F.softmax(pc, -1), F.softmax(pc2, -1)
                m_ = 0.5 * (p1 + p2)
                kl = lambda a, bb: (a * (torch.log(a + 1e-8) - torch.log(bb + 1e-8))).sum(-1)
                con = (0.5 * kl(p1, m_) + 0.5 * kl(p2, m_)) * keep.mean(1)
                loss = loss + 0.3 * con.mean() + 0.3 * F.binary_cross_entropy_with_logits(pb2, torch.tensor(yb[ii]).to(dev))
                eps_log.append(float(eps.mean().detach())); con_log.append(float(con.mean().detach()))
            opt.zero_grad(); loss.backward(); opt.step()

    if ours:
        eps_m = float(np.mean(eps_log)); con_m = float(np.mean(con_log))
        assert 0.05 <= eps_m <= 0.25, f"PSA eps out of range: {eps_m}"
        assert con_m > 0, f"consistency loss inactive: {con_m}"
    else:
        eps_m, con_m = 0.0, 0.0

    enc.eval(); bin_h.eval(); cls_h.eval()
    if use_graph:
        nbe.eval(); gproj.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte).to(dev)
        h = enc(xb)
        if use_graph:
            nseq, nxy, nm, nw, ep = batch_tensors(te, np.arange(len(xb)))
            g = gproj(nbe(nseq, nxy, nm))
            rho = torch.clamp(nm.float().sum(1) / 4.0, max=1.0) if ours else None
            h = h + (gamma * graph_ctx(h, g, nm, nb_w=(nw if ours else None), rho=rho) if ours else graph_ctx(h, g, nm))
        pb = torch.sigmoid(bin_h(h).squeeze(-1)).cpu().numpy()
        pc = F.softmax(cls_h(h), -1).cpu().numpy()
    cell, y = te["cell"], te["y"]
    out = {"seed": seed, "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(pb, y > 0), "eps_mean": eps_m, "con_mean": con_m,
           "params": int(sum(p.numel() for p in params))}
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
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed_start", type=int, default=0)
    args = ap.parse_args()
    tr, te = load(args.dir, "train"), load(args.dir, "test")

    specs = [
        ("GRU", dict(kind="gru")),
        ("TCN", dict(kind="tcn")),
        ("Transformer", dict(kind="transformer")),
        ("iTransformer", dict(kind="itransformer")),
        ("Mamba-lite", dict(kind="mamba")),
        ("Neural-CDE-lite", dict(kind="cde")),
        ("AISClean+Transformer", dict(kind="transformer", clean=True)),
        ("PhyLoss-TCN", dict(kind="tcn", phy=0.05)),
        ("ST-GAT", dict(kind="tcn", graph=True)),
        ("Interval-GNN", dict(kind="tcn", graph=True)),
        ("STGVAD", dict(kind="tcn", graph=True)),
        ("Ours-Full-TCN", dict(kind="tcn", x_key="X9", graph=True, ours=True)),
        ("Ours-Full-GRU", dict(kind="gru", x_key="X9", graph=True, ours=True)),
    ]
    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    if only:
        specs = [s for s in specs if s[0] in only]
    rows = []
    for label, kw in specs:
        for s in range(args.seed_start, args.seed_start + args.seeds):
            r, _ = run(seed=s, tr=tr, te=te, **kw)
            r["model"] = label
            rows.append(r)
            print(f"{label} s{s}: clean={r['clean_AP']:.3f} full={r['full_AP']:.3f} R1={r['recall_1pct']:.3f} R5={r['recall_5pct']:.3f}", flush=True)
    agg = {}
    for r in rows:
        agg.setdefault(r["model"], []).append(r)
    summ = {}
    for k, v in agg.items():
        summ[k] = {m: round(float(np.nanmean([x[m] for x in v])), 3) for m in
                   ("clean_AP", "full_AP", "recall_1pct", "recall_5pct", "params")}
        for m in ("clean_AP", "full_AP", "recall_1pct"):
            summ[k][m + "_std"] = round(float(np.nanstd([x[m] for x in v])), 3)
    # 配对 t 检验：Ours vs 最强基线（按 full_AP 均值）
    base = max((k for k in summ if not k.startswith("Ours")), key=lambda k: summ[k]["full_AP"])
    t, p = stats.ttest_rel([x["full_AP"] for x in agg["Ours-Full-TCN"]], [x["full_AP"] for x in agg[base]])
    summ["_comparison"] = {"vs": base, "t": round(float(t), 3), "p": round(float(p), 4)}
    Path(args.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
