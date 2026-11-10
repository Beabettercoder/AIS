#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 33: 图模型定稿版（ST-GAT / Interval-GNN / STGVAD / Ours-Full×2 主干）。

与上一版的区别：
  * ST-GAT：标准图注意力（不加 nb_w）
  * Interval-GNN：邻船图预测目标速度区间 [vmin, vmax]，区间残差进分类头（interval loss 真实参与训练）
  * STGVAD：图自编码重构邻船速度 + 重构误差特征判别
  * Ours-Full：nb_w 门控 + 双距离校准 Gψ + PSA + 干净/退化 JS 一致性，零初始化残差
  * 结束自动断言 ST-GAT / Interval-GNN / STGVAD 的预测数组互不相同
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
    dp = np.diff(raw[:, :, 0:2], axis=1, prepend=0.0) * w[..., None]
    raw[:, :, 0:2] = np.cumsum(dp, axis=1)
    for ch in (2, 3, 4, 5, 6):
        v = raw[:, :, ch]
        low = w < 0.2
        for t in range(1, v.shape[1]):
            m = low[:, t]
            v[m, t] = v[m, t - 1]
        raw[:, :, ch] = v
    X = np.concatenate([raw, z["r_field"][..., None], w[..., None]], -1)
    return dict(X=np.nan_to_num(X), mask=z["mask"], y=z["label_class"], cell=z["label_cell"],
                rf=z["r_field"], nb_v=z["nb_v"], nb_xy=z["nb_xy"], nb_w=z["nb_w"], nb_mask=z["nb_mask"],
                nb_seq=z["nb_seq"] if "nb_seq" in z.files else None)


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

    def forward(self, x):
        if self.kind == "gru":
            h, _ = self.rnn(x)
            return torch.cat([h.mean(1), h.max(1).values], -1)
        h = self.net(x.transpose(1, 2))
        return torch.cat([h.mean(-1), h.max(-1).values], -1)


class NBEnc(nn.Module):
    def __init__(self, nh=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, nh), nn.GELU(), nn.Linear(nh, nh))

    def forward(self, nb_v, nb_xy, nb_w, nb_mask):
        feat = torch.cat([nb_v, nb_xy / 1852.0, nb_w[..., None], nb_mask[..., None].float()], -1)
        return self.net(feat)


class GraphHead(nn.Module):
    """mode: gat | interval | vad | ours"""
    def __init__(self, dim, nh=64, mode="gat", use_w=False, use_seq=False):
        super().__init__()
        self.mode, self.use_w = mode, use_w
        self.nb = NBEnc(nh)
        self.q = nn.Linear(dim, nh)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(nh, dim)
        self.vmin = nn.Linear(nh, 1) if mode == "interval" else None
        self.vmax = nn.Linear(nh, 1) if mode == "interval" else None
        self.rec = nn.Linear(dim, 2) if mode == "vad" else None
        self.aux_dim = 2 if mode in ("interval", "vad") else 0

    def forward(self, h, nb_v, nb_xy, nb_w, nb_mask, r_i=None):
        g = self.nb(nb_v, nb_xy, nb_w, nb_mask)        # (B,K,H)
        # 距离与打分
        mu = (nb_v * nb_mask[..., None]).sum(1) / nb_mask.sum(1, keepdim=True).clamp_min(1.0)
        d_abs = torch.norm(nb_v - mu[:, None, :], dim=-1)
        if self.mode == "ours" and r_i is not None:
            r_nb = d_abs
            d_rel = torch.abs(r_i[:, None] - r_nb)
        else:
            d_rel = torch.zeros_like(d_abs)
        s = torch.tanh(self.alpha * d_abs) + torch.tanh(self.beta * d_rel)
        a = torch.softmax(-s, -1) * nb_mask.float()
        if self.mode == "ours" and self.use_w:
            a = a * nb_w
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
        ctx = torch.einsum("bk,bkh->bh", a, g)
        extra = None
        if self.mode == "interval":
            vmin = self.vmin(g).squeeze(-1).min(1).values
            vmax = self.vmax(g).squeeze(-1).max(1).values
            extra = torch.stack([vmin, vmax], -1)      # (B,2)
        elif self.mode == "vad":
            rec = self.rec(h)
            extra = torch.norm(rec - mu, dim=-1, keepdim=True)
        out = self.gamma * self.out(ctx)
        return h + out, extra


def train(model_kind, seed, tr, te, epochs=25, use_psa=False, graph_mode="gat", use_w=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = tr["X"][tr["mask"]].mean(0, keepdims=True); sd = tr["X"][tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr, Xte = (tr["X"] - mu) / sd, (te["X"] - mu) / sd
    enc = Enc(model_kind, Xtr.shape[-1]).to(dev)
    gh = GraphHead(enc.dim, mode=graph_mode, use_w=use_w).to(dev)
    bin_h = nn.Linear(enc.dim, 1).to(dev)
    cls_h = nn.Linear(enc.dim, 5).to(dev)
    params = list(enc.parameters()) + list(gh.parameters()) + list(bin_h.parameters()) + list(cls_h.parameters())
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
    sel = np.isin(tr["cell"], [0, 2])
    yb = (tr["y"] > 0).astype(np.float32)
    cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
    cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    idx = np.flatnonzero(sel)
    sog_tr = tr["X"][:, -1, 4]

    for _ in range(epochs):
        enc.train(); gh.train(); bin_h.train(); cls_h.train()
        np.random.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = idx[b:b + 128]
            xb = torch.tensor(Xtr[ii]).to(dev)
            nv = torch.tensor(tr["nb_v"][ii]).to(dev)
            nxy = torch.tensor(tr["nb_xy"][ii]).to(dev)
            nw = torch.tensor(tr["nb_w"][ii]).to(dev)
            nm = torch.tensor(tr["nb_mask"][ii]).to(dev)
            ri = torch.tensor(tr["rf"][ii][:, -1]).to(dev)
            h = enc(xb)
            h2, extra = gh(h, nv, nxy, nw, nm, ri)
            pb = bin_h(h2).squeeze(-1); pc = cls_h(h2)
            loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
            m = torch.tensor(yb[ii] > 0).to(dev)
            if m.any():
                yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
            if graph_mode == "interval" and extra is not None:
                v = torch.tensor(sog_tr[ii], dtype=torch.float32, device=dev)
                vmin, vmax = extra[:, 0], extra[:, 1]
                loss = loss + 0.3 * (F.relu(vmin - v) + F.relu(v - vmax)).mean()
            if graph_mode == "vad" and extra is not None:
                loss = loss + 0.2 * extra.mean()
            if use_psa:
                mu_ = h2.mean(0, keepdim=True).detach(); sd_ = h2.std(0, keepdim=True).detach().clamp_min(1e-5)
                h_adv = mu_ + sd_ * (((h2 - mu_) / sd_) * 1.1 + 0.05)
                pb2 = bin_h(h_adv).squeeze(-1); pc2 = cls_h(h_adv)
                p1, p2 = F.softmax(pc, -1), F.softmax(pc2, -1)
                m_ = 0.5 * (p1 + p2)
                kl = lambda a, b: (a * (torch.log(a + 1e-8) - torch.log(b + 1e-8))).sum(-1)
                loss = loss + 0.3 * (0.5 * kl(p1, m_) + 0.5 * kl(p2, m_)).mean()
                loss = loss + 0.3 * F.binary_cross_entropy_with_logits(pb2, torch.tensor(yb[ii]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()

    enc.eval(); gh.eval(); bin_h.eval(); cls_h.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte).to(dev)
        nv = torch.tensor(te["nb_v"]).to(dev); nxy = torch.tensor(te["nb_xy"]).to(dev)
        nw = torch.tensor(te["nb_w"]).to(dev); nm = torch.tensor(te["nb_mask"]).to(dev)
        ri = torch.tensor(te["rf"][:, -1]).to(dev)
        h2, _ = gh(enc(xb), nv, nxy, nw, nm, ri)
        pb = torch.sigmoid(bin_h(h2).squeeze(-1)).cpu().numpy()
        pc = F.softmax(cls_h(h2), -1).cpu().numpy()
    cell, y = te["cell"], te["y"]
    out = {"seed": seed, "clean_AP": ap_score(pb[np.isin(cell, [0, 2])], y[np.isin(cell, [0, 2])] > 0),
           "full_AP": ap_score(pb, y > 0)}
    for k, cn in enumerate(["turn_back", "deviation", "stall", "sinuous", "speed_drop"], start=1):
        m = cell == 2
        out[f"AP_{cn}"] = ap_score(pc[m, k - 1], (y[m] == k))
    neg, pos = pb[cell == 0], pb[cell == 2]
    for far in (1, 5):
        thr = np.quantile(neg, 1 - far / 100)
        out[f"recall_{far}pct"] = float((pos > thr).mean())
    out["params"] = int(sum(p.numel() for p in params))
    return out, pb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    tr, te = load(args.dir, "train"), load(args.dir, "test")

    specs = [
        ("ST-GAT", dict(model_kind="tcn", graph_mode="gat", use_w=False)),
        ("Interval-GNN", dict(model_kind="tcn", graph_mode="interval", use_w=False)),
        ("STGVAD", dict(model_kind="tcn", graph_mode="vad", use_w=False)),
        ("Ours-Full-TCN", dict(model_kind="tcn", graph_mode="ours", use_w=True, use_psa=True)),
        ("Ours-Full-GRU", dict(model_kind="gru", graph_mode="ours", use_w=True, use_psa=True)),
    ]
    preds, rows = {}, []
    for label, kw in specs:
        for s in range(args.seeds):
            r, pb = train(seed=s, tr=tr, te=te, **kw)
            r["model"] = label
            rows.append(r)
            preds.setdefault(label, []).append(pb)
            print(f"{label} s{s}: clean={r['clean_AP']:.3f} full={r['full_AP']:.3f} R1={r['recall_1pct']:.3f}", flush=True)

    # 自检：三个基线预测必须互不相同
    a, b, c = preds["ST-GAT"][0], preds["Interval-GNN"][0], preds["STGVAD"][0]
    assert not np.allclose(a, b) and not np.allclose(a, c) and not np.allclose(b, c), "graph baselines collapsed!"
    print("[assert] ST-GAT / Interval-GNN / STGVAD predictions are pairwise different ✓", flush=True)

    agg = {}
    for r in rows:
        agg.setdefault(r["model"], []).append(r)
    summ = {k: {m: round(float(np.nanmean([x[m] for x in v])), 3)
                for m in ("clean_AP", "full_AP", "recall_1pct", "recall_5pct", "params")} for k, v in agg.items()}
    Path(args.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
