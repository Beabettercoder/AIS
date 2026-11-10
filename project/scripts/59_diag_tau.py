#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 59: 诊断 τ / Top-K 失效原因：看 gamma、注意力分布与图贡献占比。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)

ck = torch.load(W / "ablation_4_full_cdg.pt", map_location="cpu", weights_only=False)
sd = ck["sd"]
print("gamma =", float(sd["gamma"]))
print("alpha =", float(sd["alpha"]), " beta =", float(sd["beta"]))
print("gproj weight norm =", float(sd["gproj.weight"].norm()))

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = sw.Net(ck["kind"], ck["nf"], use_graph=True).to(dev)
m.load_state_dict(sd); m.eval()
te = sw.load(BASE / "eval_v2_graph2", "test")
mu, sdv = ck["stats"]
X = (sw.build(te, True, True) - mu) / sdv
xb = torch.tensor(X, dtype=torch.float32, device=dev)
ns = torch.tensor(te["nb_seq"], device=dev); nx = torch.tensor(te["nb_xy"], device=dev)
nm = torch.tensor(te["nb_mask"], device=dev); nw = torch.tensor(te["nb_w"], device=dev)
ri = torch.tensor(te["rf"][:, -1], device=dev)

with torch.no_grad():
    h = m.enc(xb)
    b, k, s, f = ns.shape
    g, _ = m.rnn(ns.reshape(b * k, s, f))
    g = m.gproj(torch.cat([g[:, -1].reshape(b, k, -1), nx / 1852.0], -1)) * nm[..., None].float()
    d = torch.norm(g, dim=-1) * 0.1
    sco = -(torch.tanh(m.alpha * d) + torch.tanh(m.beta * torch.abs(ri[:, None] - d)))
    a = torch.softmax(sco, -1) * nm.float() * nw
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
    ctx = torch.einsum("bk,bkh->bh", a, g)
    contrib = (m.gamma * ctx).norm(dim=-1)
    hn = h.norm(dim=-1)

print("\n=== graph contribution ===")
print("mean |gamma*ctx| =", float(contrib.mean()), " mean |h| =", float(hn.mean()),
      " ratio =", float((contrib / hn.clamp_min(1e-6)).mean()))
print("valid neighbours per sample: mean", float(nm.sum(1).float().mean()), "p10", float(nm.sum(1).float().quantile(0.1)))
ent = -(a.clamp_min(1e-9).log() * a).sum(-1) / np.log(k)
print("normalised attention entropy: mean", float(ent.mean()), "p90", float(ent.quantile(0.9)))
print("attention max weight mean:", float(a.max(-1).values.mean()))
print("attention weight std across neighbours:", float(a.std(-1).mean()))
print("score spread (std of -s):", float((-sco).std(-1).mean()))
