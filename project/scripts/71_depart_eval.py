#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 71: 管线级注入的编队脱离检测（单船 vs 图）。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
sp = importlib.util.spec_from_file_location("gf", "/mnt/sda/WangZijia/AIS/scripts/60_graph_fix.py")
gf = importlib.util.module_from_spec(sp); sp.loader.exec_module(gf)


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import sys
    d = (BASE / sys.argv[1]) if len(sys.argv) > 1 else (BASE / "group_depart")
    tr = sw.load(d, "train"); te = sw.load(d, "test")
    ztr = np.load(d / "train.npz"); zte = np.load(d / "test.npz")
    tr["nb_v"] = ztr["nb_v"].astype(np.float32); te["nb_v"] = zte["nb_v"].astype(np.float32)
    ytr = ztr["label_depart"].astype(np.float32); yte = zte["label_depart"].astype(np.float32)
    # 只在无行为异常的窗上评估（排除其他注入干扰）
    mtr = np.isin(tr["cell"], [0, 1]); mte = np.isin(te["cell"], [0, 1])
    idx_tr = np.flatnonzero(mtr); idx_te = np.flatnonzero(mte)
    print("departure rate: train %.2f test %.2f" % (ytr[idx_tr].mean(), yte[idx_te].mean()))

    Xtr = np.concatenate([tr["gated"][idx_tr], tr["rf"][idx_tr][..., None], tr["w"][idx_tr][..., None]], -1)
    Xte = np.concatenate([te["gated"][idx_te], te["rf"][idx_te][..., None], te["w"][idx_te][..., None]], -1)
    mu = Xtr[tr["mask"][idx_tr]].mean(0, keepdims=True); sd = Xtr[tr["mask"][idx_tr]].std(0, keepdims=True) + 1e-6
    norm = lambda X: ((X - mu) / sd).astype(np.float32)

    def train(graph, epochs=20):
        torch.manual_seed(0)
        m = (gf.Net2("tcn", Xtr.shape[-1]) if graph else sw.Net("tcn", Xtr.shape[-1], use_graph=False)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
        xn = norm(Xtr); yy = ytr[idx_tr]
        for _ in range(epochs):
            m.train(); perm = np.random.permutation(len(xn))
            for b in range(0, len(xn), 256):
                ii = perm[b:b + 256]
                xb = torch.tensor(xn[ii]).to(dev)
                if graph:
                    pb, _, _, _ = m(xb, *gf.tensors(tr, idx_tr[ii], dev))
                else:
                    pb, _, _ = m(xb)
                loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yy[ii]).to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
        return m

    res = {}
    for name, graph in (("single-vessel", False), ("graph (Ours)", True)):
        m = train(graph); m.eval()
        with torch.no_grad():
            xb = torch.tensor(norm(Xte)).to(dev)
            if graph:
                pb, _, _, _ = m(xb, *gf.tensors(te, idx_te, dev))
            else:
                pb, _, _ = m(xb)
            p = torch.sigmoid(pb).cpu().numpy()
        res[name] = round(ap_score(p, yte[idx_te]), 3)
        print(f"{name}: departure AP = {res[name]:.3f}", flush=True)
    (BASE / "departure_pipeline_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
