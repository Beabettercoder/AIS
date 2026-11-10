#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 18: 用冻结数据重训 5 seeds 生产模型并保存 .pt 权重到冻结包。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_MAX, EPS_MIN = 0.3, 0.01


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
    return dict(X=np.nan_to_num(X), mask=z["mask"], y=z["label_class"], cell=z["label_cell"])


class Net(nn.Module):
    def __init__(self, nf=9, nh=64):
        super().__init__()
        self.c1 = nn.Conv1d(nf, nh, 5, padding=2)
        self.c2 = nn.Conv1d(nh, nh, 5, padding=4, dilation=2)
        self.c3 = nn.Conv1d(nh, nh, 5, padding=8, dilation=4)
        self.bin = nn.Linear(2 * nh, 1)
        self.cls = nn.Linear(2 * nh, 5)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.gelu(self.c1(h)); h = F.gelu(self.c2(h)); h = F.gelu(self.c3(h))
        p = torch.cat([h.mean(-1), h.max(-1).values], -1)
        return self.bin(p).squeeze(-1), self.cls(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    tr, te = load(args.dir, "train"), load(args.dir, "test")
    mu = tr["X"][tr["mask"]].mean(0, keepdims=True)
    sd = tr["X"][tr["mask"]].std(0, keepdims=True) + 1e-6
    Xtr, Xte = (tr["X"] - mu) / sd, (te["X"] - mu) / sd
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = []

    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = Net(Xtr.shape[-1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        sel = np.isin(tr["cell"], [0, 2])
        yb = (tr["y"] > 0).astype(np.float32)
        cnt = np.bincount(tr["y"][sel & (tr["y"] > 0)] - 1, minlength=5).astype(float)
        cw = torch.tensor(len(cnt) / (5.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
        idx = np.flatnonzero(sel)
        for _ in range(args.epochs):
            model.train(); np.random.shuffle(idx)
            for b in range(0, len(idx), 128):
                ii = idx[b:b + 128]
                pb, pc = model(torch.tensor(Xtr[ii]).to(dev))
                loss = F.binary_cross_entropy_with_logits(pb, torch.tensor(yb[ii]).to(dev))
                m = torch.tensor(yb[ii] > 0).to(dev)
                if m.any():
                    yy = torch.tensor(tr["y"][ii][m.cpu().numpy()] - 1).to(dev)
                    loss = loss + 0.5 * F.cross_entropy(pc[m], yy, weight=cw)
                opt.zero_grad(); loss.backward(); opt.step()
        torch.save({"state_dict": model.state_dict(), "seed": seed,
                    "norm_mean": mu, "norm_std": sd,
                    "config": {"channels": int(Xtr.shape[-1]), "tsteps": int(Xtr.shape[1]),
                               "epochs": args.epochs, "heads": "bin+cls5"}},
                   Path(args.outdir) / f"model_seed{seed}.pt")
        summary.append(seed)
        print(f"saved model_seed{seed}.pt", flush=True)
    (Path(args.outdir) / "weights_manifest.json").write_text(
        json.dumps({"seeds": summary, "source_dir": args.dir}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
