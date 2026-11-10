#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 17b: Table 6（向量化版）—— 批量法方程求解 + 5 次 warm-up。

场求解：cKDTree.query(k=K) 一次取 K 近邻 -> (N,K,3) 设计矩阵 -> 批量正规方程
    A = Xᵀ W X + λI ,  b = Xᵀ W Y ,  θ = A⁻¹ b
散度/旋度由 θ 的线性项直接给出。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

RADIUS_M = 8 * 1852.0
K_MAX = 32
LAMBDA = 1e-4


class TCN(nn.Module):
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
        return self.bin(p), self.cls(p)


def field_batch(pos, vel, K=K_MAX):
    """向量化：一次 query + 批量正规方程，返回 (div, curl, res)。"""
    tree = cKDTree(pos)
    dist, idx = tree.query(pos, k=min(K, len(pos)))
    dx = pos[idx, 0] - pos[:, None, 0]
    dy = pos[idx, 1] - pos[:, None, 1]
    d = np.hypot(dx, dy)
    w = np.exp(-(d**2) / (2 * (RADIUS_M / 2) ** 2)) * (d <= RADIUS_M) * (d > 1.0)
    X = np.stack([np.ones_like(dx), dx / RADIUS_M, dy / RADIUS_M], -1)      # (N,K,3)
    Y = vel[idx]                                                            # (N,K,2)
    W = w[..., None]
    A = np.einsum('nki,nk,nkj->nij', X, w, X) + LAMBDA * np.eye(3)
    b = np.einsum('nki,nk,nky->niy', X, w, Y)
    theta = np.linalg.solve(A, b)                                           # (N,3,2)
    div = (theta[:, 1, 0] + theta[:, 2, 1]) / RADIUS_M
    curl = (theta[:, 1, 1] - theta[:, 2, 0]) / RADIUS_M
    res = np.hypot(vel[:, 0] - theta[:, 0, 0], vel[:, 1] - theta[:, 0, 1])
    return div, curl, res


def bench(n, dev, repeats=15, warmup=5, tsteps=20):
    rng = np.random.default_rng(0)
    pos = np.column_stack([rng.uniform(0, 50000, n), rng.uniform(0, 5000, n)]).astype(np.float32)
    vel = (rng.normal(0, 1.5, (n, 2)) + np.array([0.0, 3.09])).astype(np.float32)
    X = torch.randn(n, tsteps, 9, device=dev)
    model = TCN().to(dev).eval()

    lat = []
    for it in range(warmup + repeats):
        t0 = time.perf_counter()
        field_batch(pos, vel)
        t1 = time.perf_counter()
        with torch.no_grad():
            model(X)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if it >= warmup:
            lat.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3))
    arr = np.asarray(lat)
    peak = torch.cuda.max_memory_allocated(dev) / 1e6 if dev.type == "cuda" else 0.0
    return {
        "n_vessels": n,
        "field_ms_p50": float(np.percentile(arr[:, 0], 50)),
        "field_ms_p95": float(np.percentile(arr[:, 0], 95)),
        "infer_ms_p50": float(np.percentile(arr[:, 1], 50)),
        "infer_ms_p95": float(np.percentile(arr[:, 1], 95)),
        "total_ms_p50": float(np.percentile(arr[:, 2], 50)),
        "total_ms_p95": float(np.percentile(arr[:, 2], 95)),
        "throughput_vessels_per_s": float(n / (np.percentile(arr[:, 2], 50) / 1000.0)),
        "peak_gpu_mb": float(peak),
    }


def main() -> int:
    global K_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--k", type=int, default=K_MAX)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K_MAX = args.k
    rows = [bench(n, dev, args.repeats) for n in (100, 300, 1000)]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    for r in rows:
        print(f"N={r['n_vessels']:>4}  total p50={r['total_ms_p50']:6.1f} ms  p95={r['total_ms_p95']:6.1f} ms  "
              f"(field {r['field_ms_p50']:.1f} + infer {r['infer_ms_p50']:.1f})  "
              f"thr={r['throughput_vessels_per_s']:.0f}/s  peak={r['peak_gpu_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
