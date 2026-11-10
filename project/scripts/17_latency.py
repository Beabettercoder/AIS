#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 17: Table 6 —— 端到端千船并发延迟（邻域检索 + 留一法场求解 + TCN 双头推理）。"""
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


def bench(n, dev, repeats=15, tsteps=20):
    rng = np.random.default_rng(0)
    pos = np.column_stack([rng.uniform(0, 50000, n), rng.uniform(0, 5000, n)])
    vel = rng.normal(0, 1.5, (n, 2)) + np.array([0.0, 3.09])  # ~6 kn 向北
    X = torch.randn(n, tsteps, 9, device=dev)

    model = TCN().to(dev).eval()
    lat_search, lat_field, lat_infer, lat_total = [], [], [], []
    torch.cuda.reset_peak_memory_stats(dev) if dev.type == "cuda" else None

    for _ in range(repeats):
        t0 = time.perf_counter()
        tree = cKDTree(pos)
        t1 = time.perf_counter()
        balls = tree.query_ball_point(pos, r=RADIUS_M)
        t2 = time.perf_counter()
        for i in range(n):
            cand = np.asarray(balls[i], dtype=np.int64)
            if cand.size < 5:
                continue
            px, py = pos[cand, 0], pos[cand, 1]
            vx, vy = vel[cand, 0], vel[cand, 1]
            Xd = np.column_stack([np.ones_like(px), (px - pos[i, 0]) / RADIUS_M, (py - pos[i, 1]) / RADIUS_M])
            np.linalg.lstsq(Xd, np.column_stack([vx, vy]), rcond=None)
        t3 = time.perf_counter()
        with torch.no_grad():
            model(X)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        lat_search.append((t1 - t0) * 1e3)
        lat_field.append((t3 - t2) * 1e3)
        lat_infer.append((t4 - t3) * 1e3)
        lat_total.append((t4 - t0) * 1e3)

    peak = torch.cuda.max_memory_allocated(dev) / 1e6 if dev.type == "cuda" else 0.0
    return {
        "n_vessels": n,
        "search_ms_p50": float(np.percentile(lat_search, 50)),
        "search_ms_p95": float(np.percentile(lat_search, 95)),
        "field_ms_p50": float(np.percentile(lat_field, 50)),
        "field_ms_p95": float(np.percentile(lat_field, 95)),
        "infer_ms_p50": float(np.percentile(lat_infer, 50)),
        "infer_ms_p95": float(np.percentile(lat_infer, 95)),
        "total_ms_p50": float(np.percentile(lat_total, 50)),
        "total_ms_p95": float(np.percentile(lat_total, 95)),
        "throughput_vessels_per_s": float(1000.0 / np.percentile(lat_total, 50) * n / n) if lat_total else 0.0,
        "peak_gpu_mb": float(peak),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [bench(n, dev, args.repeats) for n in (100, 300, 1000)]
    for r in rows:
        r["throughput_vessels_per_s"] = round(r["n_vessels"] / (r["total_ms_p50"] / 1000.0), 1)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    for r in rows:
        print(f"N={r['n_vessels']:>4}  total p50={r['total_ms_p50']:.1f} ms  p95={r['total_ms_p95']:.1f} ms  "
              f"(search {r['search_ms_p50']:.1f} / field {r['field_ms_p50']:.1f} / infer {r['infer_ms_p50']:.1f})  "
              f"thr={r['throughput_vessels_per_s']:.0f} vessels/s  peak={r['peak_gpu_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
