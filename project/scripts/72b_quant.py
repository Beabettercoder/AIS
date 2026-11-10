#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 72b: 本模型量化/部署分析（基于已落盘的 sw.Net 权重）。"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
W = BASE / "vis_weights"
spec = importlib.util.spec_from_file_location("sw", "/mnt/sda/WangZijia/AIS/scripts/51_save_weights.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)


def ap_score(s, y):
    ok = np.isfinite(s); s, y = s[ok], np.asarray(y)[ok].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    te = sw.load(BASE / "eval_v2_graph2", "test")
    ck = torch.load(W / "ablation_4_full_cdg.pt", map_location="cpu", weights_only=False)
    mu, sd = ck["stats"]
    X = (np.concatenate([te["gated"], te["rf"][..., None], te["w"][..., None]], -1) - mu) / sd
    X = X.astype(np.float32)
    y = te["y"] > 0
    nseq = torch.tensor(te["nb_seq"], device=dev)
    nxy = torch.tensor(te["nb_xy"], device=dev)
    nm = torch.tensor(te["nb_mask"], device=dev)
    nw = torch.tensor(te["nb_w"], device=dev)
    ri = torch.tensor(te["rf"][:, -1], device=dev)

    def build():
        m = sw.Net(ck["kind"], ck["nf"], use_graph=True)
        m.load_state_dict(ck["sd"])
        return m

    def run(model, t, tag):
        model = model.to(device=dev, dtype=t).eval()
        xb = torch.tensor(X, dtype=t, device=dev)
        a = [z.to(t) for z in (nseq, nxy, nm, nw, ri)]
        with torch.no_grad():
            for _ in range(3):
                model(xb, *a)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                pb, _, _ = model(xb, *a)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 10
            p = torch.sigmoid(pb.float()).cpu().numpy()
        size = sum(q.numel() * q.element_size() for q in model.parameters())
        return {"tag": tag, "AP": round(ap_score(p, y), 3), "size_KB": round(size / 1024, 1),
                "latency_ms_1000": round(dt * 1000, 2)}

    rows = []
    for t, tag in ((torch.float32, "FP32 (32-bit)"), (torch.float16, "FP16"), (torch.bfloat16, "BF16")):
        try:
            rows.append(run(build(), t, tag))
        except Exception as e:
            print(tag, "skipped:", str(e)[:80])
            rows.append({"tag": tag, "AP": None, "note": "dtype cast needs mixed-precision guard"})

    # INT8 动态量化（CPU）
    m = build().cpu().eval()
    q = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        pb, _, _ = q(torch.tensor(X), *[z.cpu() for z in (nseq, nxy, nm, nw, ri)])
        p = torch.sigmoid(pb).numpy()
    size = sum(v.numel() * v.element_size() for v in q.state_dict().values() if hasattr(v, "numel"))
    rows.append({"tag": "INT8 dynamic (CPU)", "AP": round(ap_score(p, y), 3),
                 "size_KB": round(size / 1024, 1), "latency_ms_1000": None})

    for r in rows:
        print(r)
    (BASE / "quantization_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
