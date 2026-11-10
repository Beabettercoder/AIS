#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 72: 本模型的量化/部署分析（FP32/FP16/INT8/INT4）——体积、显存、延迟、精度。"""
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
    te = sw.load(BASE / "eval_v2_graph2", "test")
    te["nb_v"] = np.load(BASE / "eval_v2_graph2" / "test.npz")["nb_v"].astype(np.float32)
    ck = torch.load(W / "ablation_4_full_cdg.pt", map_location="cpu", weights_only=False)
    mu, sd = ck["stats"]
    X = (np.concatenate([te["gated"], te["rf"][..., None], te["w"][..., None]], -1) - mu) / sd
    X = X.astype(np.float32)
    args = gf.tensors(te, np.arange(len(X)), dev)
    y = te["y"] > 0

    def evaluate(model, dtype, tag):
        model = model.to(device=dev, dtype=dtype if dtype else torch.float32).eval()
        xb = torch.tensor(X, dtype=dtype if dtype else torch.float32, device=dev)
        a = [t.to(dtype=dtype if dtype else torch.float32) if isinstance(t, torch.Tensor) else t for t in args]
        # warmup
        with torch.no_grad():
            for _ in range(3):
                model(xb, *a)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                pb, _, _, _ = model(xb, *a)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        dt = (time.perf_counter() - t0) / 10
        p = torch.sigmoid(pb.float()).cpu().numpy()
        nbytes = sum(q.numel() * q.element_size() for q in model.parameters())
        return {"tag": tag, "dtype": str(dtype).replace("torch.", ""), "AP": round(ap_score(p, y), 3),
                "size_KB": round(nbytes / 1024, 1), "latency_ms_1000": round(dt * 1000, 2)}

    rows = []
    rows.append(evaluate(gf.Net2("tcn", X.shape[-1]).to(dev), None, "FP32"))
    m32 = gf.Net2("tcn", X.shape[-1]); m32.load_state_dict(ck["sd"])
    rows.append(evaluate(m32, torch.float16, "FP16"))
    m16 = gf.Net2("tcn", X.shape[-1]); m16.load_state_dict(ck["sd"])
    rows.append(evaluate(m16, torch.bfloat16, "BF16"))

    # INT8 动态量化（CPU 推理）
    m8 = gf.Net2("tcn", X.shape[-1]); m8.load_state_dict(ck["sd"]); m8.eval()
    try:
        q = torch.ao.quantization.quantize_dynamic(m8.cpu(), {torch.nn.Linear}, dtype=torch.qint8)
        with torch.no_grad():
            pb, _, _, _ = q(torch.tensor(X), *[t.cpu() for t in args])
            p = torch.sigmoid(pb).numpy()
        nbytes = sum(v.numel() * v.element_size() for v in q.state_dict().values() if hasattr(v, "numel"))
        rows.append({"tag": "INT8 (dynamic, CPU)", "dtype": "qint8", "AP": round(ap_score(p, y), 3),
                     "size_KB": round(nbytes / 1024, 1), "latency_ms_1000": None})
    except Exception as e:
        print("int8 failed:", e)

    for r in rows:
        print(r)
    (BASE / "quantization_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
