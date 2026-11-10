#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 62: 分层评测——图模块只应在"需要群体参照"的样本上起作用。"""
from __future__ import annotations

import importlib.util
import json
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
    z = np.load(BASE / "eval_v2_graph2" / "test.npz")
    te["nb_v"] = z["nb_v"].astype(np.float32)

    # 上下文依赖度：目标船速度与邻船群体速度的偏离（相对量）
    nb_v = te["nb_v"]
    mask = te["nb_mask"]
    v_t = te["raw"][:, -1, 2:4]
    nb_speed = np.linalg.norm(nb_v, axis=-1)
    nb_speed_m = np.where(mask, nb_speed, np.nan)
    grp = np.nanmedian(nb_speed_m, axis=1)
    grp_std = np.nanstd(nb_speed_m, axis=1) + 1e-3
    ctx_dep = np.abs(np.linalg.norm(v_t, axis=1) - grp) / grp_std
    ctx_dep = np.nan_to_num(ctx_dep, nan=0.0)
    print("context-dependence: p10/p50/p90 =",
          np.round(np.percentile(ctx_dep, [10, 50, 90]), 3))

    # 三个模型
    def pred(name, kind="old"):
        if name == "graph_fixed":
            ck = torch.load(W / "graph_fixed.pt", map_location="cpu", weights_only=False)
            from importlib import util as _u
            sp = _u.spec_from_file_location("gf", "/mnt/sda/WangZijia/AIS/scripts/60_graph_fix.py")
            gf = _u.module_from_spec(sp); sp.loader.exec_module(gf)
            m = gf.Net2(ck["kind"], ck["nf"]).to(dev); m.load_state_dict(ck["sd"]); m.eval()
            mu, sd = ck["stats"]
            X = (np.concatenate([te["gated"], te["rf"][..., None], te["w"][..., None]], -1) - mu) / sd
            with torch.no_grad():
                pb, _, _, _ = m(torch.tensor(X, dtype=torch.float32, device=dev),
                                torch.tensor(te["nb_seq"], device=dev), torch.tensor(te["nb_v"], device=dev),
                                torch.tensor(te["nb_xy"], device=dev), torch.tensor(te["nb_mask"], device=dev),
                                torch.tensor(te["nb_w"], device=dev), torch.tensor(te["rf"][:, -1], device=dev),
                                torch.tensor(te["raw"][:, -1, 2:4], device=dev))
                return torch.sigmoid(pb).cpu().numpy()
        ck = torch.load(W / f"{name}.pt", map_location="cpu", weights_only=False)
        m = sw.Net(ck["kind"], ck["nf"], use_graph=ck["use_graph"]).to(dev)
        m.load_state_dict(ck["sd"]); m.eval()
        mu, sd = ck["stats"]
        X = (sw.build(te, ck["cfg"]["use_kg"], ck["cfg"]["use_lf"]) - mu) / sd
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=dev)
            if ck["use_graph"]:
                pb, _, _ = m(xb, torch.tensor(te["nb_seq"], device=dev), torch.tensor(te["nb_xy"], device=dev),
                             torch.tensor(te["nb_mask"], device=dev), torch.tensor(te["nb_w"], device=dev),
                             torch.tensor(te["rf"][:, -1], device=dev))
            else:
                pb, _, _ = m(xb)
            return torch.sigmoid(pb).cpu().numpy()

    preds = {"+KG+LF (no graph)": pred("ablation_2_kglf"),
             "PG-DRGN (old Gpsi)": pred("ablation_4_full_cdg"),
             "PG-DRGN (fixed Gpsi)": pred("graph_fixed")}

    y = te["y"] > 0
    order = np.argsort(-ctx_dep)
    n = len(order)
    groups = {"top 25% (context-dependent)": order[: n // 4],
              "middle 50%": order[n // 4: 3 * n // 4],
              "bottom 25% (self-evident)": order[3 * n // 4:]}
    out = {}
    for gname, idx in groups.items():
        out[gname] = {k: round(ap_score(p[idx], y[idx]), 3) for k, p in preds.items()}
        print(gname, out[gname])
    (BASE / "context_stratified.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
