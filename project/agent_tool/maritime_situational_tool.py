#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PG-DRGN maritime situational-awareness tool
===========================================

A self-contained callable module for LLM agents (OpenAI / Qwen / DeepSeek
function-calling schema + a minimal MCP-compatible dispatcher).

Only numpy + torch are required (both already present in the training env);
everything else is the Python standard library.

Public API
----------
analyze_maritime_situation(ais_window_data, region_name="Unknown", far_budget=0.01) -> dict
    Returns a three-layer situational dictionary:
      macro_flow_status      regional flow field (density, speed, divergence, curl, fallback)
      signal_quality_audit   kinematic gate audit (GPS jumps intercepted before the model saw them)
      vessel_anomaly_alerts  per-vessel alerts above the strict-FAR threshold, with tier + intent
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parent / "weights" / "pg_drgn.pt"
_SAVER = Path(__file__).resolve().parent / "pgdrgn_net.py"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_maritime_situation",
        "description": ("Run the physics-guided AIS anomaly detector (PG-DRGN, 66.8K params) on a batch of "
                        "20-step vessel windows. Returns regional flow status, a kinematic signal-quality "
                        "audit and a ranked list of vessel anomaly alerts."),
        "parameters": {
            "type": "object",
            "properties": {
                "region_name": {"type": "string", "description": "Human readable area name, e.g. 'Houston Ship Channel'."},
                "far_budget": {"type": "number", "description": "Tolerated false-alarm rate (0.01 = 1%).", "default": 0.01},
                "window_path": {"type": "string", "description": "Optional .npz path holding the AIS windows."},
            },
            "required": ["region_name"],
        },
    },
}

MCP_TOOL_DEF = {
    "name": "analyze_maritime_situation",
    "description": TOOL_SCHEMA["function"]["description"],
    "inputSchema": TOOL_SCHEMA["function"]["parameters"],
}

TIER_I = "Tier-I Self-Referential"
TIER_II = "Tier-II Formation Departure"
TIER_III = "Tier-III Group Non-Compliance"
BEHAVIOURS = ["Turn-back", "Deviation", "Stall", "Sinuous", "Speed-drop"]

_MODEL_CACHE: dict = {}


def _load_model(device="cpu"):
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE
    import importlib.util
    import torch

    spec = importlib.util.spec_from_file_location("sw", str(_SAVER))
    sw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sw)
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model = sw.Net(ck["kind"], ck["nf"], use_graph=ck.get("use_graph", False)).to(device)
    model.load_state_dict(ck["sd"])
    model.eval()
    _MODEL_CACHE.update({"model": model, "sw": sw, "ck": ck, "device": device})
    return _MODEL_CACHE


def _local_field_div_curl(nb_xy, nb_v, nb_mask, own_xy):
    """Least-squares local linear field from neighbours; returns (div, curl, residual)."""
    xy = nb_xy[nb_mask]
    vv = nb_v[nb_mask]
    if len(xy) < 4:
        return None
    R = 8.0 * 1852.0
    X = np.concatenate([np.ones((len(xy), 1)), (xy - own_xy) / R], axis=1)
    try:
        theta, *_ = np.linalg.lstsq(X, vv, rcond=None)
    except np.linalg.LinAlgError:
        return None
    div = (theta[1, 0] + theta[2, 1]) / R
    curl = (theta[2, 0] - theta[1, 1]) / R
    return float(div), float(curl), theta[0]


def analyze_maritime_situation(ais_window_data, region_name: str = "Unknown", far_budget: float = 0.01) -> dict:
    """Main entry point. See module docstring for the output layout."""
    t_start = time.perf_counter()
    c = _load_model()
    model, sw, ck = c["model"], c["sw"], c["ck"]
    import torch

    raw = np.asarray(ais_window_data["raw"], dtype=np.float32)
    w = np.asarray(ais_window_data["w"], dtype=np.float32)
    rf = np.asarray(ais_window_data["r_field"], dtype=np.float32)
    nb_xy = np.asarray(ais_window_data.get("nb_xy", np.zeros((len(raw), 1, 2))), dtype=np.float32)
    nb_v = np.asarray(ais_window_data.get("nb_v", np.zeros((len(raw), 1, 2))), dtype=np.float32)
    nb_mask = np.asarray(ais_window_data.get("nb_mask", np.zeros((len(raw), 1))), dtype=bool)
    ids = ais_window_data.get("mmsi")
    N = len(raw)

    # --- features and forward pass -------------------------------------------------
    gated = raw.copy()
    dp = np.diff(gated[:, :, 0:2], axis=1, prepend=0.0) * w[..., None]
    gated[:, :, 0:2] = np.cumsum(dp, axis=1)
    for ch in (2, 3, 4, 5, 6):
        v = gated[:, :, ch]
        lowgt = w < 0.2
        for t in range(1, v.shape[1]):
            msk = lowgt[:, t]
            v[msk, t] = v[msk, t - 1]
        gated[:, :, ch] = v
    X = sw.build({"raw": raw, "gated": gated, "rf": rf, "w": w}, ck["cfg"]["use_kg"], ck["cfg"]["use_lf"])
    mu, sd = ck["stats"]
    with torch.no_grad():
        xb = torch.tensor((X - mu) / sd, dtype=torch.float32)
        logits, cls_logits, _ = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy()
        cls_p = torch.softmax(cls_logits, dim=-1).cpu().numpy()

    # --- threshold: strict-FAR quantile over the batch (assumed mostly normal) ------
    thr = float(np.quantile(prob, 1.0 - far_budget))

    # --- layer 1: macro flow -------------------------------------------------------
    sog = raw[:, :, 4]
    speeds = sog.reshape(-1)
    divs, curls, hit = [], [], 0
    for i in range(N):
        out = _local_field_div_curl(nb_xy[i], nb_v[i], nb_mask[i], np.zeros(2, dtype=np.float32))
        if out is not None:
            hit += 1
            divs.append(out[0])
            curls.append(out[1])
    valid_ratio = hit / max(N, 1)
    div_mean = float(np.mean(divs)) if divs else 0.0
    curl_mean = float(np.mean(curls)) if curls else 0.0
    fallback = 1.0 - valid_ratio
    if fallback > 0.5:
        macro_label = "Pelagic Sparse Mode"
    elif div_mean < -2e-5 and float(np.mean(speeds)) < 7.0:
        macro_label = "Group Congestion Slowdown"
    else:
        macro_label = "Normal Flow"

    # --- layer 2: signal quality audit --------------------------------------------
    low = w < 0.3
    flagged = []
    for i in range(N):
        steps = np.flatnonzero(low[i])
        if len(steps):
            flagged.append({
                "vessel_id": (str(ids[i]) if ids is not None else f"WIN-{i:04d}"),
                "intercepted_steps": int(len(steps)),
                "step_index": [int(s) for s in steps[:8]],
                "min_gate_weight": round(float(w[i].min()), 3),
                "estimated_kinematic_residual_mps": round(float((1.0 - w[i].min()) * 6.0), 2),
                "repair": "position re-integrated with gate weight; speed/course held from previous step",
            })
    flagged.sort(key=lambda d: d["min_gate_weight"])

    # --- layer 3: vessel alerts ----------------------------------------------------
    alerts = []
    for i in np.argsort(-prob):
        if prob[i] < thr:
            break
        rid = float(np.mean(np.abs(rf[i])))
        nb_spd = float(np.mean(np.linalg.norm(nb_v[i][nb_mask[i]], axis=-1))) if nb_mask[i].any() else 0.0
        own_spd = float(np.mean(sog[i]))
        if nb_spd > 1e-3 and abs(own_spd - nb_spd) > 0.45 * max(nb_spd, 1e-3):
            tier = TIER_III
        elif rid > 0.6:
            tier = TIER_II
        else:
            tier = TIER_I
        k = int(np.argmax(cls_p[i]))
        alerts.append({
            "vessel_id": (str(ids[i]) if ids is not None else f"WIN-{i:04d}"),
            "anomaly_probability": round(float(prob[i]), 4),
            "tier": tier,
            "behaviour_intent": BEHAVIOURS[k],
            "behaviour_confidence": round(float(cls_p[i][k]), 4),
            "field_residual_mps": round(rid, 3),
            "mean_gate_weight": round(float(w[i].mean()), 3),
        })

    elapsed = (time.perf_counter() - t_start) * 1000.0
    return {
        "region_name": region_name,
        "macro_flow_status": {
            "vessels_in_window": int(N),
            "mean_sog_kn": round(float(np.mean(speeds)), 2),
            "divergence_mean": round(div_mean, 8),
            "curl_mean": round(curl_mean, 8),
            "field_valid_ratio": round(valid_ratio, 3),
            "fallback_rate": round(fallback, 3),
            "macro_label": macro_label,
        },
        "signal_quality_audit": {
            "intercepted_vessels": len(flagged),
            "intercepted_step_ratio": round(float(low.mean()), 4),
            "vessels": flagged[:10],
        },
        "vessel_anomaly_alerts": {
            "threshold_prob": round(thr, 4),
            "far_budget": far_budget,
            "alerts": alerts[:10],
        },
        "meta": {
            "model": "PG-DRGN (TCN, 66.8K params, BF16-ready)",
            "weights": str(WEIGHTS),
            "inference_ms": round(elapsed, 1),
            "windows": int(N),
        },
    }


# ------------------------------------------------------------------ agent adapters
def openai_tool_call(ais_window_data, region_name="Unknown", far_budget=0.01) -> str:
    """Return an OpenAI-style tool message payload (JSON string)."""
    return json.dumps({"role": "tool", "name": "analyze_maritime_situation",
                       "content": json.dumps(analyze_maritime_situation(ais_window_data, region_name, far_budget),
                                             ensure_ascii=False)}, ensure_ascii=False)


def mcp_handle(request: dict) -> dict:
    """Minimal MCP `tools/call` handler: {"name":..., "arguments": {...}} -> result envelope."""
    if request.get("method") == "tools/list":
        return {"tools": [MCP_TOOL_DEF]}
    args = request.get("params", {}).get("arguments", {}) or request.get("arguments", {})
    data = args.pop("ais_window_data", None)
    if data is None:
        return {"isError": True, "content": [{"type": "text", "text": "ais_window_data is required"}]}
    res = analyze_maritime_situation(data, args.get("region_name", "Unknown"), args.get("far_budget", 0.01))
    return {"isError": False, "content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}]}


if __name__ == "__main__":
    z = np.load("/mnt/sda/WangZijia/AIS/outputs/phase1/eval_v2_graph2/test.npz")
    sub = {k: z[k][:40] for k in ("raw", "w_self", "r_field", "nb_xy", "nb_v", "nb_mask")}
    out = analyze_maritime_situation({"raw": sub["raw"], "w": sub["w_self"], "r_field": sub["r_field"],
                                      "nb_xy": sub["nb_xy"], "nb_v": sub["nb_v"], "nb_mask": sub["nb_mask"]},
                                     region_name="Houston Ship Channel", far_budget=0.01)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:2500])
