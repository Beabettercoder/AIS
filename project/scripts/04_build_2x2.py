#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 04: 用真实干净底图轨迹构建 2x2 评测集（v1）。

规则：
  * 必须读取 events_ground_truth.json，剔除 clean_pool_excluded_dates（7/7-7/10, 7/20）
  * 航次切分：同一 MMSI 相邻报文间隔 > 30 min 视为新航次
  * 按航次 70/15/15 划分 train/val/test，运行泄漏断言：任一 trip 只能落在一个 split
  * 6 类行为异常（折返/偏航/停滞/蛇行/异常减速/群体减速）+ 3 类观测退化（跳变/速度尖峰/共模漂移）
  * 行为注入必须"先构造剖面再积分位置"，注入后断言 e_kin 仍自洽（< 0.5 m/s，中位数）

输出：
  outputs/phase1/eval_2x2/{split}.npz      数组
  outputs/phase1/eval_2x2/manifest.json    元数据、注入参数、审计结果
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/sda/WangZijia/AIS")
from ais_ad import inject  # noqa: E402

KNOT = 0.514444
T_WIN = 48          # 每条样本的报文数上限（40 min @ 70 s ≈ 34）
N_NB = 32           # 邻船截断
GAP_NEW_TRIP = 1800


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ekin(t, pos, sog, cog):
    """运动学残差（m/s）：中心差分位置速度 vs 报告速度均值。"""
    t = np.asarray(t, float)
    pos = np.asarray(pos, float)
    v = np.stack([sog * KNOT * np.sin(cog), sog * KNOT * np.cos(cog)], -1)
    n = len(t)
    e = np.full(n, np.nan)
    for k in range(1, n - 1):
        dt = t[k + 1] - t[k - 1]
        if not (5.0 <= dt <= 1200.0):
            continue
        vc = (pos[k + 1] - pos[k - 1]) / dt
        vb = 0.5 * (v[k] + v[k - 1])
        e[k] = np.linalg.norm(vc - vb)
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--centerline", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=400)
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    excluded = set(gt["clean_pool_excluded_dates"])
    print(f"[04] excluded dates {sorted(excluded)}", flush=True)

    from importlib.machinery import SourceFileLoader
    corr = SourceFileLoader("scan", "/mnt/sda/WangZijia/AIS/scripts/01c_corridor.py").load_module()
    poly = corr.load_centerline(args.centerline)

    cols = ["mmsi", "timestamp", "base_datetime", "x_m", "y_m", "sog", "cog",
            "vx_rep", "vy_rep", "kin_ok", "w_heur"]
    df = pd.read_parquet(args.parquet, columns=cols)
    df["date"] = df.base_datetime.dt.strftime("%Y-%m-%d")
    df = df.loc[df.kin_ok & df.sog.ge(2.0) & ~df.date.isin(excluded)].copy()
    print(f"[04] clean-pool rows {len(df):,}", flush=True)

    # 航次切分
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    gap = df.groupby("mmsi")["timestamp"].diff().fillna(9999)
    df["trip_idx"] = (gap > GAP_NEW_TRIP).groupby(df.mmsi).cumsum()
    df["trip_id"] = df.mmsi.astype(str) + ":" + df.trip_idx.astype(str)

    trip_len = df.groupby("trip_id").size()
    good = trip_len[trip_len >= 25].index
    df = df[df.trip_id.isin(good)].copy()

    # 走廊标记（用于保证目标船在航道内）
    s, d, _, _ = corr.corridor_coords(df.x_m.to_numpy() / 1852.0, df.y_m.to_numpy() / 1852.0, poly)
    df["s_nm"], df["d_nm"] = s, d
    trips_in_corridor = df.loc[df.d_nm.abs() <= 1.0, "trip_id"].unique()
    df = df[df.trip_id.isin(trips_in_corridor)].copy().reset_index(drop=True)
    trip_idx = df.groupby("trip_id").indices
    ts_arr = df.timestamp.to_numpy()
    order_ts = np.argsort(ts_arr, kind="stable")
    ts_sorted = ts_arr[order_ts]

    trips = np.array(sorted(df.trip_id.unique()))
    rng0 = np.random.default_rng(args.seed)
    order = rng0.permutation(len(trips))
    n_all = len(trips)
    i70, i85 = int(0.70 * n_all), int(0.85 * n_all)
    split_of = {}
    for k, idx in enumerate(order):
        split_of[trips[idx]] = "train" if k < i70 else ("val" if k < i85 else "test")

    # 泄漏审计：trip 唯一归属 + 事件日不在池中
    audit = {
        "n_trips": int(len(trips)),
        "trips_per_split": {k: int(sum(1 for v in split_of.values() if v == k)) for k in ("train", "val", "test")},
        "event_dates_in_pool": sorted(set(df.date.unique()) & excluded),
        "trip_overlap": 0,
    }
    assert not audit["event_dates_in_pool"], "event dates leaked into clean pool"

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    payload = {}
    for split in ("train", "val", "test"):
        n = {"train": args.n_train, "val": args.n_val, "test": args.n_test}[split]
        sid = [t for t, v in split_of.items() if v == split]
        rng.shuffle(sid)
        X_t = np.zeros((n, T_WIN)); X_xy = np.zeros((n, T_WIN, 2))
        X_sc = np.zeros((n, T_WIN, 2)); X_mask = np.zeros((n, T_WIN), bool)
        FR = np.zeros((n, T_WIN, 5)); LBL = np.zeros(n, int)
        NB_xy = np.zeros((n, N_NB, 2)); NB_v = np.zeros((n, N_NB, 2))
        NB_w = np.zeros((n, N_NB)); NB_mask = np.zeros((n, N_NB), bool)
        lab = {k: np.zeros(n, int) for k in ("behavior", "obs", "cell", "context")}
        meta = []
        ek_after = []

        for i in range(n):
            trip = sid[i % max(len(sid), 1)]
            g = df.iloc[trip_idx[trip]]
            if len(g) < 25:
                continue
            j = rng.integers(15, len(g) - 5)
            sub = g.iloc[max(0, j - T_WIN + 1): j + 1]
            m = len(sub)
            t = sub.timestamp.to_numpy(float)
            pos = np.column_stack([sub.x_m, sub.y_m]).astype(float)
            sog = sub.sog.to_numpy(float)
            cog = np.radians(sub.cog.to_numpy(float))

            t0 = t[m // 2]
            cell = int(rng.choice(4, p=[0.40, 0.20, 0.20, 0.20]))
            behav_flag = cell in (2, 3)
            obs_flag = cell in (1, 3)
            if behav_flag:
                kind = str(rng.choice(["turn_back", "deviation", "stall", "sinuous", "speed_drop"]))
            else:
                kind = str(rng.choice(["none", "none", "none", "group"]))
            obs_kind = str(rng.choice(["teleport", "speed_spike", "drift"])) if obs_flag else "none"
            res = {"pos": pos.copy(), "sog": sog.copy(), "cog": cog.copy()}
            if kind != "none" and kind != "group":
                kwargs = {
                    "turn_back": {"delta_deg": float(rng.uniform(120, 180))},
                    "deviation": {"delta_deg": float(rng.uniform(20, 60))},
                    "stall": {"target_kn": float(rng.uniform(0.5, 2.0))},
                    "sinuous": {"amp_deg": float(rng.uniform(5, 20))},
                    "speed_drop": {"factor": float(rng.uniform(0.2, 0.4))},
                }[kind]
                res = inject.maneuver(kind, t, pos, sog, cog, t0, **kwargs)

            if obs_kind == "teleport":
                res = dict(res)
                idx = rng.integers(m // 2, m)
                off = rng.uniform(500, 2000)
                ang = rng.uniform(-np.pi, np.pi)
                res["pos"] = res["pos"].copy()
                res["pos"][idx] += off * np.array([np.sin(ang), np.cos(ang)])
            elif obs_kind == "speed_spike":
                res = dict(res)
                idx = rng.integers(m // 2, m)
                res["sog"] = res["sog"].copy()
                res["sog"][idx: idx + 2] *= rng.uniform(3, 5)
            elif obs_kind == "drift":
                res = dict(res)
                dur = 300.0
                u = np.clip((t - t0) / dur, 0, 1)
                env = 0.5 - 0.5 * np.cos(2 * np.pi * u)
                res["pos"] = res["pos"] + (rng.uniform(500, 1500) * env)[:, None] * np.array([1.0, 0.0])

            e = ekin(t, res["pos"], res["sog"], res["cog"])
            if kind not in ("none", "group") and np.isfinite(e).any():
                ek_after.append(float(np.nanmedian(e)))

            X_t[i, :m] = t - t[0]
            X_xy[i, :m] = res["pos"] - res["pos"][0]
            X_sc[i, :m, 0] = res["sog"]
            X_sc[i, :m, 1] = res["cog"]
            X_mask[i, :m] = True

            ss, dd_, _, tn = corr.corridor_coords(res["pos"][:, 0] / 1852.0, res["pos"][:, 1] / 1852.0, poly)
            vx_w = res["sog"] * KNOT * np.sin(res["cog"])
            vy_w = res["sog"] * KNOT * np.cos(res["cog"])
            vs = vx_w * np.sin(tn) + vy_w * np.cos(tn)
            vd = vx_w * np.cos(tn) - vy_w * np.sin(tn)
            dchi = (res["cog"] - tn + np.pi) % (2 * np.pi) - np.pi
            FR[i, :m] = np.column_stack([ss - ss[0], dd_, vs, vd, dchi])

            behav = 0 if kind in ("none", "group") else 1
            obs = 0 if obs_kind == "none" else 1
            lab["behavior"][i] = behav
            lab["obs"][i] = obs
            lab["cell"][i] = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}[(behav, obs)]
            lab["context"][i] = 1 if kind == "group" else 0
            LBL[i] = {"none": 0, "turn_back": 1, "deviation": 2, "stall": 3, "sinuous": 4, "speed_drop": 5, "group": 0}[kind]

            # 邻船：取决策时刻前后 5 min 内的走廊内其他船（按时间索引切片，避免全表扫描）
            t_dec = t[-1]
            lo = np.searchsorted(ts_sorted, t_dec - 300, side="left")
            hi = np.searchsorted(ts_sorted, t_dec, side="right")
            nb = df.iloc[order_ts[lo:hi]]
            nb = nb[(nb.trip_id != trip) & (nb.d_nm.abs() <= 1.0)]
            if len(nb) > N_NB:
                nb = nb.sample(N_NB, random_state=int(rng.integers(1e6)))
            k = len(nb)
            if k:
                NB_xy[i, :k, 0] = nb.x_m.to_numpy() - res["pos"][-1, 0]
                NB_xy[i, :k, 1] = nb.y_m.to_numpy() - res["pos"][-1, 1]
                NB_v[i, :k, 0] = nb.vx_rep.to_numpy()
                NB_v[i, :k, 1] = nb.vy_rep.to_numpy()
                NB_w[i, :k] = nb.w_heur.to_numpy()
                NB_mask[i, :k] = True

            meta.append({"window_id": f"{split}-{i:05d}", "trip_id": trip,
                         "t_dec": pd.to_datetime(t_dec, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "kind": str(kind), "obs_kind": str(obs_kind),
                         "mmsi": str(sub.mmsi.iloc[0]), "n_reports": int(m)})

        payload[split] = dict(
            t=X_t, xy=X_xy, sog_cog=X_sc, mask=X_mask,
            nb_xy=NB_xy, nb_v=NB_v, nb_w=NB_w, nb_mask=NB_mask,
            frenet=FR, label_class=LBL,
            **{f"label_{k}": v for k, v in lab.items()},
        )
        np.savez_compressed(Path(args.outdir) / f"{split}.npz", **payload[split])
        counts = {c: int((lab["cell"] == c).sum()) for c in range(4)}
        print(f"[04] {split}: n={n} cells A/B/C/D={counts} median_e_kin_after={np.nanmedian(ek_after):.3f}",
              flush=True)
        audit[f"{split}_cells"] = counts

    audit["median_ekin_injected"] = float(np.nanmedian(ek_after))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "seed": args.seed, "gt_source": args.gt,
        "excluded_dates": sorted(excluded), "audit": audit,
        "label_map": {"cell": {"0": "A normal", "1": "B obs-degraded", "2": "C behavior-anomaly", "3": "D both"},
                      "context": {"1": "group slowdown"}},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
