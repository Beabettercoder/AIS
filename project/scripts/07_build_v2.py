#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 07: 构建 v2 评测集 —— 决策点聚焦窗口 + 动态场残差序列。

每个样本 = (目标船, 决策时刻)：
  窗口 = [t_dec - 23 min, t_dec]，约 20 步（Δt 中位 70 s）
  行为注入的 onset t0 = t_dec - delay，delay ∈ {3, 5, 10} min
  逐时间步计算：留一法局部线性场残差 r_field(t)、未加权邻船均值残差 r_mean(t)、
  自身观测可信度 w_self(t)，作为三套特征的对照。

输出 outputs/phase1/eval_v2/{split}.npz + manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/sda/WangZijia/AIS")
from ais_ad import inject  # noqa: E402

KNOT = 0.514444
NM = 1852.0
RADIUS_M = 8.0 * NM
T_STEPS = 20
WIN_SEC = 23 * 60
DELAYS = [180, 300, 600]
MIN_NBRS = 4
MAX_DCOG = np.pi / 4


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--centerline", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_train", type=int, default=4500)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--amp_level", type=int, default=3, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--split_by", default="trip", choices=["trip", "mmsi"])
    ap.add_argument("--inject_mode", default="smooth", choices=["smooth", "crude"])
    ap.add_argument("--inject_departure", type=float, default=0.0,
                    help="probability of injecting a formation-departure manoeuvre")
    ap.add_argument("--dep_angle", type=float, default=18.0)
    ap.add_argument("--dep_steps", type=int, default=4)
    ap.add_argument("--dep_loss", type=float, default=0.10)
    ap.add_argument("--zone", default="all", choices=["all", "a", "b"])
    ap.add_argument("--radius_nm", type=float, default=8.0)
    ap.add_argument("--dcog_deg", type=float, default=45.0)
    ap.add_argument("--min_nbrs", type=int, default=4)
    args = ap.parse_args()

    global RADIUS_M, MIN_NBRS, MAX_DCOG
    RADIUS_M = args.radius_nm * NM
    MIN_NBRS = args.min_nbrs
    MAX_DCOG = np.radians(args.dcog_deg)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    excluded = set(gt["clean_pool_excluded_dates"])

    from importlib.machinery import SourceFileLoader
    corr = SourceFileLoader("scan", "/mnt/sda/WangZijia/AIS/scripts/01c_corridor.py").load_module()
    poly = corr.load_centerline(args.centerline)

    df = pd.read_parquet(args.parquet, columns=["mmsi", "timestamp", "base_datetime", "x_m", "y_m",
                                                "sog", "cog", "vx_rep", "vy_rep", "kin_ok"])
    df["date"] = df.base_datetime.dt.strftime("%Y-%m-%d")
    df = df.loc[df.kin_ok & df.sog.ge(2.0) & ~df.date.isin(excluded)].copy()
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    gap = df.groupby("mmsi")["timestamp"].diff().fillna(9999)
    df["trip_idx"] = (gap > 1800).groupby(df.mmsi).cumsum()
    df["trip_id"] = df.mmsi.astype(str) + ":" + df.trip_idx.astype(str)

    s, d, _, _ = corr.corridor_coords(df.x_m.to_numpy() / NM, df.y_m.to_numpy() / NM, poly)
    df["s_nm"], df["d_nm"] = s, d
    keep = df.groupby("trip_id")["d_nm"].transform(lambda x: (x.abs() <= 1.0).any())
    df = df[keep & (df.groupby("trip_id").mmsi.transform("size") >= 22)].copy().reset_index(drop=True)

    if args.zone != "all":
        trip_s = df.groupby("trip_id")["s_nm"].median()
        keep_ids = trip_s[trip_s <= 16.0].index if args.zone == "a" else trip_s[trip_s > 16.0].index
        df = df[df.trip_id.isin(keep_ids)].copy().reset_index(drop=True)

    trips = np.array(sorted(df.trip_id.unique()))
    rng0 = np.random.default_rng(args.seed)
    if getattr(args, "split_by", "trip") == "mmsi":
        # vessel-level split: every trip of a vessel goes to the same split
        vessels = np.array(sorted(df.mmsi.unique()))
        vorder = rng0.permutation(len(vessels))
        nv = len(vessels)
        v70, v85 = int(0.70 * nv), int(0.85 * nv)
        vsplit = {vessels[idx]: ("train" if k < v70 else "val" if k < v85 else "test")
                  for k, idx in enumerate(vorder)}
        split_of = {t: vsplit[df.loc[df.trip_id == t, "mmsi"].iloc[0]] for t in trips}
    else:
        order = rng0.permutation(len(trips))
        n_all = len(trips)
        i70, i85 = int(0.70 * n_all), int(0.85 * n_all)
        split_of = {trips[idx]: ("train" if k < i70 else "val" if k < i85 else "test")
                    for k, idx in enumerate(order)}

    mmsi_idx = df.groupby("mmsi").indices
    trip_idx = df.groupby("trip_id").indices
    ts_arr = df.timestamp.to_numpy()
    order_ts = np.argsort(ts_arr, kind="stable")
    ts_sorted = ts_arr[order_ts]
    dfx = df.x_m.to_numpy(); dfy = df.y_m.to_numpy()
    dfcog = np.radians(df.cog.to_numpy())
    dfvx = df.vx_rep.to_numpy(); dfvy = df.vy_rep.to_numpy()
    dfs = df.s_nm.to_numpy(); dfd = df.d_nm.to_numpy()
    df_sog = df.sog.to_numpy(); df_cog = df.cog.to_numpy()

    rng = np.random.default_rng(args.seed)
    audit = {"n_trips": int(len(trips)),
             "trips_per_split": {k: int(sum(1 for v in split_of.values() if v == k)) for k in ("train", "val", "test")},
             "event_dates_in_pool": sorted(set(df.date.unique()) & excluded), "trip_overlap": 0}
    assert not audit["event_dates_in_pool"]

    for split, n_target in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        sid = np.array([t for t, v in split_of.items() if v == split])
        rng.shuffle(sid)
        X_raw = np.zeros((n_target, T_STEPS, 7), np.float32)
        X_rf = np.zeros((n_target, T_STEPS), np.float32)
        X_rm = np.zeros((n_target, T_STEPS), np.float32)
        X_w = np.zeros((n_target, T_STEPS), np.float32)
        X_nb = np.zeros((n_target, T_STEPS), np.int16)
        EP = np.zeros((n_target, T_STEPS, 4), np.float32)   # |h_c|, |dh_c|/dt, D, C
        RV = np.zeros((n_target, T_STEPS, 2), np.float32)   # 场残差矢量 (r_x, r_y)
        FRN = np.zeros((n_target, T_STEPS, 3), np.float32)  # s-s0, d, dchi
        K_NB = 32
        NB_XY = np.zeros((n_target, K_NB, 2), np.float32)
        NB_V = np.zeros((n_target, K_NB, 2), np.float32)
        NB_R = np.zeros((n_target, K_NB), np.float32)
        NB_W = np.zeros((n_target, K_NB), np.float32)
        NB_M = np.zeros((n_target, K_NB), bool)
        NB_SEQ = np.zeros((n_target, K_NB, 5, 4), np.float32)
        DEP = np.zeros(n_target, np.int8)
        X_t = np.zeros((n_target, T_STEPS), np.float32)
        X_mask = np.zeros((n_target, T_STEPS), bool)
        lab = {k: np.zeros(n_target, int) for k in ("class", "cell", "delay")}
        meta = []
        i = 0
        tries = 0
        while i < n_target and tries < n_target * 6:
            tries += 1
            trip = sid[tries % len(sid)]
            idx_tr = trip_idx[trip]
            ts_g = ts_arr[idx_tr]
            if len(ts_g) < 27:
                continue
            j = int(rng.integers(24, len(ts_g)))
            t_dec = float(ts_g[j])
            sel_w = (ts_g > t_dec - WIN_SEC) & (ts_g <= t_dec)
            w_idx = idx_tr[sel_w][-T_STEPS:]
            if len(w_idx) < 8:
                continue
            m = len(w_idx)

            cell = int(rng.choice(4, p=[0.40, 0.20, 0.20, 0.20]))
            behav_flag, obs_flag = cell in (2, 3), cell in (1, 3)
            delay = int(rng.choice(DELAYS)) if behav_flag else 0
            kind = str(rng.choice(["turn_back", "deviation", "stall", "sinuous", "speed_drop"])) if behav_flag else "none"

            t = ts_arr[w_idx].astype(float)
            pos = np.column_stack([dfx[w_idx], dfy[w_idx]]).astype(float)
            sog = df_sog[w_idx].astype(float)
            cog = np.radians(df_cog[w_idx].astype(float))
            res = {"pos": pos, "sog": sog, "cog": cog}
            if behav_flag:
                t0 = t_dec - delay
                if t0 < t[0] + 60:
                    continue
                A = {1: dict(tb=120, dv=20, st=2.0, sn=5, sd=0.60),
                     2: dict(tb=135, dv=40, st=1.5, sn=10, sd=0.50),
                     3: dict(tb=150, dv=60, st=1.0, sn=15, sd=0.40),
                     4: dict(tb=165, dv=90, st=0.7, sn=18, sd=0.30),
                     5: dict(tb=180, dv=120, st=0.5, sn=20, sd=0.20)}[args.amp_level]
                kw = {"turn_back": {"delta_deg": A["tb"]},
                      "deviation": {"delta_deg": A["dv"]},
                      "stall": {"target_kn": A["st"]},
                      "sinuous": {"amp_deg": A["sn"]},
                      "speed_drop": {"factor": A["sd"]}}[kind]
                res = inject.maneuver(kind, t, pos, sog, cog, t0, **kw)
                if args.inject_mode == "crude":
                    # hard-setting injection: instantaneous COG step + SOG step, positions untouched
                    dpsi = np.deg2rad(A["tb"] if kind == "turn_back" else A["dv"])
                    step = (t >= t0).astype(float)
                    res["cog"] = cog + dpsi * step
                    res["sog"] = sog * (1.0 - 0.7 * step)

            # formation departure, injected at observation level before feature computation
            if args.inject_departure > 0 and rng.random() < args.inject_departure:
                k0 = max(1, m - args.dep_steps - 1)
                A = np.deg2rad(args.dep_angle) * (1.0 if rng.random() < 0.5 else -1.0)
                cut = np.clip((np.arange(m) - k0) / float(args.dep_steps), 0.0, 1.0)
                psi2 = res["cog"] + A * cut
                spd2 = res["sog"] * 0.514444 * (1.0 - args.dep_loss * cut)
                pos2 = res["pos"].copy()
                for k in range(k0 + 1, m):
                    dt = max(1.0, t[k] - t[k - 1])
                    v = 0.5 * (spd2[k] + spd2[k - 1]) * dt
                    pos2[k] = pos2[k - 1] + v * np.array([np.sin(psi2[k]), np.cos(psi2[k])])
                res = dict(res)
                res["pos"], res["cog"] = pos2, psi2
                res["sog"] = spd2 / 0.514444
                DEP[i] = 1

            if obs_flag:
                ok = str(rng.choice(["teleport", "speed_spike", "drift"]))
                k = int(rng.integers(m // 2, m))
                res = {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in res.items()}
                if ok == "teleport":
                    off, ang = rng.uniform(500, 2000), rng.uniform(-np.pi, np.pi)
                    res["pos"][k] += off * np.array([np.sin(ang), np.cos(ang)])
                elif ok == "speed_spike":
                    res["sog"][k:k + 2] *= rng.uniform(3, 5)
                else:
                    u = np.clip((t - (t_dec - 300)) / 300.0, 0, 1)
                    env = 0.5 - 0.5 * np.cos(2 * np.pi * u)
                    res["pos"] += (rng.uniform(500, 1500) * env)[:, None] * np.array([1.0, 0.0])

            vx = res["sog"] * KNOT * np.sin(res["cog"])
            vy = res["sog"] * KNOT * np.cos(res["cog"])

            # 逐时间步：邻船残差 + 自身可信度
            for k in range(m):
                tk = t[k]
                lo = np.searchsorted(ts_sorted, tk - 300, "left")
                hi = np.searchsorted(ts_sorted, tk, "right")
                cand = order_ts[lo:hi]
                if len(cand) == 0:
                    continue
                dd = np.hypot(dfx[cand] - res["pos"][k, 0], dfy[cand] - res["pos"][k, 1])
                sel = (dd <= RADIUS_M) & (np.abs(dfd[cand]) <= 1.0)
                if sel.sum() < MIN_NBRS:
                    X_nb[i, k] = int(sel.sum())
                    continue
                cnd = cand[sel]
                dsel = np.abs(wrap(dfcog[cnd] - res["cog"][k])) <= MAX_DCOG
                cnd = cnd[dsel]
                if len(cnd) < MIN_NBRS:
                    X_nb[i, k] = len(cnd)
                    continue
                px, py = dfx[cnd], dfy[cnd]
                nvx, nvy = dfvx[cnd], dfvy[cnd]
                X_rm[i, k] = np.hypot(vx[k] - nvx.mean(), vy[k] - nvy.mean())
                _bgx, _bgy = nvx.mean(), nvy.mean()
                ux = (px - res["pos"][k, 0]) / RADIUS_M
                uy = (py - res["pos"][k, 1]) / RADIUS_M
                Xm = np.column_stack([np.ones_like(ux), ux, uy])
                try:
                    th, *_ = np.linalg.lstsq(Xm, np.column_stack([nvx, nvy]), rcond=None)
                    X_rf[i, k] = np.hypot(vx[k] - th[0, 0], vy[k] - th[0, 1])
                    RV[i, k, 0] = vx[k] - th[0, 0]
                    RV[i, k, 1] = vy[k] - th[0, 1]
                    EP[i, k, 0] = np.hypot(th[0, 0], th[0, 1])
                    EP[i, k, 2] = (th[1, 0] + th[2, 1]) / RADIUS_M
                    EP[i, k, 3] = (th[1, 1] - th[2, 0]) / RADIUS_M
                except np.linalg.LinAlgError:
                    pass
                X_nb[i, k] = len(cnd)

                if k == m - 1:
                    idx_nb = cnd[:K_NB]
                    kk = len(idx_nb)
                    NB_XY[i, :kk, 0] = dfx[idx_nb] - res["pos"][k, 0]
                    NB_XY[i, :kk, 1] = dfy[idx_nb] - res["pos"][k, 1]
                    NB_V[i, :kk, 0] = dfvx[idx_nb]
                    NB_V[i, :kk, 1] = dfvy[idx_nb]
                    NB_M[i, :kk] = True
                    for jj, gi in enumerate(idx_nb):
                        w_j = 1.0
                        if cell in (1, 3) and rng.random() < 0.15:
                            if rng.random() < 0.5:
                                off = rng.uniform(500, 2000)
                                ang = rng.uniform(-np.pi, np.pi)
                                NB_XY[i, jj, 0] += off * np.sin(ang)
                                NB_XY[i, jj, 1] += off * np.cos(ang)
                                e_j = off / 70.0
                            else:
                                mult = rng.uniform(3.0, 5.0)
                                NB_V[i, jj, 0] *= mult
                                NB_V[i, jj, 1] *= mult
                                e_j = abs(mult - 1.0) * float(np.hypot(dfvx[gi], dfvy[gi]))
                            w_j = float(np.exp(-max(0.0, e_j - 2.5) ** 2 / 8.0))
                        NB_W[i, jj] = w_j
                        mm = int(df.mmsi.iloc[gi])
                        seg = mmsi_idx.get(mm)
                        if seg is not None:
                            pin = int(np.searchsorted(ts_arr[seg], t[k], side="right"))
                            hist_idx = seg[max(0, pin - 5):pin]
                            if len(hist_idx):
                                pad = 5 - len(hist_idx)
                                NB_SEQ[i, jj, pad:, 0] = dfx[hist_idx] - dfx[gi]
                                NB_SEQ[i, jj, pad:, 1] = dfy[hist_idx] - dfy[gi]
                                NB_SEQ[i, jj, pad:, 2] = dfvx[hist_idx]
                                NB_SEQ[i, jj, pad:, 3] = dfvy[hist_idx]

            # 自身运动学可信度（三点中心差分）
            e = np.full(m, np.nan)
            for k in range(1, m - 1):
                dt = t[k + 1] - t[k - 1]
                if 5 <= dt <= 1200:
                    vc = (res["pos"][k + 1] - res["pos"][k - 1]) / dt
                    vb = 0.5 * (np.array([vx[k], vy[k]]) + np.array([vx[k - 1], vy[k - 1]]))
                    e[k] = np.linalg.norm(vc - vb)
            X_w[i, :m] = np.exp(-np.maximum(0.0, np.nan_to_num(e, nan=0.0) - 2.5) ** 2 / 8.0)

            ss, dd_, _, tn = corr.corridor_coords(res["pos"][:, 0] / NM, res["pos"][:, 1] / NM, poly)
            if args.zone == "a" and ss[-1] > 16.0:
                continue
            if args.zone == "b" and ss[-1] <= 16.0:
                continue
            FRN[i, :m, 0] = ss - ss[0]
            FRN[i, :m, 1] = dd_
            FRN[i, :m, 2] = (res["cog"] - tn + np.pi) % (2 * np.pi) - np.pi
            for k in range(1, m):
                dtk = max(t[k] - t[k - 1], 1.0)
                EP[i, k, 1] = abs(EP[i, k, 0] - EP[i, k - 1, 0]) / dtk
            X_t[i, :m] = t - t[0]
            X_raw[i, :m, 0] = res["pos"][:, 0] - res["pos"][0, 0]
            X_raw[i, :m, 1] = res["pos"][:, 1] - res["pos"][0, 1]
            X_raw[i, :m, 2] = vx
            X_raw[i, :m, 3] = vy
            X_raw[i, :m, 4] = res["sog"]
            X_raw[i, :m, 5] = np.sin(res["cog"])
            X_raw[i, :m, 6] = np.cos(res["cog"])
            X_mask[i, :m] = True
            lab["class"][i] = {"none": 0, "turn_back": 1, "deviation": 2, "stall": 3, "sinuous": 4, "speed_drop": 5}[kind]
            lab["cell"][i] = cell
            lab["delay"][i] = delay
            meta.append({"i": i, "trip": trip, "cell": cell, "kind": kind, "delay": delay})
            i += 1

        out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{split}.npz", raw=X_raw, r_field=X_rf, r_mean=X_rm,
                            w_self=X_w, n_nb=X_nb, e_phys=EP, r_vec=RV, frenet3=FRN,
                            nb_xy=NB_XY, nb_v=NB_V, nb_w=NB_W, nb_mask=NB_M, nb_seq=NB_SEQ, t=X_t, mask=X_mask,
                            label_class=lab["class"], label_cell=lab["cell"], label_delay=lab["delay"], label_depart=DEP)
        counts = {c: int((lab["cell"] == c).sum()) for c in range(4)}
        print(f"[07] {split}: {i} samples, cells {counts}", flush=True)
        audit[f"{split}_cells"] = counts

    (Path(args.outdir) / "manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
