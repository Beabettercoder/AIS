"""运动学相容性残差。

核心思想：相邻两次有效报文之间的位置差分速度，应当与报告的对地速度自洽。
不相容只能说明"这条报文/这段轨迹与船体运动学不一致"，不能直接等同于欺骗——
论文里请一律写作"运动学相容性评分"，不要写"欺骗判定"。
"""
from __future__ import annotations

import numpy as np

from .geo import KNOT_MPS, wrap_pi


def velocity_vectors(sog_kn, cog_rad):
    """(s, chi) -> 平面速度向量 (..., 2)，x 东 y 北。"""
    s = np.asarray(sog_kn, dtype=float) * KNOT_MPS
    chi = np.asarray(cog_rad, dtype=float)
    return np.stack([s * np.sin(chi), s * np.cos(chi)], axis=-1)


def pseudo_huber(x, delta=1.0):
    """平滑的 |x| 近似，避免个别跳变点主导损失。"""
    x = np.asarray(x, dtype=float)
    return delta * (np.sqrt(1.0 + (x / delta) ** 2) - 1.0)


def kinematic_residual(pos, t, sog_kn, cog_rad, valid=None, min_dt=5.0, max_dt=600.0):
    """逐报文的运动学残差。

    Parameters
    ----------
    pos : (T, 2) 局部平面坐标（米）
    t   : (T,)   时间戳（秒，严格递增）
    sog_kn : (T,) 对地航速（节）
    cog_rad: (T,) 对地航向（弧度，自北顺时针）
    valid  : (T,) bool，报文是否可用；None 视为全可用
    min_dt : 小于该间隔的差分视为噪声，不计算残差
    max_dt : 大于该间隔视为断点，不计算残差（另记 long_gap）

    Returns
    -------
    dict
        e_pos      (T,) 位置-速度不相容度（m/s，已做 pseudo-Huber 平滑）
        e_raw      (T,) 原始不相容度（m/s）
        turn_rate  (T,) |dchi/dt|（rad/s）
        dt         (T,) 与上一条有效报文的间隔（秒）
        ok         (T,) 该步残差是否有效
        long_gap   (T,) 该步是否跨越长断点
    """
    pos = np.asarray(pos, dtype=float)
    t = np.asarray(t, dtype=float)
    sog_kn = np.asarray(sog_kn, dtype=float)
    cog_rad = np.asarray(cog_rad, dtype=float)
    T = len(t)
    if valid is None:
        valid = np.ones(T, dtype=bool)
    valid = np.asarray(valid, dtype=bool)

    dt = np.diff(t, prepend=t[0]) if T else np.zeros(0)
    dp = np.diff(pos, axis=0, prepend=pos[:1]) if T else np.zeros((0, 2))
    v = velocity_vectors(sog_kn, cog_rad)

    # 用前后两条报文速度的均值代表区间平均速度，降低单点噪声
    v_prev = np.roll(v, 1, axis=0)
    v_bar = 0.5 * (v + v_prev)
    v_bar[0] = v[0]

    ok = np.zeros(T, dtype=bool)
    if T > 1:
        ok[1:] = valid[1:] & valid[:-1] & (dt[1:] >= min_dt) & (dt[1:] <= max_dt)

    safe_dt = np.maximum(dt, 1e-6)
    pred = np.where(ok[:, None], dp / safe_dt[:, None], 0.0)
    e_raw = np.linalg.norm(pred - v_bar, axis=-1)
    e_pos = pseudo_huber(e_raw, delta=1.0)

    dchi = np.abs(wrap_pi(np.diff(cog_rad, prepend=cog_rad[0]))) if T else np.zeros(0)
    turn_rate = np.where(ok, dchi / safe_dt, 0.0)

    long_gap = np.zeros(T, dtype=bool)
    if T > 1:
        long_gap[1:] = dt[1:] > max_dt

    return {
        "e_pos": e_pos,
        "e_raw": e_raw,
        "turn_rate": turn_rate,
        "dt": dt,
        "ok": ok,
        "long_gap": long_gap,
    }


def obs_features(res, sog_kn, cog_rad, gap_flag=None):
    """把残差整理成观测门控的输入特征 (T, F)。

    列：e_pos, e_raw, turn_rate, log1p(dt), speed(m/s), gap_flag, ok
    """
    speed = np.abs(np.asarray(sog_kn, dtype=float)) * KNOT_MPS
    gap = np.zeros_like(res["ok"], dtype=float) if gap_flag is None else np.asarray(gap_flag, dtype=float)
    return np.stack(
        [
            res["e_pos"],
            res["e_raw"],
            res["turn_rate"],
            np.log1p(np.maximum(res["dt"], 0.0)),
            speed,
            gap,
            res["ok"].astype(float),
        ],
        axis=-1,
    )
