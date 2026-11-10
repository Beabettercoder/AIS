"""观测退化注入。

四类退化，对应方案 5.2 节：
1. 随机丢点          drop_reports
2. 独立位置跳变      inject_spikes
3. 区域共同漂移      common_mode_offset   （可用于"仅位置"或"位置+速度自洽"）
4. 自洽漂移          consistent_drift     （位置与速度同时平移，运动学残差仍很小）

论文里必须把 3 拆成两个子设置分别报告，因为"位置+速度自洽"的退化在
单船运动学层面几乎不可辨识，只能靠岸线/邻船几何等外部参照。
"""
from __future__ import annotations

import numpy as np

from .geo import KNOT_MPS


def drop_reports(rng, T, rate=0.3):
    """返回 keep mask (T,)，True 表示保留。"""
    return rng.random(T) >= rate


def inject_spikes(rng, pos, rate=0.01, lo=500.0, hi=2000.0):
    """按比例给单条报文叠加随机方向位移，返回 (pos2, spike_mask)。"""
    pos = np.asarray(pos, dtype=float)
    T = len(pos)
    hit = rng.random(T) < rate
    ang = rng.uniform(-np.pi, np.pi, T)
    mag = rng.uniform(lo, hi, T)
    off = np.stack([mag * np.sin(ang), mag * np.cos(ang)], axis=-1)
    return pos + hit[:, None] * off, hit


def raised_cosine(t, t0, duration):
    """[t0, t0+duration] 上的 0->1->0 平滑包络。"""
    t = np.asarray(t, dtype=float)
    u = (t - t0) / max(duration, 1e-6)
    inside = (u >= 0.0) & (u <= 1.0)
    s = np.where(inside, 0.5 * (1.0 - np.cos(2.0 * np.pi * u)), 0.0)
    return s


def common_mode_offset(t, t0, duration=300.0, peak=1000.0, direction=(1.0, 0.0)):
    """区域共同漂移的位移向量场 (T,2)，对所有船施加同一偏移。"""
    s = raised_cosine(t, t0, duration) * peak
    d = np.asarray(direction, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    return s[:, None] * d[None, :]


def recompute_sog_cog(pos, t):
    """由漂移后的位置重新计算 SOG/COG，使报文内部自洽。"""
    pos = np.asarray(pos, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(pos) < 2:
        return np.zeros(len(pos)), np.zeros(len(pos))
    dp = np.gradient(pos, t, axis=0)
    sog = np.linalg.norm(dp, axis=1) / KNOT_MPS
    cog = np.arctan2(dp[:, 0], dp[:, 1])
    return sog, cog


def consistent_drift(t, pos, t0, duration=300.0, peak=1000.0, direction=(1.0, 0.0)):
    """位置与速度自洽的共同漂移：位移后重算 SOG/COG。

    返回 (pos2, sog2, cog2)。这类退化不应被包装成"我们已能检测"，
    它需要岸线掩膜或跨船几何一致性等额外证据。
    """
    off = common_mode_offset(t, t0, duration=duration, peak=peak, direction=direction)
    pos2 = np.asarray(pos, dtype=float) + off
    sog2, cog2 = recompute_sog_cog(pos2, t)
    return pos2, sog2, cog2


def degrade_window(rng, t, pos, sog_kn, cog_rad, cfg):
    """把多种退化组合到一个窗口，返回新数组与逐条标签。

    返回 dict: pos, sog, cog, keep, spike, drift, obs_label(T,)
    obs_label: 1 表示该条报文被污染（用于观测头监督），0 表示未污染。
    """
    pos = np.asarray(pos, dtype=float).copy()
    sog = np.asarray(sog_kn, dtype=float).copy()
    cog = np.asarray(cog_rad, dtype=float).copy()
    T = len(t)
    obs = np.zeros(T, dtype=np.int8)

    if cfg.get("drift_peak", 0.0) > 0:
        off = common_mode_offset(
            t,
            cfg["drift_t0"],
            duration=cfg.get("drift_duration", 300.0),
            peak=cfg["drift_peak"],
            direction=cfg.get("drift_direction", (1.0, 0.0)),
        )
        pos = pos + off
        if cfg.get("drift_consistent", False):
            sog, cog = recompute_sog_cog(pos, t)
        window = raised_cosine(t, cfg["drift_t0"], cfg.get("drift_duration", 300.0)) > 1e-3
        obs[window] = 1

    if cfg.get("spike_rate", 0.0) > 0:
        pos, hit = inject_spikes(
            rng, pos, rate=cfg["spike_rate"],
            lo=cfg.get("spike_lo", 500.0), hi=cfg.get("spike_hi", 2000.0),
        )
        obs[hit] = 1

    keep = np.ones(T, dtype=bool)
    if cfg.get("drop_rate", 0.0) > 0:
        keep = drop_reports(rng, T, cfg["drop_rate"])
        obs[~keep] = 0  # 丢点不是"错误值"，观测头不该学成异常值
    return {"pos": pos, "sog": sog, "cog": cog, "keep": keep, "obs_label": obs}
