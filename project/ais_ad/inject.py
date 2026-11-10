"""行为异常注入：在真实正常轨迹上生成"物理自洽"的机动异常。

这是本课题测试集的核心。关键原则：

**不要直接改 COG/SOG 字段，先构造航向/速度剖面，再积分出位置。**

否则一个"折返"会同时表现为运动学不相容，被观测门控抓走——行为异常就被
错标成了观测异常，2×2 实验设计（运动学一致 × 行为异常）直接失效。

所有注入都受转向率与加速度上限约束，生成的轨迹与真实操纵在运动学上不可区分。
"""
from __future__ import annotations

import numpy as np

from .kinematics import velocity_vectors


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def ramp(t, t0, dur):
    """0 → 1 的平滑阶跃，dur 之后保持 1。"""
    return _smoothstep((np.asarray(t, dtype=float) - t0) / max(dur, 1e-6))


def _index(t, t0):
    i = int(np.searchsorted(t, t0, side="left"))
    return min(max(i, 0), len(t) - 1)


def reintegrate(t, pos, sog, cog, i0):
    """从 i0 开始，用给定的速度/航向序列重新积分位置（梯形法）。"""
    v = velocity_vectors(sog, cog)
    p = np.array(pos, dtype=float, copy=True)
    if i0 >= len(t) - 1:
        return p
    dt = np.diff(np.asarray(t, dtype=float)[i0:])
    avg = 0.5 * (v[i0 + 1:] + v[i0:-1])
    p[i0 + 1:] = p[i0] + np.cumsum(avg * dt[:, None], axis=0)
    return p


def _base(t, pos, sog, cog, t0):
    t = np.asarray(t, dtype=float)
    return (
        t,
        np.asarray(pos, dtype=float),
        np.asarray(sog, dtype=float),
        np.asarray(cog, dtype=float),
        _index(t, t0),
    )


def turn_back(t, pos, sog, cog, t0, delta_deg=180.0, dur=600.0,
              rot_max_deg_s=3.0, sign=1):
    """平滑转向 delta_deg 并保持新航向（折返/大角度偏转）。"""
    t, pos, sog, cog, i0 = _base(t, pos, sog, cog, t0)
    delta = np.deg2rad(delta_deg) * sign
    need = 1.5 * abs(delta) / np.deg2rad(rot_max_deg_s)
    dur = float(max(dur, need))
    env = ramp(t, t0, dur)
    chi = cog.copy()
    chi[i0:] = cog[i0] + delta * env[i0:]
    return {
        "pos": reintegrate(t, pos, sog, chi, i0),
        "sog": sog.copy(),
        "cog": chi,
        "t0": t0,
        "t1": t0 + dur,
        "kind": "turn_back",
        "consistent": True,
    }


def deviation(t, pos, sog, cog, t0, delta_deg=30.0, dur=180.0, hold=900.0,
              rot_max_deg_s=3.0, sign=1):
    """偏出航道 delta_deg，保持 hold 秒后回到原航向。"""
    t, pos, sog, cog, i0 = _base(t, pos, sog, cog, t0)
    delta = np.deg2rad(delta_deg) * sign
    need = 1.5 * abs(delta) / np.deg2rad(rot_max_deg_s)
    dur = float(max(dur, need))
    out = ramp(t, t0, dur)
    back = ramp(t, t0 + hold, dur)
    chi = cog.copy()
    chi[i0:] = cog[i0] + delta * (out[i0:] - back[i0:])
    return {
        "pos": reintegrate(t, pos, sog, chi, i0),
        "sog": sog.copy(),
        "cog": chi,
        "t0": t0,
        "t1": t0 + hold + dur,
        "kind": "deviation",
        "consistent": True,
    }


def stall(t, pos, sog, cog, t0, target_kn=0.8, dur=240.0):
    """平滑减速到 target_kn 并保持（停车等待/被扣）。"""
    t, pos, sog, cog, i0 = _base(t, pos, sog, cog, t0)
    env = ramp(t, t0, dur)
    s = sog.copy()
    s[i0:] = sog[i0] * (1.0 - env[i0:]) + target_kn * env[i0:]
    return {
        "pos": reintegrate(t, pos, s, cog, i0),
        "sog": s,
        "cog": cog.copy(),
        "t0": t0,
        "t1": t0 + dur,
        "kind": "stall",
        "consistent": True,
    }


def sinuous(t, pos, sog, cog, t0, amp_deg=12.0, period=600.0, dur=1800.0):
    """蛇行：航向正弦摆动，包络在两端回到 0。"""
    t, pos, sog, cog, i0 = _base(t, pos, sog, cog, t0)
    u = (t - t0) / max(dur, 1e-6)
    env = np.where((u >= 0.0) & (u <= 1.0), np.sin(np.pi * np.clip(u, 0, 1)) ** 2, 0.0)
    osc = np.sin(2.0 * np.pi * (t - t0) / max(period, 1e-6))
    chi = cog.copy()
    chi[i0:] = cog[i0] + np.deg2rad(amp_deg) * osc[i0:] * env[i0:]
    return {
        "pos": reintegrate(t, pos, sog, chi, i0),
        "sog": sog.copy(),
        "cog": chi,
        "t0": t0,
        "t1": t0 + dur,
        "kind": "sinuous",
        "consistent": True,
    }


def speed_drop(t, pos, sog, cog, t0, factor=0.3, dur=120.0):
    """速度乘以 factor 并保持（异常减速）。位置随之重积分。"""
    t, pos, sog, cog, i0 = _base(t, pos, sog, cog, t0)
    env = ramp(t, t0, dur)
    s = sog.copy()
    s[i0:] = sog[i0] * (1.0 - (1.0 - factor) * env[i0:])
    return {
        "pos": reintegrate(t, pos, s, cog, i0),
        "sog": s,
        "cog": cog.copy(),
        "t0": t0,
        "t1": t0 + dur,
        "kind": "speed_drop",
        "consistent": True,
    }


def speed_spike(t, pos, sog, cog, t0, factor=5.0, dur=60.0):
    """只改报告速度、不改位置：运动学不相容，属于观测异常而非行为异常。"""
    t, pos, sog, cog, i0 = _base(t, pos, sog, cog, t0)
    env = ramp(t, t0, dur) - ramp(t, t0 + dur, dur)
    s = sog.copy()
    s[i0:] = sog[i0] * (1.0 + (factor - 1.0) * env[i0:])
    return {
        "pos": pos.copy(),
        "sog": s,
        "cog": cog.copy(),
        "t0": t0,
        "t1": t0 + 2 * dur,
        "kind": "speed_spike",
        "consistent": False,
    }


def teleport(t, pos, sog, cog, t0, offset_m=1500.0, angle_deg=45.0):
    """单条报文位置跳变：观测异常。"""
    t, pos, sog, cog, i0 = _base(t, pos, sog, cog, t0)
    p = pos.copy()
    a = np.deg2rad(angle_deg)
    p[i0] = p[i0] + offset_m * np.array([np.sin(a), np.cos(a)])
    return {
        "pos": p,
        "sog": sog.copy(),
        "cog": cog.copy(),
        "t0": t0,
        "t1": t0,
        "kind": "teleport",
        "consistent": False,
    }


MANEUVERS = {
    "turn_back": turn_back,
    "deviation": deviation,
    "stall": stall,
    "sinuous": sinuous,
    "speed_drop": speed_drop,
    "speed_spike": speed_spike,
    "teleport": teleport,
}


def maneuver(kind, t, pos, sog, cog, t0, **kwargs):
    """按名称调用注入器；未知类型直接报错，避免静默生成错误样本。"""
    if kind not in MANEUVERS:
        raise KeyError(f"unknown maneuver: {kind}")
    return MANEUVERS[kind](t, pos, sog, cog, t0, **kwargs)


def group_deceleration(t, ships, t0, factor=0.6, ramp_dur=120.0):
    """群体减速：只改邻船速度并重积分位置，目标船保持原样。

    ships: [{"pos": (T,2), "sog": (T,), "cog": (T,)}, ...]
    这是六宫格里最关键的负类：整片海域慢下来，但每艘船都没做异常机动。
    """
    out = []
    for s in ships:
        t_arr = np.asarray(t, dtype=float)
        i0 = _index(t_arr, t0)
        env = ramp(t_arr, t0, ramp_dur)
        sog2 = np.array(s["sog"], dtype=float, copy=True)
        sog2[i0:] = np.asarray(s["sog"], dtype=float)[i0] * (
            1.0 - (1.0 - factor) * env[i0:]
        )
        pos2 = reintegrate(t_arr, s["pos"], sog2, s["cog"], i0)
        out.append({"pos": pos2, "sog": sog2, "cog": np.array(s["cog"], dtype=float, copy=True)})
    return out


def segment_labels(t, t0, t1, pad=0.0):
    """注入段的 0/1 标签（1 = 异常段内）。"""
    t = np.asarray(t, dtype=float)
    return ((t >= t0 - pad) & (t <= t1 + pad)).astype(np.int64)
