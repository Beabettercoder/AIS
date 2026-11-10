"""留一法（leave-one-out）同向背景场。

v0 用**加权局部线性回归**，而不是一上来就写神经势函数场：
    v_j ≈ a + B (p_j - p_i),  j 属于目标 i 的同向可信邻船
于是
    V_hat_i = a                   （把邻船流场外推到目标位置）
    div_i   = tr(B)
    curl_i  = B[1, 0] - B[0, 1]   （dv_y/dx - dv_x/dy）
    r_i     = v_i - V_hat_i       （个体残差）

优点：闭式解、无需训练、可解释、可并行；它本身就是 ABL2「普通邻船均值」的
自然升级版，也是神经势场（v1）必须打败的基线。
"""
from __future__ import annotations

import numpy as np

from .geo import wrap_pi


def _weighted_linear_fit(dp, v, w, scale, max_cond=1e3):
    """加权最小二乘拟合 v ≈ a + B dp，返回 (a, grad_x, grad_y, cond)。

    坐标先除以 scale 再拟合：否则设计矩阵的条件数随米制量级膨胀，
    "有效/无效"的判断会变成对单位的依赖。梯度再除回 scale。
    """
    sw = np.sqrt(np.maximum(w, 0.0))
    A = np.stack(
        [np.ones_like(dp[:, 0]), dp[:, 0] / scale, dp[:, 1] / scale], axis=1
    ) * sw[:, None]
    Y = v * sw[:, None]
    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > max_cond:
        return None
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)  # (3, 2)
    return coef[0], coef[1] / scale, coef[2] / scale, cond


def loo_background(
    pos,
    vel,
    cog,
    valid=None,
    w=None,
    radius=9260.0,
    heading_tol=np.deg2rad(45.0),
    min_neighbors=4,
    min_spread_ratio=0.15,
    max_cond=1e3,
    kernel_scale=0.5,
):
    """对每一艘目标船，用它**之外**的同向邻船拟合局部背景场。

    Parameters
    ----------
    pos : (N, 2) 米
    vel : (N, 2) m/s
    cog : (N,)   对地航向（弧度）
    valid : (N,) bool，报文/船舶是否可用
    w   : (N,) 观测可信度 [0,1]，None 时全 1；调用方应传入 stop-grad 后的值
    radius : 邻域半径（米），默认 5 海里

    Returns
    -------
    dict，键均为长度 N 的数组：
        v_hat (N,2) 背景速度外推值；无效位置为 nan
        h_bar (N,2) 邻船可信加权平均速度（不剥离趋势，用于对照）
        resid (N,2) v_i - v_hat
        div   (N,)  散度
        curl  (N,)  旋度
        n_nb  (N,)  参与拟合的邻船数
        spread(N,)  邻船空间展布（加权位置协方差迹的平方根，米）
        ok    (N,)  场是否有效（邻居数、展布、条件数都通过）
        cond  (N,)  设计矩阵条件数
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    cog = np.asarray(cog, dtype=float)
    N = len(pos)
    if valid is None:
        valid = np.ones(N, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if w is None:
        w = np.ones(N, dtype=float)
    w = np.clip(np.asarray(w, dtype=float), 1e-3, 1.0)

    v_hat = np.full((N, 2), np.nan)
    h_bar = np.full((N, 2), np.nan)
    resid = np.full((N, 2), np.nan)
    div = np.full(N, np.nan)
    curl = np.full(N, np.nan)
    n_nb = np.zeros(N, dtype=int)
    spread = np.zeros(N)
    ok = np.zeros(N, dtype=bool)
    cond = np.full(N, np.nan)

    for i in range(N):
        if not valid[i]:
            continue
        d = np.linalg.norm(pos - pos[i], axis=1)
        dh = np.abs(wrap_pi(cog - cog[i]))
        m = valid & (d > 1.0) & (d < radius) & (dh < heading_tol)
        m[i] = False
        idx = np.flatnonzero(m)
        n_nb[i] = idx.size
        if idx.size < min_neighbors:
            continue

        ww = w[idx] * np.exp(-0.5 * (d[idx] / (kernel_scale * radius)) ** 2)
        if ww.sum() <= 0:
            continue
        h_bar[i] = (ww[:, None] * vel[idx]).sum(0) / ww.sum()

        # 邻船空间展布：加权位置协方差
        dp = pos[idx] - pos[i]
        mu = (ww[:, None] * dp).sum(0) / ww.sum()
        cov = ((dp - mu) * ww[:, None]).T @ (dp - mu) / ww.sum()
        spread[i] = float(np.sqrt(max(np.trace(cov), 0.0)))
        if spread[i] < min_spread_ratio * radius:
            continue

        fit = _weighted_linear_fit(dp, vel[idx], ww, scale=radius, max_cond=max_cond)
        if fit is None:
            continue
        a, gx, gy, c = fit
        cond[i] = c
        grad = np.stack([gx, gy], axis=1)  # (2,2)：grad[:,0]=dv/dx, grad[:,1]=dv/dy
        v_hat[i] = a
        resid[i] = vel[i] - a
        div[i] = grad[0, 0] + grad[1, 1]
        curl[i] = grad[1, 0] - grad[0, 1]
        ok[i] = True

    return {
        "v_hat": v_hat,
        "h_bar": h_bar,
        "resid": resid,
        "div": div,
        "curl": curl,
        "n_nb": n_nb,
        "spread": spread,
        "ok": ok,
        "cond": cond,
    }


def background_series(pos, vel, cog, valid=None, w=None, stride=1, **kwargs):
    """对 (T, N, 2) 的序列逐时刻调用 loo_background，返回堆叠后的结果。

    注意：这是参考实现，按时刻循环；生产代码应把 N 维批量化，或用 KD-tree
    预先建邻域，再对小矩阵批量求解。
    """
    pos = np.asarray(pos, dtype=float)
    T, N = pos.shape[:2]
    keys = ("v_hat", "h_bar", "resid", "div", "curl", "n_nb", "spread", "ok", "cond")
    out = {k: [] for k in keys}
    for t in range(0, T, stride):
        v = None if valid is None else valid[t]
        ww = None if w is None else w[t]
        r = loo_background(pos[t], vel[t], cog[t], valid=v, w=ww, **kwargs)
        for k in keys:
            out[k].append(r[k])
    return {k: np.stack(v, axis=0) for k, v in out.items()}


def physical_context(bg, dt=1.0, hbar_prev=None):
    """四维物理态势：||h||、||dh/dt||、散度、旋度。

    同步加速度优先用相邻时刻的 h_bar 差分；缺失时置 0 并由掩码传递。
    """
    h = bg["h_bar"]
    speed = np.linalg.norm(h, axis=-1)
    if hbar_prev is None:
        accel = np.zeros_like(speed)
    else:
        accel = np.linalg.norm(h - hbar_prev, axis=-1) / max(dt, 1e-6)
    return np.stack([speed, accel, bg["div"], bg["curl"]], axis=-1)
