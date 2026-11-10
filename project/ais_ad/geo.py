"""坐标与角度工具。

约定（全仓库统一）：
- 局部平面坐标 x 向东、y 向北，单位米；
- 角度为弧度，方位角自正北顺时针；
- 速度向量 v = s * [sin(chi), cos(chi)]，s 为 m/s。
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_000.0
KNOT_MPS = 0.514444


def wrap_pi(angle):
    """把角度环绕到 [-pi, pi)。"""
    a = np.asarray(angle, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def to_local_xy(lat, lon, lat0=None, lon0=None):
    """经纬度 -> 局部米制平面（等距圆柱近似，适用于 < 100 km 的窗口）。"""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat0 is None:
        lat0 = float(np.nanmean(lat))
    if lon0 is None:
        lon0 = float(np.nanmean(lon))
    x = np.deg2rad(lon - lon0) * EARTH_R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * EARTH_R
    return np.stack([x, y], axis=-1)


def haversine_m(lat1, lon1, lat2, lon2):
    """球面距离（米），用于校验投影误差。"""
    lat1, lon1, lat2, lon2 = (np.deg2rad(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
