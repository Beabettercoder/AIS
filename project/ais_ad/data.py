"""数据契约、航次切分与泄漏审计。

真实数据接入时，先冻结本文件里的字段名与单位，再写任何模型代码。
"""
from __future__ import annotations

import numpy as np

# 动态报文最小字段集（缺一不可）
SCHEMA = {
    "mmsi": "int64",
    "ts": "datetime64[s, UTC]",
    "lat": "float64",
    "lon": "float64",
    "sog": "float32",   # 节
    "cog": "float32",   # 度，0-360
    "heading": "float32",  # 度，511 表示不可用
    "nav_status": "int8",
    "rot": "float32",   # 度/分，-128 表示不可用
    "msg_type": "int8",
}

# 静态报文（用于船型、尺度，可选）
STATIC_SCHEMA = {
    "mmsi": "int64",
    "ship_type": "int16",
    "length": "float32",
    "width": "float32",
    "draught": "float32",
}


def segment_trips(ts, gap_s=1800.0):
    """按时间断点切分航次，返回 trip 编号（0..K-1，按时间递增）。"""
    ts = np.asarray(ts, dtype="datetime64[s]").astype("int64")
    if ts.size == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.argsort(ts, kind="mergesort")
    ts_sorted = ts[order]
    brk = np.zeros(ts.size, dtype=bool)
    brk[0] = True
    brk[1:] = np.diff(ts_sorted) > gap_s
    trip_sorted = np.cumsum(brk) - 1
    trip = np.empty(ts.size, dtype=np.int32)
    trip[order] = trip_sorted
    return trip


def assign_time_splits(trip_start, fractions=(0.6, 0.2, 0.2)):
    """按航次起始时间先后划分，返回每个航次的 'train'/'val'/'test'。"""
    trip_start = np.asarray(trip_start, dtype="datetime64[s]").astype("int64")
    n = len(trip_start)
    order = np.argsort(trip_start, kind="mergesort")
    f_train, f_val = fractions[0], fractions[1]
    n_train = int(round(n * f_train))
    n_val = int(round(n * f_val))
    out = np.empty(n, dtype=object)
    for k, idx in enumerate(order):
        if k < n_train:
            out[idx] = "train"
        elif k < n_train + n_val:
            out[idx] = "val"
        else:
            out[idx] = "test"
    return out


def leakage_report(split_by_trip, trip_of_sample, sample_name="sample"):
    """检查同一航次是否跨集合，返回冲突列表（空列表表示无泄漏）。"""
    split_by_trip = dict(split_by_trip)
    trip_of_sample = np.asarray(trip_of_sample)
    bad = []
    seen = {}
    for s, trip in enumerate(trip_of_sample):
        sp = split_by_trip.get(trip)
        if trip in seen and seen[trip] != sp:
            bad.append((sample_name, int(s), trip, seen[trip], sp))
        seen[trip] = sp
    return bad


def window_starts(start_s, end_s, window_s=1200.0, stride_s=30.0, min_history_s=600.0):
    """给定航次的时间范围，生成决策时刻序列（秒）。"""
    t = start_s + min_history_s
    out = []
    while t <= end_s:
        out.append(t)
        t += stride_s
    return np.asarray(out, dtype=float)
