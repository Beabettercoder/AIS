"""事件级指标。

这一层比模型更容易出错，而且直接决定审稿人信不信你的结论：
- 窗口级 F1 会被高度重叠的窗口放大，不能单独作为主指标；
- 主指标应落在"事件"上：召回、误报/千船小时、提前量；
- 置信区间必须按航次/事件聚类重采样，不能把窗口当独立样本。
"""
from __future__ import annotations

import numpy as np


def roc_auc(y_true, score):
    """秩和法 AUC，含并列处理。"""
    y = np.asarray(y_true).ravel()
    s = np.asarray(score, dtype=float).ravel()
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks_sorted = np.arange(1, y.size + 1, dtype=float)

    i = 0
    while i < y.size:
        j = i
        while j + 1 < y.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks_sorted[i:j + 1] = ranks_sorted[i:j + 1].mean()
        i = j + 1

    ranks = np.empty(y.size, dtype=float)
    ranks[order] = ranks_sorted
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def macro_f1(y_true, y_pred, n_classes=None):
    """宏平均 F1（忽略测试集中未出现的类别）。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if n_classes is None:
        n_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1
    f1s = []
    for c in range(n_classes):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else float("nan")


def match_alarms_to_events(alarms, events, allow_pre=0.0, require_type=False):
    """一对一把报警与真值事件配对。

    alarms : dict(mmsi=(n,), t=(n,), score=(n,))
    events : dict(mmsi=(m,), t0=(m,), t1=(m,)[, type=(m,)])
    allow_pre : 允许报警早于 t0 的秒数（敏感性分析用，主协议取 0）

    配对规则：同船、t ∈ [t0 - allow_pre, t1]、一对一、每个真值取最早可用报警。
    返回 matched / fp / fn / lead（t1 - t_alarm，秒）。
    """
    a_mmsi = np.asarray(alarms["mmsi"])
    a_t = np.asarray(alarms["t"], dtype=float)
    e_mmsi = np.asarray(events["mmsi"])
    e_t0 = np.asarray(events["t0"], dtype=float)
    e_t1 = np.asarray(events["t1"], dtype=float)

    used = np.zeros(len(a_t), dtype=bool)
    matched, lead = [], []
    order = np.argsort(e_t0, kind="mergesort")
    for ei in order:
        cand = np.flatnonzero(
            (~used)
            & (a_mmsi == e_mmsi[ei])
            & (a_t >= e_t0[ei] - allow_pre)
            & (a_t <= e_t1[ei])
        )
        if cand.size == 0:
            continue
        ai = cand[np.argmin(a_t[cand])]
        used[ai] = True
        matched.append((int(ei), int(ai)))
        lead.append(float(e_t1[ei] - a_t[ai]))

    fp = np.flatnonzero(~used)
    matched_events = {m[0] for m in matched}
    fn = [int(i) for i in range(len(e_t0)) if i not in matched_events]
    return {
        "matched": matched,
        "fp": fp,
        "fn": np.asarray(fn, dtype=int),
        "lead": np.asarray(lead, dtype=float),
    }


def far_per_1000h(n_false_alarms, vessel_hours, dedup_window_s=300.0, n_dup=0):
    """误报率：每千船小时。n_false_alarms 应为去重后的事件数。"""
    if vessel_hours <= 0:
        return float("nan")
    return 1000.0 * float(n_false_alarms) / float(vessel_hours)


def dedup_alarms(mmsi, t, window_s=300.0):
    """同船、同窗口内的连续报警合并为一个事件，返回保留索引。"""
    mmsi = np.asarray(mmsi)
    t = np.asarray(t, dtype=float)
    keep = []
    last = {}
    for i in np.argsort(t, kind="mergesort"):
        m = mmsi[i]
        if m in last and t[i] - last[m] < window_s:
            continue
        keep.append(i)
        last[m] = t[i]
    return np.asarray(keep, dtype=int)


def cluster_bootstrap(values, clusters, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """按 cluster（航次/事件）重采样的百分位置信区间。"""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in pick])
        stats[b] = stat(values[idx])
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return float(stat(values)), lo, hi


def ece(probs, labels, n_bins=15):
    """期望校准误差（多分类取预测类置信度）。"""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels).ravel()
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)
