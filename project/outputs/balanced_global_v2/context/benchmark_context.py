"""Synthetic scale check, not model-inference latency or a paper result."""
import gc
import json
import time
from pathlib import Path

import numpy as np
from context_features import BackgroundIndex, ContextConfig, KNOT_MPS, build_window


def run(n_reports):
    ships, windows = 1000, 100
    points = n_reports // ships
    rng = np.random.default_rng(42)
    # 1,000 coexisting ships in 200 x 200 km; independent spatial layout;
    # 60-second reports with 1,000 reports in each minute bucket.
    ids = np.repeat(np.arange(200000000, 200000000+ships).astype(str), points)
    times = np.tile(np.arange(points)*60., ships)
    origins = rng.uniform(-100000, 100000, (ships, 2))
    x = np.repeat(origins[:, 0], points)
    y = np.repeat(origins[:, 1], points) + times * 10 * KNOT_MPS
    obs = dict(mmsi=ids, time_s=times, x_m=x, y_m=y,
               sog_kn=np.full(n_reports, 10.), cog_deg=np.zeros(n_reports))
    cfg = ContextConfig()
    start = time.perf_counter()
    index = BackgroundIndex(obs, cfg)
    construction_seconds = time.perf_counter()-start
    index_bytes = sum(a.nbytes for a in index.data.values()) + sum(a.nbytes for a in index.values.values())
    del obs, ids, times, x, y
    gc.collect()
    query_rows, accepted, timings = [], [], []
    for j in range(windows):
        ship = j % ships
        end = (points-1-j)*60.
        target_times = np.arange(end-20*60., end+1, 60.)
        target = dict(mmsi=np.full(len(target_times), str(200000000+ship)),
                      time_s=target_times, x_m=np.full(len(target_times), origins[ship, 0]),
                      y_m=origins[ship, 1]+target_times*10*KNOT_MPS,
                      sog_kn=np.full(len(target_times), 10.), cog_deg=np.zeros(len(target_times)))
        start = time.perf_counter()
        result = build_window(target, index, decision_time_s=end)
        timings.append(time.perf_counter()-start)
        for t in result["time_s"][result["mask"]]:
            lo, hi = np.searchsorted(index.data["time_s"], [t-cfg.freshness_seconds, t], side="left")
            hi = np.searchsorted(index.data["time_s"], t, side="right")
            query_rows.append(int(hi-lo))
        accepted.extend(result["valid_neighbor_count"].tolist())
    return dict(synthetic_only=True, reports=n_reports, ships=ships, measured_windows=windows,
                reports_interval_s=60, steps_per_window=20, index_construction_s=construction_seconds,
                immutable_array_bytes=index_bytes, per_window_median_s=float(np.median(timings)),
                per_window_p95_s=float(np.quantile(timings, .95)),
                measured_total_window_s=float(sum(timings)),
                estimated_3000_windows_s=float(np.mean(timings)*3000),
                query_time_slice_rows_median=float(np.median(query_rows)),
                valid_neighbor_count_median=float(np.median(accepted)),
                query_strategy="two binary searches on time, deduplicate 300s slice, spatial filter; never scan all reports")


if __name__ == "__main__":
    rows = []
    for n in (1000000, 2000000):
        rows.append(run(n))
        print(json.dumps(rows[-1]), flush=True)
        gc.collect()
    Path(__file__).with_name("performance_check.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
