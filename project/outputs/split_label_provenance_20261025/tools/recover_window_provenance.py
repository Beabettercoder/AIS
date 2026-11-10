#!/usr/bin/env python3
"""Recover immutable NPZ window provenance by exact source matching.

Reads existing source/NPZ data; never changes either. New evidence is written
only to --out. Unmatched/ambiguous samples remain unresolved (exit status 2).
Example: python recover_window_provenance.py --parquet FILE --dataset DIR --out DIR
"""
import argparse
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(sog, sine):
    s = np.ascontiguousarray(sog, dtype=np.float32).view(np.uint32)
    c = np.ascontiguousarray(sine, dtype=np.float32).view(np.uint32)
    return (s.astype(np.uint64) << np.uint64(32)) | c.astype(np.uint64)


def recover(parquet, dataset, outdir, excluded_dates=(), departure_steps=None):
    parquet, dataset, outdir = Path(parquet), Path(dataset), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    columns = ['mmsi', 'timestamp', 'x_m', 'y_m', 'sog', 'cog', 'kin_ok']
    df = pd.read_parquet(parquet, columns=columns)
    df = df[df.kin_ok & df.sog.ge(2.0)].copy()
    if excluded_dates:
        dates = pd.to_datetime(df.timestamp, unit='s', utc=True).dt.strftime('%Y-%m-%d')
        df = df[~dates.isin(excluded_dates)].copy()
    df = df.sort_values(['mmsi', 'timestamp']).reset_index(drop=True)
    ts = df.timestamp.to_numpy()
    mmsi = df.mmsi.to_numpy()
    gaps = df.groupby('mmsi')['timestamp'].diff().fillna(9999)
    trip_number = (gaps > 1800).groupby(df.mmsi).cumsum().to_numpy()
    xy = df[['x_m', 'y_m']].to_numpy().astype(np.float64)
    # 07_build_v2 explicitly casts sog/cog to float64 before trigonometry.
    sog = df.sog.to_numpy().astype(np.float64)
    cog = np.radians(df.cog.to_numpy().astype(np.float64))
    sin = np.sin(cog)
    cos = np.cos(cog)
    features = np.column_stack([sog * 0.514444 * sin, sog * 0.514444 * cos,
                                sog, sin, cos]).astype(np.float32)
    keys = signature(features[:, 2], features[:, 3])
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    del df, sog, cog, sin, cos, keys
    source_sha = sha256(parquet)
    all_rows, summary = {}, {}
    for split in ['train', 'val', 'test']:
        path = dataset / (split + '.npz')
        d = np.load(path)
        raw_all, mask_all, time_all = d['raw'], d['mask'], d['t']
        cells, classes, delays = d['label_cell'], d['label_class'], d['label_delay']
        departures = d['label_depart'] if 'label_depart' in d.files else np.zeros(len(raw_all), dtype=int)
        npz_sha = sha256(path)
        rows, unresolved, counts = [], [], collections.Counter()
        for i in range(len(raw_all)):
            mask = mask_all[i]
            m = int(mask.sum())
            if m < 2 or not np.array_equal(mask, np.arange(len(mask)) < m):
                unresolved.append({'sample_index': i, 'reason': 'invalid_mask'})
                continue
            raw, rel_t = raw_all[i, :m], time_all[i, :m]
            cell, cls, delay, depart = int(cells[i]), int(classes[i]), int(delays[i]), int(departures[i])
            if depart and departure_steps is None:
                unresolved.append({'sample_index': i, 'reason': 'departure_config_unknown'})
                continue
            # Exact key match, followed by exact all-five first velocity features.
            key = signature(raw[0, 4:5], raw[0, 5:6])[0]
            lo = np.searchsorted(sorted_keys, key, side='left')
            hi = np.searchsorted(sorted_keys, key, side='right')
            candidates = order[lo:hi]
            counts['first_signature_candidates'] += len(candidates)
            # Both observation perturbations and behavior leave this velocity prefix intact.
            vel_prefix = rel_t < (rel_t[-1] - delay if cell in (2, 3) else np.inf)
            if cell in (1, 3):
                vel_prefix &= np.arange(m) < m // 2
            if depart:
                vel_prefix &= np.arange(m) <= max(1, m - departure_steps - 1)
            # Reintegrate retains the onset anchor's position even when its velocity changes.
            pos_prefix = np.ones(m, dtype=bool)
            if cell in (2, 3):
                anchor = min(m - 1, int(np.searchsorted(rel_t, rel_t[-1] - delay)))
                pos_prefix &= np.arange(m) <= anchor
            if cell in (1, 3):
                pos_prefix &= (np.arange(m) < m // 2) & (rel_t <= rel_t[-1] - 300)
            if depart:
                pos_prefix &= np.arange(m) <= max(1, m - departure_steps - 1)
            hits = []
            for jj in candidates:
                j = int(jj)
                if j + m > len(ts) or mmsi[j] != mmsi[j + m - 1]:
                    continue
                if not np.array_equal(features[j], raw[0, 2:7]):
                    continue
                candidate_t = (ts[j:j + m] - ts[j]).astype(np.float32)
                if not np.array_equal(candidate_t, rel_t):
                    continue
                counts['full_time_candidates'] += 1
                # At least two protected samples must corroborate identity.
                if max(int(vel_prefix.sum()), int(pos_prefix.sum())) < 2:
                    continue
                if not np.array_equal(features[j:j + m][vel_prefix], raw[vel_prefix, 2:7]):
                    continue
                if int(pos_prefix.sum()) >= 2:
                    rel_xy = (xy[j:j + m] - xy[j]).astype(np.float32)
                    if not np.array_equal(rel_xy[pos_prefix], raw[pos_prefix, :2]):
                        continue
                    method = 'exact_time_velocity_and_geometry'
                else:
                    method = 'exact_time_and_velocity_no_clean_geometry'
                hits.append((j, method))
            if len(hits) != 1:
                unresolved.append({'sample_index': i, 'reason': 'unmatched' if not hits else 'ambiguous',
                                   'match_count': len(hits), 'first_signature_candidates': len(candidates),
                                   'matched_mmsi': sorted(set(int(mmsi[j]) for j, _ in hits)),
                                   'protected_velocity_points': int(vel_prefix.sum()),
                                   'protected_geometry_points': int(pos_prefix.sum())})
                continue
            j, method = hits[0]
            timestamps = [int(t) for t in ts[j:j + m]]
            identity_text = str(int(mmsi[j])) + ':' + ','.join(map(str, timestamps))
            row = {'dataset': dataset.name, 'split': split, 'sample_index': i,
                   'mmsi': int(mmsi[j]), 'start_timestamp': timestamps[0],
                   'trip_id': str(int(mmsi[j])) + ':' + str(int(trip_number[j])),
                   'start_utc': datetime.fromtimestamp(timestamps[0], timezone.utc).isoformat(),
                   'decision_utc': datetime.fromtimestamp(timestamps[-1], timezone.utc).isoformat(),
                   'source_daily_files': sorted(set(datetime.fromtimestamp(t, timezone.utc).strftime('AIS_%Y_%m_%d.zip') for t in timestamps)),
                   'decision_timestamp': timestamps[-1], 'timestamps': timestamps,
                   'source_window_sha256': hashlib.sha256(identity_text.encode()).hexdigest(),
                   'valid_steps': m, 'label_cell': cell, 'label_class': cls,
                   'label_delay': delay, 'label_depart': depart,
                   'behavior_onset_timestamp': timestamps[-1] - delay if cell in (2, 3) else None,
                   'match_method': method, 'protected_velocity_points': int(vel_prefix.sum()),
                   'protected_geometry_points': int(pos_prefix.sum()),
                   'source_parquet': str(parquet), 'source_sha256': source_sha,
                   'npz_file': str(path), 'npz_sha256': npz_sha}
            rows.append(row)
            counts[method] += 1
        with open(outdir / (split + '_provenance.jsonl'), 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
        (outdir / (split + '_unresolved.json')).write_text(json.dumps(unresolved, indent=2))
        identities = collections.Counter(row['source_window_sha256'] for row in rows)
        summary[split] = {'total_samples': len(raw_all), 'unique_matched': len(rows),
                          'unresolved': len(unresolved), 'unique_mmsi': len(set(r['mmsi'] for r in rows)),
                          'unique_source_windows': len(identities),
                          'repeated_source_window_samples': sum(n - 1 for n in identities.values()),
                          'mmsi': sorted(set(r['mmsi'] for r in rows)),
                          'counts': dict(counts), 'npz_sha256': npz_sha}
        all_rows[split] = rows
        print(json.dumps({'dataset': dataset.name, 'split': split, **{k: v for k, v in summary[split].items() if k != 'mmsi'}}), flush=True)
    overlaps = {}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        va, vb = {r['mmsi'] for r in all_rows[a]}, {r['mmsi'] for r in all_rows[b]}
        wa = {r['source_window_sha256'] for r in all_rows[a]}
        wb = {r['source_window_sha256'] for r in all_rows[b]}
        overlaps[a + '__' + b] = {'mmsi': sorted(va & vb), 'mmsi_count': len(va & vb),
                                 'identical_source_window_count': len(wa & wb)}
    summary['cross_split_overlap'] = overlaps
    summary['source_parquet'] = str(parquet)
    summary['source_sha256'] = source_sha
    summary['matching_scope'] = ('All kin_ok and SOG>=2 source records. No assumed split or vessel '
                                  'exclusions are used to force uniqueness. Exact float32 checks; '
                                  'unmatched and ambiguous windows are not assigned.')
    summary['excluded_dates'] = list(excluded_dates)
    summary['complete'] = all(summary[s]['unresolved'] == 0 for s in ['train', 'val', 'test'])
    (outdir / 'provenance_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--excluded-dates', nargs='*', default=[])
    p.add_argument('--departure-steps', type=int, default=None)
    a = p.parse_args()
    result = recover(a.parquet, a.dataset, a.out, a.excluded_dates, a.departure_steps)
    raise SystemExit(0 if result['complete'] else 2)
