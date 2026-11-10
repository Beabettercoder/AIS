"""Causal observed stopping-place context, never an official port designation."""
from __future__ import annotations

from datetime import datetime
import math

import numpy as np

RADIUS_M = 200.
MIN_OTHER_VESSELS = 3
MEANING = '历史上多艘船稳定停留过的地点，不是官方港口资料，也不代表已确认正常。'


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _epoch(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return result.timestamp() if result.tzinfo is not None else None
    except (TypeError, AttributeError, ValueError, OverflowError, OSError):
        return None


def _xyz(latlon):
    lat, lon = np.deg2rad(latlon).T
    return 6371008.8 * np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def _observations(record):
    obs = record.get('motion_observations')
    if not isinstance(obs, dict):
        return None
    try:
        sog = np.asarray(obs.get('sog_kn'), dtype=float)
        times = np.asarray(obs.get('time_seconds'), dtype=float)
        gates = np.asarray(obs.get('gate_weights'), dtype=float)
        xy = np.asarray(obs.get('xy_m'), dtype=float)
    except (TypeError, ValueError):
        return None
    if sog.ndim != 1 or len(sog) < 2 or times.shape != sog.shape or gates.shape != sog.shape or xy.shape != (len(sog), 2):
        return None
    speed_valid = np.isfinite(sog) & (sog >= 0) & (sog < 102.2)
    if 'sog_reported_mask' in obs:
        try:
            mask = np.asarray(obs['sog_reported_mask'], dtype=float)
        except (TypeError, ValueError):
            return None
        if mask.shape != sog.shape:
            return None
        speed_valid &= np.isfinite(mask) & (mask > .5)
    good = speed_valid & np.isfinite(times) & np.isfinite(gates) & (gates >= .3) & np.isfinite(xy).all(1)
    dt = np.diff(times)
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    plausible = np.maximum(6., (sog[1:] + sog[:-1]) * .514444 + 3.)
    links = good[1:] & good[:-1] & (dt >= 1) & (dt <= 600)
    moving_jump = links & (distance / np.maximum(dt, 1) > plausible)
    links &= ~moving_jump
    return {'sog': sog, 'times': times, 'good': good, 'speed_valid': speed_valid,
            'xy': xy, 'distance': distance, 'links': links, 'moving_jump': moving_jump}


def _positions(record):
    try:
        ll = np.asarray(record.get('latlon'), dtype=float)
        gates = np.asarray(record.get('gate_weights'), dtype=float)
    except (TypeError, ValueError):
        return None
    if ll.ndim != 2 or ll.shape[1] != 2 or gates.shape != (len(ll),) or not len(ll):
        return None
    valid = np.isfinite(ll).all(1) & (np.abs(ll[:, 0]) <= 90) & (np.abs(ll[:, 1]) <= 180)
    valid &= np.isfinite(gates) & (gates >= .3)
    return ll, valid


def _reference(record):
    observed = _observations(record)
    positioned = _positions(record)
    if observed is None or positioned is None:
        return None
    good, sog, t, xy = (observed[key] for key in ['good', 'sog', 'times', 'xy'])
    if good.sum() < 8 or np.ptp(t[good]) < 600 or np.median(sog[observed['speed_valid']]) >= .5:
        return None
    path = float(observed['distance'][observed['links']].sum())
    if path >= 300 or observed['moving_jump'].any():
        return None
    # An intercepted jump must not create two distant, apparently stationary
    # clusters whose link was omitted from a short credible-path total.
    if np.max(np.linalg.norm(xy[good] - np.median(xy[good], axis=0), axis=1)) > 300:
        return None
    ll, valid = positioned
    if valid.sum() < 3:
        return None
    centers = _xyz(ll[valid])
    center = np.median(centers, axis=0)
    if np.max(np.linalg.norm(centers - center, axis=1)) > 300:
        return None
    end = _epoch(record.get('end_utc'))
    if end is None:
        return None
    return {'id': str(record['id']), 'vessel': str(record['vessel_id']),
            'end': end, 'utc': record['end_utc'], 'position': center}


def _target(record):
    policy = record.get('maritime_policy') or {}
    if not isinstance(policy, dict):
        return None
    # The median of the window's last 35% can still be high when a genuine
    # stop occupies its final two minutes. The original continuous tail below
    # is the evidence for this terminal state, rather than that coarse median.
    observed, positioned = _observations(record), _positions(record)
    start = _epoch(record.get('start_utc'))
    if observed is None or positioned is None or start is None:
        return None
    sog, t, good = (observed[key] for key in ['sog', 'times', 'good'])
    if not good[-1] or sog[-1] >= 1.5:
        return None
    first = len(sog) - 1
    while first > 0 and good[first - 1] and sog[first - 1] < 1.5 and observed['links'][first - 1]:
        first -= 1
    duration = float(t[-1] - t[first])
    if duration < 90:
        return None
    ll, valid = positioned
    if not valid[-1]:
        return None
    return {'start': start, 'position': _xyz(ll[-1:])[0], 'duration': duration}


def _empty():
    return {'matches': False, 'reference_ids': [], 'reference_mmsi': [], 'reference_utc': [],
            'distances_m': [], 'radius_m': RADIUS_M, 'basis': 'past_observations_only', 'meaning': MEANING}


def annotate_stopping_context(records: list[dict]) -> dict[str, dict]:
    """Find at least three other ships' earlier observed stable stopping places.

    Target speeds are original reported values, not derived model inputs. Each
    other MMSI contributes at most one reference example; repeated windows of
    one berthed ship cannot manufacture three-ship support. No input is mutated.
    """
    unique = {}
    for record in records:
        if isinstance(record, dict) and record.get('id') and record.get('vessel_id'):
            unique.setdefault(str(record['id']), record)
    references = [reference for row in unique.values() if (reference := _reference(row)) is not None]
    positions = np.array([r['position'] for r in references])
    ends = np.array([r['end'] for r in references])
    vessels = np.array([r['vessel'] for r in references])
    output = {}
    for rid, record in unique.items():
        if not (record.get('motion_candidate') or record.get('is_candidate')):
            continue
        context = output[rid] = _empty()
        target = _target(record)
        if target is None or not references:
            continue
        distances = np.linalg.norm(positions - target['position'], axis=1)
        selected = np.flatnonzero((ends < target['start']) & (vessels != str(record['vessel_id'])) & (distances <= RADIUS_M))
        distinct = {}
        for index in sorted(selected, key=lambda i: (distances[i], -ends[i])):
            distinct.setdefault(references[index]['vessel'], (references[index], float(distances[index])))
        context['terminal_low_speed_seconds'] = round(target['duration'], 1)
        if len(distinct) < MIN_OTHER_VESSELS:
            continue
        examples = list(distinct.values())
        context.update(matches=True, reference_ids=[r['id'] for r, _ in examples],
                       reference_mmsi=[r['vessel'] for r, _ in examples],
                       reference_utc=[r['utc'] for r, _ in examples],
                       distances_m=[round(distance, 2) for _, distance in examples])
    return output
