"""Causal, vessel-specific repetition evidence; repetition is not normality.

Only observed, reliable continuous track pieces enter the reference history.
Reference passages belong to the same vessel, precede the target, and are at
least one hour apart. No model score or business classification establishes a
reference match. The output can lower review priority, never confirm normality.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
import math

import numpy as np

EARTH_M = 6371008.8
MIN_SEPARATION_SECONDS = 3600.0
MAX_HISTORY_PER_VESSEL = 512
MAX_REFERENCE_EXAMPLES = 8
MIN_PATH_M = 500.0
MATCH_DISTANCE_M = 300.0
RESAMPLE_POINTS = 20


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _epoch(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return result.timestamp() if result.tzinfo is not None else None
    except (ValueError, OverflowError, OSError):
        return None


def _array(value, n=None):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.ndim != 1 or (n is not None and len(result) != n):
        return None
    return result


def _absolute_meters(latlon):
    # Earth-centred metres give every window the same absolute reference and
    # also handle tracks straddling the longitude date line without a jump.
    latitude, longitude = np.deg2rad(latlon).T
    c = np.cos(latitude)
    return EARTH_M * np.column_stack((c * np.cos(longitude), c * np.sin(longitude), np.sin(latitude)))


def _descriptor(record):
    policy = record.get('maritime_policy')
    if not isinstance(policy, dict) or policy.get('stationary') or policy.get('is_stationary'):
        return None
    declared_path = _number(policy.get('continuous_path_length_m'))
    if declared_path is None or declared_path < MIN_PATH_M:
        return None
    start = _epoch(record.get('start_utc'))
    end = _epoch(record.get('end_utc'))
    if start is None or end is None or end < start:
        return None
    try:
        ll = np.asarray(record.get('latlon'), dtype=float)
    except (TypeError, ValueError):
        return None
    if ll.ndim != 2 or ll.shape[1] != 2 or len(ll) < 8:
        return None
    n = len(ll)
    times = _array(record.get('time_seconds'), n)
    weights = _array(record.get('gate_weights'), n)
    sog = _array(record.get('sog_kn'), n)
    if times is None or weights is None or sog is None:
        return None
    valid_position = np.isfinite(ll).all(1) & (np.abs(ll[:, 0]) <= 90) & (np.abs(ll[:, 1]) <= 180)
    valid = valid_position & np.isfinite(times) & np.isfinite(weights) & (weights >= .3)
    valid &= np.isfinite(sog) & (sog >= 0) & (sog < 102.2)
    # Invalid coordinates remain invalid; never repair or interpolate a jump.
    xyz = _absolute_meters(np.where(np.isfinite(ll), ll, 0))
    dt = np.diff(times)
    distances = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    reported_mps = .5 * (sog[1:] + sog[:-1]) * .514444
    max_plausible_mps = np.maximum(6., 2. * reported_mps + 3.)
    links = valid[1:] & valid[:-1] & (dt >= 1) & (dt <= 600)
    links &= distances / np.maximum(dt, 1) <= max_plausible_mps
    segments, current = [], []
    for i in range(n):
        if not valid[i]:
            if current:
                segments.append(current)
                current = []
        elif current and links[i - 1]:
            current.append(i)
        else:
            if current:
                segments.append(current)
            current = [i]
    if current:
        segments.append(current)
    eligible = []
    for indices in segments:
        if len(indices) < 8:
            continue
        points = xyz[indices]
        along = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if along[-1] >= MIN_PATH_M:
            eligible.append((float(along[-1]), indices, points, along))
    if not eligible:
        return None
    path, indices, points, along = max(eligible, key=lambda part: part[0])
    # Repeated stationary samples contribute no artificial extra arc length.
    distinct = np.r_[True, np.diff(along) > 1e-6]
    points, along = points[distinct], along[distinct]
    if len(points) < 2:
        return None
    target = np.linspace(0, path, RESAMPLE_POINTS)
    shape = np.column_stack([np.interp(target, along, points[:, axis]) for axis in range(3)])
    initial, late = _number(policy.get('initial_sog_kn')), _number(policy.get('late_sog_kn'))
    early_path = _number(policy.get('early_path_length_m'))
    arrival = (initial is not None and initial >= 3 and late is not None and 0 <= late < 1.5 and
               early_path is not None and early_path >= 100 and indices[-1] == n - 1)
    return {'id': str(record['id']), 'vessel': str(record['vessel_id']), 'start': start, 'end': end,
            'utc': record['start_utc'], 'shape': shape, 'path': path, 'initial': initial, 'late': late,
            'arrival': bool(arrival), 'endpoint': xyz[-1] if arrival else None}


def _independent(matches):
    selected = []
    # Recent examples are more useful than distant operating history. All have
    # already passed the target's one-hour causal separation check.
    for descriptor, distance in sorted(matches, key=lambda item: item[0]['start'], reverse=True):
        if all(abs(descriptor['start'] - other['start']) >= MIN_SEPARATION_SECONDS for other, _ in selected):
            selected.append((descriptor, distance))
            if len(selected) >= MAX_REFERENCE_EXAMPLES:
                break
    return sorted(selected, key=lambda item: item[0]['start'])


def _same_direction(first, second):
    # Endpoint distance alone would falsely match opposite traversals of a
    # compact closed loop. Require similar observed entry and exit directions.
    for start, end in [(0, 3), (-4, -1)]:
        a, b = first[end] - first[start], second[end] - second[start]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if min(na, nb) < 30 or float(a @ b) / (na * nb) < math.cos(math.radians(45)):
            return False
    return True


def _context(matches, kind=None):
    repeated = len(matches) >= 2
    chosen = matches if repeated else []
    return {'repeated': repeated, 'kind': kind if repeated else None,
            'reference_ids': [entry['id'] for entry, _ in chosen],
            'reference_utc': [entry['utc'] for entry, _ in chosen],
            'reference_distances_m': [round(float(distance), 2) for _, distance in chosen],
            'independent_passages': len(chosen), 'basis': 'past_observations_only'}


def annotate_recurrence(records: list[dict]) -> dict[str, dict]:
    """Return repetition evidence for motion candidates, without mutating input.

    ``independent_passages`` counts other historical passages (not the target),
    capped at eight displayed examples. Distances parallel ``reference_ids``:
    the largest endpoint/median shape error for a trajectory match, or terminal
    distance for a repeated slowdown endpoint. Invalid/missing observations do
    not become synthetic reference tracks. Historical storage is bounded to
    512 valid observed windows per vessel.
    """
    output, unique = {}, {}
    for record in records:
        if not isinstance(record, dict) or not record.get('id'):
            continue
        rid = str(record['id'])
        if record.get('motion_candidate') or record.get('is_candidate'):
            output[rid] = _context([])
        if rid not in unique and record.get('vessel_id'):
            unique[rid] = record
    ordered = sorted(unique.values(), key=lambda record: (_epoch(record.get('start_utc')) or float('inf'), str(record['id'])))
    history = defaultdict(lambda: deque(maxlen=MAX_HISTORY_PER_VESSEL))
    for record in ordered:
        descriptor = _descriptor(record)
        if descriptor is None:
            continue
        prior = history[descriptor['vessel']]
        if descriptor['id'] in output:
            available = [item for item in prior if descriptor['start'] - item['start'] >= MIN_SEPARATION_SECONDS and item['end'] < descriptor['start']]
            trajectory_matches, endpoint_matches = [], []
            for item in available:
                initial, late = descriptor['initial'], descriptor['late']
                if initial is not None and late is not None and item['initial'] is not None and item['late'] is not None:
                    if (abs(initial - item['initial']) <= 2 and abs(late - item['late']) <= 2 and
                            .67 < item['path'] / descriptor['path'] < 1.5 and _same_direction(descriptor['shape'], item['shape'])):
                        distance = np.linalg.norm(descriptor['shape'] - item['shape'], axis=1)
                        error = max(float(distance[0]), float(distance[-1]), float(np.median(distance)))
                        if error <= MATCH_DISTANCE_M:
                            trajectory_matches.append((item, error))
                if descriptor['arrival'] and item['arrival']:
                    speed_tolerance = max(2., descriptor['initial'] * .3)
                    if abs(descriptor['initial'] - item['initial']) <= speed_tolerance:
                        endpoint_error = float(np.linalg.norm(descriptor['endpoint'] - item['endpoint']))
                        if endpoint_error <= MATCH_DISTANCE_M:
                            endpoint_matches.append((item, endpoint_error))
            trajectories = _independent(trajectory_matches)
            arrivals = _independent(endpoint_matches)
            output[descriptor['id']] = _context(trajectories, 'trajectory') if len(trajectories) >= 2 else _context(arrivals, 'slowdown_endpoint')
        prior.append(descriptor)
    return output
