"""Merge only physically continuous vessel windows into multi-label episodes.

This module groups evidence; it never promotes a motion clue into a priority
candidate or changes the representative window's original priority score.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import math


MAX_REPORT_GAP_SECONDS = 600.0
POSITION_TOLERANCE_M = 300.0
MAX_ENDPOINT_DISTANCE_M = 10_000.0
KNOT_TO_MPS = 0.514444


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _window_index(record):
    value = _number(record.get('window_index'))
    return int(value) if value is not None and value >= 0 and value.is_integer() else None


def _record_id(record):
    value = record.get('id') or record.get('window_id')
    return str(value) if value is not None and str(value) else None


def _endpoint(record, last):
    """Use actual report endpoints, never repaired or interpolated positions."""
    track = record.get('latlon')
    if not isinstance(track, (list, tuple)) or not track:
        return None
    point = track[-1 if last else 0]
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    lat, lon = map(_number, point)
    if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return None
    weights = record.get('gate_weights')
    if weights is not None:
        if not isinstance(weights, (list, tuple)) or len(weights) != len(track):
            return None
        weight = _number(weights[-1 if last else 0])
        if weight is None or weight < .3:
            return None
    speeds = record.get('sog_kn')
    speed = None
    if isinstance(speeds, (list, tuple)) and len(speeds) == len(track):
        speed = _number(speeds[-1 if last else 0])
        if speed is not None and not 0 <= speed < 102.2:
            speed = None
    return (lat, lon), speed


def _distance_m(a, b):
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlat = lat2 - lat1
    dlon = math.radians((b[1] - a[1] + 180) % 360 - 180)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6_371_000.0 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _can_join(left, right):
    ship = left.get('vessel_id')
    if ship is None or str(ship) == '' or str(ship) != str(right.get('vessel_id')):
        return False
    li, ri = _window_index(left), _window_index(right)
    # One actual record per vessel per window is expected. Duplicate windows
    # and skipped non-candidate windows never establish a continuous episode.
    if li is None or ri is None or ri - li != 1:
        return False
    if _record_id(left) is None or _record_id(right) is None:
        return False
    start_left, end_left = _timestamp(left.get('start_utc')), _timestamp(left.get('end_utc'))
    start_right, end_right = _timestamp(right.get('start_utc')), _timestamp(right.get('end_utc'))
    if any(t is None for t in (start_left, end_left, start_right, end_right)):
        return False
    if end_left < start_left or end_right < start_right:
        return False
    gap = start_right - end_left
    if not 0 <= gap <= MAX_REPORT_GAP_SECONDS:
        return False
    endpoint_left, endpoint_right = _endpoint(left, True), _endpoint(right, False)
    if endpoint_left is None or endpoint_right is None:
        return False
    distance = _distance_m(endpoint_left[0], endpoint_right[0])
    if distance > MAX_ENDPOINT_DISTANCE_M:
        return False
    # Both trustworthy endpoint speeds constrain a reachable bridge. Missing
    # speed permits only the fixed 300 m spatial tolerance, not guessed motion.
    speeds = (endpoint_left[1], endpoint_right[1])
    speed_allowance = max(speeds) * KNOT_TO_MPS * gap if all(s is not None for s in speeds) else 0.0
    return distance <= POSITION_TOLERANCE_M + speed_allowance


def _score(record):
    value = _number(record.get('score'))
    return value if value is not None else -math.inf


def _episode(windows):
    priority = [r for r in windows if r.get('is_candidate') is True]
    representative = max(priority or windows, key=_score)
    result = deepcopy(representative)
    for key in list(result):
        if key.startswith('event_') or key.startswith('related_event') or key in ('priority_window_ids', 'review_level'):
            result.pop(key)
    types, labels = [], []
    for row in windows:
        if row.get('type') and row['type'] not in types:
            types.append(row['type'])
        if row.get('type_label') and row['type_label'] not in labels:
            labels.append(row['type_label'])
    result.update(
        event_start_utc=windows[0].get('start_utc'),
        event_end_utc=windows[-1].get('end_utc'),
        event_window_ids=[rid for row in windows if (rid := _record_id(row)) is not None],
        priority_window_ids=[rid for row in priority if (rid := _record_id(row)) is not None],
        event_types=types,
        event_type_labels=labels,
        is_candidate=bool(priority),
        motion_candidate=True,
        review_level='priority' if priority else 'clue',
    )
    return result


def merge_motion_episodes(records: list[dict]) -> list[dict]:
    """Return chronologically ordered episodes, without modifying input records.

    Only adjacent candidate-window indices with fresh reports and reachable
    actual positions merge. Ranking/recurrence qualification belongs to the
    caller; all constituent clue IDs and priority IDs remain traceable.
    """
    groups = {}
    for serial, row in enumerate(records):
        vessel = row.get('vessel_id')
        key = ('ship', str(vessel)) if vessel is not None and str(vessel) else ('unknown', serial)
        groups.setdefault(key, []).append(row)
    episodes = []
    for rows in groups.values():
        rows = sorted(rows, key=lambda r: (_window_index(r) if _window_index(r) is not None else math.inf,
                                           _timestamp(r.get('start_utc')) or 0, _record_id(r) or ''))
        run = []
        for row in rows:
            if run and not _can_join(run[-1], row):
                episodes.append(_episode(run))
                run = []
            run.append(row)
        if run:
            episodes.append(_episode(run))
    return sorted(episodes, key=lambda r: (_timestamp(r.get('event_start_utc')) or 0,
                                            str(r.get('vessel_id', '')), _record_id(r) or ''))
