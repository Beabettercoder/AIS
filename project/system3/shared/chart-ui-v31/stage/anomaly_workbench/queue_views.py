"""Filter the complete saved evidence before limiting a vessel review queue."""
from __future__ import annotations


def _number(value):
    try:
        return float(value) if value is not None else 0.
    except (TypeError, ValueError):
        return 0.


def _stationary(row):
    policy = row.get('maritime_policy') or {}
    return bool(policy.get('stationary') or policy.get('is_stationary'))


def _matches(row, kind, category, hide_stationary):
    if kind == 'candidate' and not row.get('is_candidate'):
        return False
    if kind == 'clue' and row.get('review_level') != 'clue':
        return False
    if kind == 'quality' and not row.get('has_quality_issue'):
        return False
    if kind == 'all' and hide_stationary and _stationary(row):
        return False
    return not category or (bool(row.get('has_quality_issue')) if category == 'observation_glitch' else row.get('type') == category)


def _rank(row, kind):
    if kind == 'quality':
        return (_number(row.get('e_kin_mps')), _number(row.get('score')), str(row.get('id', '')))
    return (bool(row.get('is_candidate')), bool(row.get('motion_candidate')), _number(row.get('score')), str(row.get('id', '')))


def build_queue(scan, *, kind='candidate', category='', limit=20, window=None,
                hide_stationary=False, focus_record=None):
    """Keep window/episode counts separate from the displayed unique vessels."""
    source = (window or {}).get('records') if window is not None else scan.get('scatter', [])
    source = source or []
    by_id = {row['id']: row for row in source}
    rows = []
    matching_unit = '船舶窗口'
    if window is None and kind in {'candidate', 'clue'}:
        matching_unit = '运动段'
        for event in scan.get('motion_events', []):
            if event.get('review_level') != ('priority' if kind == 'candidate' else 'clue'):
                continue
            ids = event.get('priority_window_ids', []) if kind == 'candidate' else event.get('event_window_ids', [])
            members = [by_id[rid] for rid in ids if rid in by_id and _matches(by_id[rid], kind, category, hide_stationary)]
            if not members:
                continue
            preferred = next((row for row in members if row['id'] == focus_record), None)
            row = dict(preferred or max(members, key=lambda item: _rank(item, kind)))
            for key in ['event_start_utc', 'event_end_utc', 'event_window_ids', 'priority_window_ids', 'event_types', 'event_type_labels']:
                if key in event:
                    row[key] = event[key]
            row['episode_representative_id'] = event.get('id')
            rows.append(row)
    else:
        rows = [dict(row) for row in source if _matches(row, kind, category, hide_stationary)]
    total_matching = len(rows)
    grouped = {}
    counts = {}
    for row in rows:
        vessel = str(row.get('vessel_id', ''))
        counts[vessel] = counts.get(vessel, 0) + 1
        current = grouped.get(vessel)
        if current is None or row.get('id') == focus_record or (current.get('id') != focus_record and _rank(row, kind) > _rank(current, kind)):
            grouped[vessel] = row
    displayed = sorted(grouped.values(), key=lambda item: _rank(item, kind), reverse=True)
    preferred = next((row for row in displayed if row.get('id') == focus_record), None)
    if preferred is not None:
        displayed = [preferred] + [row for row in displayed if row is not preferred]
    for row in displayed:
        row['related_event_count'] = counts[str(row.get('vessel_id', ''))]
        row['related_matching_unit'] = matching_unit
    displayed = displayed[:limit]
    return {'records': displayed, 'total_matching': total_matching, 'total_vessels': len(grouped),
            'returned': len(displayed), 'matching_unit': matching_unit,
            'scope': 'window' if window is not None else 'scan',
            'focused_record': preferred.get('id') if preferred else None}
