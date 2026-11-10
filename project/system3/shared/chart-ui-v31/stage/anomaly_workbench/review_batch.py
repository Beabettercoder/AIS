"""Separate observed manoeuvres from independently justified review tasks."""
from __future__ import annotations

from collections import Counter
import gzip
import json

from .event_episodes import merge_motion_episodes
from .recurrent_patterns import annotate_recurrence
from .review_policy import assess_review_evidence
from .stopping_context import annotate_stopping_context
from .maneuver_intent import assess_maneuver_intent

POLICY_VERSION = 5


def qualify_records(records):
    """Keep motion facts, adding a distinct, auditable review qualification."""
    for row in records:
        row.setdefault('motion_candidate', bool(row.get('is_candidate')))
    recurrence = annotate_recurrence(records)
    stopping = annotate_stopping_context(records)
    for row in records:
        if not row['motion_candidate']:
            row['review_level'] = 'background'
            continue
        assessment = assess_review_evidence(row)
        repeated = recurrence.get(row['id'], {'repeated': False, 'reference_ids': [], 'basis': 'past_observations_only'})
        stop_context = stopping.get(row['id'], {'matches': False, 'basis': 'past_observations_only'})
        # A repeated self-manoeuvre is not new evidence of abnormal behaviour.
        # Independent, sustained group contrast is still retained for review.
        independent_group = (bool(assessment.get('evidence', {}).get('local_corroboration')) or
                             any(code.startswith('confirmed_group') or code in {'supported_group_relation', 'locally_corroborated_maneuver'} for code in assessment.get('codes', [])))
        priority = bool(assessment.get('eligible')) and (not repeated['repeated'] or independent_group)
        reasons = list(assessment.get('reasons', []))
        ordinary_stop_context = (stop_context.get('matches') is True and
                                 assessment.get('codes') == ['sustained_rapid_deceleration'])
        if ordinary_stop_context:
            priority = False
            reasons.append('本船在此前已有至少三艘其他船稳定停留的地点附近减速至持续低速，先列一般停留线索；局部船群继续通行本身不足以证明本船异常。')
        if repeated['repeated'] and not independent_group:
            reasons.append('此前至少两次独立经过出现相近轨迹或减速终点，暂列重复运动线索；这不代表已经确认正常。')
            references = repeated.get('reference_utc', [])[:2]
            if references:
                reasons.append('历史参照世界时：' + '、'.join(str(value) for value in references) + '。')
        row['review_assessment']=assessment
        row['recurrence_evidence']=repeated
        row['stopping_context']=stop_context
        intent=assess_maneuver_intent(row)
        ordinary_maneuver=bool(intent.get('evidence',{}).get('normal_maneuver_pattern'))
        if ordinary_maneuver and not independent_group:
            priority=False
            reasons.append('当前动作也符合减速通行、回转后停留等常见操纵；缺少独立异常参照，暂列一般线索，不能据动作强度认定异常意图。')
        row['intent_assessment']=intent
        row['review_assessment'] = {**assessment, 'eligible': priority, 'reasons': reasons}
        row['review_level'] = 'priority' if priority else 'clue'
        row['is_candidate'] = priority
        row.setdefault('motion_type', row.get('type'))
        row.setdefault('motion_tier', row.get('tier'))
        row['type'] = row['motion_type']
        if priority and assessment.get('review_type'):
            row['type'] = assessment['review_type']
            row['tier'] = {'formation_departure': 'Tier-II', 'group_inconsistency': 'Tier-III'}.get(row['type'], 'Tier-I')
            row['type_label'] = '疑似' + {
                'turn_back': '折返', 'sinuous': '持续不规则转向', 'stall': '快速减速后持续低速',
                'speed_drop': '持续快速减速', 'formation_departure': '编队脱离',
                'group_inconsistency': '群体运动不一致',
            }.get(row['type'], '运动变化')
        row['original_motion_score'] = row.get('score')
        row['review_priority_score'] = row.get('score') if priority else None
        row['review_reason'] = ' '.join(reasons)
        if not priority:
            row['type_label'] = {
                'deviation': '航向变化线索', 'stall': '减速或停止线索', 'speed_drop': '减速线索',
                'sinuous': '转向变化线索', 'turn_back': '折返线索',
            }.get(row.get('type'), '一般运动线索')
            if ordinary_maneuver:
                row['type_label']=intent.get('observed_action','一般运动线索')+'（意图待核实）'
        row.setdefault('motion_why_flagged', row.get('why_flagged', '').split('\n重点核查资格：')[0])
        row['why_flagged'] = row['motion_why_flagged'] + '\n重点核查资格：' + row['review_reason']
        row['selection_role'] = 'candidate' if priority else 'motion_clue'
    return records


def event_brief(event):
    keys = ['id', 'vessel_id', 'type', 'type_label', 'score', 'event_start_utc', 'event_end_utc',
            'event_window_ids', 'priority_window_ids', 'event_types', 'event_type_labels', 'review_level']
    return {key: event.get(key) for key in keys}


def vessel_queue(episodes):
    queue = {}
    for event in sorted(episodes, key=lambda row: row.get('score', 0), reverse=True):
        queue.setdefault(event['vessel_id'], {**event, 'related_events': []})['related_events'].append(event_brief(event))
    rows = list(queue.values())
    for row in rows:
        row['related_event_count'] = len(row['related_events'])
    return rows


def refine_saved_windows(folder, timeline, top_k):
    """Apply phase two to actual computed windows; no model rerun or DB writes."""
    references = []
    raw_counts = Counter()
    for index in range(len(timeline)):
        path = folder / f'window-{index}.json.gz'
        if not path.is_file():
            continue
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            window = json.load(stream)
        for row in window['records']:
            motion = row.get('maritime_policy') or {}
            if row.get('motion_candidate', row.get('is_candidate')):
                raw_counts[row.get('motion_type', row['type'])] += 1
            if (row.get('motion_candidate', row.get('is_candidate')) or
                    (not motion.get('stationary') and motion.get('continuous_path_length_m', 0) >= 500) or
                    (motion.get('stationary') and (motion.get('source_sog_median_kn') or 0) < .5 and motion.get('path_length_m', 0) < 300)):
                references.append(row)
    qualify_records(references)
    by_id = {row['id']: row for row in references}
    motion_rows = [row for row in references if row['motion_candidate']]
    episodes = merge_motion_episodes(motion_rows)
    priorities = [event for event in episodes if event['is_candidate']]
    clues = [event for event in episodes if not event['is_candidate']]
    ranked = vessel_queue(priorities)
    clue_queue = vessel_queue(clues)
    priority_counts = Counter()
    scatter = []
    quality = []
    background = []
    for index in range(len(timeline)):
        path = folder / f'window-{index}.json.gz'
        if not path.is_file():
            continue
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            window = json.load(stream)
        rows = [by_id.get(row['id'], row) for row in window['records']]
        for row in rows:
            row.setdefault('motion_candidate', False)
            row.setdefault('review_level', 'background')
            row['policy_version'] = POLICY_VERSION
            if row['is_candidate']:
                priority_counts[row['type']] += 1
            if row.get('has_quality_issue'):
                quality.append(row)
            if not row['motion_candidate'] and len(background) < 12:
                background.append(row)
            keys = ['id', 'vessel_id', 'score', 'model_raw_score', 'maritime_policy', 'type', 'type_label', 'tier',
                    'field_residual_mps', 'e_kin_mps', 'mean_gate_weight', 'is_candidate', 'has_quality_issue',
                    'window_index', 'start_utc', 'end_utc', 'motion_candidate', 'review_level', 'review_reason',
                    'recurrence_evidence', 'review_assessment', 'stopping_context', 'intent_assessment']
            scatter.append({key: row.get(key) for key in keys})
        row_summary = timeline[index]
        row_summary['motion_clue_count'] = sum(row['motion_candidate'] for row in rows)
        row_summary['candidate_count'] = sum(row['is_candidate'] for row in rows)
        row_summary['repeated_pattern_count'] = sum(bool(row.get('recurrence_evidence', {}).get('repeated')) for row in rows)
        window['records'] = rows
        window['summary'] = row_summary
        with gzip.open(path, 'wt', encoding='utf-8', compresslevel=3) as stream:
            json.dump(window, stream, ensure_ascii=False, allow_nan=False)
    priority_counts['observation_glitch'] = len(quality)
    priority_counts['congestion'] = sum(row.get('macro_label') == '拥堵积压迹象' for row in timeline)
    quality.sort(key=lambda row: row.get('e_kin_mps') or 0, reverse=True)
    summary = {
        'candidate_count': len(priorities), 'candidate_windows': sum(row['is_candidate'] for row in motion_rows),
        'candidate_vessel_count': len(ranked), 'motion_clue_windows': len(motion_rows),
        'motion_episode_count': len(episodes), 'general_clue_event_count': len(clues),
        'repeated_pattern_windows': sum(bool(row.get('recurrence_evidence', {}).get('repeated')) for row in motion_rows),
        'historical_stopping_area_windows': sum(bool(row.get('stopping_context', {}).get('matches')) for row in motion_rows),
        'ordinary_maneuver_windows': sum(bool(row.get('intent_assessment', {}).get('evidence',{}).get('normal_maneuver_pattern')) for row in motion_rows),
        'qualification_rejected_windows': sum(not row['is_candidate'] for row in motion_rows),
    }
    return {
        'summary': summary, 'records': ranked[:top_k], 'clue_records': clue_queue[:max(top_k, 30)],
        'candidate_events': [event_brief(event) for event in priorities],
        'motion_events': [event_brief(event) for event in episodes],
        'quality_records': quality[:max(top_k, 30)], 'context_records': background,
        'type_counts': dict(priority_counts), 'motion_type_counts': dict(raw_counts), 'scatter': scatter,
    }
