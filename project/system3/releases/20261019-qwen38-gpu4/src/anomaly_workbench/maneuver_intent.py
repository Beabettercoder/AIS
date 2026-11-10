"""Describe observed manoeuvre patterns and limited possible operational intents.

Only the current saved window is used. These hypotheses never confirm normality,
port arrival, a destination, or an anomaly, and never alter detection scores.
"""
from __future__ import annotations

import math
import numpy as np

from .review_policy import _trajectory, _stable_moving_legs, _deceleration_evidence


def _dict(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _terminal_run(speed, times, segment, low, high):
    if len(segment) < 2 or not low <= speed[segment[-1]] < high:
        return 0., []
    first = len(segment) - 1
    while first > 0 and low <= speed[segment[first - 1]] < high:
        first -= 1
    selected = segment[first:]
    return float(times[selected[-1]] - times[selected[0]]), selected.tolist()


def assess_maneuver_intent(record: dict) -> dict:
    """Return bounded, auditable descriptions, not a final intent prediction."""
    if not isinstance(record, dict):
        record = {}
    policy = _dict(record.get('maritime_policy'))
    review = _dict(record.get('review_assessment'))
    supplied = _dict(review.get('evidence'))
    recurrence = _dict(record.get('recurrence_evidence'))
    history_available = bool(recurrence.get('reference_ids'))
    evidence = {
        'normal_maneuver_pattern': False, 'pattern_code': None,
        'basis': 'current_window_original_ais_only', 'as_of_utc': record.get('end_utc'),
        'terminal_low_speed_seconds': None, 'terminal_speed_kn': None,
        'terminal_slow_transit_seconds': None, 'terminal_slow_transit_median_kn': None,
        'net_path_ratio': None, 'credible_path_m': None, 'reliable_terminal_observations': 0,
        'verified_direction_reversal_deg': None, 'verified_deceleration': False,
        'historical_reference_available': history_available,
        'historical_reference_status': 'reference_supplied' if history_available else 'no_comparable_reference_supplied',
        'neighbor_reference_available': policy.get('neighbor_supported') is True,
        'local_motion_corroboration': supplied.get('local_corroboration') is True,
        'reference_meaning': '邻船仅提供运动对照，不证明作业意图；没有历史匹配不表示罕见或异常。',
    }
    missing = ['官方港口或泊位边界', '原始导航状态', '目的港与预计到达时间', '作业或调度计划']
    if not history_available:
        missing.append('可比较的本船历史航程参照')
    result = {'observed_action': '原始连续轨迹证据不足', 'possible_intents': [],
              'explanation': '当前窗口不足以确认连续运动过程，航行意图未确认，不能仅凭模型分数推断航行目的。',
              'normal_operation_not_excluded': True, 'missing_context': missing, 'evidence': evidence}
    obs = _dict(record.get('motion_observations'))
    if not obs:
        missing.append('连续原始航速、位置和时间观测')
        return result
    # Source fields take precedence even when absent or invalid; derived speeds
    # from the display/model arrays must not replace missing original reports.
    original = dict(obs)
    if 'source_sog' in obs:
        original['sog_kn'] = obs['source_sog']
    if 'source_cog' in obs:
        original['cog_deg'] = obs['source_cog']
    try:
        raw_gates = np.asarray(original.get('gate_weights'), dtype=float)
        if raw_gates.ndim == 1:
            original['gate_weights'] = np.where(np.isfinite(raw_gates), raw_gates, 0.).tolist()
        trajectory = _trajectory({**record, 'motion_observations': original})
    except (TypeError, ValueError, AttributeError, IndexError):
        missing.append('可解析的原始运动观测')
        return result
    speed, times, xy, heading, valid_heading, dt, dp, distance, segments = trajectory
    if not len(speed):
        missing.append('连续原始航速、位置和时间观测')
        return result
    terminal = next((segment for segment in segments if len(segment) and segment[-1] == len(speed) - 1), None)
    if terminal is None or len(terminal) < 2:
        missing.append('可信的窗口末端连续轨迹')
        return result
    terminal_speed = float(speed[-1])
    path = float(distance[terminal[1:]].sum())
    net = float(np.linalg.norm(xy[terminal[-1]] - xy[terminal[0]]))
    ratio = net / path if path > 1 else None
    stop_seconds, stop_indices = _terminal_run(speed, times, terminal, 0., 1.5)
    slow_seconds, slow_indices = _terminal_run(speed, times, terminal, 1.5, 6.000001)
    slow_median = float(np.median(speed[slow_indices])) if slow_indices else None
    evidence.update(terminal_low_speed_seconds=stop_seconds, terminal_speed_kn=terminal_speed,
                    terminal_slow_transit_seconds=slow_seconds, terminal_slow_transit_median_kn=slow_median,
                    net_path_ratio=ratio, credible_path_m=path, reliable_terminal_observations=len(terminal))
    if len(terminal) < 8:
        result.update(observed_action='仅确认窗口末端局部运动',
                      explanation=f'末端报告航速为{terminal_speed:.2f}节，但连续可信观测不足，航行意图未确认，不能据此推断此前动作或作业目的。')
        missing.append('覆盖完整动作的连续可信观测')
        return result
    legs = _stable_moving_legs(speed, times, xy, heading, valid_heading, dt, dp, distance, [terminal])
    angles = [abs((later['heading_deg'] - earlier['heading_deg'] + 180.) % 360. - 180.)
              for i, earlier in enumerate(legs) for later in legs[i + 1:] if later['start'] > earlier['end']]
    actual_reversal = max(angles, default=0.)
    evidence['verified_direction_reversal_deg'] = actual_reversal if angles else None
    declared_angle = _number(supplied.get('turnback_stable_angle_deg'))
    declared_reversal = supplied.get('turnback_confirmed') is True or (declared_angle is not None and declared_angle >= 150)
    reversal = declared_reversal and actual_reversal >= 150
    verified_drop = _deceleration_evidence(speed, times, distance, [terminal])
    declared_drop = _dict(supplied.get('deceleration')).get('confirmed') is True
    before = _number(verified_drop.get('before_median_kn'))
    after = _number(verified_drop.get('after_median_kn'))
    drop = bool(declared_drop and verified_drop.get('confirmed') and before is not None and before >= 5)
    evidence['verified_deceleration'] = drop
    evidence['before_speed_kn'] = before if drop else None
    evidence['after_stable_speed_kn'] = after if drop else None
    if reversal and stop_seconds >= 90 and path >= 500 and ratio is not None and ratio <= .35:
        evidence.update(normal_maneuver_pattern=True, pattern_code='turn_then_hold')
        result.update(observed_action='回转后低速停留',
                      possible_intents=['等待或待命', '靠泊操纵', '局部作业'],
                      explanation=f'当前可信轨迹回转约{actual_reversal:.1f}度后，末端航速{terminal_speed:.2f}节并持续低速{stop_seconds:.0f}秒，可解释为回转后停留；等待、靠泊操纵或作业都有可能，具体目的尚未确认。')
    elif (drop and before is not None and slow_median is not None and slow_seconds >= 90 and
          1.5 <= terminal_speed <= 6 and 1.5 <= slow_median <= 6 and (before - slow_median) / before >= .5):
        evidence.update(normal_maneuver_pattern=True, pattern_code='fast_to_slow_transit')
        result.update(observed_action='由高速转入持续低速通行',
                      possible_intents=['低速通行', '让行或避让操纵', '进出港操纵'],
                      explanation=f'当前由稳定在航约{before:.2f}节转为低速，末端仍以{terminal_speed:.2f}节通行且低速段持续{slow_seconds:.0f}秒；可能是通行、让行或进出港操纵，不能据此称已靠泊或已确认异常。')
    elif drop and stop_seconds >= 90 and path >= 500:
        evidence.update(normal_maneuver_pattern=True, pattern_code='decelerate_then_hold')
        result.update(observed_action='在航减速后低速停留',
                      possible_intents=['等待或待命', '靠泊操纵', '局部作业'],
                      explanation=f'当前先以约{before:.2f}节稳定在航，随后末端降至{terminal_speed:.2f}节并持续低速{stop_seconds:.0f}秒；等待、靠泊操纵或作业均可产生此动作，具体目的尚未确认。')
    elif reversal:
        result.update(observed_action='在航回转，末端尚未确认停留', possible_intents=[],
                      explanation=f'当前连续轨迹存在约{actual_reversal:.1f}度回转，末端航速{terminal_speed:.2f}节；尚不能判断是航路调整、作业操纵还是需要进一步核查的变化。')
    elif stop_seconds >= 90:
        result.update(observed_action='窗口末端低速停留', possible_intents=[],
                      explanation=f'当前只确认末端以{terminal_speed:.2f}节低速保持{stop_seconds:.0f}秒，缺少完整且明确的前后动作证据，不能确定靠泊、待命或其他作业目的。')
    else:
        result.update(observed_action='在航或局部机动过程', possible_intents=[],
                      explanation=f'当前窗口末端航速为{terminal_speed:.2f}节，只能描述实际运动，不能仅凭转弯或速度变化确定航行目的。')
    if evidence['normal_maneuver_pattern']:
        result['explanation'] += '以上仅为可能解释，未确认靠泊，也未确认正常或异常。'
    if not history_available:
        result['explanation'] += '当前未提供可比较历史参照，这不表示该动作罕见或异常。'
    return result
