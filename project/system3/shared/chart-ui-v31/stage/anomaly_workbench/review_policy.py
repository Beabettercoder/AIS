"""Second-stage evidence qualification for a limited human review queue.

The first stage detects observable manoeuvres.  This stage asks whether the
window contains a substantial, reproducible change or an independent local
comparison.  It does not declare an event abnormal, change a score, or infer
that an unqualified manoeuvre is normal.
"""
from __future__ import annotations

import math
import numpy as np


def _number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _array(value, n=None, default=np.nan):
    if value is None:
        return np.full(n or 0, default, float)
    try:
        result = np.asarray(value, float).reshape(-1)
    except (TypeError, ValueError):
        result = np.array([], float)
    if n is None:
        return result
    out = np.full(n, default, float)
    out[:min(n, len(result))] = result[:n]
    return out


def _angle(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.) % 360. - 180.


def _mean_heading(value):
    value = np.asarray(value, float)
    value = value[np.isfinite(value)]
    if not len(value):
        return None
    rad = np.deg2rad(value)
    x, y = float(np.sin(rad).mean()), float(np.cos(rad).mean())
    return float(np.rad2deg(np.arctan2(x, y)) % 360) if math.hypot(x, y) > 1e-8 else None


def _trajectory(record):
    observations = record.get('motion_observations') or {}
    speed = _array(observations.get('sog_kn'))
    n = len(speed)
    times = _array(observations.get('time_seconds'), n)
    heading = _array(observations.get('cog_deg'), n)
    weights = _array(observations.get('gate_weights'), n, 0.)
    xy = np.full((n, 2), np.nan)
    try:
        raw_xy = np.asarray(observations.get('xy_m', []), float)
        if raw_xy.ndim == 2 and raw_xy.shape[1] == 2:
            xy[:min(n, len(raw_xy))] = raw_xy[:n]
    except (TypeError, ValueError):
        pass
    valid = (np.isfinite(speed) & (speed >= 0) & (speed < 102.2) &
             np.isfinite(times) & np.isfinite(xy).all(1) & (weights >= .3))
    if 'sog_reported_mask' in observations:
        valid &= _array(observations['sog_reported_mask'], n, 0) > .5
    valid_heading = np.isfinite(heading) & (heading >= 0) & (heading < 360)
    if 'cog_reported_mask' in observations:
        valid_heading &= _array(observations['cog_reported_mask'], n, 0) > .5
    dt = np.r_[0., np.diff(times)] if n else np.array([])
    dp = np.vstack([np.zeros((1, 2)), np.diff(xy, axis=0)]) if n else np.empty((0, 2))
    distance = np.linalg.norm(dp, axis=1)
    links = np.zeros(n, bool)
    if n > 1:
        average = .5 * (speed[1:] + speed[:-1]) * .514444
        limit = np.maximum(6., 2. * average + 3.)
        links[1:] = (valid[1:] & valid[:-1] & (dt[1:] >= 1) & (dt[1:] <= 300) &
                     (distance[1:] / np.maximum(dt[1:], 1) <= limit))
    segments, current = [], []
    for i in range(n):
        if not valid[i]:
            if current:
                segments.append(np.asarray(current, int)); current = []
        elif current and links[i]:
            current.append(i)
        else:
            if current:
                segments.append(np.asarray(current, int))
            current = [i]
    if current:
        segments.append(np.asarray(current, int))
    return speed, times, xy, heading, valid_heading, dt, dp, distance, segments


def _stable_moving_legs(speed, times, xy, heading, heading_valid, dt, dp, distance, segments):
    """Extract actual straight moving legs; two consistent reports can suffice."""
    legs = []
    for segment in segments:
        current = []

        def finish():
            if len(current) < 2:
                return
            indices = np.asarray(current, int)
            duration = float(times[indices[-1]] - times[indices[0]])
            path = float(distance[indices[1:]].sum())
            vector = xy[indices[-1]] - xy[indices[0]]
            net = float(np.linalg.norm(vector))
            if duration >= 90 and net >= 150 and net >= .9 * path:
                legs.append({'start':int(indices[0]), 'end':int(indices[-1]),
                             'heading_deg':float(np.rad2deg(np.arctan2(vector[0], vector[1])) % 360),
                             'duration_seconds':duration, 'distance_m':net,
                             'median_speed_kn':float(np.median(speed[indices]))})

        for i in segment[1:]:
            previous = i - 1
            actual_speed = float(distance[i] / dt[i])
            reported_speed = float(.5 * (speed[i] + speed[previous]) * .514444)
            direction = float(np.rad2deg(np.arctan2(dp[i, 0], dp[i, 1])) % 360)
            reported_heading = _mean_heading(heading[[previous, i]])
            usable = (min(speed[previous], speed[i]) >= 3 and heading_valid[previous] and heading_valid[i] and
                      reported_heading is not None and abs(float(_angle(direction, reported_heading))) <= 30 and
                      abs(actual_speed - reported_speed) <= max(1.5, .5 * reported_speed))
            if not usable:
                finish(); current = []
                continue
            if current:
                vector = xy[current[-1]] - xy[current[0]]
                before = float(np.rad2deg(np.arctan2(vector[0], vector[1])) % 360)
                if abs(float(_angle(direction, before))) > 25:
                    finish(); current = []
            if not current:
                current = [int(previous)]
            current.append(int(i))
        finish()
    return legs


def _rdp_indices(points, epsilon=30.):
    keep = {0, len(points)-1}
    stack = [(0, len(points)-1)]
    while stack:
        left, right = stack.pop()
        if right <= left + 1:
            continue
        vector = points[right] - points[left]
        length = float(np.linalg.norm(vector))
        if length > 0:
            offset = points[left+1:right] - points[left]
            distances = np.abs(vector[0]*offset[:, 1]-vector[1]*offset[:, 0]) / length
        else:
            distances = np.linalg.norm(points[left+1:right] - points[left], axis=1)
        index = int(np.argmax(distances)) + left + 1
        if float(distances[index-left-1]) > epsilon:
            keep.add(index); stack.extend([(left, index), (index, right)])
    return sorted(keep)


def _sinuous_evidence(speed, times, xy, segments, initial_speed):
    best = {'alternating_turns':0, 'turn_angles_deg':[], 'net_path_ratio':None,
            'minimum_leg_distance_m':None, 'minimum_leg_seconds':None, 'confirmed':False}
    if initial_speed is None or initial_speed < 3:
        return best
    for segment in segments:
        # Low-speed/idle geometry cannot supply the turning evidence.
        moving_runs, current = [], []
        for index in segment:
            if speed[index] >= 3:
                current.append(int(index))
            else:
                if current:
                    moving_runs.append(current)
                current = []
        if current:
            moving_runs.append(current)
        for indices in moving_runs:
            if len(indices) < 5:
                continue
            simplified = [indices[k] for k in _rdp_indices(xy[indices])]
            nodes = [simplified[0]]
            for k in simplified[1:]:
                if times[k] - times[nodes[-1]] >= 30 and np.linalg.norm(xy[k] - xy[nodes[-1]]) >= 100:
                    nodes.append(k)
            if len(nodes) < 5:
                continue
            delta = np.diff(xy[nodes], axis=0)
            lengths = np.linalg.norm(delta, axis=1)
            headings = np.rad2deg(np.arctan2(delta[:, 0], delta[:, 1]))
            turns = _angle(headings[1:], headings[:-1])
            major = turns[np.abs(turns) >= 30]
            run, maximum = 0, 0
            previous = 0.
            for value in major:
                sign = float(np.sign(value))
                run = run + 1 if sign != previous else 1
                previous = sign
                maximum = max(maximum, run)
            path = float(lengths.sum())
            ratio = float(np.linalg.norm(xy[nodes[-1]]-xy[nodes[0]]) / max(path, 1))
            if maximum > best['alternating_turns']:
                best = {'alternating_turns':maximum, 'turn_angles_deg':[float(x) for x in major],
                        'net_path_ratio':ratio, 'minimum_leg_distance_m':float(lengths.min()),
                        'minimum_leg_seconds':float(np.diff(times[nodes]).min()),
                        'confirmed':bool(maximum >= 3 and ratio <= .8)}
    return best


def _deceleration_evidence(speed, times, distance, segments):
    best = {'confirmed':False, 'before_median_kn':None, 'after_median_kn':None,
            'transition_seconds':None, 'deceleration_kn_per_min':None, 'low_speed_seconds':None}
    for segment in segments:
        if len(segment) < 5:
            continue
        for position in range(2, len(segment)-1):
            last = segment[position]
            before = segment[:position+1]
            before = before[times[last] - times[before] <= 240]
            if len(before) < 3 or times[before[-1]] - times[before[0]] < 90:
                continue
            baseline = float(np.median(speed[before]))
            spread = float(np.quantile(speed[before], .75) - np.quantile(speed[before], .25))
            if (baseline < 5 or spread > max(1., .2*baseline) or
                    abs(float(speed[last])-baseline) > max(1., .2*baseline) or
                    float(distance[before[1:]].sum()) < 100):
                continue
            threshold = max(1.5, .4*baseline)
            for after_position in range(position+1, len(segment)):
                first = segment[after_position]
                transition = float(times[first] - times[last])
                if transition > 180:
                    break
                if speed[first] > threshold:
                    continue
                after = []
                for index in segment[after_position:]:
                    if speed[index] > threshold:
                        break
                    after.append(int(index))
                    if times[index] - times[first] >= 90:
                        break
                if len(after) < 2 or times[after[-1]] - times[first] < 90:
                    continue
                final = float(np.median(speed[after]))
                drop = baseline - final
                rate = drop / max(transition, 1) * 60.
                confirmed = bool(drop >= 3 and (final < 1.5 or drop/baseline >= .6) and rate >= 2)
                if confirmed and (best['deceleration_kn_per_min'] is None or rate > best['deceleration_kn_per_min']):
                    best = {'confirmed':True, 'before_median_kn':baseline, 'after_median_kn':final,
                            'transition_seconds':transition, 'deceleration_kn_per_min':rate,
                            'before_stable_seconds':float(times[last]-times[before[0]]),
                            'low_speed_seconds':float(times[after[-1]]-times[first]),
                            'before_observations':len(before), 'after_observations':len(after)}
    return best


def assess_review_evidence(record: dict) -> dict:
    """Qualify a first-stage movement for focused review, without score changes."""
    policy = record.get('maritime_policy') or {}
    evidence = {'review_rule_version':1}
    answer = {'eligible':False, 'review_type':None, 'codes':[], 'reasons':[], 'evidence':evidence}
    if not bool(record.get('motion_candidate', record.get('is_candidate', False))):
        answer['codes'] = ['not_motion_candidate']
        answer['reasons'] = ['本窗口没有第一阶段运动线索，不新增重点核查项。']
        return answer
    if policy.get('stationary') or policy.get('is_stationary'):
        answer['codes'] = ['stationary_excluded']
        answer['reasons'] = ['全程静止背景不进入重点行为核查。']
        return answer
    data = _trajectory(record)
    speed, times, xy, heading, heading_valid, dt, dp, distance, segments = data
    continuous = max((float(distance[ix[1:]].sum()) for ix in segments if len(ix)>1), default=0.)
    initial = _number(policy.get('initial_sog_kn'))
    if initial is None and len(speed):
        good = np.concatenate(segments) if segments else np.array([], int)
        if len(good):
            early = good[times[good] <= times[good[0]] + .25*(times[good[-1]]-times[good[0]])]
            if len(early) >= 3:
                initial = float(np.median(speed[early]))
    evidence.update(continuous_path_m=continuous, initial_speed_kn=initial,
                    reliable_segments=len(segments), reliable_observations=sum(len(ix) for ix in segments))
    if continuous < 150 or not len(speed):
        answer['codes'] = ['insufficient_reliable_trajectory']
        answer['reasons'] = ['没有足够的连续可信在航轨迹，不能把定位跳点或短片段升级为重点核查。']
        return answer
    legs = _stable_moving_legs(*data)
    reversal = 0.
    for left_index, left in enumerate(legs):
        for right in legs[left_index+1:]:
            # Never join legs across an observation-quality gap.
            same_segment = next((ix for ix in segments if ix[0] <= left['start'] and right['end'] <= ix[-1]), None)
            long_segment = same_segment is not None and float(distance[same_segment[1:]].sum()) >= 500
            if left['end'] <= right['start'] and long_segment:
                reversal = max(reversal, abs(float(_angle(right['heading_deg'], left['heading_deg']))))
    turnback = bool(reversal >= 150 and continuous >= 500)
    sinuous = _sinuous_evidence(speed, times, xy, segments, initial)
    slowing = _deceleration_evidence(speed, times, distance, segments)
    residual = _number(record.get('field_residual_mps'))
    count = int(_number(policy.get('scoring_neighbor_count'), 0))
    steps = int(_number(policy.get('valid_local_steps'), 0))
    local = bool(policy.get('neighbor_supported') and count >= 3 and steps >= 3 and
                 residual is not None and residual >= 2 and initial is not None and initial >= 3 and continuous >= 500)
    supported = set(policy.get('supported_types') or [])
    physical = policy.get('evidence') or {}
    supported.update(key for key, field in [('turn_back','turn_back'),('sinuous','sinuous'),
                                           ('stall','low_speed_loiter'),('speed_drop','dynamic_speed_drop')] if physical.get(field))
    kind = record.get('motion_type') or record.get('type') or policy.get('event_type')
    relation = kind in {'formation_departure','group_inconsistency'}
    local_specific = local and bool(supported & {'turn_back','sinuous','stall','speed_drop'})
    evidence.update(stable_moving_legs=legs[:12], turnback_stable_angle_deg=reversal,
                    turnback_confirmed=turnback, sinuous=sinuous, deceleration=slowing,
                    local_corroboration=local, local_neighbor_count=count,
                    local_valid_steps=steps, local_difference_mps=residual)
    if turnback:
        answer.update(eligible=True, review_type='turn_back', codes=['stable_direction_reversal'])
        answer['reasons'] = [f'转向前后均有至少90秒、150米的可信在航运动腿，实际方向反转{reversal:.1f}度。']
    elif sinuous['confirmed']:
        answer.update(eligible=True, review_type='sinuous', codes=['sustained_geometric_oscillation'])
        answer['reasons'] = [f'去除小幅位置抖动后，轨迹仍有{sinuous["alternating_turns"]}次交替大转向，各运动腿至少100米、30秒。']
    elif slowing['confirmed']:
        review_type = 'stall' if slowing['after_median_kn'] < 1.5 else 'speed_drop'
        answer.update(eligible=True, review_type=review_type, codes=['sustained_rapid_deceleration'])
        answer['reasons'] = [f'稳定在航段航速中位数{slowing["before_median_kn"]:.2f}节降至{slowing["after_median_kn"]:.2f}节，转变耗时{slowing["transition_seconds"]:.0f}秒，低速状态持续至少{slowing["low_speed_seconds"]:.0f}秒。']
    elif local and relation:
        answer.update(eligible=True, review_type=kind, codes=['supported_group_relation'])
        answer['reasons'] = [f'已有持续有效的近距离同向邻船比较，运动差异{residual:.2f}米/秒，并有真实在航轨迹。']
    elif local_specific:
        review_type = next((key for key in ['turn_back','sinuous','stall','speed_drop'] if key in supported), kind)
        answer.update(eligible=True, review_type=review_type, codes=['locally_corroborated_maneuver'])
        answer['reasons'] = [f'实际在航动作同时得到至少3艘可信近邻、3个有效时间步和{residual:.2f}米/秒运动差异的独立对照。']
    else:
        answer['codes'] = ['general_course_change_only'] if kind == 'deviation' else ['movement_without_focus_evidence']
        answer['reasons'] = [('仅相对自身初始方向变化，缺少指定航路或独立参照，保留运动线索而不自动升级为重点。' if kind == 'deviation' else
                              '存在运动变化，但没有满足稳定在航前后段、持续强动作或有效近邻对照的重点核查证据。')]
    return answer
