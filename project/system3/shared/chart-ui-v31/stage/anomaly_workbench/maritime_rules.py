"""Operational motion gates and transparent review-priority scoring.

Only original AIS reports and reliable contiguous position differences establish
an event. Neural outputs never establish motion, override a stationary gate, or
raise the priority of a vessel without adequate local neighbours. Scores are
review priorities, not calibrated anomaly or illegality probabilities.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

MIN_OBSERVATIONS = 8
MIN_MOVING_SOG_KN = 2.0
INITIAL_UNDERWAY_KN = 3.0
MIN_MOVING_PATH_M = 500.0
STATIONARY_PATH_M = 300.0
MIN_EARLY_PATH_M = 100.0


def _array(values: Any, n: int | None = None, default: float = math.nan) -> np.ndarray:
    if values is None:
        return np.full(n or 0, default, dtype=float)
    try:
        out = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        out = np.array([math.nan if x is None else float(x) for x in values], dtype=float)
    if n is None:
        return out
    result = np.full(n, default, dtype=float)
    result[:min(n, len(out))] = out[:n]
    return result


def _finite(value: float | None) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def _circular_mean(degrees: np.ndarray) -> float | None:
    if not len(degrees):
        return None
    angle = np.deg2rad(degrees)
    v = np.array([np.sin(angle).mean(), np.cos(angle).mean()])
    if np.linalg.norm(v) < 1e-8:
        return None
    return float(np.rad2deg(np.arctan2(v[0], v[1])) % 360)


def _wrapped_delta(degrees: np.ndarray | float) -> np.ndarray:
    return (np.asarray(degrees) + 180.0) % 360.0 - 180.0


def _saturate(value: float, scale: float) -> float:
    # Continuous diminishing returns preserve differences beyond the reference
    # scale. A hard clip at that scale erased real differences between e.g.
    # 12/24 kn initial speeds or 3/6 km tracks and produced a shared top score.
    # These are fixed physical reference scales, never batch ranks or jitter.
    x = max(0.0, float(value))
    reference = max(float(scale), 1e-9)
    if not math.isfinite(x):
        return 1.0 if x > 0 else 0.0
    return float(x / (reference + x))


def _priority(strength: float) -> float:
    # Smooth, monotone, independent of batch rank. Finite event strengths in
    # [0,1] yield [.50,.95], leaving headroom instead of rounding all rows to 1.
    return float(min(.99, .5 + .49 * (1.0 - math.exp(-2.5 * max(0.0, strength)))))


def _observations(meta: dict) -> tuple[dict, str]:
    if isinstance(meta.get('motion_observations'), dict):
        return meta['motion_observations'], 'all_original_window_observations'
    # Compatibility is explicit; callers must not present sampled legacy data
    # as a fresh full-window operational assessment.
    return {
        'sog_kn': meta.get('source_sog', meta.get('source_sog_kn', meta.get('sog_kn', []))),
        'cog_deg': meta.get('source_cog', meta.get('source_cog_deg', meta.get('cog_deg', []))),
        'time_seconds': meta.get('time_seconds', []),
        'xy_m': meta.get('raw_xy', []),
        'gate_weights': meta.get('gate_weights', []),
    }, 'legacy_sampled_observations_only'


def assess_motion(meta: dict, exclude_stationary: bool = True) -> dict:
    """Assess observed movement without consulting model or neighbour scores.

    Expected ``motion_observations`` contains original ``sog_kn``, ``cog_deg``,
    ``time_seconds``, local raw ``xy_m`` and kinematic ``gate_weights`` for every
    de-duplicated report in the chosen window. Missing report fields stay null.
    """
    obs, source = _observations(meta)
    sog = _array(obs.get('sog_kn', obs.get('source_sog')))
    n = len(sog)
    cog = _array(obs.get('cog_deg', obs.get('source_cog')), n)
    times = _array(obs.get('time_seconds'), n)
    weights = np.clip(_array(obs.get('gate_weights'), n, 1.0), 0, 1)
    weights[~np.isfinite(weights)] = 0
    try:
        raw_xy = np.asarray(obs.get('xy_m', []), dtype=float)
    except (TypeError, ValueError):
        raw_xy = np.empty((0, 2))
    xy = np.full((n, 2), np.nan)
    if raw_xy.ndim == 2 and raw_xy.shape[1] == 2:
        xy[:min(n, len(raw_xy))] = raw_xy[:n]
    speed_valid = np.isfinite(sog) & (sog >= 0) & (sog < 102.2)
    if 'sog_reported_mask' in obs:
        speed_valid &= _array(obs['sog_reported_mask'], n, 0) > .5
    heading_valid = np.isfinite(cog) & (cog >= 0) & (cog < 360)
    if 'cog_reported_mask' in obs:
        heading_valid &= _array(obs['cog_reported_mask'], n, 0) > .5
    # Combined reported_mask is intentionally not used to discard a valid SOG
    # of zero merely because COG is absent at a stationary vessel.
    reliable = speed_valid & np.isfinite(times) & np.isfinite(xy).all(1) & (weights >= .3)
    links = np.zeros(n, dtype=bool)
    link_distance = np.zeros(n, dtype=float)
    dt = np.zeros(n, dtype=float)
    dp = np.zeros((n, 2), dtype=float)
    if n > 1:
        dt[1:] = np.diff(times)
        dp[1:] = np.diff(xy, axis=0)
        distances = np.linalg.norm(dp, axis=1)
        average_report_mps = .5 * (sog + np.roll(sog, 1)) * .514444
        maximum_plausible = np.maximum(6.0, 2.0 * np.nan_to_num(average_report_mps, nan=0) + 3.0)
        links[1:] = (reliable[1:] & reliable[:-1] & (dt[1:] >= 1) & (dt[1:] <= 600) &
                     (distances[1:] / np.maximum(dt[1:], 1) <= maximum_plausible[1:]))
        link_distance[links] = distances[links]
    # Never bridge over an intercepted jump or a long observation gap to claim
    # a large movement. Each connected reliable segment is assessed separately.
    segments: list[np.ndarray] = []
    current: list[int] = []
    for i in range(n):
        if not reliable[i]:
            if current:
                segments.append(np.asarray(current)); current = []
        elif current and links[i]:
            current.append(i)
        else:
            if current:
                segments.append(np.asarray(current))
            current = [i]
    if current:
        segments.append(np.asarray(current))
    segment_metrics = []
    for indices in segments:
        if len(indices) < 2:
            continue
        positions = xy[indices]
        path = float(link_distance[indices[1:]].sum())
        net = float(np.linalg.norm(positions[-1] - positions[0]))
        extent = float(np.linalg.norm(positions - positions[0], axis=1).max())
        segment_metrics.append((path, net, extent, indices))
    path_length = float(link_distance.sum())
    longest = max(segment_metrics, key=lambda item: item[0], default=(0., 0., 0., np.array([], int)))
    continuous_path, net_displacement = longest[0], longest[1]
    track_extent = max((item[2] for item in segment_metrics), default=0.)
    values = sog[speed_valid]
    maximum = float(values.max()) if len(values) else None
    median = float(np.median(values)) if len(values) else None
    sufficient_sog = len(values) >= MIN_OBSERVATIONS and len(values) / max(n, 1) >= .8
    stationary_max = sufficient_sog and maximum is not None and maximum < MIN_MOVING_SOG_KN
    stationary_median = (sufficient_sog and median is not None and median < .5 and
                         len(longest[3]) >= 4 and path_length < STATIONARY_PATH_M)
    stationary = bool(stationary_max or stationary_median)
    valid_times = times[np.isfinite(times)]
    t_start = float(valid_times.min()) if len(valid_times) else 0.
    duration = float(valid_times.max() - t_start) if len(valid_times) else 0.
    early = reliable & (times <= t_start + duration * .25)
    late = reliable & (times >= t_start + duration * .65)
    initial = float(np.median(sog[early])) if early.sum() >= 3 else None
    final = float(np.median(sog[late])) if late.sum() >= 3 else None
    early_path = float(link_distance[links & (times <= t_start + duration * .3)].sum())
    decrease = max(0., (initial - final) / initial) if initial and final is not None else 0.
    decrease_kn = max(0., initial - final) if initial is not None and final is not None else 0.
    deceleration = np.zeros(n)
    if n > 1:
        differences = np.nan_to_num(np.diff(sog), nan=0)
        deceleration[1:] = np.where(links[1:], np.maximum(0, -differences) / np.maximum(dt[1:], 1) * 60., 0)
    max_deceleration = float(deceleration.max()) if n else 0.
    moving = reliable & (sog >= 2) & heading_valid
    heading_links = links & moving & np.r_[False, moving[:-1]] if n else links
    delta_heading = np.zeros(n)
    if n > 1:
        delta_heading[1:] = _wrapped_delta(np.diff(cog))
    total_turn = float(np.abs(delta_heading[heading_links]).sum())
    rot = np.where(heading_links, delta_heading / np.maximum(dt, 1) * 60., 0)
    # Geometry confirms that a heading oscillation is a curved movement,
    # rather than a noisy COG field along a straight observed trajectory.
    geo_heading = np.rad2deg(np.arctan2(dp[:, 0], dp[:, 1])) % 360
    geo_good = links & (link_distance >= 10)
    geo_turns = np.zeros(n)
    geo_consecutive = geo_good & np.r_[False, geo_good[:-1]] if n else geo_good
    if n > 1:
        geo_turns[1:] = _wrapped_delta(np.diff(geo_heading))
    active_turns = geo_turns[geo_consecutive & (np.abs(geo_turns) >= 5)]
    active_signs = np.sign(active_turns)
    alternations = int(np.sum(active_signs[1:] != active_signs[:-1])) if len(active_signs) > 1 else 0
    geometric_turn = float(np.abs(geo_turns[geo_consecutive]).sum())
    moving_indices = np.flatnonzero(moving)
    first_heading = _circular_mean(cog[moving_indices[:max(2, len(moving_indices)//5)]]) if len(moving_indices) else None
    last_heading = _circular_mean(cog[moving_indices[-max(2, len(moving_indices)//5):]]) if len(moving_indices) else None
    opposite_angle = abs(float(_wrapped_delta(last_heading-first_heading))) if first_heading is not None and last_heading is not None else 0.
    heading_span = float(np.ptp(np.unwrap(np.deg2rad(cog[moving_indices]))) * 180/math.pi) if len(moving_indices) else 0.
    cross_track = 0.
    if first_heading is not None and len(longest[3]) >= 2:
        a = math.radians(first_heading)
        perpendicular = np.array([math.cos(a), -math.sin(a)])
        cross_track = float(np.abs((xy[longest[3]] - xy[longest[3][0]]) @ perpendicular).max())
    enough = (len(values) >= MIN_OBSERVATIONS and reliable.sum() >= MIN_OBSERVATIONS and
              links.sum() >= 4 and duration >= 60)
    genuinely_underway = initial is not None and initial >= INITIAL_UNDERWAY_KN and early_path >= MIN_EARLY_PATH_M
    dynamic_drop = bool(enough and genuinely_underway and decrease_kn >= 1.5 and
                        (decrease > .5 or (final is not None and final < 1.5)))
    moving_path = continuous_path >= MIN_MOVING_PATH_M
    loiter = bool(enough and genuinely_underway and moving_path and total_turn > 120 and
                  geometric_turn > 100 and net_displacement < min(300., .25*continuous_path) and
                  final is not None and final < 2.)
    turn_back = bool(enough and moving_path and total_turn > 110 and opposite_angle > 110 and geometric_turn > 90)
    sinuous = bool(enough and moving_path and alternations >= 3 and heading_span >= 25 and geometric_turn >= 90)
    deviation = bool(enough and moving_path and opposite_angle >= 30 and
                     cross_track >= max(100., .1*continuous_path) and geometric_turn >= 25)
    supported: list[str] = []
    strengths: dict[str, float] = {}
    if dynamic_drop:
        strength = (.38*_saturate(decrease, 1) + .22*_saturate(initial or 0, 12) +
                    .22*_saturate(max_deceleration, 4) + .18*_saturate(continuous_path, 3000))
        if final is not None and final < 1.5:
            supported.append('stall'); strengths['stall'] = strength
        supported.append('speed_drop'); strengths['speed_drop'] = strength
    if loiter:
        supported.append('stall')
        strengths['stall'] = max(strengths.get('stall', 0), .4*_saturate(total_turn, 360)+.3*_saturate(geometric_turn, 360)+.3*_saturate(continuous_path, 2000))
    if turn_back:
        supported.append('turn_back'); strengths['turn_back'] = .35*_saturate(total_turn,180)+.3*_saturate(opposite_angle,180)+.2*_saturate(float(np.max(np.abs(rot))),60)+.15*_saturate(continuous_path,3000)
    if sinuous:
        supported.append('sinuous'); strengths['sinuous'] = .35*_saturate(alternations,8)+.3*_saturate(heading_span,90)+.2*_saturate(geometric_turn,360)+.15*_saturate(continuous_path,3000)
    if deviation:
        supported.append('deviation'); strengths['deviation'] = .45*_saturate(cross_track,1000)+.3*_saturate(opposite_angle,90)+.15*_saturate(geometric_turn,180)+.1*_saturate(continuous_path,3000)
    supported = list(dict.fromkeys(supported))
    # Prefer specific turning mechanisms to a generic relative-course change.
    preference = ['turn_back','sinuous','stall','speed_drop','deviation']
    event = next((key for key in preference if key in supported), None)
    if stationary or not enough:
        event = None; supported = []; strengths = {}
    strength = float(strengths.get(event, 0.))
    reasons: list[str] = []
    if stationary:
        reasons.append('符合靠泊/锚泊静止特征；不作为行为异常候选，也不能用远处过路船抬高排序。')
    elif not enough:
        reasons.append('原始有效速度、连续可信位置或时间覆盖不足，不能确认动态异常。')
    elif event is None:
        reasons.append('没有满足在航起点、真实位移及对应转向/减速条件的动态事件，不进入高优候选。')
    elif event in ('stall','speed_drop'):
        reasons.append(f'前段中位航速 {initial:.2f} kn，后段 {final:.2f} kn，降幅 {100*decrease:.1f}%，前段可信位移 {early_path:.1f} m。')
        reasons.append('此运动变化也可能来自正常到港、排队或避让，需结合靠泊与调度信息人工确认。')
    else:
        reasons.append(f'连续可信轨迹 {continuous_path:.1f} m，累计报告转向 {total_turn:.1f}°，真实位置转向 {geometric_turn:.1f}°。')
    if event == 'deviation':
        reasons.append('偏离仅相对自身初始航向，未提供官方航线，不能认定偏离指定航路。')
    if source != 'all_original_window_observations':
        reasons.append('仅有历史采样窗口，不能把本次检查宣称为全窗原始报文审查。')
    return {
        'is_stationary':stationary, 'stationary_exclusion_enabled':bool(exclude_stationary),
        'stationary_reason':'max_sog_below_2kn' if stationary_max else 'median_below_0_5kn_and_path_below_300m' if stationary_median else None,
        'movement_state':'berthed_anchored_features' if stationary else 'insufficient_observations' if not enough else 'underway_dynamic' if event else 'underway_no_supported_event' if moving_path else 'low_speed_or_local_movement',
        'is_underway':bool(enough and moving_path and np.any(reliable & (sog>=2)) and not stationary),
        'eligible':bool(event and not stationary and enough), 'event_type':event,
        'supported_types':supported, 'event_strengths':strengths, 'kinematic_strength':strength,
        'self_priority_score':_priority(strength) if event else 0.0 if stationary else .1 if enough else 0.0,
        'max_sog_kn':_finite(maximum),'median_sog_kn':_finite(median),
        'initial_sog_kn':_finite(initial),'late_sog_kn':_finite(final),'drop_fraction':decrease,
        'path_length_m':path_length,'continuous_path_length_m':continuous_path,
        'net_displacement_m':net_displacement,'track_extent_m':track_extent,'early_path_length_m':early_path,
        'max_deceleration_kn_per_min':max_deceleration,'total_heading_change_deg':total_turn,
        'peak_abs_rot_deg_per_min':float(np.max(np.abs(rot))) if n else 0.,
        'heading_span_deg':heading_span,
        'geometric_heading_change_deg':geometric_turn,'turnback_angle_deg':opposite_angle,
        'rot_sign_alternations':alternations,'cross_track_from_initial_heading_m':cross_track,
        'source_observations':n,'valid_sog_observations':int(speed_valid.sum()),'valid_sog_count':int(speed_valid.sum()),'reliable_observations':int(reliable.sum()),
        'valid_motion_links':int(links.sum()),'source_basis':source,'reasons':reasons,
        'evidence':{'initial_underway':genuinely_underway,'dynamic_speed_drop':dynamic_drop,'low_speed_loiter':loiter,
                    'turn_back':turn_back,'sinuous':sinuous,'relative_course_deviation':deviation},
    }


def score_priority(motion: dict, model_raw_score: float | None, neighbor_supported: bool,
                   local_residual: float | None = None, model_reference: Any = None) -> dict:
    """Rank only an already-supported motion event; sparse scores are self-only."""
    raw = _finite(model_raw_score)
    if raw is not None:
        raw = float(np.clip(raw,0,1))
    eligible = bool(motion.get('eligible')) and not motion.get('is_stationary',False)
    if not eligible:
        value=0.0 if motion.get('is_stationary') else min(.49,max(0.,float(motion.get('self_priority_score',0))))
        basis='stationary_excluded' if motion.get('is_stationary') else 'no_supported_dynamic_event'
        return {'score':value,'priority_score':value,'is_behavior_candidate':False,'score_basis':basis,
                'raw_model_score':raw,'model_used':False,'neighbor_used':False,
                'score_components':{'self_kinematic_strength':0.0},
                'priority_notice':'静止/无充分动态证据，不进入行为候选；模型高分不能覆盖业务门控。'}
    own=float(np.clip(motion.get('kinematic_strength',0),0,1))
    strength=own;components={'self_kinematic_strength':own}
    supported=bool(neighbor_supported)
    model_used=False
    if supported:
        reference=model_reference.get('scores') if isinstance(model_reference,dict) else model_reference
        percentile=raw if raw is not None else 0.
        if reference is not None and not np.isscalar(reference):
            values=np.sort(np.asarray(reference,float));values=values[np.isfinite(values)]
            if len(values) and raw is not None:percentile=float(np.searchsorted(values,raw,side='right')/len(values))
        elif reference is not None and np.isscalar(reference):
            percentile=float(np.clip(reference,0,1))
        residual=_finite(local_residual)
        residual_support=1-math.exp(-max(0,residual)/3) if residual is not None else 0.
        confirmation=.5*float(percentile)+.5*residual_support
        strength=.8*own+.2*confirmation
        components.update(model_percentile_or_raw_reference=float(percentile),local_residual_confirmation=residual_support,
                          self_weight=.8,confirmation_weight=.2)
        model_used=raw is not None
    value=_priority(strength)
    return {'score':value,'priority_score':value,'is_behavior_candidate':True,
            'score_basis':'self_kinematics_with_local_confirmation' if supported else 'self_kinematics',
            'raw_model_score':raw,'model_used':model_used,'neighbor_used':supported,
            'score_components':components,
            'priority_notice':'仅作可核查事件的平滑核查优先级，不是校准概率。' if supported else '可信同向邻船不足3艘；排序完全依据自身速度、转向与可信位移，不使用模型分或邻船速差。',
            'score_mapping':'0.50 + 0.49 * (1 - exp(-2.5 * physical_strength)); not a probability',
    }


def apply_policy(meta: dict, raw_model_score: float | None, neighbor_supported: bool,
                 local_residual: float | None = None, model_reference: Any = None,
                 exclude_stationary: bool = True) -> dict:
    motion=assess_motion(meta,exclude_stationary)
    return {'motion':motion,**score_priority(motion,raw_model_score,neighbor_supported,local_residual,model_reference)}
