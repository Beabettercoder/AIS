"""Controlled maneuvers that preserve the source AIS innovation.

All arrays are observed reports: seconds, local metric positions, knots and
clockwise-from-north degrees.  Position changes use the SAME endpoint-average
velocity operator as the shared kinematic gate. Thus behavior generation does
not make anomalous examples artificially cleaner than their source examples.

PhysicalLimits are configurable benchmark budgets, not vessel-type-calibrated
claims. Smooth examples are accepted only after both observed-grid and 1-second
reported-profile checks. hard_step is an explicit discontinuous comparison and
only records budget violations; it does not claim physical feasibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

KNOT_MPS = 1852.0 / 3600.0
KINDS = ("none", "turn_back", "deviation", "stall", "sinuous", "speed_drop")
DEFAULTS = {
    "turn_back": {"delta_deg": 150.0, "duration_s": 600.0, "sign": 1},
    "deviation": {"delta_deg": 45.0, "duration_s": 180.0, "hold_s": 240.0, "sign": 1},
    "stall": {"target_kn": 0.8, "duration_s": 240.0},
    "sinuous": {"amplitude_deg": 15.0, "period_s": 600.0, "duration_s": 120.0,
                "active_duration_s": 1800.0, "ramp_down_s": 120.0,
                "phase_rad": 0.0, "sign": 1},
    "speed_drop": {"factor": 0.4, "duration_s": 180.0},
}


@dataclass(frozen=True)
class PhysicalLimits:
    max_longitudinal_accel_mps2: float = 0.15
    max_turn_rate_deg_s: float = 1.5
    max_lateral_accel_mps2: float = 0.30
    max_speed_kn: float = 60.0
    max_duration_s: float = 1800.0
    tolerance: float = 1e-8

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")


def _input(t, pos, sog_kn, cog_deg):
    t = np.asarray(t, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    sog = np.asarray(sog_kn, dtype=np.float64)
    cog = np.asarray(cog_deg, dtype=np.float64)
    if t.ndim != 1 or len(t) < 2 or pos.shape != (len(t), 2):
        raise ValueError("Expected t:(N,), pos:(N,2), N>=2")
    if sog.shape != t.shape or cog.shape != t.shape:
        raise ValueError("Speed/course must match t")
    if not all(np.isfinite(v).all() for v in (t, pos, sog, cog)):
        raise ValueError("Target reports must be finite")
    if np.any(np.diff(t) <= 0) or np.any(sog < 0) or np.any(sog >= 102.2):
        raise ValueError("Times must increase and SOG be in [0,102.2) knots")
    if np.any(cog < 0) or np.any(cog >= 360):
        raise ValueError("COG must be in [0,360) degrees")
    return tuple(v.copy() for v in (t, pos, sog, cog))


def velocity_vectors(sog_kn, cog_deg):
    angle = np.deg2rad(np.asarray(cog_deg, float))
    speed = np.asarray(sog_kn, float) * KNOT_MPS
    return speed[..., None] * np.stack((np.sin(angle), np.cos(angle)), axis=-1)


def innovation_vectors(t, pos, sog_kn, cog_deg):
    """Vector innovation in m/s; do not conflate this with speed increments."""
    v = velocity_vectors(sog_kn, cog_deg)
    return np.diff(np.asarray(pos, float), axis=0) / np.diff(np.asarray(t, float))[:, None] - (v[1:] + v[:-1]) / 2.0


def physical_statistics(t, sog_kn, cog_deg):
    t = np.asarray(t, float); sog = np.asarray(sog_kn, float); cog = np.asarray(cog_deg, float)
    dt = np.diff(t)
    if len(dt) == 0:
        return {"max_longitudinal_accel_mps2": 0.0, "max_turn_rate_deg_s": 0.0,
                "max_lateral_accel_mps2": 0.0, "max_speed_kn": float(sog.max()),
                "min_speed_kn": float(sog.min())}
    dc = (np.diff(cog) + 180.0) % 360.0 - 180.0
    accel = np.abs(np.diff(sog)) * KNOT_MPS / dt
    rot = np.abs(dc) / dt
    lateral = (sog[1:] + sog[:-1]) * 0.5 * KNOT_MPS * np.deg2rad(rot)
    return {"max_longitudinal_accel_mps2": float(accel.max()),
            "max_turn_rate_deg_s": float(rot.max()),
            "max_lateral_accel_mps2": float(lateral.max()),
            "max_speed_kn": float(sog.max()), "min_speed_kn": float(sog.min())}


def _violations(stats, limits):
    return [name for name in ("max_longitudinal_accel_mps2", "max_turn_rate_deg_s",
                              "max_lateral_accel_mps2", "max_speed_kn")
            if stats[name] > getattr(limits, name) + limits.tolerance]


def _ramp(times, onset, duration, profile):
    if profile == "hard_step":
        return (times >= onset).astype(np.float64)
    u = np.clip((times - onset) / duration, 0.0, 1.0)
    return u*u*(3.0-2.0*u)


def _params(kind, supplied):
    p = dict(DEFAULTS[kind]); supplied = dict(supplied or {})
    if kind == 'sinuous' and 'ramp_up_s' in supplied:
        ramp_up = supplied.pop('ramp_up_s')
        if 'duration_s' in supplied and supplied['duration_s'] != ramp_up:
            raise ValueError('For sinuous, duration_s and ramp_up_s must agree')
        supplied['duration_s'] = ramp_up
    unknown = set(supplied)-set(p)
    if unknown:
        raise ValueError(f"Unknown {kind} parameters: {sorted(unknown)}")
    p.update(supplied)
    if not all(np.isfinite(v) for v in p.values()):
        raise ValueError("Parameters must be finite")
    if p["duration_s"] <= 0:
        raise ValueError("duration_s must be positive")
    if "sign" in p and p["sign"] not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    if "delta_deg" in p and not 0 < p["delta_deg"] <= 180:
        raise ValueError("delta_deg must be in (0,180]")
    if "hold_s" in p and p["hold_s"] < 0:
        raise ValueError("hold_s must be nonnegative")
    if kind == "stall" and not 0 <= p["target_kn"] < 102.2:
        raise ValueError("target_kn must be a valid nonnegative AIS speed")
    if kind == "speed_drop" and not 0 < p["factor"] < 1:
        raise ValueError("factor must be in (0,1)")
    if kind == "sinuous":
        if not 0 < p["amplitude_deg"] < 90 or p["period_s"] <= 0:
            raise ValueError("Require 0<amplitude_deg<90 and positive period_s")
        if p['ramp_down_s'] <= 0 or p['active_duration_s'] < p['duration_s']+p['ramp_down_s']:
            raise ValueError('Sinuous active_duration_s must contain the two ramps')
        p['ramp_up_s'] = p['duration_s']
    return p


def _profile(times, source_speed, source_course, kind, onset, p, profile):
    """Apply relative changes, preserving the observed baseline variations."""
    ramp = _ramp(times, onset, p["duration_s"], profile)
    speed = np.array(source_speed, dtype=np.float64, copy=True)
    heading_offset = np.zeros(len(times), dtype=np.float64)
    if kind == "turn_back":
        heading_offset = p["sign"]*p["delta_deg"]*ramp
    elif kind == "deviation":
        return_at = onset+p["duration_s"]+p["hold_s"]
        heading_offset = p["sign"]*p["delta_deg"]*(ramp-_ramp(times, return_at, p["duration_s"], profile))
    elif kind == "stall":
        speed = source_speed*(1-ramp)+p["target_kn"]*ramp
    elif kind == "speed_drop":
        speed = source_speed*(1-(1-p["factor"])*ramp)
    elif kind == "sinuous":
        phase = 2*np.pi*(times-onset)/p["period_s"]+p['phase_rad']
        end_at = onset+p['active_duration_s']
        down_at = end_at if profile == 'hard_step' else end_at-p['ramp_down_s']
        down = _ramp(times,down_at,p['ramp_down_s'],profile)
        heading_offset = p["sign"]*p["amplitude_deg"]*ramp*(1-down)*np.sin(phase)
    return speed, (source_course+heading_offset) % 360.0, heading_offset, ramp


def sample_parameters(kind, seed):
    """A deterministic example parameter sampler; final ranges belong in protocol."""
    if kind not in DEFAULTS:
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    sign = int(rng.choice([-1, 1]))
    if kind == "turn_back":
        return dict(delta_deg=float(rng.uniform(120,180)),duration_s=float(rng.uniform(360,720)),sign=sign)
    if kind == "deviation":
        return dict(delta_deg=float(rng.uniform(25,80)),duration_s=float(rng.uniform(120,300)),hold_s=float(rng.uniform(120,420)),sign=sign)
    if kind == "stall":
        return dict(target_kn=float(rng.uniform(.4,1.5)),duration_s=float(rng.uniform(180,420)))
    if kind == "sinuous":
        return dict(amplitude_deg=float(rng.uniform(8,25)),period_s=float(rng.uniform(300,720)),duration_s=float(rng.uniform(90,240)),phase_rad=float(rng.uniform(0,2*np.pi)),sign=sign)
    return dict(factor=float(rng.uniform(.25,.65)),duration_s=float(rng.uniform(120,360)))


def generate_maneuver(t, pos, sog_kn, cog_deg, *, kind, onset_s=None, params=None,
                      profile="smooth", limits=None, adapt=True,
                      max_adaptations=8, adaptation_factor=1.35):
    """Return an accepted candidate or unchanged input with an explicit rejection.

    The off-grid onset uses a disclosed discrete bridge: changes are zero at all
    reports before onset. The first changed report and its predecessor delimit
    the onset interval. The endpoint-average delta-velocity is integrated on that
    ENTIRE report interval, consistently with the shared discrete gate. No claim
    of sub-report onset-time resolution is made. Prefer observed report onsets
    when constructing strictly matched temporal tests.
    """
    t, pos, sog, cog = _input(t,pos,sog_kn,cog_deg)
    if kind not in KINDS or profile not in ("smooth","hard_step"):
        raise ValueError("Unknown maneuver kind/profile")
    limits = limits or PhysicalLimits()
    if not isinstance(limits, PhysicalLimits):
        limits = PhysicalLimits(**dict(limits))
    if max_adaptations < 0 or int(max_adaptations) != max_adaptations or not np.isfinite(adaptation_factor) or adaptation_factor <= 1:
        raise ValueError("Invalid adaptation policy")
    original = dict(t=t.copy(),pos=pos.copy(),sog_kn=sog.copy(),cog_deg=cog.copy())
    baseline_stats = physical_statistics(t,sog,cog)
    if kind == "none":
        return dict(accepted=True,reason="identity",**original,
                    metadata=dict(schema="innovation_preserving_maneuver_v1",kind="none",profile="identity",
                                  requested_onset_s=None,onset_bracket_s=None,
                                  requested_params={},applied_params={},physical_observed=baseline_stats,
                                  constraints_enforced=False,violations=_violations(baseline_stats,limits),
                                  realized=dict(max_heading_offset_deg=0.,max_speed_change_kn=0.,
                                                transition_progress_at_decision=0.),
                                  innovation_preservation_max_mps=0.0,
                                  input_mutated=False,behavior_injected=False))
    if onset_s is None or not np.isfinite(onset_s) or not t[0] < onset_s < t[-1]:
        raise ValueError("A maneuver requires t[0] < onset_s < t[-1]")
    onset = float(onset_s); requested = _params(kind,params); applied = dict(requested)
    if requested["duration_s"] > limits.max_duration_s:
        raise ValueError("Requested duration exceeds configured maximum")
    i0 = int(np.searchsorted(t,onset,side="left"));before = i0-1
    temporal_period_floor = float(4*np.median(np.diff(t))) if kind == 'sinuous' else None
    temporal_violation = False
    if kind == 'sinuous' and applied['period_s'] < temporal_period_floor:
        if adapt:
            applied['period_s'] = temporal_period_floor
        else:
            temporal_violation = True
    # Dense checks interpolate *reported* baseline states, not hidden true motion.
    # This is an explicit benchmark feasibility check rather than ship dynamics.
    dense_t = np.unique(np.r_[t,np.arange(t[0],t[-1],1.0),onset])
    dense_speed = np.interp(dense_t,t,sog)
    source_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(cog)))
    dense_course = np.interp(dense_t,t,source_unwrapped) % 360
    source_v = velocity_vectors(sog,cog)
    source_innovation = innovation_vectors(t,pos,sog,cog)
    attempts=[]; accepted=False; candidate=None
    for attempt in range(int(max_adaptations)+1):
        aug_sog,aug_cog,offset,progress = _profile(t,sog,cog,kind,onset,applied,profile)
        delta_v=velocity_vectors(aug_sog,aug_cog)-source_v
        delta_pos=np.zeros_like(pos)
        delta_pos[1:]=np.cumsum(.5*(delta_v[1:]+delta_v[:-1])*np.diff(t)[:,None],axis=0)
        aug_pos=pos+delta_pos
        dense_sog,dense_cog,_,_ = _profile(dense_t,dense_speed,dense_course,kind,onset,applied,profile)
        obs_stats=physical_statistics(t,aug_sog,aug_cog)
        dense_stats=physical_statistics(dense_t,dense_sog,dense_cog)
        violations=sorted(set(_violations(obs_stats,limits)+_violations(dense_stats,limits)))
        if temporal_violation:violations.append('period_below_four_median_report_intervals')
        attempts.append(dict(duration_s=float(applied['duration_s']),applied_params=dict(applied),violations=violations,
                             physical_observed=obs_stats,physical_dense=dense_stats))
        candidate=(aug_pos,aug_sog,aug_cog,offset,progress,obs_stats,dense_stats)
        if not temporal_violation and (profile == "hard_step" or not violations):
            accepted=True;break
        if not adapt or attempt>=max_adaptations:break
        next_duration=applied['duration_s']*adaptation_factor
        if next_duration>limits.max_duration_s:break
        proposed=dict(applied);proposed['duration_s']=float(next_duration)
        if kind == 'sinuous':
            # Longer activation alone cannot reduce the eventual oscillation ROT.
            proposed['period_s']=float(applied['period_s']*adaptation_factor)
            proposed['ramp_up_s']=proposed['duration_s']
            if proposed['active_duration_s'] < proposed['duration_s']+proposed['ramp_down_s']:
                break
        applied=proposed
    aug_pos,aug_sog,aug_cog,offset,progress,obs_stats,dense_stats=candidate
    innovation_error = np.linalg.norm(innovation_vectors(t,aug_pos,aug_sog,aug_cog)-source_innovation,axis=1)
    preservation=float(innovation_error.max())
    if preservation>1e-7:
        raise FloatingPointError(f"Source innovation was not preserved: {preservation}")
    assert np.array_equal(aug_pos[:i0],pos[:i0])
    changed=np.flatnonzero((np.abs(aug_sog-sog)>1e-10)|(np.abs(offset)>1e-10))
    realized=dict(max_heading_offset_deg=float(np.abs(offset).max()),
                  terminal_heading_offset_deg=float(offset[-1]),
                  max_speed_change_kn=float(np.abs(aug_sog-sog).max()),
                  terminal_speed_change_kn=float(aug_sog[-1]-sog[-1]),
                  terminal_speed_factor=float(aug_sog[-1]/sog[-1]) if sog[-1]>0 else None,
                  transition_progress_at_decision=float(progress[-1]),
                  reports_before_onset=i0,reports_at_or_after_onset=len(t)-i0,
                  first_changed_report_s=None if not len(changed) else float(t[changed[0]]))
    reference_speed=float(np.interp(onset,t,sog))
    realized.update(reference_speed_at_onset_kn=reference_speed,
                    terminal_speed_change_from_onset_kn=float(aug_sog[-1]-reference_speed),
                    minimum_speed_change_from_onset_kn=float(aug_sog[t>=onset].min()-reference_speed))
    if kind == 'sinuous':
        post_offset=offset[t>=onset]
        realized.update(realized_sinuous_amplitude_deg=float(np.abs(post_offset).max()),
                        realized_sinuous_peak_to_peak_deg=float(np.ptp(post_offset)),
                        observed_cycles=float((t[-1]-onset)/applied['period_s']))
    metadata=dict(schema="innovation_preserving_maneuver_v1",kind=kind,profile=profile,
                  requested_onset_s=onset,onset_bracket_s=[float(t[before]),float(t[i0])],
                  onset_on_observed_report=bool(np.any(t==onset)),
                  bridge_policy="Actual AIS intervals; trapezoidal integration of augmented-minus-source endpoint velocities including the predecessor-to-first-affected interval",
                  sub_report_physical_onset_resolution_claimed=False,
                  requested_params=requested,applied_params=applied,adaptation_attempts=attempts,
                  limits=asdict(limits),budget_interpretation="Configurable reported-profile benchmark budgets; not calibrated vessel dynamics",
                  physical_baseline=baseline_stats,physical_observed=obs_stats,physical_dense=dense_stats,
                  physical_dense_grid="<=1 second between samples, source speed linear and source course shortest-angle interpolated",
                  temporal_period_floor_s=temporal_period_floor,
                  constraints_enforced=profile=="smooth",violations=attempts[-1]['violations'],
                  realized=realized,innovation_preservation_max_mps=preservation,
                  behavior_injected=accepted,input_mutated=False)
    if not accepted:
        # A failed feasibility check must not silently produce a labeled anomaly.
        return dict(accepted=False,reason="temporal_resolution_exceeded" if temporal_violation else "physical_budget_exceeded",**original,metadata=metadata)
    return dict(accepted=True,reason="accepted" if profile=="smooth" else "matched_discontinuous_control",
                t=t.copy(),pos=aug_pos,sog_kn=aug_sog,cog_deg=aug_cog,metadata=metadata)


def apply_observation_noise(t,pos,sog_kn,cog_deg,*,noise="none",seed=0,params=None):
    """Label-independent observation corruption; never changes behavior labels."""
    t,p,s,c=_input(t,pos,sog_kn,cog_deg);rng=np.random.default_rng(seed);params=dict(params or {})
    if noise not in ('none','teleport','speed_spike','drift'):
        raise ValueError(noise)
    allowed={'none':set(),'teleport':{'report_index','offset_m','angle_deg'},
             'speed_spike':{'report_index','reports','factor'},
             'drift':{'onset_s','duration_s','offset_m','angle_deg'}}[noise]
    if set(params)-allowed:raise ValueError(f'Unknown {noise} parameters: {sorted(set(params)-allowed)}')
    before=innovation_vectors(t,p,s,c)
    applied={};n=len(t)
    if noise=='teleport':
        ix=int(params.get('report_index',rng.integers(1,n)))
        if not 0<=ix<n:raise ValueError('Invalid report_index')
        amplitude=float(params.get('offset_m',rng.uniform(500,1500)))
        angle=float(params.get('angle_deg',rng.uniform(0,360)))
        if not np.isfinite(amplitude) or amplitude<0 or not np.isfinite(angle):raise ValueError('Invalid teleport')
        p[ix]+=amplitude*np.array([np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))])
        applied=dict(report_index=ix,time_s=float(t[ix]),offset_m=amplitude,angle_deg=angle)
    elif noise=='speed_spike':
        ix=int(params.get('report_index',rng.integers(1,n)))
        count=int(params.get('reports',2));factor=float(params.get('factor',rng.uniform(3,5)))
        if not 0<=ix<n or count<1 or not np.isfinite(factor) or factor<=1:raise ValueError('Invalid speed spike')
        # 102.1 remains below the unavailable-speed threshold after float32 IO.
        end=min(n,ix+count);s[ix:end]=np.minimum(s[ix:end]*factor,102.1)
        applied=dict(report_index=ix,reports=end-ix,factor=factor,time_s=float(t[ix]),ais_valid_speed_cap_kn=102.1)
    elif noise=='drift':
        onset=float(params.get('onset_s',t[max(0,n//2-1)]));duration=float(params.get('duration_s',300))
        amplitude=float(params.get('offset_m',rng.uniform(500,1500)));angle=float(params.get('angle_deg',rng.uniform(0,360)))
        if not all(np.isfinite(v) for v in [onset,duration,amplitude,angle]) or duration<=0 or amplitude<0:raise ValueError('Invalid drift')
        u=(t-onset)/duration;env=np.where((u>=0)&(u<=1),.5-.5*np.cos(2*np.pi*np.clip(u,0,1)),0.)
        p+=amplitude*env[:,None]*np.array([np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))])
        applied=dict(onset_s=onset,duration_s=duration,offset_m=amplitude,angle_deg=angle)
    after=innovation_vectors(t,p,s,c)
    return dict(t=t,pos=p,sog_kn=s,cog_deg=c,
                metadata=dict(schema="independent_observation_corruption_v1",noise=noise,seed=int(seed),
                              parameters=applied,behavior_label_changed=False,input_mutated=False,
                              observations_changed=bool(not np.array_equal(p,np.asarray(pos)) or not np.array_equal(s,np.asarray(sog_kn))),
                              innovation_before_max_mps=float(np.linalg.norm(before,axis=1).max()),
                              innovation_after_max_mps=float(np.linalg.norm(after,axis=1).max())))


def paired_controls(t,pos,sog_kn,cog_deg,*,kind,onset_s,params=None,limits=None,adapt=True):
    """Adapt smooth feasibility first, then use its EXACT applied plan for hard."""
    smooth=generate_maneuver(t,pos,sog_kn,cog_deg,kind=kind,onset_s=onset_s,params=params,
                             limits=limits,adapt=adapt,profile='smooth')
    if not smooth['accepted']:
        return {'accepted':False,'smooth':smooth,'hard_step':None}
    hard=generate_maneuver(t,pos,sog_kn,cog_deg,kind=kind,onset_s=onset_s,
                          params=smooth['metadata']['applied_params'],limits=limits,
                          adapt=False,profile='hard_step')
    assert smooth['metadata']['applied_params']==hard['metadata']['applied_params']
    return {'accepted':True,'smooth':smooth,'hard_step':hard}
