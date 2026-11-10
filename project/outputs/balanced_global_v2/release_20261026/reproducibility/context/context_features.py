"""Causal, shared reliability-aware context features for trajectory benchmarks.

Input columns are ``mmsi, time_s, x_m, y_m, sog_kn, cog_deg``. Times are
absolute UTC epoch seconds; coordinates use ONE common metric projection;
COG is degrees clockwise from north. No labels enter this module.

``BackgroundIndex`` is an immutable observation snapshot. Rebuild it after
changing ANY background observation. Call ``build_window`` again after any
target perturbation. Indexes may be reused for unchanged background data.

Missing fields are represented by zero plus explicit validity masks, never by
a claimed valid field. The returned JSON metadata includes every selected
neighbor identity/time, including rejected candidates and rejection reasons.
All neural pooling/loss code MUST consume the returned step mask. Train-only
neighbor eligibility is supplied explicitly by the split manager.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable, Mapping

import numpy as np

KNOT_MPS = 1852.0 / 3600.0
NM_M = 1852.0
COLUMNS = ("mmsi", "time_s", "x_m", "y_m", "sog_kn", "cog_deg")
RAW_NAMES = ("dx_over_R", "dy_over_R", "vx_over_vscale", "vy_over_vscale",
             "dt_over_window", "sin_cog", "cos_cog")
CONTEXT_NAMES = RAW_NAMES + (
    "residual_vx_over_vscale", "residual_vy_over_vscale",
    "kinematic_residual_over_vscale", "self_reliability", "kinematic_valid",
    "context_valid", "support_fraction", "divergence_times_R_over_vscale",
    "curl_times_R_over_vscale")


@dataclass(frozen=True)
class ContextConfig:
    max_steps: int = 20
    min_steps: int = 8
    window_seconds: float = 23 * 60.0
    radius_nm: float = 8.0
    distance_sigma_nm: float = 4.0
    heading_limit_deg: float = 45.0
    freshness_seconds: float = 300.0
    max_kinematic_gap_seconds: float = 300.0
    min_heading_speed_kn: float = 0.5
    gate_deadzone_mps: float = 2.5
    gate_denominator_mps2: float = 8.0
    min_neighbor_reliability: float = 0.3
    min_neighbors: int = 4
    min_geometry_spread_m: float = 50.0
    max_design_condition: float = 1e5
    ridge: float = 1e-6
    velocity_scale_mps: float = 5 * KNOT_MPS

    def __post_init__(self):
        for name in ("max_steps", "min_steps", "window_seconds", "radius_nm",
                     "distance_sigma_nm", "freshness_seconds",
                     "max_kinematic_gap_seconds", "gate_denominator_mps2",
                     "min_neighbors", "min_geometry_spread_m",
                     "max_design_condition", "velocity_scale_mps"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_steps > self.max_steps or self.min_neighbors < 3:
            raise ValueError("min_steps <= max_steps and min_neighbors >= 3 required")
        if not 0 <= self.heading_limit_deg <= 180:
            raise ValueError("heading_limit_deg must be in [0,180]")
        if not 0 <= self.min_neighbor_reliability <= 1:
            raise ValueError("min_neighbor_reliability must be in [0,1]")
        if self.gate_deadzone_mps < 0 or self.ridge < 0:
            raise ValueError("gate_deadzone_mps and ridge must be nonnegative")
        for name in ("max_steps", "min_steps", "min_neighbors"):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f"{name} must be an integer")


def _canonicalize(observations: Mapping[str, Iterable]) -> dict:
    missing = set(COLUMNS) - set(observations)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = {k: np.asarray(observations[k], dtype=str if k == "mmsi" else float)
            for k in COLUMNS}
    if any(v.ndim != 1 for v in data.values()):
        raise ValueError("All input columns must be 1-D")
    n = len(data["mmsi"])
    if any(len(v) != n for v in data.values()):
        raise ValueError("Input columns have different lengths")
    if np.any(~np.isfinite(data["time_s"])) or np.any(data["mmsi"] == ""):
        raise ValueError("Every observation needs a finite time and nonempty identity")
    # Duplicate timestamps have a deterministic last-input-row policy. Such
    # duplicates are counted; simultaneous reports never count as extra ships.
    order = np.lexsort((np.arange(n), data["time_s"], data["mmsi"]))
    data = {k: v[order] for k, v in data.items()}
    keep = np.ones(n, bool)
    if n > 1:
        keep[:-1] = ((data["mmsi"][:-1] != data["mmsi"][1:]) |
                     (data["time_s"][:-1] != data["time_s"][1:]))
    data = {k: v[keep] for k, v in data.items()}
    data["duplicates_removed"] = n - int(keep.sum())
    return data


def _kinematics(data: dict, cfg: ContextConfig) -> dict:
    """Raw increments and trust are independent; residual uses m/s units."""
    n = len(data["time_s"])
    pos = np.column_stack((data["x_m"], data["y_m"]))
    heading_ok = np.isfinite(data["cog_deg"]) & (data["cog_deg"] >= 0) & (data["cog_deg"] < 360)
    speed_ok = np.isfinite(data["sog_kn"]) & (data["sog_kn"] >= 0) & (data["sog_kn"] < 102.2)
    observed = np.isfinite(pos).all(axis=1) & heading_ok & speed_ok
    theta = np.deg2rad(np.where(heading_ok, data["cog_deg"], 0))
    speed = np.where(speed_ok, data["sog_kn"], 0) * KNOT_MPS
    velocity = speed[:, None] * np.column_stack((np.sin(theta), np.cos(theta)))
    dt = np.zeros(n)
    delta = np.zeros((n, 2))
    time_interval_valid = np.zeros(n, bool)
    position_increment_valid = np.zeros(n, bool)
    pair_valid = np.zeros(n, bool)
    if n > 1:
        dt[1:] = np.diff(data["time_s"])
        delta[1:] = np.diff(pos, axis=0)
        time_interval_valid[1:] = ((data["mmsi"][1:] == data["mmsi"][:-1]) & (dt[1:] > 0))
        finite_position = np.isfinite(pos).all(axis=1)
        position_increment_valid[1:] = (time_interval_valid[1:] &
                                        finite_position[1:] & finite_position[:-1])
        pair_valid[1:] = (observed[1:] & observed[:-1] & time_interval_valid[1:] &
                          (dt[1:] <= cfg.max_kinematic_gap_seconds))
    error = np.zeros(n)
    ix = np.flatnonzero(pair_valid)
    if len(ix):
        measured = delta[ix] / dt[ix, None]
        reported = (velocity[ix] + velocity[ix-1]) / 2
        error[ix] = np.linalg.norm(measured - reported, axis=1)
    reliability = np.zeros(n)
    reliability[ix] = np.exp(-np.maximum(0, error[ix] - cfg.gate_deadzone_mps)**2 /
                              cfg.gate_denominator_mps2)
    # Preserve actual sampling intervals and finite position increments even
    # when the interval is too long to certify kinematic consistency. Unknown
    # reliability is distinct from a measured low reliability.
    delta[~position_increment_valid] = 0
    dt[~time_interval_valid] = 0
    return dict(position=pos, velocity=velocity, theta=theta, observed=observed,
                dt=dt, delta=delta, time_interval_valid=time_interval_valid,
                position_increment_valid=position_increment_valid, kinematic_valid=pair_valid,
                e_kin_mps=error, reliability=reliability)


class BackgroundIndex:
    """Precomputed causal per-report reliability for an unchanged background."""

    def __init__(self, observations: Mapping[str, Iterable], config: ContextConfig | None = None):
        self.config = config or ContextConfig()
        data = _canonicalize(observations)
        self.duplicates_removed = data.pop("duplicates_removed")
        values = _kinematics(data, self.config)
        order = np.argsort(data["time_s"], kind="stable")
        self.data = {k: v[order] for k, v in data.items()}
        self.values = {k: v[order] for k, v in values.items()}
        digest = hashlib.sha256()
        for key in COLUMNS:
            digest.update(key.encode())
            digest.update(self.data[key].tobytes())
        self.fingerprint = digest.hexdigest()
        for arr in list(self.data.values()) + list(self.values.values()):
            arr.setflags(write=False)

    def latest(self, time_s: float, target_id: str, eligible: set[str] | None):
        times = self.data["time_s"]
        lo = np.searchsorted(times, time_s - self.config.freshness_seconds, side="left")
        hi = np.searchsorted(times, time_s, side="right")
        candidates = np.arange(lo, hi)
        if not len(candidates):
            return candidates
        _, reverse_first = np.unique(self.data["mmsi"][candidates][::-1], return_index=True)
        latest = candidates[len(candidates) - 1 - reverse_first]
        latest = latest[self.data["mmsi"][latest] != target_id]
        if eligible is not None:
            latest = latest[np.fromiter((str(x) in eligible for x in self.data["mmsi"][latest]), bool)]
        return latest

    def window_observations(self, decision_time_s: float) -> dict:
        """Copy the causal reports and history margins for one augmented window.

        Mutate this copy, then rebuild BackgroundIndex, avoiding a full-region
        copy for each augmented example. Includes predecessor-gate history.
        """
        cfg = self.config
        earliest = (decision_time_s - cfg.window_seconds - cfg.freshness_seconds -
                    cfg.max_kinematic_gap_seconds)
        lo = np.searchsorted(self.data["time_s"], earliest, side="left")
        hi = np.searchsorted(self.data["time_s"], decision_time_s, side="right")
        return {key: arr[lo:hi].copy() for key, arr in self.data.items()}


def _affine_fit(positions, velocities, weights, cfg):
    """Normalized local affine fit; weights sum to one, positions relative to target."""
    result = dict(field=np.zeros(2), divergence=0., curl=0., valid=False,
                  reason="sparse", condition=None, min_geometry_std_m=None)
    if len(positions) < cfg.min_neighbors:
        return result
    centered = positions - np.sum(weights[:, None] * positions, axis=0)
    covariance = centered.T @ (weights[:, None] * centered)
    minimum_eigenvalue = max(0., float(np.linalg.eigvalsh(covariance)[0]))
    result["min_geometry_std_m"] = float(np.sqrt(minimum_eigenvalue))
    if minimum_eigenvalue < cfg.min_geometry_spread_m**2:
        result["reason"] = "degenerate_geometry"
        return result
    X = np.column_stack((np.ones(len(positions)), positions / (cfg.radius_nm * NM_M)))
    gram = X.T @ (weights[:, None] * X)
    condition = float(np.linalg.cond(gram))
    result["condition"] = condition if np.isfinite(condition) else None
    if not np.isfinite(condition) or condition > cfg.max_design_condition:
        result["reason"] = "ill_conditioned"
        return result
    try:
        coefficients = np.linalg.solve(gram + cfg.ridge * np.diag([0., 1., 1.]),
                                       X.T @ (weights[:, None] * velocities))
    except np.linalg.LinAlgError:
        result["reason"] = "singular_solve"
        return result
    if not np.isfinite(coefficients).all():
        result["reason"] = "nonfinite_solution"
        return result
    R = cfg.radius_nm * NM_M
    result.update(field=coefficients[0],
                  divergence=float((coefficients[1, 0] + coefficients[2, 1]) / R),
                  curl=float((coefficients[1, 1] - coefficients[2, 0]) / R),
                  valid=True, reason="valid")
    return result


def _context_at(index, time_s, target_id, position, cog_deg, sog_kn, eligible, cfg):
    empty = dict(mean=np.zeros(2), field=np.zeros(2), residual=np.zeros(2),
                 divergence=0., curl=0., valid=False, mean_valid=False,
                 support_fraction=0., count=0, reason="sparse", neighbors=[],
                 condition=None, min_geometry_std_m=None,
                 plain_mean=np.zeros(2), plain_mean_valid=False, plain_count=0,
                 plain_residual=np.zeros(2), plain_valid=False,
                 plain_affine=_affine_fit(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), cfg))
    if not np.isfinite(position).all() or not np.isfinite(cog_deg) or sog_kn < cfg.min_heading_speed_kn:
        empty["reason"] = "target_heading_unavailable"
        return empty
    indices = index.latest(time_s, target_id, eligible)
    if not len(indices):
        return empty
    p = index.values["position"][indices]
    distances = np.linalg.norm(p - position, axis=1)
    # Unknown/nonfinite positions cannot be spatial neighbors. They are counted
    # in index metadata but not silently placed at the target position.
    spatial = np.isfinite(distances) & (distances <= cfg.radius_nm * NM_M)
    indices, distances = indices[spatial], distances[spatial]
    accepted, weights, plain_accepted = [], [], []
    for ix, distance in zip(indices, distances):
        heading = index.data["cog_deg"][ix]
        heading_difference = abs((heading - cog_deg + 180) % 360 - 180)
        w = float(index.values["reliability"][ix])
        distance_weight = float(np.exp(-0.5 * (distance / (cfg.distance_sigma_nm * NM_M))**2))
        reason = None
        if not index.values["observed"][ix]:
            reason = "invalid_observation"
        elif index.data["sog_kn"][ix] < cfg.min_heading_speed_kn:
            reason = "heading_unavailable_at_low_speed"
        elif heading_difference > cfg.heading_limit_deg:
            reason = "heading_mismatch"
        plain_ok = reason is None
        if plain_ok:
            plain_accepted.append(ix)
            if not index.values["kinematic_valid"][ix]:
                reason = "kinematics_unavailable"
            elif w < cfg.min_neighbor_reliability or w == 0:
                reason = "unreliable_observation"
            elif distance_weight <= np.finfo(float).tiny:
                reason = "negligible_distance_weight"
        effective_weight = w * distance_weight if reason is None else 0.
        empty["neighbors"].append(dict(
            mmsi=str(index.data["mmsi"][ix]), time_s=float(index.data["time_s"][ix]),
            age_s=float(time_s - index.data["time_s"][ix]), distance_m=float(distance),
            heading_difference_deg=float(heading_difference) if np.isfinite(heading_difference) else None,
            reliability=w, distance_weight=distance_weight, weight=effective_weight,
            e_kin_mps=float(index.values["e_kin_mps"][ix]),
            kinematic_valid=bool(index.values["kinematic_valid"][ix]),
            accepted=reason is None, plain_accepted=plain_ok, rejection=reason))
        if reason is None:
            accepted.append(ix)
            weights.append(effective_weight)
    target_velocity = sog_kn * KNOT_MPS * np.array([np.sin(np.deg2rad(cog_deg)), np.cos(np.deg2rad(cog_deg))])
    np_plain = len(plain_accepted)
    empty["plain_count"] = np_plain
    if np_plain:
        plain_positions = index.values["position"][plain_accepted] - position
        plain_velocities = index.values["velocity"][plain_accepted]
        empty["plain_mean"] = plain_velocities.mean(axis=0)
        empty["plain_mean_valid"] = True
        empty["plain_valid"] = np_plain >= cfg.min_neighbors
        if empty["plain_valid"]:
            empty["plain_residual"] = target_velocity - empty["plain_mean"]
        empty["plain_affine"] = _affine_fit(plain_positions, plain_velocities,
                                             np.full(np_plain, 1. / np_plain), cfg)
    n = len(accepted)
    empty["count"] = n
    empty["support_fraction"] = min(1., n / cfg.min_neighbors)
    if n == 0:
        return empty
    weights = np.asarray(weights)
    weights /= weights.sum()
    positions = index.values["position"][accepted] - position
    velocities = index.values["velocity"][accepted]
    empty["mean"] = np.sum(weights[:, None] * velocities, axis=0)
    empty["mean_valid"] = True
    fit = _affine_fit(positions, velocities, weights, cfg)
    empty.update(fit)
    if fit["valid"]:
        empty["residual"] = target_velocity - fit["field"]
    return empty


def build_window(target: Mapping[str, Iterable], background: BackgroundIndex | Mapping[str, Iterable],
                 *, decision_time_s: float, config: ContextConfig | None = None,
                 eligible_neighbor_mmsi: Iterable[str] | None = None,
                 known_coverage_bbox_m: tuple[float, float, float, float] | None = None,
                 split: str = "unspecified") -> dict:
    """Build a single right-padded chronological window; no future inputs used.

    For ``split='train'``, an explicit allowed-neighbor identity set is required.
    The caller supplies that set after removing every global val/test target ID.
    For held-out windows, eligibility must follow the declared inference protocol.
    Return arrays plus JSON-serializable ``metadata`` (including context exposure).
    """
    cfg = config or (background.config if isinstance(background, BackgroundIndex) else ContextConfig())
    if not np.isfinite(decision_time_s):
        raise ValueError("decision_time_s must be finite")
    if split == "train" and eligible_neighbor_mmsi is None:
        raise ValueError("Training requires explicit eligible_neighbor_mmsi")
    coverage = None
    if known_coverage_bbox_m is not None:
        coverage = np.asarray(known_coverage_bbox_m, float)
        if coverage.shape != (4,) or not np.isfinite(coverage).all():
            raise ValueError("known_coverage_bbox_m must contain four finite bounds")
        if coverage[0] >= coverage[2] or coverage[1] >= coverage[3]:
            raise ValueError("Coverage needs xmin < xmax and ymin < ymax")
    eligible = None if eligible_neighbor_mmsi is None else {str(x) for x in eligible_neighbor_mmsi}
    index = background if isinstance(background, BackgroundIndex) else BackgroundIndex(background, cfg)
    if index.config != cfg:
        raise ValueError("BackgroundIndex/config mismatch; rebuild the index")
    data = _canonicalize(target)
    duplicates_removed = data.pop("duplicates_removed")
    ids = np.unique(data["mmsi"])
    if len(ids) != 1:
        raise ValueError("A target window requires exactly one target identity")
    target_id = str(ids[0])
    kin = _kinematics(data, cfg)
    selected = np.flatnonzero((data["time_s"] <= decision_time_s) &
                             (data["time_s"] >= decision_time_s - cfg.window_seconds))[-cfg.max_steps:]
    T = cfg.max_steps
    out = dict(mask=np.zeros(T, bool), kinematic_valid=np.zeros(T, bool),
               time_interval_valid=np.zeros(T, bool), position_increment_valid=np.zeros(T, bool),
               time_s=np.zeros(T), raw=np.zeros((T, len(RAW_NAMES))),
               gated=np.zeros((T, len(RAW_NAMES))),
               context=np.zeros((T, len(CONTEXT_NAMES))),
               plain_context=np.zeros((T, len(CONTEXT_NAMES))),
               plain_affine_context=np.zeros((T, len(CONTEXT_NAMES))),
               e_kin_mps=np.zeros(T), self_reliability=np.zeros(T), self_gate_weight=np.zeros(T),
               background_mean_mps=np.zeros((T, 2)), background_mean_valid=np.zeros(T, bool),
               background_field_mps=np.zeros((T, 2)), residual_mps=np.zeros((T, 2)),
               divergence_per_s=np.zeros(T), curl_per_s=np.zeros(T),
               context_valid=np.zeros(T, bool), support_fraction=np.zeros(T),
               boundary_context_valid=np.zeros(T, bool),
               valid_neighbor_count=np.zeros(T, np.int32),
               plain_neighbor_count=np.zeros(T, np.int32),
               plain_context_valid=np.zeros(T, bool),
               plain_background_mean_mps=np.zeros((T, 2)),
               plain_background_mean_valid=np.zeros(T, bool),
               plain_affine_valid=np.zeros(T, bool))
    audits, consumed_ids, plain_consumed_ids = [], set(), set()
    R, vscale = cfg.radius_nm * NM_M, cfg.velocity_scale_mps
    for step, ix in enumerate(selected):
        out["time_s"][step] = data["time_s"][ix]
        out["mask"][step] = kin["observed"][ix]
        out["kinematic_valid"][step] = kin["kinematic_valid"][ix]
        out["time_interval_valid"][step] = kin["time_interval_valid"][ix]
        out["position_increment_valid"][step] = kin["position_increment_valid"][ix]
        out["e_kin_mps"][step] = kin["e_kin_mps"][ix]
        out["self_reliability"][step] = kin["reliability"][ix]
        if not kin["observed"][ix]:
            audits.append(dict(step=step, time_s=float(data["time_s"][ix]),
                               reason="invalid_target_observation", neighbors=[]))
            continue
        raw = np.r_[kin["delta"][ix] / R, kin["velocity"][ix] / vscale,
                    kin["dt"][ix] / cfg.window_seconds,
                    np.sin(kin["theta"][ix]), np.cos(kin["theta"][ix])]
        gated = raw.copy()
        target_gate = kin["reliability"][ix] if kin["kinematic_valid"][ix] else 1.
        out["self_gate_weight"][step] = target_gate
        gated[[0, 1, 2, 3, 5, 6]] *= target_gate
        out["raw"][step], out["gated"][step] = raw, gated
        ctx = _context_at(index, data["time_s"][ix], target_id, kin["position"][ix],
                          data["cog_deg"][ix], data["sog_kn"][ix], eligible, cfg)
        px, py = kin["position"][ix]
        boundary_ok = coverage is None or (px-R >= coverage[0] and py-R >= coverage[1] and
                                            px+R <= coverage[2] and py+R <= coverage[3])
        out["boundary_context_valid"][step] = boundary_ok
        unmasked_reason = ctx["reason"]
        if not boundary_ok:
            # Missing ROI coverage is separate from natural traffic sparsity.
            # Keep candidate audit/counts but suppress fields in every arm.
            ctx.update(mean=np.zeros(2), field=np.zeros(2), residual=np.zeros(2),
                       divergence=0., curl=0., valid=False, mean_valid=False,
                       plain_mean=np.zeros(2), plain_mean_valid=False,
                       plain_residual=np.zeros(2), plain_valid=False,
                       reason="coverage_boundary")
            ctx["plain_affine"].update(field=np.zeros(2), divergence=0., curl=0.,
                                       valid=False, reason="coverage_boundary")
        out["background_mean_mps"][step] = ctx["mean"]
        out["background_mean_valid"][step] = ctx["mean_valid"]
        out["background_field_mps"][step] = ctx["field"]
        out["residual_mps"][step] = ctx["residual"]
        out["divergence_per_s"][step], out["curl_per_s"][step] = ctx["divergence"], ctx["curl"]
        out["context_valid"][step] = ctx["valid"]
        out["support_fraction"][step] = ctx["support_fraction"]
        out["valid_neighbor_count"][step] = ctx["count"]
        out["plain_neighbor_count"][step] = ctx["plain_count"]
        out["plain_context_valid"][step] = ctx["plain_valid"]
        out["plain_background_mean_mps"][step] = ctx["plain_mean"]
        out["plain_background_mean_valid"][step] = ctx["plain_mean_valid"]
        plain_support = min(1., ctx["plain_count"] / cfg.min_neighbors)
        # B uses the SAME causal/spatial/heading candidate pool, but no target
        # gate, neighbor kinematic gate, reliability or distance weights. The
        # primary B is an ordinary uniform mean; the extra affine control
        # isolates reliability weighting while retaining the spatial estimator.
        out["plain_context"][step] = np.r_[raw, ctx["plain_residual"] / vscale,
            0., 0., 0., ctx["plain_valid"], plain_support, 0., 0.]
        plain_fit = ctx["plain_affine"]
        out["plain_affine_valid"][step] = plain_fit["valid"]
        plain_residual = kin["velocity"][ix] - plain_fit["field"] if plain_fit["valid"] else np.zeros(2)
        out["plain_affine_context"][step] = np.r_[raw, plain_residual / vscale,
            0., 0., 0., plain_fit["valid"], plain_support,
            plain_fit["divergence"] * R / vscale, plain_fit["curl"] * R / vscale]
        # The field has its own neighbor-support validity. Target residual
        # attenuation uses the measured gate when available; an unavailable
        # consistency estimate preserves observed motion with an explicit mask.
        w = kin["reliability"][ix]
        out["context"][step] = np.r_[gated, ctx["residual"] / vscale * target_gate,
            kin["e_kin_mps"][ix] / vscale, w, kin["kinematic_valid"][ix],
            ctx["valid"], ctx["support_fraction"],
            ctx["divergence"] * R / vscale, ctx["curl"] * R / vscale]
        audits.append(dict(step=step, time_s=float(data["time_s"][ix]),
                           reason=ctx["reason"], condition=ctx["condition"],
                           boundary_context_valid=bool(boundary_ok),
                           context_reason_before_boundary_mask=unmasked_reason,
                           min_geometry_std_m=ctx["min_geometry_std_m"],
                           plain_mean_valid=ctx["plain_valid"],
                           plain_affine_reason=plain_fit["reason"],
                           plain_affine_condition=plain_fit["condition"],
                           neighbors=ctx["neighbors"]))
        consumed_ids.update(x["mmsi"] for x in ctx["neighbors"] if x["accepted"])
        plain_consumed_ids.update(x["mmsi"] for x in ctx["neighbors"] if x["plain_accepted"])
    for k in ("raw", "gated", "context", "plain_context", "plain_affine_context"):
        if not np.isfinite(out[k]).all():
            raise FloatingPointError(f"Nonfinite output in {k}")
    out["metadata"] = dict(
        schema_version="reliable_context_v2", config=asdict(cfg), split=split,
        target_mmsi=target_id, decision_time_s=float(decision_time_s),
        selected_reports=len(selected), valid_steps=int(out["mask"].sum()),
        usable_window=bool(out["mask"].sum() >= cfg.min_steps), padding="right_zero",
        target_duplicates_removed=duplicates_removed,
        background_duplicates_removed=index.duplicates_removed,
        background_fingerprint=index.fingerprint,
        raw_feature_names=list(RAW_NAMES), context_feature_names=list(CONTEXT_NAMES),
        plain_feature_policy="same 16 slots; raw ungated; reliability-specific slots 9:12 zero; mean and unweighted affine controls",
        unknown_target_reliability_policy="self_reliability=0 with kinematic_valid=false; self_gate_weight=1 preserves valid observed target motion; invalid/padded observations remain zero",
        raw_increment_policy="actual same-identity dt and finite position increments are retained independently of the maximum kinematic-consistency interval",
        motion_units="m/s; positions m; divergence/curl s^-1",
        gate_equation=f"exp(-max(0,e_kin_mps-{cfg.gate_deadzone_mps})^2/{cfg.gate_denominator_mps2})",
        eligibility_mode="explicit_allowlist" if eligible is not None else "all_available_except_self",
        coverage_status="unknown" if coverage is None else "known_source_roi",
        known_coverage_bbox_m=None if coverage is None else coverage.tolist(),
        coverage_policy="entire R-circle must fit known source ROI; unknown ROI uses available observations and is explicitly marked unknown",
        eligible_neighbor_count=None if eligible is None else len(eligible),
        context_exposure_mmsi=sorted(consumed_ids),
        plain_context_exposure_mmsi=sorted(plain_consumed_ids), steps=audits)
    return out


def masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked summary helper; empty support returns zeros, never NaN."""
    values, mask = np.asarray(values), np.asarray(mask, bool)
    if values.shape[0] != len(mask):
        raise ValueError("Mask length does not match values")
    return values[mask].mean(axis=0) if mask.any() else np.zeros(values.shape[1:])
