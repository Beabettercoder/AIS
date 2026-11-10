"""Continuous meteorological wind field modeling using Barnes Objective Analysis.

This module upgrades isolated point-based weather observations into a continuous
spatial field across the maritime domain. It provides:
1. Orthogonal wind vector (u, v) decomposition and synthesis to prevent the classic
   360°/0° angular scalar-averaging inversion failure.
2. Barnes objective analysis using Gaussian distance-decay kernel weighting.
3. Physically sound spatial confidence attenuation decaying exponentially with
   distance from the nearest authoritative observation station.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def haversine_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute great-circle distance in kilometers between two WGS84 coordinates."""
    r_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r_km * c


def decompose_wind(speed_mps: float, direction_deg: float) -> tuple[float, float]:
    """Decompose wind speed and meteorological direction into zonal (u) and meridional (v) components.

    Meteorological wind direction defines the direction FROM which the wind blows.
    - 0° (North wind): blows towards South -> u = 0, v = -speed
    - 90° (East wind): blows towards West -> u = -speed, v = 0
    - 180° (South wind): blows towards North -> u = 0, v = speed
    - 270° (West wind): blows towards East -> u = speed, v = 0
    """
    rad = math.radians(direction_deg % 360.0)
    u = -speed_mps * math.sin(rad)
    v = -speed_mps * math.cos(rad)
    return u, v


def synthesize_wind(u: float, v: float) -> tuple[float, float]:
    """Synthesize zonal (u) and meridional (v) vector components into wind speed and meteorological direction.

    Wind comes from opposite direction of velocity vector (-u, -v).
    """
    speed = math.hypot(u, v)
    if speed < 1e-6:
        return 0.0, 0.0

    # Meteorological direction: atan2(-u, -v) returns angle from North (clockwise)
    direction = math.degrees(math.atan2(-u, -v)) % 360.0
    if abs(direction - 360.0) < 1e-4:
        direction = 0.0
    return round(speed, 2), round(direction, 1)


def interpolate_point_weather(
    station_obs: Sequence[dict[str, Any]],
    target_lon: float,
    target_lat: float,
    *,
    length_scale_km: float = 20.0,
    decay_scale_km: float = 30.0,
    exact_match_threshold_km: float = 0.05,
) -> dict[str, Any]:
    """Perform Barnes objective analysis to interpolate weather at target coordinate.

    Args:
        station_obs: List of observation records with 'lon', 'lat' and weather attributes.
        target_lon: Target longitude (WGS84).
        target_lat: Target latitude (WGS84).
        length_scale_km: Characteristic filtering scale L for Gaussian kernel (w_i = exp(-d_i^2 / (2 L^2))).
        decay_scale_km: Distance attenuation scale for spatial confidence estimation.
        exact_match_threshold_km: Distance threshold (km) within which observation is taken directly.
    """
    valid_stations: list[dict[str, Any]] = []
    distances: list[float] = []

    for s in station_obs:
        lon = s.get("lon")
        lat = s.get("lat")
        if lon is None or lat is None:
            continue
        try:
            flon, flat = float(lon), float(lat)
        except (ValueError, TypeError):
            continue
        d_km = haversine_distance_km(target_lon, target_lat, flon, flat)
        valid_stations.append(s)
        distances.append(d_km)

    if not valid_stations:
        return {
            "target_lon": target_lon,
            "target_lat": target_lat,
            "spatial_confidence": 0.0,
            "error": "No valid observation stations available for interpolation",
        }

    # If within exact match threshold, preserve authoritative station observation directly
    min_dist = min(distances)
    nearest_idx = distances.index(min_dist)
    nearest_station = valid_stations[nearest_idx]
    nearest_id = (nearest_station.get("location_id") or nearest_station.get("station_id")
                  or nearest_station.get("source_id") or "unknown")

    if min_dist <= exact_match_threshold_km:
        wind_speed = nearest_station.get("wind_speed_mps")
        wind_dir = nearest_station.get("wind_direction_deg")
        u_val, v_val = (decompose_wind(float(wind_speed), float(wind_dir))
                        if wind_speed is not None and wind_dir is not None else (None, None))
        return {
            "target_lon": target_lon,
            "target_lat": target_lat,
            "wind_speed_mps": float(wind_speed) if wind_speed is not None else None,
            "wind_direction_deg": float(wind_dir) if wind_dir is not None else None,
            "u_component_mps": round(u_val, 2) if u_val is not None else None,
            "v_component_mps": round(v_val, 2) if v_val is not None else None,
            "wind_gust_mps": float(nearest_station["wind_gust_mps"]) if nearest_station.get("wind_gust_mps") is not None else None,
            "air_pressure_hpa": float(nearest_station["air_pressure_hpa"]) if nearest_station.get("air_pressure_hpa") is not None else None,
            "air_temperature_c": float(nearest_station["air_temperature_c"]) if nearest_station.get("air_temperature_c") is not None else None,
            "relative_humidity_pct": float(nearest_station["relative_humidity_pct"]) if nearest_station.get("relative_humidity_pct") is not None else None,
            "spatial_confidence": 1.0,
            "nearest_station_km": round(min_dist, 3),
            "nearest_station_id": str(nearest_id),
            "contributing_station_count": 1,
            "interpolation_mode": "exact_station_match",
            "provenance": {
                "method": "authoritative_station_ground_truth",
                "station_id": str(nearest_id),
            },
        }

    # Barnes Gaussian kernel weights: w_i = exp(-d_i^2 / (2 * L^2))
    two_l_sq = 2.0 * (length_scale_km ** 2)
    weights = [math.exp(-(d ** 2) / two_l_sq) for d in distances]
    total_weight = sum(weights)

    if total_weight < 1e-12:
        # Fallback to inverse distance weighting if far beyond scale
        inv_weights = [1.0 / max(d, 1e-3) for d in distances]
        sum_inv = sum(inv_weights)
        norm_weights = [w / sum_inv for w in inv_weights]
    else:
        norm_weights = [w / total_weight for w in weights]

    # Vector wind interpolation
    weighted_u = 0.0
    weighted_v = 0.0
    wind_weight_sum = 0.0

    for s, w in zip(valid_stations, norm_weights):
        ws = s.get("wind_speed_mps")
        wd = s.get("wind_direction_deg")
        if ws is not None and wd is not None:
            try:
                u, v = decompose_wind(float(ws), float(wd))
                weighted_u += w * u
                weighted_v += w * v
                wind_weight_sum += w
            except (ValueError, TypeError):
                continue

    if wind_weight_sum > 0:
        interp_u = weighted_u / wind_weight_sum
        interp_v = weighted_v / wind_weight_sum
        synth_speed, synth_dir = synthesize_wind(interp_u, interp_v)
    else:
        synth_speed, synth_dir, interp_u, interp_v = None, None, None, None

    # Scalar attributes interpolation helper
    def _interpolate_scalar(field_name: str) -> float | None:
        val_sum = 0.0
        w_sum = 0.0
        for st, wt in zip(valid_stations, norm_weights):
            val = st.get(field_name)
            if val is not None:
                try:
                    val_sum += wt * float(val)
                    w_sum += wt
                except (ValueError, TypeError):
                    continue
        if w_sum > 0:
            return round(val_sum / w_sum, 2)
        return None

    gust = _interpolate_scalar("wind_gust_mps")
    pressure = _interpolate_scalar("air_pressure_hpa")
    temp = _interpolate_scalar("air_temperature_c")
    humidity = _interpolate_scalar("relative_humidity_pct")

    # Spatial confidence: exponential decay with distance to nearest observation station
    spatial_confidence = round(math.exp(-min_dist / decay_scale_km), 4)

    # Top contributing stations
    contributions = []
    ranked_stations = sorted(zip(valid_stations, distances, norm_weights), key=lambda x: x[2], reverse=True)
    for st, d_st, w_st in ranked_stations[:4]:
        st_id = st.get("location_id") or st.get("station_id") or st.get("source_id") or "unknown"
        contributions.append({
            "station_id": str(st_id),
            "distance_km": round(d_st, 2),
            "weight": round(w_st, 4),
        })

    return {
        "target_lon": target_lon,
        "target_lat": target_lat,
        "wind_speed_mps": synth_speed,
        "wind_direction_deg": synth_dir,
        "u_component_mps": round(interp_u, 2) if interp_u is not None else None,
        "v_component_mps": round(interp_v, 2) if interp_v is not None else None,
        "wind_gust_mps": gust,
        "air_pressure_hpa": pressure,
        "air_temperature_c": temp,
        "relative_humidity_pct": humidity,
        "spatial_confidence": spatial_confidence,
        "nearest_station_km": round(min_dist, 2),
        "nearest_station_id": str(nearest_id),
        "contributing_station_count": len(valid_stations),
        "interpolation_mode": "barnes_objective_analysis",
        "contributions": contributions,
        "provenance": {
            "method": "barnes_objective_analysis",
            "length_scale_km": length_scale_km,
            "decay_scale_km": decay_scale_km,
            "vector_orthogonal_decomposition": True,
        },
    }


def generate_grid_wind_field(
    station_obs: Sequence[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    *,
    cols: int = 8,
    rows: int = 8,
    length_scale_km: float = 20.0,
    decay_scale_km: float = 30.0,
) -> dict[str, Any]:
    """Generate a continuous regular grid wind and weather field over a bounding box.

    Args:
        station_obs: Observation records for the time window.
        bbox: Bounding box as (west, south, east, north).
        cols: Number of grid columns.
        rows: Number of grid rows.
        length_scale_km: Barnes Gaussian characteristic radius.
        decay_scale_km: Spatial confidence attenuation distance.
    """
    west, south, east, north = bbox
    cols = max(2, min(32, cols))
    rows = max(2, min(32, rows))

    dx = (east - west) / float(cols)
    dy = (north - south) / float(rows)

    grid_cells: list[dict[str, Any]] = []
    total_speed = 0.0
    valid_speeds = 0
    max_speed = 0.0
    min_confidence = 1.0

    for r in range(rows):
        for c in range(cols):
            # Center of the cell
            c_lon = round(west + (c + 0.5) * dx, 5)
            c_lat = round(south + (r + 0.5) * dy, 5)
            cell_box = (
                round(west + c * dx, 5),
                round(south + r * dy, 5),
                round(west + (c + 1) * dx, 5),
                round(south + (r + 1) * dy, 5),
            )
            interp = interpolate_point_weather(
                station_obs,
                c_lon,
                c_lat,
                length_scale_km=length_scale_km,
                decay_scale_km=decay_scale_km,
            )
            interp["col"] = c
            interp["row"] = r
            interp["cell_bbox"] = cell_box
            grid_cells.append(interp)

            sp = interp.get("wind_speed_mps")
            if sp is not None:
                total_speed += sp
                valid_speeds += 1
                if sp > max_speed:
                    max_speed = sp
            conf = interp.get("spatial_confidence", 0.0)
            if conf < min_confidence:
                min_confidence = conf

    avg_speed = round(total_speed / valid_speeds, 2) if valid_speeds > 0 else 0.0

    return {
        "bbox": bbox,
        "grid_dimensions": {"cols": cols, "rows": rows, "total_cells": len(grid_cells)},
        "summary": {
            "avg_wind_speed_mps": avg_speed,
            "max_wind_speed_mps": round(max_speed, 2),
            "min_spatial_confidence": round(min_confidence, 4),
            "station_count": len(station_obs),
        },
        "cells": grid_cells,
        "method": "barnes_objective_analysis",
    }


def estimate_weather_tendency(history_states: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Estimate time tendencies (derivative rates per hour) from a sequence of temporal states.

    Args:
        history_states: Chronologically ordered state snapshots (oldest to newest).
            Each item must include 'timestamp_seconds' or 'valid_time_utc', plus weather fields.
    """
    if len(history_states) < 2:
        return {
            "u_rate_per_hour": 0.0,
            "v_rate_per_hour": 0.0,
            "speed_rate_per_hour": 0.0,
            "pressure_rate_per_hour": 0.0,
            "pressure_trend": "steady",
            "wind_trend": "stable",
            "sample_count": len(history_states),
        }

    # Extract time and fields
    t_hours: list[float] = []
    t_start = history_states[0].get("timestamp_seconds", 0.0)
    for st in history_states:
        sec = st.get("timestamp_seconds")
        if sec is None and st.get("valid_time_utc"):
            from datetime import datetime
            sec = datetime.fromisoformat(str(st["valid_time_utc"]).replace("Z", "+00:00")).timestamp()
        t_hours.append((float(sec or 0.0) - float(t_start)) / 3600.0)

    delta_t = t_hours[-1] - t_hours[0]
    if delta_t <= 1e-4:
        delta_t = 1.0

    def _fit_rate(field_name: str) -> float:
        values = [st.get(field_name) for st in history_states if st.get(field_name) is not None]
        if len(values) < 2:
            return 0.0
        # Simple weighted difference between latest and earliest
        v_diff = float(values[-1]) - float(values[0])
        return round(v_diff / delta_t, 3)

    u_rate = max(-15.0, min(15.0, _fit_rate("u_component_mps")))
    v_rate = max(-15.0, min(15.0, _fit_rate("v_component_mps")))
    speed_rate = max(-12.0, min(12.0, _fit_rate("wind_speed_mps")))
    pressure_rate = max(-20.0, min(20.0, _fit_rate("air_pressure_hpa")))

    # Trend classifications
    if pressure_rate <= -2.0:
        pressure_trend = "rapidly_falling"
    elif pressure_rate <= -0.5:
        pressure_trend = "falling"
    elif pressure_rate >= 0.5:
        pressure_trend = "rising"
    else:
        pressure_trend = "steady"

    if speed_rate >= 1.5:
        wind_trend = "rapidly_strengthening"
    elif speed_rate >= 0.5:
        wind_trend = "strengthening"
    elif speed_rate <= -0.5:
        wind_trend = "weakening"
    else:
        wind_trend = "stable"

    return {
        "u_rate_per_hour": u_rate,
        "v_rate_per_hour": v_rate,
        "speed_rate_per_hour": speed_rate,
        "pressure_rate_per_hour": pressure_rate,
        "pressure_trend": pressure_trend,
        "wind_trend": wind_trend,
        "sample_count": len(history_states),
        "history_window_hours": round(delta_t, 2),
    }


def predict_weather_nowcast(
    history_states: Sequence[dict[str, Any]],
    lead_hours: float,
    *,
    relaxation_hours: float = 2.5,
    lead_decay_hours: float = 1.5,
) -> dict[str, Any]:
    """Perform kinetic advection nowcast (0~3 hours) with physical damping.

    Args:
        history_states: Chronologically ordered history states up to the prediction baseline t0.
        lead_hours: Forecast lead time forward in hours (tau in [0.0, 3.0]).
        relaxation_hours: Relaxation time scale for exponential damping kernel.
        lead_decay_hours: Characteristic time scale for temporal confidence decay.
    """
    if not history_states:
        raise ValueError("历史气象序列不能为空")

    base = history_states[-1]
    tendency = estimate_weather_tendency(history_states)

    lead_hours = max(0.0, float(lead_hours))
    # Damping factor: exp(-tau / tau_relax)
    gamma = math.exp(-lead_hours / relaxation_hours)

    u0 = base.get("u_component_mps")
    v0 = base.get("v_component_mps")
    s0 = base.get("wind_speed_mps") or 0.0

    if u0 is None or v0 is None:
        # If missing u/v, decompose from speed and direction
        dir0 = base.get("wind_direction_deg") or 0.0
        u0, v0 = decompose_wind(float(s0), float(dir0))

    pred_u = u0 + gamma * tendency["u_rate_per_hour"] * lead_hours
    pred_v = v0 + gamma * tendency["v_rate_per_hour"] * lead_hours
    pred_speed, pred_dir = synthesize_wind(pred_u, pred_v)

    # Pressure prediction
    p0 = base.get("air_pressure_hpa")
    pred_p = round(p0 + gamma * tendency["pressure_rate_per_hour"] * lead_hours, 2) if p0 is not None else None

    # Gust prediction: preserve dynamic gust ratio
    g0 = base.get("wind_gust_mps")
    gust_ratio = (g0 / s0) if (g0 is not None and s0 > 1e-3) else 1.35
    pred_gust = round(pred_speed * gust_ratio, 2)

    # Temperature prediction
    temp0 = base.get("air_temperature_c")
    pred_temp = temp0

    # Spatiotemporal dual confidence attenuation
    c_spatial = base.get("spatial_confidence", 1.0)
    c_temporal = math.exp(-lead_hours / lead_decay_hours)
    c_combined = round(float(c_spatial) * c_temporal, 4)

    # Maritime gale warning level
    if pred_speed >= 32.7 or pred_gust >= 37.0:
        warning_level = "typhoon_warning_red"
        warning_desc = "红色台风风暴警报（风力≥12级，极大阵风险情）"
    elif pred_speed >= 24.5 or pred_gust >= 28.5:
        warning_level = "storm_warning_orange"
        warning_desc = "橙色狂风暴雨警报（风力10-11级，船舶应进港避风）"
    elif pred_speed >= 17.2 or pred_gust >= 20.8:
        warning_level = "gale_warning_yellow"
        warning_desc = "黄色大风预警（风力8-9级烈风，海面波涛汹涌）"
    elif pred_speed >= 13.9:
        warning_level = "gale_warning_blue"
        warning_desc = "蓝色强风警报（风力7级疾风，小型船只应停止作业）"
    else:
        warning_level = "normal_navigable"
        warning_desc = "适航海况（风力≤6级）"

    return {
        "lead_hours": round(lead_hours, 2),
        "target_lon": base.get("target_lon"),
        "target_lat": base.get("target_lat"),
        "predicted_weather": {
            "wind_speed_mps": pred_speed,
            "wind_direction_deg": pred_dir,
            "u_component_mps": round(pred_u, 2),
            "v_component_mps": round(pred_v, 2),
            "wind_gust_mps": pred_gust,
            "air_pressure_hpa": pred_p,
            "air_temperature_c": pred_temp,
        },
        "baseline_weather": {
            "wind_speed_mps": base.get("wind_speed_mps"),
            "wind_direction_deg": base.get("wind_direction_deg"),
            "air_pressure_hpa": base.get("air_pressure_hpa"),
            "air_temperature_c": base.get("air_temperature_c"),
        },
        "tendency": tendency,
        "confidence": {
            "spatial_confidence": round(float(c_spatial), 4),
            "temporal_confidence": round(c_temporal, 4),
            "combined_confidence": c_combined,
        },
        "warning_assessment": {
            "warning_level": warning_level,
            "warning_description": warning_desc,
        },
        "method": "damped_kinetic_nowcast",
        "parameters": {
            "relaxation_hours": relaxation_hours,
            "lead_decay_hours": lead_decay_hours,
            "damping_factor": round(gamma, 4),
        },
    }

