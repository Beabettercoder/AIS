"""Explainable candidate generation and validation for schema alignment.

The scores in this module are evidence scores, not calibrated probabilities.
They are intentionally deterministic so a model suggestion can never bypass
the same value and conflict checks used for a manual mapping.
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any, Iterable

from .datetime_parser import is_datetime_like


AIS_FIELD_DESCRIPTIONS = {
    "timestamp_utc": "UTC observation timestamp",
    "mmsi": "nine digit Maritime Mobile Service Identity",
    "lon": "WGS84 longitude between -180 and 180",
    "lat": "WGS84 latitude between -90 and 90",
    "sog_knots": "speed over ground, normalized to knots",
    "cog_degrees": "course over ground in degrees from 0 to 360",
    "vessel_name": "human readable vessel name",
    "vessel_type": "AIS vessel type code or vessel type label",
    "received_at_utc": "UTC reception timestamp",
    "heading_degrees": "true heading in degrees from 0 to 359",
    "rot_degrees_per_minute": "rate of turn in degrees per minute",
    "nav_status": "AIS navigational status code or label",
    "position_accuracy": "AIS position accuracy flag",
    "raim_flag": "receiver autonomous integrity monitoring flag",
    "maneuver_indicator": "AIS special maneuver indicator",
    "imo": "seven digit IMO vessel number",
    "callsign": "vessel radio call sign",
    "cargo_type": "AIS cargo type code or label",
    "ais_class": "AIS transceiver or mobile station class",
    "length_m": "vessel length normalized to metres",
    "width_m": "vessel width or beam normalized to metres",
    "dimension_to_bow_m": "distance from AIS antenna to bow in metres",
    "dimension_to_stern_m": "distance from AIS antenna to stern in metres",
    "dimension_to_port_m": "distance from AIS antenna to port side in metres",
    "dimension_to_starboard_m": "distance from AIS antenna to starboard side in metres",
    "position_fixing_device": "electronic position fixing device",
    "draught_m": "vessel draught normalized to metres",
    "destination": "reported voyage destination",
    "eta_utc": "estimated time of arrival normalized to UTC",
    "dte": "AIS data terminal equipment status",
    "message_type": "AIS message type number",
    "repeat_indicator": "AIS repeat indicator",
    "ais_version": "AIS protocol version indicator",
    "receiver_id": "receiving station identifier",
    "radio_channel": "AIS radio channel",
    "source_type": "terrestrial, satellite, or provider source type",
}

GIS_FIELD_DESCRIPTIONS = {
    "feature_id": "stable source feature identifier",
    "object_name": "human readable geographic object name",
    "object_type": "canonical geographic object category",
    "source_date": "source publication or dataset date",
    "valid_from": "date or timestamp when the boundary becomes valid",
    "valid_to": "date or timestamp when the boundary stops being valid",
    "updated_at": "source feature update timestamp",
    "source_object_id": "identifier assigned by the upstream GIS provider",
    "source_url": "upstream dataset or feature URL",
    "authority": "organization responsible for the feature",
    "status": "operational or lifecycle status",
    "description": "free-text feature description or remarks",
    "country_code": "country name or ISO country code",
    "port_code": "UN/LOCODE or provider port code",
    "depth_m": "water or feature depth normalized to metres",
    "min_depth_m": "minimum depth normalized to metres",
    "max_depth_m": "maximum depth normalized to metres",
    "width_m": "feature or channel width normalized to metres",
    "length_m": "feature length normalized to metres",
    "area_m2": "feature area normalized to square metres",
    "elevation_m": "elevation normalized to metres",
    "clearance_height_m": "vertical clearance normalized to metres",
}

WEATHER_FIELD_DESCRIPTIONS = {
    "valid_time_utc": "observation time or forecast valid time normalized to UTC",
    "issue_time_utc": "forecast issue or model run time normalized to UTC",
    "location_id": "station, buoy, site, or grid identifier",
    "location_name": "human readable station or location name",
    "lon": "WGS84 longitude between -180 and 180",
    "lat": "WGS84 latitude between -90 and 90",
    "wind_speed_mps": "wind speed normalized to metres per second",
    "wind_direction_deg": "wind direction in degrees from 0 to 360",
    "wind_gust_mps": "wind gust normalized to metres per second",
    "air_pressure_hpa": "air pressure normalized to hectopascals",
    "air_temperature_c": "air temperature normalized to degrees Celsius",
    "precipitation_mm": "precipitation normalized to millimetres",
    "visibility_m": "visibility normalized to metres",
    "relative_humidity_pct": "relative humidity from 0 to 100 percent",
    "cloud_cover_pct": "cloud cover from 0 to 100 percent",
    "wave_height_m": "wave height normalized to metres",
    "wave_period_s": "wave period normalized to seconds",
    "wave_direction_deg": "wave direction in degrees from 0 to 360",
    "water_temperature_c": "water temperature normalized to degrees Celsius",
    "water_level_m": "water level normalized to metres",
}

CRITICAL_AIS_FIELDS = {"timestamp_utc", "mmsi", "lon", "lat"}
AUTO_MAP_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.65


def normalize_name(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum())


def _name_score(column: str, aliases: Iterable[str]) -> float:
    source = normalize_name(column)
    if not source:
        return 0.0
    best = 0.0
    for alias in aliases:
        target = normalize_name(alias)
        if not target:
            continue
        if source == target:
            return 1.0
        score = SequenceMatcher(None, source, target).ratio()
        if len(target) >= 3 and (source.startswith(target) or source.endswith(target)):
            score = max(score, 0.9)
        if len(source) >= 4 and len(target) >= 4 and (source in target or target in source):
            score = max(score, min(len(source), len(target)) / max(len(source), len(target)))
        best = max(best, score)
    return round(best, 6)


def _present(values: Iterable[Any]) -> list[str]:
    return [str(value).strip() for value in values if str(value or "").strip()]


def _ratio(values: list[str], predicate: Any) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if predicate(value)) / len(values)


def _number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _time_like(value: str) -> bool:
    return is_datetime_like(value)


def value_compatibility(target_field: str, values: Iterable[Any]) -> float:
    """Return deterministic compatibility in [0, 1] for representative values."""
    samples = _present(values)
    if not samples:
        return 0.0
    if target_field in {"timestamp_utc", "received_at_utc", "eta_utc",
                        "source_date", "valid_from", "valid_to", "updated_at",
                        "valid_time_utc", "issue_time_utc"}:
        return _ratio(samples, _time_like)
    if target_field == "mmsi":
        return _ratio(samples, lambda value: bool(re.fullmatch(r"\d{9}", value)))
    if target_field == "lon":
        return _ratio(samples, lambda value: (number := _number(value)) is not None and -180 <= number <= 180)
    if target_field == "lat":
        return _ratio(samples, lambda value: (number := _number(value)) is not None and -90 <= number <= 90)
    if target_field == "sog_knots":
        return _ratio(samples, lambda value: (number := _number(value)) is not None and 0 <= number <= 500)
    if target_field in {"cog_degrees", "heading_degrees"}:
        return _ratio(samples, lambda value: (number := _number(value)) is not None and 0 <= number <= 511)
    if target_field == "rot_degrees_per_minute":
        return _ratio(samples, lambda value: (number := _number(value)) is not None and -720 <= number <= 720)
    if target_field in {"position_accuracy", "raim_flag", "dte"}:
        return _ratio(samples, lambda value: normalize_name(value) in {
            "0", "1", "true", "false", "yes", "no", "high", "low",
            "available", "notavailable", "ready", "notready", "是", "否"
        })
    if target_field == "maneuver_indicator":
        return _ratio(samples, lambda value: (number := _number(value)) is not None and 0 <= number <= 2)
    if target_field == "imo":
        return _ratio(samples, lambda value: bool(re.fullmatch(r"(?:IMO)?\d{7}", value.upper())))
    if target_field == "vessel_name":
        return _ratio(samples, lambda value: not _time_like(value)
                      and bool(re.search(r"[^\d.\-+ ]", value)))
    if target_field == "vessel_type":
        return _ratio(samples, lambda value: not _time_like(value) and (
                      bool(re.search(r"[A-Za-z\u4e00-\u9fff]", value)) or
                      ((number := _number(value)) is not None and 0 <= number <= 99))
                      )
    if target_field in {"length_m", "width_m", "dimension_to_bow_m", "dimension_to_stern_m",
                        "dimension_to_port_m", "dimension_to_starboard_m", "draught_m",
                        "depth_m", "min_depth_m", "max_depth_m", "elevation_m",
                        "clearance_height_m"}:
        return _ratio(samples, lambda value: (number := _number(value)) is not None and 0 <= number <= 1000)
    if target_field == "area_m2":
        return _ratio(samples, lambda value: (number := _number(value)) is not None and number >= 0)
    weather_ranges = {
        "wind_speed_mps": (0, 250), "wind_direction_deg": (0, 360),
        "wind_gust_mps": (0, 250), "air_pressure_hpa": (0, 120000),
        "air_temperature_c": (-150, 200), "precipitation_mm": (0, 10000),
        "visibility_m": (0, 500000), "relative_humidity_pct": (0, 100),
        "cloud_cover_pct": (0, 100), "wave_height_m": (0, 100),
        "wave_period_s": (0, 120), "wave_direction_deg": (0, 360),
        "water_temperature_c": (-20, 100), "water_level_m": (-100, 100),
    }
    if target_field in weather_ranges:
        lower, upper = weather_ranges[target_field]
        return _ratio(samples, lambda value: (number := _number(value)) is not None
                      and lower <= number <= upper)
    if target_field == "message_type":
        return _ratio(samples, lambda value: (number := _number(value)) is not None
                      and number.is_integer() and 1 <= number <= 63)
    if target_field in {"repeat_indicator", "ais_version"}:
        return _ratio(samples, lambda value: (number := _number(value)) is not None
                      and number.is_integer() and 0 <= number <= 3)
    if target_field == "nav_status":
        return _ratio(samples, lambda value: bool(re.search(r"[A-Za-z\u4e00-\u9fff]", value)) or
                      ((number := _number(value)) is not None and number.is_integer() and 0 <= number <= 15))
    if target_field in {"callsign", "cargo_type", "ais_class", "position_fixing_device",
                        "destination", "receiver_id", "radio_channel", "source_type",
                        "feature_id", "object_name", "object_type", "source_object_id",
                        "source_url", "authority", "status", "description", "country_code",
                        "port_code", "location_id", "location_name"}:
        return 0.6 * _ratio(samples, lambda value: bool(value.strip()))
    return 0.0


def _unit_score(column: str, target_field: str) -> float:
    name = normalize_name(column)
    if target_field == "sog_knots":
        known = ("sog", "speed", "knot", "kn", "kts", "kmh", "kph", "mps", "航速")
        return 1.0 if any(token in name for token in known) else 0.5
    if target_field in {"length_m", "width_m", "dimension_to_bow_m", "dimension_to_stern_m",
                        "dimension_to_port_m", "dimension_to_starboard_m", "draught_m",
                        "depth_m", "min_depth_m", "max_depth_m", "elevation_m",
                        "clearance_height_m"}:
        known = ("length", "width", "beam", "draught", "draft", "depth", "elevation",
                 "height", "clearance", "dim", "size", "ref", "meter", "metre", "ft")
        return 1.0 if any(token in name for token in known) else 0.5
    if target_field == "area_m2":
        return 1.0 if any(token in name for token in ("area", "square", "m2", "km2", "hectare", "ha")) else 0.5
    weather_tokens = {
        "wind_speed_mps": ("wind", "speed", "mps", "ms", "knot", "kts", "kmh", "mph", "风速"),
        "wind_gust_mps": ("gust", "mps", "knot", "kts", "kmh", "阵风"),
        "air_pressure_hpa": ("pressure", "hpa", "mb", "kpa", "pa", "气压"),
        "air_temperature_c": ("temperature", "temp", "celsius", "fahrenheit", "气温"),
        "precipitation_mm": ("precip", "rain", "mm", "inch", "降水"),
        "visibility_m": ("visibility", "meter", "km", "nm", "能见度"),
        "wave_height_m": ("wave", "height", "meter", "ft", "浪高"),
        "wave_period_s": ("wave", "period", "second", "周期"),
        "water_temperature_c": ("water", "sea", "temperature", "temp", "sst", "水温"),
        "water_level_m": ("water", "sea", "tide", "level", "meter", "ft", "水位"),
    }
    if target_field in weather_tokens:
        return 1.0 if any(token in name for token in weather_tokens[target_field]) else 0.5
    if target_field in {"cog_degrees", "heading_degrees", "rot_degrees_per_minute"}:
        known = ("cog", "heading", "rot", "course", "deg", "rad")
        return 1.0 if any(token in name for token in known) else 0.5
    return 1.0


def build_ais_candidates(columns: list[str], sample_rows: list[dict[str, Any]],
                         aliases: dict[str, Iterable[str]],
                         mapped_fields: dict[str, str] | None = None,
                         field_groups: dict[str, list[str]] | None = None,
                         top_k: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Route unknown columns to one semantic group, then build Top-K candidates."""
    return _build_candidates(columns, sample_rows, aliases, mapped_fields, field_groups,
                             top_k, AIS_FIELD_DESCRIPTIONS)


def build_gis_candidates(columns: list[str], sample_rows: list[dict[str, Any]],
                         aliases: dict[str, Iterable[str]],
                         mapped_fields: dict[str, str] | None = None,
                         field_groups: dict[str, list[str]] | None = None,
                         top_k: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Build explainable Top-K candidates for unknown GeoJSON property names."""
    return _build_candidates(columns, sample_rows, aliases, mapped_fields, field_groups,
                             top_k, GIS_FIELD_DESCRIPTIONS)


def build_weather_candidates(columns: list[str], sample_rows: list[dict[str, Any]],
                             aliases: dict[str, Iterable[str]],
                             mapped_fields: dict[str, str] | None = None,
                             field_groups: dict[str, list[str]] | None = None,
                             top_k: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Build explainable Top-K candidates for unknown weather CSV columns."""
    return _build_candidates(columns, sample_rows, aliases, mapped_fields, field_groups,
                             top_k, WEATHER_FIELD_DESCRIPTIONS)


def _build_candidates(columns: list[str], sample_rows: list[dict[str, Any]],
                      aliases: dict[str, Iterable[str]],
                      mapped_fields: dict[str, str] | None,
                      field_groups: dict[str, list[str]] | None,
                      top_k: int, descriptions: dict[str, str]
                      ) -> dict[str, list[dict[str, Any]]]:
    mapped_fields = mapped_fields or {}
    field_groups = field_groups or {"all": list(aliases)}
    target_groups = {field: group for group, fields in field_groups.items() for field in fields}
    used_columns = {column for column in mapped_fields.values() if column}
    available_targets = [field for field in aliases if not mapped_fields.get(field)]
    result: dict[str, list[dict[str, Any]]] = {}
    for column in columns:
        if column in used_columns:
            continue
        values = [row.get(column) for row in sample_rows]
        candidates = []
        for target in available_targets:
            name_score = _name_score(column, aliases[target])
            value_score = value_compatibility(target, values)
            unit_score = _unit_score(column, target)
            retrieval_score = 0.55 * name_score + 0.35 * value_score + 0.10 * unit_score
            normalized_column = normalize_name(column)
            has_eta_hint = any(token in normalized_column for token in ("eta", "arrival", "arrive", "到达"))
            if target == "timestamp_utc" and value_score >= 0.8 and not has_eta_hint:
                retrieval_score += 0.15
            elif target == "eta_utc" and value_score >= 0.8 and not has_eta_hint:
                retrieval_score -= 0.15
            candidates.append({
                "target_field": target,
                "field_group": target_groups.get(target, "unclassified"),
                "description": descriptions.get(target, target),
                "name_score": round(name_score, 6),
                "value_score": round(value_score, 6),
                "unit_score": round(unit_score, 6),
                "retrieval_score": round(retrieval_score, 6),
            })
        group_scores: dict[str, float] = {}
        for candidate in candidates:
            group = candidate["field_group"]
            group_scores[group] = max(group_scores.get(group, 0.0), candidate["retrieval_score"])
        ranked_groups = sorted(group_scores.items(), key=lambda item: (-item[1], item[0]))
        if not ranked_groups or ranked_groups[0][1] < 0.45:
            continue
        selected_group, group_score = ranked_groups[0]
        grouped = [candidate for candidate in candidates if candidate["field_group"] == selected_group]
        for candidate in grouped:
            candidate["group_score"] = round(group_score, 6)
        result[column] = sorted(grouped, key=lambda item: (-item["retrieval_score"],
                                                            item["target_field"]))[:top_k]
    return result


def validate_ais_suggestion(source_column: str, target_field: str, values: Iterable[Any],
                            candidate: dict[str, Any], model_score: float,
                            selected_mapping: dict[str, str]) -> dict[str, Any]:
    """Validate one model suggestion and produce an uncalibrated evidence score."""
    checks: list[dict[str, Any]] = []
    samples = _present(values)
    value_score = value_compatibility(target_field, samples)
    conflicts = sorted(field for field, column in selected_mapping.items()
                       if column == source_column and field != target_field)
    if conflicts:
        checks.append({"check": "source_column_conflict", "status": "block",
                       "detail": f"同一来源列还被映射到: {', '.join(conflicts)}"})
    if not samples:
        checks.append({"check": "representative_values", "status": "warning",
                       "detail": "没有非空样例值"})
    elif target_field in CRITICAL_AIS_FIELDS and value_score < 0.8:
        checks.append({"check": "value_compatibility", "status": "block",
                       "detail": f"关键字段样例兼容比例仅为 {value_score:.1%}"})
    elif value_score < 0.5:
        checks.append({"check": "value_compatibility", "status": "warning",
                       "detail": f"样例兼容比例为 {value_score:.1%}"})
    else:
        checks.append({"check": "value_compatibility", "status": "pass",
                       "detail": f"样例兼容比例为 {value_score:.1%}"})
    status = "block" if any(item["status"] == "block" for item in checks) else (
        "warning" if any(item["status"] == "warning" for item in checks) else "pass")
    bounded_model_score = min(1.0, max(0.0, float(model_score)))
    raw_confidence = (0.35 * float(candidate.get("name_score", 0.0)) +
                      0.40 * value_score +
                      0.10 * float(candidate.get("unit_score", 1.0)) +
                      0.15 * bounded_model_score)
    if target_field == "mmsi" and value_score == 1.0:
        raw_confidence = min(1.0, raw_confidence + 0.10)
    elif target_field in {"timestamp_utc", "received_at_utc", "eta_utc"} and value_score == 1.0:
        raw_confidence = min(1.0, raw_confidence + 0.05)
    if status == "block":
        raw_confidence = 0.0
    decision = "blocked" if status == "block" else (
        "auto_map" if raw_confidence >= AUTO_MAP_THRESHOLD and status == "pass" else
        "needs_review" if raw_confidence >= REVIEW_THRESHOLD else "keep_unknown")
    return {
        "status": status,
        "checks": checks,
        "raw_confidence": round(raw_confidence, 6),
        "confidence_kind": "uncalibrated_evidence_score",
        "decision": decision,
    }


def validate_gis_suggestion(source_column: str, target_field: str, values: Iterable[Any],
                            candidate: dict[str, Any], model_score: float,
                            selected_mapping: dict[str, str]) -> dict[str, Any]:
    """Validate one GIS property mapping with deterministic evidence checks."""
    checks: list[dict[str, Any]] = []
    samples = _present(values)
    value_score = value_compatibility(target_field, samples)
    conflicts = sorted(field for field, column in selected_mapping.items()
                       if column == source_column and field != target_field)
    if conflicts:
        checks.append({"check": "source_column_conflict", "status": "block",
                       "detail": f"同一来源属性还被映射到: {', '.join(conflicts)}"})
    selected_source = selected_mapping.get(target_field)
    if selected_source and selected_source != source_column:
        checks.append({"check": "target_field_conflict", "status": "block",
                       "detail": f"目标字段已由另一属性映射: {selected_source}"})
    if not samples:
        checks.append({"check": "representative_values", "status": "warning",
                       "detail": "没有非空样例值"})
    elif value_score < 0.5:
        checks.append({"check": "value_compatibility", "status": "warning",
                       "detail": f"样例兼容比例为 {value_score:.1%}"})
    else:
        checks.append({"check": "value_compatibility", "status": "pass",
                       "detail": f"样例兼容比例为 {value_score:.1%}"})
    status = "block" if any(item["status"] == "block" for item in checks) else (
        "warning" if any(item["status"] == "warning" for item in checks) else "pass")
    bounded_model_score = min(1.0, max(0.0, float(model_score)))
    raw_confidence = (0.40 * float(candidate.get("name_score", 0.0)) +
                      0.30 * value_score +
                      0.15 * float(candidate.get("unit_score", 1.0)) +
                      0.15 * bounded_model_score)
    if status == "block":
        raw_confidence = 0.0
    decision = "blocked" if status == "block" else (
        "auto_map" if raw_confidence >= AUTO_MAP_THRESHOLD and status == "pass" else
        "needs_review" if raw_confidence >= REVIEW_THRESHOLD else "keep_unknown")
    return {"status": status, "checks": checks, "raw_confidence": round(raw_confidence, 6),
            "confidence_kind": "uncalibrated_evidence_score", "decision": decision}


def validate_weather_suggestion(source_column: str, target_field: str, values: Iterable[Any],
                                candidate: dict[str, Any], model_score: float,
                                selected_mapping: dict[str, str]) -> dict[str, Any]:
    """Validate a weather-column suggestion before it may affect ingestion."""
    checks: list[dict[str, Any]] = []
    samples = _present(values)
    value_score = value_compatibility(target_field, samples)
    conflicts = sorted(field for field, column in selected_mapping.items()
                       if column == source_column and field != target_field)
    if conflicts:
        checks.append({"check": "source_column_conflict", "status": "block",
                       "detail": f"同一来源列还被映射到: {', '.join(conflicts)}"})
    selected_source = selected_mapping.get(target_field)
    if selected_source and selected_source != source_column:
        checks.append({"check": "target_field_conflict", "status": "block",
                       "detail": f"目标字段已由另一列映射: {selected_source}"})
    critical = {"valid_time_utc", "lon", "lat"}
    if not samples:
        checks.append({"check": "representative_values", "status": "warning",
                       "detail": "没有非空样例值"})
    elif target_field in critical and value_score < 0.8:
        checks.append({"check": "value_compatibility", "status": "block",
                       "detail": f"必要字段样例兼容比例仅为 {value_score:.1%}"})
    elif value_score < 0.5:
        checks.append({"check": "value_compatibility", "status": "warning",
                       "detail": f"样例兼容比例为 {value_score:.1%}"})
    else:
        checks.append({"check": "value_compatibility", "status": "pass",
                       "detail": f"样例兼容比例为 {value_score:.1%}"})
    status = "block" if any(item["status"] == "block" for item in checks) else (
        "warning" if any(item["status"] == "warning" for item in checks) else "pass")
    bounded_model_score = min(1.0, max(0.0, float(model_score)))
    raw_confidence = (0.40 * float(candidate.get("name_score", 0.0)) +
                      0.30 * value_score +
                      0.15 * float(candidate.get("unit_score", 1.0)) +
                      0.15 * bounded_model_score)
    if status == "block":
        raw_confidence = 0.0
    decision = "blocked" if status == "block" else (
        "auto_map" if raw_confidence >= AUTO_MAP_THRESHOLD and status == "pass" else
        "needs_review" if raw_confidence >= REVIEW_THRESHOLD else "keep_unknown")
    return {"status": status, "checks": checks, "raw_confidence": round(raw_confidence, 6),
            "confidence_kind": "uncalibrated_evidence_score", "decision": decision}
