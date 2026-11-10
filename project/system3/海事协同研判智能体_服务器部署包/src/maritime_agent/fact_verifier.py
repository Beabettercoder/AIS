from __future__ import annotations

import math
import re
from typing import Any, NamedTuple


class MetricClaim(NamedTuple):
    category: str      # "wind_speed", "distance", "pressure", "vessel_count", "record_count", "percentage", "wave_height"
    raw_text: str      # e.g. "48.2 节"
    value: float       # 48.2
    unit: str          # "节"


CATEGORY_LABELS: dict[str, str] = {
    "wind_speed": "风速/航速",
    "distance": "距离/尺度",
    "pressure": "气压",
    "vessel_count": "船舶/目标数量",
    "record_count": "数据/记录数量",
    "percentage": "比率/百分比",
    "wave_height": "浪高",
}

# Conversion constants
KNOTS_TO_MPS = 0.514444444
MPS_TO_KNOTS = 1.943844492
NM_TO_KM = 1.852
KM_TO_NM = 1.0 / 1.852
NM_TO_M = 1852.0
KM_TO_M = 1000.0

SKIP_KEYS = {
    "evidence_id", "id", "tool_call_id", "call_id",
    "mmsi", "imo", "callsign", "hash",
    "timestamp", "date", "datetime", "created_at", "updated_at",
    "start_utc", "end_utc", "first_utc", "last_utc", "date_utc",
    "target_day_utc", "baseline_day_utc",
    "geometry", "coordinates", "provenance"
}

CATEGORY_KEY_MAP: dict[str, set[str]] = {
    "wind_speed": {"wind", "speed", "gust", "sog", "knots", "velocity", "wind_speed", "wind_kts"},
    "distance": {"distance", "drift", "range", "length", "width", "radius", "drift_distance_km", "drift_km", "drift_m"},
    "pressure": {"pressure", "slp", "baro", "hpa", "min_pressure"},
    "vessel_count": {"vessels", "vessel_count", "ship_count", "ships", "targets", "match_count", "unique_vessels"},
    "record_count": {"record_count", "records", "total_records", "point_count", "track_count", "row_count", "count", "total"},
    "percentage": {"confidence", "score", "probability", "percent", "ratio"},
    "wave_height": {"wave", "wave_height", "swell", "sea_state"},
}


def extract_metric_claims(text: str) -> list[MetricClaim]:
    """
    Extract physical and quantitative metric claims from text.
    Filters out dates, clock times, ordinals ('第1轮'), and non-physical values.
    """
    claims: list[MetricClaim] = []

    def _is_skipped_prefix(start: int) -> bool:
        prefix = text[:start].rstrip()
        return prefix.endswith(("第", "前", "top", "Top", "TOP", "共计前", "仅前"))

    # 1. Wind speed & ship speed: e.g. 48 节, 48.2节, 25 m/s, 24.8米/秒, 65 knots, 12.5 kts
    pattern_speed = re.finditer(
        r'(?<!\d)(\d+(?:,\d{3})*(?:\.\d+)?)\s*(节|knots?|kts?|m/s|米/秒)(?![a-zA-Z0-9])',
        text, re.IGNORECASE
    )
    for m in pattern_speed:
        if _is_skipped_prefix(m.start()):
            continue
        num_str = m.group(1).replace(",", "")
        unit = m.group(2).lower()
        try:
            val = float(num_str)
            claims.append(MetricClaim(category="wind_speed", raw_text=m.group(0).strip(), value=val, unit=unit))
        except ValueError:
            pass

    # 2. Wave height: e.g. 4.5米浪高, 3.2m浪高, 4米涌浪
    pattern_wave = re.finditer(
        r'(?<!\d)(\d+(?:,\d{3})*(?:\.\d+)?)\s*(米浪高|m浪高|米涌浪|米高海浪)',
        text, re.IGNORECASE
    )
    for m in pattern_wave:
        if _is_skipped_prefix(m.start()):
            continue
        num_str = m.group(1).replace(",", "")
        try:
            val = float(num_str)
            claims.append(MetricClaim(category="wave_height", raw_text=m.group(0).strip(), value=val, unit=m.group(2)))
        except ValueError:
            pass

    # 3. Distance & Drift: e.g. 3.2公里, 3200米, 1.8海里, 2.5 km, 1500 m
    # Must NOT match 'm/s', '米/秒', '米浪高', 'm浪高', or dimensions like '10米宽'
    pattern_distance = re.finditer(
        r'(?<!\d)(\d+(?:,\d{3})*(?:\.\d+)?)\s*(海里|nm|nmi|公里|千米|km|(?:米|m)(?!/|秒|浪|分|宽|长|深|高|厚))(?![a-zA-Z0-9])',
        text, re.IGNORECASE
    )
    for m in pattern_distance:
        if _is_skipped_prefix(m.start()):
            continue
        num_str = m.group(1).replace(",", "")
        unit = m.group(2).lower()
        try:
            val = float(num_str)
            claims.append(MetricClaim(category="distance", raw_text=m.group(0).strip(), value=val, unit=unit))
        except ValueError:
            pass

    # 4. Atmospheric pressure: e.g. 978 hPa, 978百帕, 980 mb, 980毫巴
    pattern_pressure = re.finditer(
        r'(?<!\d)(\d+(?:,\d{3})*(?:\.\d+)?)\s*(hpa|百帕|mb|毫巴)(?![a-zA-Z0-9])',
        text, re.IGNORECASE
    )
    for m in pattern_pressure:
        if _is_skipped_prefix(m.start()):
            continue
        num_str = m.group(1).replace(",", "")
        unit = m.group(2).lower()
        try:
            val = float(num_str)
            claims.append(MetricClaim(category="pressure", raw_text=m.group(0).strip(), value=val, unit=unit))
        except ValueError:
            pass

    # 5. Vessel / Target counts: e.g. 12艘, 12 艘船, 5条船, 3个目标
    pattern_vessels = re.finditer(
        r'(?<!\d)(\d+(?:,\d{3})*)\s*(艘船舶|艘遇险船|艘船|条船|只船|艘|个目标)(?![a-zA-Z0-9])',
        text
    )
    for m in pattern_vessels:
        if _is_skipped_prefix(m.start()):
            continue
        num_str = m.group(1).replace(",", "")
        try:
            val = float(num_str)
            claims.append(MetricClaim(category="vessel_count", raw_text=m.group(0).strip(), value=val, unit=m.group(2)))
        except ValueError:
            pass

    # 6. Record / Observation counts: e.g. 12,109条记录, 350条轨迹, 45条数据, 5000条AIS
    pattern_records = re.finditer(
        r'(?<!\d)(\d+(?:,\d{3})*)\s*(条记录|条数据|条点位|条轨迹|条ais|个轨迹点)(?![a-zA-Z0-9])',
        text, re.IGNORECASE
    )
    for m in pattern_records:
        if _is_skipped_prefix(m.start()):
            continue
        num_str = m.group(1).replace(",", "")
        try:
            val = float(num_str)
            claims.append(MetricClaim(category="record_count", raw_text=m.group(0).strip(), value=val, unit=m.group(2)))
        except ValueError:
            pass

    # 7. Probability / Percentage: e.g. 85%, 92.5%, 15.2%
    pattern_percentage = re.finditer(
        r'(?<!\d)(\d+(?:\.\d+)?)\s*(%|百分点)(?![a-zA-Z0-9])',
        text
    )
    for m in pattern_percentage:
        if _is_skipped_prefix(m.start()):
            continue
        num_str = m.group(1)
        try:
            val = float(num_str)
            claims.append(MetricClaim(category="percentage", raw_text=m.group(0).strip(), value=val, unit=m.group(2)))
        except ValueError:
            pass

    return claims


def extract_evidence_numbers_and_claims(evidence: Any) -> tuple[set[float], dict[str, set[float]]]:
    """
    Recursively inspect evidence items and extract:
    1. all numerical values (floats/ints), skipping calendar dates, times, ids, mmsi.
    2. categorized numerical values by inspecting dict keys.
    3. metric claims present within text strings in the evidence.
    """
    all_numbers: set[float] = set()
    categorized: dict[str, set[float]] = {cat: set() for cat in CATEGORY_LABELS}

    def _walk(obj: Any, parent_key: str = ""):
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            val = float(obj)
            all_numbers.add(val)
            low_k = parent_key.lower()
            for cat, keywords in CATEGORY_KEY_MAP.items():
                if any(kw in low_k for kw in keywords):
                    categorized[cat].add(val)
        elif isinstance(obj, str):
            # Skip pure ISO timestamps like 2024-07-06T17:00:00Z
            if re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?', obj):
                return
            # Parse metric claims inside text
            inner_claims = extract_metric_claims(obj)
            for ic in inner_claims:
                all_numbers.add(ic.value)
                categorized[ic.category].add(ic.value)
            # Standalone number string
            num_match = re.match(r'^-?\d+(?:,\d{3})*(?:\.\d+)?$', obj.strip())
            if num_match:
                try:
                    all_numbers.add(float(obj.strip().replace(",", "")))
                except ValueError:
                    pass
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in SKIP_KEYS:
                    continue
                _walk(v, parent_key=k)
        elif isinstance(obj, (list, tuple)):
            # Record list length as an available count
            if len(obj) > 0:
                all_numbers.add(float(len(obj)))
                categorized["vessel_count"].add(float(len(obj)))
                categorized["record_count"].add(float(len(obj)))
            for item in obj:
                _walk(item, parent_key=parent_key)

    _walk(evidence)
    return all_numbers, categorized


def is_claim_grounded(
    claim: MetricClaim,
    all_numbers: set[float],
    categorized_numbers: dict[str, set[float]],
    context_numbers: set[float],
    rel_tol: float = 0.05,
    abs_tol: float = 0.6,
) -> bool:
    """
    Check if a metric claim is supported by evidence numbers, unit conversions, or question context.
    """
    def _matches(target: float, candidates: set[float]) -> bool:
        for c in candidates:
            if math.isclose(target, c, rel_tol=rel_tol, abs_tol=abs_tol):
                return True
        return False

    # 1. Direct match in evidence or context
    if _matches(claim.value, all_numbers) or _matches(claim.value, context_numbers):
        return True

    # 2. Unit conversion checks
    cat = claim.category
    cat_candidates = categorized_numbers.get(cat, set()) | all_numbers

    if cat == "wind_speed":
        # If claimed in knots, candidate might be in m/s
        if any(claim.unit.startswith(u) for u in ("节", "knot", "kt")):
            if _matches(claim.value * KNOTS_TO_MPS, cat_candidates):
                return True
        # If claimed in m/s, candidate might be in knots
        elif any(claim.unit.startswith(u) for u in ("m/s", "米/秒")):
            if _matches(claim.value * MPS_TO_KNOTS, cat_candidates):
                return True

    elif cat == "distance":
        # If claimed in km
        if any(claim.unit.startswith(u) for u in ("公里", "千米", "km")):
            if _matches(claim.value * KM_TO_M, cat_candidates) or _matches(claim.value * KM_TO_NM, cat_candidates):
                return True
        # If claimed in meters
        elif any(claim.unit.startswith(u) for u in ("米", "m")):
            if _matches(claim.value / KM_TO_M, cat_candidates) or _matches(claim.value / NM_TO_M, cat_candidates):
                return True
        # If claimed in nm (海里)
        elif any(claim.unit.startswith(u) for u in ("海里", "nm", "nmi")):
            if _matches(claim.value * NM_TO_KM, cat_candidates) or _matches(claim.value * NM_TO_M, cat_candidates):
                return True

    elif cat == "percentage":
        # 85% claimed vs 0.85 in evidence
        if _matches(claim.value / 100.0, cat_candidates):
            return True
        # 0.85% claimed vs 85 in evidence (rare)
        if _matches(claim.value * 100.0, cat_candidates):
            return True

    return False


def verify_numerical_grounding(
    text: str,
    evidence: list[dict[str, Any]] | dict[str, Any],
    context_text: str = "",
    rel_tol: float = 0.05,
    abs_tol: float = 0.6,
) -> list[str]:
    """
    Verifies that all quantitative/metric claims in `text` are grounded in `evidence` or `context_text`.
    Returns a list of warning strings for any ungrounded claims.
    """
    claims = extract_metric_claims(text)
    if not claims:
        return []

    all_numbers, categorized = extract_evidence_numbers_and_claims(evidence)

    # Extract numbers from context text (e.g. user question)
    context_numbers: set[float] = set()
    if context_text:
        for match in re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)', context_text):
            try:
                context_numbers.add(float(match.replace(",", "")))
            except ValueError:
                pass

    warnings: list[str] = []
    seen_claims: set[tuple[str, float, str]] = set()

    for claim in claims:
        key = (claim.category, round(claim.value, 2), claim.unit)
        if key in seen_claims:
            continue
        seen_claims.add(key)

        if not is_claim_grounded(claim, all_numbers, categorized, context_numbers, rel_tol, abs_tol):
            label = CATEGORY_LABELS.get(claim.category, "定量指标")
            warnings.append(f"数值事实存疑: 声称{label} '{claim.raw_text}' 未在引用证据中找到对应数值支持")

    return warnings
