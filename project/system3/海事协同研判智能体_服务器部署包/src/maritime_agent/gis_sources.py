"""Bounded GeoJSON intake and one allowlisted NOAA ENC GIS connector."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .alignment_rules import rules_for
from .database import _initialize, open_readonly


NOAA_CHANNELS_URL = (
    "https://gis.charttools.noaa.gov/arcgis/rest/services/"
    "NavigationChartData/MarineTransportation/MapServer/1/query"
)
NOAA_LAYER_URL = NOAA_CHANNELS_URL.removesuffix("/query")
MAX_GIS_FEATURES = 10_000
MAX_GIS_COORDINATES = 1_000_000
MAX_GIS_BYTES = 200 * 1024 * 1024
MAX_EXTERNAL_FEATURES = 100
GIS_RULES = rules_for("gis_geojson")
GIS_ALIASES = {field: tuple(aliases) for field, aliases in GIS_RULES["aliases"].items()}


def _norm(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum())


def _field_mapping(columns: list[str], explicit: dict[str, str] | None = None) -> dict[str, str]:
    by_norm = {_norm(column): column for column in columns}
    result = {field: next((by_norm[_norm(alias)] for alias in aliases if _norm(alias) in by_norm), "")
              for field, aliases in GIS_ALIASES.items()}
    result.update({key: value for key, value in (explicit or {}).items()
                   if key in GIS_ALIASES and value in columns})
    return result


def _source_crs(payload: dict[str, Any]) -> str:
    crs = payload.get("crs")
    if not crs:
        return "EPSG:4326"
    text = json.dumps(crs, ensure_ascii=False).casefold()
    if "3857" in text or "900913" in text:
        return "EPSG:3857"
    if "4326" in text or "crs84" in text:
        return "EPSG:4326"
    raise ValueError("当前 GIS 接入仅能确定性转换 EPSG:3857，其他原始 CRS 需要先明确转换")


def _wgs84_coordinates(value: Any, source_crs: str) -> Any:
    if isinstance(value, list) and len(value) >= 2 and all(isinstance(x, (float, int)) for x in value[:2]):
        if source_crs == "EPSG:4326":
            return list(value)
        x, y = float(value[0]), float(value[1])
        if not (math.isfinite(x) and math.isfinite(y)
                and abs(x) <= 20037508.342789244 and abs(y) <= 20037508.342789244):
            raise ValueError("EPSG:3857 坐标超出 Web Mercator 有效范围")
        lon = x * 180.0 / 20037508.342789244
        lat = math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)
        return [lon, lat, *value[2:]]
    if isinstance(value, list) and value:
        return [_wgs84_coordinates(part, source_crs) for part in value]
    raise ValueError("GIS 要素坐标结构无效")


def _canonical_object_type(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return "unknown"
    normalized = _norm(text)
    for canonical, aliases in GIS_RULES.get("object_type_aliases", {}).items():
        if normalized in {_norm(alias) for alias in aliases}:
            return canonical
    return "unknown"


def _canonical_date(value: Any, field: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        return datetime.strptime(text.replace("/", "-"), "%Y-%m-%d").date().isoformat()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} 不是可识别的 ISO 日期或时间") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} 含时分秒但没有时区")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite_number(value: Any, field: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 不是数字") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} 不是有限数值")
    return number


def _measurement(value: Any, column: str, field: str) -> tuple[float | None, dict[str, Any] | None]:
    number = _finite_number(value, field)
    if number is None:
        return None, None
    normalized_column = _norm(column)
    length_units = {_norm(alias): unit for alias, unit in GIS_RULES.get("length_unit_by_alias", {}).items()}
    area_units = {_norm(alias): unit for alias, unit in GIS_RULES.get("area_unit_by_alias", {}).items()}
    if field == "area_m2":
        unit = area_units.get(normalized_column)
        if unit is None:
            unit = "km2" if "km2" in normalized_column else ("ha" if normalized_column.endswith("ha") else "m2")
        factor = {"m2": 1.0, "km2": 1_000_000.0, "ha": 10_000.0}[unit]
        target_unit = "m2"
    else:
        unit = length_units.get(normalized_column)
        if unit is None:
            unit = "ft" if any(token in normalized_column for token in ("feet", "foot", "ft")) else "m"
        factor = 0.3048 if unit == "ft" else 1.0
        target_unit = "m"
    converted = number * factor
    if converted < 0 and field not in {"depth_m", "min_depth_m", "max_depth_m", "elevation_m"}:
        raise ValueError(f"{field} 不能为负数")
    audit = None if factor == 1.0 else {
        "field": field, "source_column": column, "source_unit": unit,
        "target_unit": target_unit, "rule": f"gis_{field}_{unit}_to_{target_unit}"
    }
    return converted, audit


def _canonical_properties(props: dict[str, Any], mapping: dict[str, str]
                          ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    canonical: dict[str, Any] = {}
    audits: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    date_fields = {"source_date", "valid_from", "valid_to", "updated_at"}
    measurement_fields = {"depth_m", "min_depth_m", "max_depth_m", "width_m", "length_m",
                          "area_m2", "elevation_m", "clearance_height_m"}
    core = {"feature_id", "object_name", "object_type"}
    for field, column in mapping.items():
        if not column or field in core:
            continue
        value = props.get(column)
        if value in (None, ""):
            continue
        try:
            audit = None
            if field in date_fields:
                normalized = _canonical_date(value, field)
            elif field in measurement_fields:
                normalized, audit = _measurement(value, column, field)
            else:
                normalized = " ".join(str(value).strip().split())[:1000] or None
            if normalized is not None:
                canonical[field] = normalized
            if audit is not None:
                audits.append(audit)
        except ValueError as exc:
            errors.append({"field": field, "source_column": column, "error": str(exc)})
    return canonical, audits, errors


def _issue(code: str, severity: str, dimension: str, message: str, count: int,
           total: int, samples: list[dict] | None = None, blocking: bool = False) -> dict[str, Any]:
    return {"code": code, "severity": severity, "dimension": dimension, "message": message,
            "affected_count": count, "affected_ratio": round(count / total, 6) if total else 0,
            "samples": (samples or [])[:5], "blocking": blocking}


def _bbox(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("GIS 范围必须是 [西,南,东,北]")
    try:
        west, south, east, north = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("GIS 范围含非数值坐标") from exc
    if not all(map(math.isfinite, (west, south, east, north))):
        raise ValueError("GIS 范围含非有限数值")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("GIS 范围经纬度或顺序无效")
    if east - west > 5 or north - south > 5:
        raise ValueError("单次外部 GIS 查询范围不得超过经纬度各 5 度")
    return [west, south, east, north]


def _coordinates(value: Any, points: list[tuple[float, float]]) -> None:
    if isinstance(value, list) and len(value) >= 2 and all(isinstance(x, (float, int)) for x in value[:2]):
        lon, lat = float(value[0]), float(value[1])
        if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("GIS 要素坐标超出 WGS84 经纬度范围")
        points.append((lon, lat))
        if len(points) > MAX_GIS_COORDINATES:
            raise ValueError("GIS 要素坐标点过多")
    elif isinstance(value, list) and value:
        for part in value:
            _coordinates(part, points)
    else:
        raise ValueError("GIS 要素坐标结构无效")


def parse_geojson(data: bytes, provider: str = "user", mapping: dict[str, str] | None = None,
                  repair_actions: set[str] | None = None) -> tuple[list[tuple], dict[str, Any]]:
    repair_actions = set(repair_actions or ())
    if not data or len(data) > MAX_GIS_BYTES:
        raise ValueError("GIS GeoJSON 文件不得超过 200 MB")
    raw_text = None
    for enc in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            raw_text = data.decode(enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if raw_text is None:
        raise ValueError("GIS 文件编码无法识别（支持 UTF-8、GB18030、UTF-16）")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("GIS 文件必须是合法的 GeoJSON") from exc

    features: list[Any] | None = None
    if isinstance(payload, dict):
        p_type = payload.get("type")
        if p_type == "FeatureCollection":
            features = payload.get("features")
        elif p_type == "Feature":
            features = [payload]
        elif p_type in {"Polygon", "MultiPolygon", "LineString", "MultiLineString", "Point", "MultiPoint", "GeometryCollection"}:
            features = [{"type": "Feature", "geometry": payload, "properties": {}}]
    elif isinstance(payload, list) and all(isinstance(item, dict) and item.get("type") == "Feature" for item in payload):
        features = payload

    if not isinstance(features, list) or not 1 <= len(features) <= MAX_GIS_FEATURES:
        raise ValueError(f"GIS 文件须包含 1 至 {MAX_GIS_FEATURES} 个 GeoJSON Feature")
    source_crs = _source_crs(payload)
    columns = sorted({str(key) for feature in features if isinstance(feature, dict)
                      for key in (feature.get("properties") or {}).keys()})
    resolved = _field_mapping(columns, mapping)
    rows: list[tuple] = []
    all_points: list[tuple[float, float]] = []
    ids: set[str] = set()
    invalid_count = 0
    invalid_samples: list[dict] = []
    missing_name = 0
    unknown_type = 0
    generated_id = 0
    geometry_counts: dict[str, int] = {}
    preview_features: list[dict] = []
    normalization_audits: list[dict[str, Any]] = []
    property_errors: list[dict[str, Any]] = []
    for index, feature in enumerate(features, 1):
        try:
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise ValueError("不是 GeoJSON Feature")
            source_geometry = feature.get("geometry")
            props = feature.get("properties") or {}
            if not isinstance(source_geometry, dict) or not isinstance(props, dict):
                raise ValueError("缺少几何或属性")
            kind = source_geometry.get("type")
            if kind not in {"Point", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}:
                raise ValueError("几何类型不支持")
            geometry = {**source_geometry,
                        "coordinates": _wgs84_coordinates(source_geometry.get("coordinates"), source_crs)}
            points: list[tuple[float, float]] = []
            _coordinates(geometry.get("coordinates"), points)
            if kind in {"Polygon", "MultiPolygon"}:
                polygons = [geometry["coordinates"]] if kind == "Polygon" else geometry["coordinates"]
                for polygon in polygons:
                    if not isinstance(polygon, list) or not polygon:
                        raise ValueError("多边形缺少环")
                    for ring in polygon:
                        if not isinstance(ring, list) or len(ring) < 4 or ring[0][:2] != ring[-1][:2]:
                            raise ValueError("多边形环必须闭合且至少含 4 个点")
            raw_id = feature.get("id") or (props.get(resolved["feature_id"]) if resolved["feature_id"] else None)
            if raw_id in (None, ""):
                raw_id = f"generated-{index}"
                generated_id += 1
            feature_id = hashlib.sha256(str(raw_id).encode()).hexdigest()[:16]
            if feature_id in ids:
                raise ValueError("对象 ID 重复")
            ids.add(feature_id)
            name = str(props.get(resolved["object_name"], "") if resolved["object_name"] else "").strip()[:250]
            if not name:
                missing_name += 1
            raw_type = ("maintained_channel" if provider == "noaa" else
                        props.get(resolved["object_type"], "") if resolved["object_type"] else "")
            object_type = _canonical_object_type(raw_type)
            if object_type == "unknown":
                unknown_type += 1
            canonical_properties, audits, field_errors = _canonical_properties(props, resolved)
            source_date = str(canonical_properties.get("source_date") or "")[:40]
            normalization_audits.extend(audit for audit in audits if audit not in normalization_audits)
            property_errors.extend({"feature": index, **error} for error in field_errors)
            aligned_properties = {
                **props,
                **canonical_properties,
                "feature_id": feature_id,
                "object_name": name,
                "object_type": object_type,
                "geometry_type": kind,
                "crs": "EPSG:4326",
                "_raw_properties": props,
            }
            all_points.extend(points)
            if len(all_points) > MAX_GIS_COORDINATES:
                raise ValueError("GIS 文件坐标点过多")
            geometry_counts[kind] = geometry_counts.get(kind, 0) + 1
            rows.append((feature_id, name, object_type, source_date,
                         json.dumps(geometry, ensure_ascii=False, separators=(",", ":")),
                         json.dumps(aligned_properties, ensure_ascii=False, separators=(",", ":"))))
            if len(preview_features) < 100:
                preview_features.append({"type": "Feature", "id": feature_id,
                                         "properties": {"object_name": name, "object_type": object_type},
                                         "geometry": geometry})
        except ValueError as exc:
            invalid_count += 1
            if len(invalid_samples) < 20:
                invalid_samples.append({"feature": index, "error": str(exc)})
    if not rows:
        raise ValueError("GeoJSON 没有可用要素: " + json.dumps(invalid_samples, ensure_ascii=False))
    bbox = [min(x for x, _ in all_points), min(y for _, y in all_points),
            max(x for x, _ in all_points), max(y for _, y in all_points)]
    issues = []
    dropping = "drop_invalid_features" in repair_actions
    if invalid_count:
        issues.append(_issue("GIS_INVALID_FEATURE", "warning" if dropping else "error", "validity",
                             "存在无效几何、重复 ID 或不支持的要素。", invalid_count, len(features),
                             invalid_samples, not dropping))
    if missing_name:
        issues.append(_issue("GIS_MISSING_NAME", "warning", "completeness",
                             "部分要素没有名称，仍可按空间位置和要素 ID 使用。", missing_name, len(features)))
    if unknown_type:
        issues.append(_issue("GIS_UNKNOWN_OBJECT_TYPE", "warning", "semantics",
                             "部分要素没有统一对象类别，已保留为 unknown。", unknown_type, len(features)))
    if generated_id:
        issues.append(_issue("GIS_GENERATED_ID", "info", "identity",
                             "部分要素没有原始 ID，系统按文件内顺序生成稳定 ID。", generated_id, len(features)))
    if property_errors:
        issues.append(_issue("GIS_PROPERTY_NORMALIZATION_FAILED", "warning", "semantics",
                             "部分已映射属性无法标准化；已保留原值并跳过对应标准字段。",
                             len(property_errors), len(features), property_errors))
    blockers = sum(1 for item in issues if item["blocking"])
    quality = {"version": 1, "status": "blocked" if blockers else "ready",
               "summary": {"raw_rows": len(features), "accepted_rows": len(rows),
                           "rejected_rows": invalid_count, "duplicates_removed": 0,
                           "blockers": blockers,
                           **{level: sum(1 for item in issues if item["severity"] == level)
                              for level in ("error", "warning", "info")}},
               "issues": issues, "geometry_types": geometry_counts, "crs": "EPSG:4326",
               "source_crs": source_crs}
    repairs = ([{"action_id": "drop_invalid_features", "label": "拒绝无效 GIS 要素",
                 "category": "needs_confirmation", "affected_count": invalid_count,
                 "selected": dropping,
                 "description": "不写入无效几何、重复 ID 或不支持的要素；原始文件保持不变。"}]
               if invalid_count else [])
    return rows, {"bbox_wgs84": bbox, "feature_count": len(rows), "raw_row_count": len(features),
                  "rejected_rows": invalid_count, "duplicates_removed": 0,
                  "schema_version": GIS_RULES["schema_version"],
                  "ruleset_version": GIS_RULES["ruleset_version"],
                  "columns": columns, "mapping": resolved, "geometry_types": geometry_counts,
                  "extra_columns": [column for column in columns if column not in set(resolved.values())],
                  "normalizations": ([{"field": "geometry", "source_crs": source_crs,
                                        "target_crs": "EPSG:4326", "rule": "gis_epsg3857_to_epsg4326"}]
                                     if source_crs == "EPSG:3857" else []) + normalization_audits,
                  "sample": [{"feature_id": row[0], "object_name": row[1], "object_type": row[2]}
                             for row in rows[:5]],
                  "quality_note": "已完成属性对齐、类别与单位标准化，并检查 GeoJSON 类型、WGS84 经纬度、支持的几何、闭合环和对象 ID；未自动修复多边形自相交。",
                  "quality_report": quality, "repair_plan": repairs,
                  "repair_preview": {"before": {"rows": len(features)},
                                     "after": {"rows": len(rows), "rejected_rows": invalid_count,
                                               "duplicates_removed": 0},
                                     "selected_actions": sorted(repair_actions)},
                  "map_preview": {"type": "FeatureCollection", "features": preview_features}}


def store_geojson(db_path: Path, data: bytes, kind: str, region_label: str,
                  source_url: str = "", request_bbox: list[float] | None = None,
                  mapping: dict[str, str] | None = None,
                  repair_actions: set[str] | None = None) -> dict[str, Any]:
    if kind not in {"gis_geojson", "noaa_enc_channels"}:
        raise ValueError("未知 GIS 来源类型")
    provider = "noaa" if kind == "noaa_enc_channels" else "user"
    rows, report = parse_geojson(data, provider, mapping, repair_actions)
    if report["quality_report"]["summary"]["blockers"]:
        raise ValueError("GIS 质量报告仍有阻断问题，请先确认修复方案并重新校验")
    digest = hashlib.sha256(data).hexdigest()
    source_id = "gis:" + hashlib.sha256((kind + region_label + source_url + digest).encode()).hexdigest()[:20]
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    conn = sqlite3.connect(db_path)
    try:
        _initialize(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        if conn.execute("SELECT 1 FROM external_gis_sources WHERE source_id=?", (source_id,)).fetchone():
            return {"status": "already_imported", "source_id": source_id, **report}
        with conn:
            conn.execute("INSERT INTO external_gis_sources VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (source_id, kind, region_label, "NOAA ENC Direct to GIS" if provider == "noaa" else "用户上传",
                          source_url, digest, fetched, json.dumps(report["bbox_wgs84"]),
                          json.dumps(request_bbox) if request_bbox else None,
                          report["feature_count"], report["quality_note"], data.decode("utf-8-sig")))
            conn.executemany("INSERT INTO external_gis_features VALUES (?,?,?,?,?,?,?)",
                             [(source_id, *row) for row in rows])
            conn.execute("INSERT INTO gis_intake_quality_audits VALUES (?,?,?,?,?,?,?,?,?)",
                         (source_id, report["quality_report"]["version"],
                          json.dumps(report["quality_report"], ensure_ascii=False),
                          json.dumps(report["repair_plan"], ensure_ascii=False),
                          json.dumps(report["repair_preview"]["selected_actions"], ensure_ascii=False),
                          report["raw_row_count"], report["feature_count"], report["rejected_rows"], fetched))
        return {"status": "imported", "source_id": source_id, **report}
    finally:
        conn.close()


def fetch_noaa_channels(bbox: list[float], limit: int = 50) -> bytes:
    bbox = _bbox(bbox)
    if not 1 <= limit <= MAX_EXTERNAL_FEATURES:
        raise ValueError("NOAA 单次获取要素数须为 1 至 100")
    params = {"where": "1=1", "geometry": ",".join(map(str, bbox)),
              "geometryType": "esriGeometryEnvelope", "inSR": "4326",
              "spatialRel": "esriSpatialRelIntersects", "outFields": "*",
              "returnGeometry": "true", "outSR": "4326", "f": "geojson",
              "resultRecordCount": str(limit + 1)}
    request = Request(NOAA_CHANNELS_URL + "?" + urlencode(params),
                      headers={"User-Agent": "maritime-situation-agent/0.11"})
    with urlopen(request, timeout=20) as response:
        raw = response.read(MAX_GIS_BYTES + 1)
    if len(raw) > MAX_GIS_BYTES:
        raise ValueError("NOAA 返回数据过大，请缩小区域")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("error") or payload.get("type") != "FeatureCollection":
        raise ValueError("NOAA GIS 接口未返回有效 GeoJSON")
    if payload.get("exceededTransferLimit") or len(payload.get("features", [])) > limit:
        raise ValueError("NOAA 结果超过单次上限，请缩小区域")
    if not payload.get("features"):
        raise ValueError("NOAA 航道图层在该范围没有要素")
    parse_geojson(raw, "noaa")
    return raw


def ensure_noaa_channels(db_path: Path, bbox: list[float], region_label: str = "Houston",
                         limit: int = 50) -> dict[str, Any]:
    bbox = _bbox(bbox)
    if not region_label.strip():
        raise ValueError("区域名称不能为空")
    with closing(open_readonly(db_path)) as conn:
        rows = conn.execute("SELECT source_id,fetched_at_utc,feature_count,request_bbox_json "
                            "FROM external_gis_sources WHERE source_kind='noaa_enc_channels' "
                            "ORDER BY fetched_at_utc DESC").fetchall()
    for row in rows:
        if row["request_bbox_json"] and json.loads(row["request_bbox_json"]) == bbox:
            fetched = datetime.fromisoformat(row["fetched_at_utc"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - fetched < timedelta(days=7):
                return {"status": "cached", "source_id": row["source_id"],
                        "feature_count": row["feature_count"], "bbox_wgs84": bbox,
                        "fetched_at_utc": row["fetched_at_utc"], "source_url": NOAA_LAYER_URL,
                        "temporal_caveat": "当前 NOAA 图层快照不证明历史 AIS 日期使用相同边界。"}
    raw = fetch_noaa_channels(bbox, limit)
    result = store_geojson(db_path, raw, "noaa_enc_channels", region_label, NOAA_LAYER_URL, bbox)
    result["source_url"] = NOAA_LAYER_URL
    result["temporal_caveat"] = "当前 NOAA 图层快照不证明历史 AIS 日期使用相同边界。"
    return result
