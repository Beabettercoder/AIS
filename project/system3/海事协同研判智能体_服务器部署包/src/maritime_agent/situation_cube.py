"""Task-adaptive, evidence-backed maritime situation cubes.

The cube is a derived logical index.  Native modality tables and raster assets
remain authoritative; this module only stores stable references, scores and
provenance.  Time values are never rounded to an hour.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .database import migrate_database
from .fusion import (associate_remote_detections, build_deterministic_relations,
                     build_open_set_associations, geometry_bbox, rebuild_fusion_index)
from .region_registry import normalize_region_name
from .weather_field import interpolate_point_weather


METHOD_VERSION = "adaptive_situation_cube_v1"
MODALITIES = ("ais", "gis", "weather", "remote_sensing", "event")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _refresh_fusion(db_path: Path, dataset_id: str) -> None:
    """Refresh nodes and restore all existing relation products after the rebuild."""
    rebuild_fusion_index(db_path, dataset_id)
    build_deterministic_relations(db_path, dataset_id)
    build_open_set_associations(db_path, dataset_id)
    associate_remote_detections(db_path, dataset_id)


def _stable_id(prefix: str, *parts: Any) -> str:
    return prefix + ":" + hashlib.sha256(_json(parts).encode("utf-8")).hexdigest()[:20]


def _utc(value: str, name: str) -> str:
    text = (value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} 必须是带时区的 ISO 8601 时间") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} 必须明确时区")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _seconds(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        west, south, east, north = map(float, value)
    except (TypeError, ValueError):
        return None
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        return None
    return west, south, east, north


def _bbox_geometry(box: tuple[float, float, float, float]) -> dict[str, Any]:
    west, south, east, north = box
    return {"type": "Polygon", "coordinates": [[[west, south], [east, south],
             [east, north], [west, north], [west, south]]]}


def _overlaps(left: tuple[float, float, float, float] | None,
              right: tuple[float, float, float, float] | None) -> bool:
    if left is None or right is None:
        return False
    return not (left[2] < right[0] or right[2] < left[0]
                or left[3] < right[1] or right[3] < left[1])


def _contains(box: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _intersection_ratio(container: tuple[float, float, float, float],
                        observed: tuple[float, float, float, float] | None) -> float:
    if observed is None:
        return 0.0
    if observed[0] == observed[2] and observed[1] == observed[3]:
        return 1.0 if _contains(container, (observed[0], observed[1])) else 0.0
    west, south = max(container[0], observed[0]), max(container[1], observed[1])
    east, north = min(container[2], observed[2]), min(container[3], observed[3])
    if west >= east or south >= north:
        return 0.0
    intersection = (east - west) * (north - south)
    observed_area = max((observed[2] - observed[0]) * (observed[3] - observed[1]), 1e-12)
    return min(1.0, intersection / observed_area)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _distance_m(left: tuple[float, float], right: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*left, *right))
    value = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371008.8 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _time_relation(node: sqlite3.Row, start: str, end: str) -> tuple[str, float | None, float]:
    if not node["start_utc"]:
        return "static_or_unknown", None, 1.0 if node["modality"] == "gis" else 0.45
    node_start = _seconds(node["start_utc"])
    node_end = _seconds(node["end_utc"] or node["start_utc"])
    unit_start, unit_end = _seconds(start), _seconds(end)
    if node_end >= unit_start and node_start <= unit_end:
        overlap = max(0.0, min(node_end, unit_end) - max(node_start, unit_start))
        duration = max(node_end - node_start, 1.0)
        # Instant observations inside the window are exact evidence, not zero-duration failures.
        score = 1.0 if node_start == node_end else min(1.0, max(0.55, overlap / duration))
        return "overlaps", 0.0, score
    delta = unit_start - node_end if node_end < unit_start else node_start - unit_end
    return ("before" if node_end < unit_start else "after"), delta, max(0.0, 1.0 - delta / 86400.0)


def _node_key(node: sqlite3.Row) -> str:
    try:
        native = json.loads(node["native_key_json"] or "{}")
    except json.JSONDecodeError:
        native = {}
    return _json(native)


def _candidate_nodes(conn: sqlite3.Connection, dataset_id: str, region_id: str,
                     region_box: tuple[float, float, float, float], start: str,
                     end: str) -> list[sqlite3.Row]:
    bindings = {row["source_id"]: row["region_id"] for row in conn.execute(
        "SELECT source_id,region_id FROM source_region_bindings WHERE dataset_id=?", (dataset_id,))}
    region = conn.execute("SELECT canonical_name,aliases_json FROM regions WHERE dataset_id=? AND region_id=?",
                          (dataset_id, region_id)).fetchone()
    region_terms: list[str] = []
    if region:
        labels = [region["canonical_name"], *json.loads(region["aliases_json"] or "[]")]
        for label in labels:
            normalized = normalize_region_name(str(label))
            pieces = [normalize_region_name(item) for item in str(label).replace("-", " ").split()]
            region_terms.extend(item for item in [normalized, *pieces] if len(item) >= 4)
    rows = conn.execute(
        "SELECT * FROM fusion_nodes WHERE dataset_id=? AND modality IN ('ais','gis','weather','remote_sensing','event') "
        "AND (start_utc IS NULL OR COALESCE(end_utc,start_utc)>=? AND start_utc<=?) "
        "AND (bbox_min_lon IS NULL OR NOT (bbox_max_lon<? OR bbox_min_lon>? OR bbox_max_lat<? OR bbox_min_lat>?)) "
        "ORDER BY COALESCE(start_utc,''),modality,node_id",
        (dataset_id, start, end, region_box[0], region_box[2], region_box[1], region_box[3])).fetchall()
    selected: list[sqlite3.Row] = []
    for row in rows:
        bound_region = bindings.get(row["source_id"]) if row["source_id"] else None
        if bound_region and bound_region != region_id:
            continue
        node_box = ((row["bbox_min_lon"], row["bbox_min_lat"], row["bbox_max_lon"], row["bbox_max_lat"])
                    if row["bbox_min_lon"] is not None else None)
        # Text-only events may be region-bound without geometry.  Other unlocated
        # records are not silently assigned to a place.
        if node_box is None:
            relation_located = False
            text_located = False
            if row["modality"] == "event":
                relation_located = conn.execute(
                    "SELECT 1 FROM fusion_relations r JOIN fusion_nodes p "
                    "ON p.dataset_id=r.dataset_id AND p.node_id=r.object_node_id "
                    "WHERE r.dataset_id=? AND r.subject_node_id=? AND r.predicate='OCCURS_IN' "
                    "AND r.decision='accepted' AND p.bbox_min_lon IS NOT NULL "
                    "AND NOT (p.bbox_max_lon<? OR p.bbox_min_lon>? OR p.bbox_max_lat<? OR p.bbox_min_lat>?) LIMIT 1",
                    (dataset_id, row["node_id"], region_box[0], region_box[2],
                     region_box[1], region_box[3])).fetchone() is not None
                try:
                    attributes = json.loads(row["attributes_json"] or "{}")
                except json.JSONDecodeError:
                    attributes = {}
                location_text = " ".join(str(attributes.get(key, "")) for key in
                                         ("location_text", "location", "region", "area"))
                normalized_location = normalize_region_name(location_text)
                text_located = any(term in normalized_location for term in region_terms)
            if not (row["modality"] == "event" and
                    (bound_region == region_id or relation_located or text_located)):
                continue
        if node_box is not None and not _overlaps(region_box, node_box):
            continue
        selected.append(row)
    return selected


def _derive_time_range(nodes: Iterable[sqlite3.Row]) -> tuple[str, str]:
    starts = [row["start_utc"] for row in nodes if row["start_utc"]]
    ends = [row["end_utc"] or row["start_utc"] for row in nodes if row["start_utc"]]
    if not starts:
        raise ValueError("研究区域没有带真实时间的观测，必须明确 start_time_utc 和 end_time_utc")
    start, end = min(starts), max(ends)
    if start == end:
        # Preserve the exact observation while creating a valid half-open query envelope.
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp() + 1
        end = datetime.fromtimestamp(end_dt, timezone.utc).isoformat().replace("+00:00", "Z")
    return start, end


def _unit_specs(cube_id: str, region_id: str, region_geometry: dict[str, Any],
                region_box: tuple[float, float, float, float], nodes: list[sqlite3.Row],
                start: str, end: str, task_type: str) -> list[dict[str, Any]]:
    specs = [{"unit_id": _stable_id("unit", cube_id, "region"), "geometry": region_geometry,
              "bbox": region_box, "spatial_cell_id": None, "spatial_level": "region",
              "creation_reason": "query_window"}]
    zones: list[tuple[str, dict[str, Any], tuple[float, float, float, float]]] = []
    for row in nodes:
        if row["modality"] != "gis" or not row["geometry_json"]:
            continue
        try:
            geometry = json.loads(row["geometry_json"])
        except json.JSONDecodeError:
            continue
        box = geometry_bbox(geometry)
        if box is None or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        zones.append((row["node_id"], geometry, box))
    for node_id, geometry, box in zones[:100]:
        specs.append({"unit_id": _stable_id("unit", cube_id, "zone", node_id),
                      "geometry": geometry, "bbox": box, "spatial_cell_id": node_id,
                      "spatial_level": "functional_zone", "creation_reason": "gis_functional_zone"})

    # Uncovered dynamic observations receive deterministic adaptive cells.  Cell
    # size follows region extent and is never a fixed global resolution.
    width, height = max(region_box[2] - region_box[0], 1e-6), max(region_box[3] - region_box[1], 1e-6)
    cell_width, cell_height = width / 8.0, height / 8.0
    occupied: set[tuple[int, int]] = set()
    for row in nodes:
        if row["modality"] == "gis" or row["bbox_min_lon"] is None:
            continue
        box = (row["bbox_min_lon"], row["bbox_min_lat"], row["bbox_max_lon"], row["bbox_max_lat"])
        point = _center(box)
        if any(_contains(zone_box, point) for _, _, zone_box in zones):
            continue
        col = min(7, max(0, int((point[0] - region_box[0]) / cell_width)))
        row_index = min(7, max(0, int((point[1] - region_box[1]) / cell_height)))
        occupied.add((col, row_index))
    for col, row_index in sorted(occupied):
        box = (region_box[0] + col * cell_width, region_box[1] + row_index * cell_height,
               region_box[0] + (col + 1) * cell_width, region_box[1] + (row_index + 1) * cell_height)
        cell_id = f"adaptive-8x8-{col}-{row_index}"
        specs.append({"unit_id": _stable_id("unit", cube_id, cell_id),
                      "geometry": _bbox_geometry(box), "bbox": box,
                      "spatial_cell_id": cell_id, "spatial_level": "adaptive_cell",
                      "creation_reason": "uncovered_observation_extent"})
    return specs


def _quality_scores(conn: sqlite3.Connection, dataset_id: str, region_id: str,
                    node: sqlite3.Row) -> tuple[float, float, dict[str, Any]]:
    source_score, schema_score = 0.8, 0.75
    details: dict[str, Any] = {"source_binding": "dataset_native", "schema_audit": "node_quality"}
    if node["source_id"]:
        binding = conn.execute(
            "SELECT validation_status,overlap_ratio FROM source_region_bindings "
            "WHERE dataset_id=? AND region_id=? AND source_id=?",
            (dataset_id, region_id, node["source_id"])).fetchone()
        if binding:
            source_score = {"verified": 1.0, "partial": 0.7, "legacy": 0.55,
                            "unverified": 0.4}.get(binding["validation_status"], 0.4)
            details["source_binding"] = binding["validation_status"]
            details["source_region_overlap"] = binding["overlap_ratio"]
        audit = conn.execute("SELECT schema_version,ruleset_version FROM source_alignment_metadata "
                             "WHERE source_id=?", (node["source_id"],)).fetchone()
        if audit:
            schema_score = 1.0
            details["schema_audit"] = {"schema_version": audit[0], "ruleset_version": audit[1]}
    try:
        quality = json.loads(node["quality_json"] or "{}")
    except json.JSONDecodeError:
        quality = {}
        schema_score = min(schema_score, 0.5)
    details["node_quality"] = quality
    return source_score, schema_score, details


def _link_node(conn: sqlite3.Connection, dataset_id: str, region_id: str,
               unit: dict[str, Any], node: sqlite3.Row, start: str, end: str,
               now: str) -> bool:
    unit_box = unit["bbox"]
    node_box = ((node["bbox_min_lon"], node["bbox_min_lat"], node["bbox_max_lon"], node["bbox_max_lat"])
                if node["bbox_min_lon"] is not None else None)
    spatial_score = _intersection_ratio(unit_box, node_box)
    if node_box is None:
        if unit["spatial_level"] != "region" or node["modality"] != "event":
            return False
        spatial_relation, spatial_score, distance = "region_text_reference", 0.35, None
    else:
        if spatial_score <= 0:
            return False
        spatial_relation = "contained" if spatial_score >= 0.999 else "intersects"
        distance = _distance_m(_center(unit_box), _center(node_box))
    temporal_relation, delta, temporal_score = _time_relation(node, start, end)
    if node["start_utc"] and temporal_relation != "overlaps":
        return False
    source_score, schema_score, quality_details = _quality_scores(
        conn, dataset_id, region_id, node)
    final = round(0.35 * temporal_score + 0.35 * spatial_score
                  + 0.15 * source_score + 0.15 * schema_score, 6)
    native_key = _node_key(node)
    attributes = json.loads(node["attributes_json"] or "{}")
    asset_id = attributes.get("scene_id") or attributes.get("asset_id")
    provenance = {"method_version": METHOD_VERSION,
                  "native_key": json.loads(native_key),
                  "node_provenance": json.loads(node["provenance_json"] or "{}"),
                  "score_components": quality_details,
                  "native_time": {"start_utc": node["start_utc"], "end_utc": node["end_utc"],
                                  "time_precision": node["time_precision"]},
                  "no_hour_rounding": True}
    conn.execute(
        "INSERT OR REPLACE INTO situation_evidence_links VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (dataset_id, unit["unit_id"], node["node_id"], node["modality"], node["native_table"],
         native_key, asset_id, temporal_relation, spatial_relation, delta, distance,
         spatial_score, temporal_score, spatial_score, source_score, schema_score, final,
         "explainable_rule_score_not_probability", METHOD_VERSION, _json(provenance), now))
    return True


def build_situation_cube(db_path: Path, dataset_id: str, region_id: str, *,
                         start_time_utc: str = "", end_time_utc: str = "",
                         task_type: str = "general_situation",
                         refresh_fusion_index: bool = False) -> dict[str, Any]:
    """Create or incrementally refresh one task-bounded logical cube."""
    migrate_database(db_path)
    if refresh_fusion_index:
        _refresh_fusion(db_path, dataset_id)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run_id = "cube-run:" + uuid4().hex
    try:
        region = conn.execute("SELECT * FROM regions WHERE dataset_id=? AND region_id=? AND status!='archived'",
                              (dataset_id, region_id)).fetchone()
        if region is None:
            raise ValueError("研究区域不存在或已经归档")
        region_box = _bbox(region["bbox_json"])
        if region_box is None:
            raise ValueError("研究区域缺少有效 WGS84 空间范围，暂不能建立态势立方体")
        region_geometry = (json.loads(region["boundary_geojson"]) if region["boundary_geojson"]
                           else _bbox_geometry(region_box))
        # Use a broad first pass only to derive an exact native observation range.
        if not start_time_utc or not end_time_utc:
            broad = _candidate_nodes(conn, dataset_id, region_id, region_box,
                                     "0001-01-01T00:00:00Z", "9999-12-31T23:59:59Z")
            derived_start, derived_end = _derive_time_range(broad)
            start_time_utc = start_time_utc or derived_start
            end_time_utc = end_time_utc or derived_end
        start, end = _utc(start_time_utc, "start_time_utc"), _utc(end_time_utc, "end_time_utc")
        if start == end:
            end = datetime.fromtimestamp(_seconds(end) + 1, timezone.utc).isoformat().replace("+00:00", "Z")
        if start > end:
            raise ValueError("start_time_utc 必须早于 end_time_utc")
        nodes = _candidate_nodes(conn, dataset_id, region_id, region_box, start, end)
        cube_id = _stable_id("cube", dataset_id, region_id, task_type, start, end)
        existing = conn.execute("SELECT 1 FROM situation_cubes WHERE dataset_id=? AND cube_id=?",
                                (dataset_id, cube_id)).fetchone() is not None
        now = _now()
        parameters = {"native_time_precision": True, "fixed_hour_bins": False,
                      "spatial_hierarchy": ["region", "functional_zone", "adaptive_cell"],
                      "refresh_fusion_index": refresh_fusion_index}
        with conn:
            conn.execute(
                "INSERT INTO situation_cubes VALUES (?,?,?,?,?,?,'building',?,?,?,?,?,?) "
                "ON CONFLICT(dataset_id,cube_id) DO UPDATE SET status='building',method_version=excluded.method_version,"
                "parameters_json=excluded.parameters_json,updated_at_utc=excluded.updated_at_utc",
                (dataset_id, cube_id, region_id, task_type, start, end, METHOD_VERSION,
                 _json(parameters), "[]", _json(list(MODALITIES)), now, now))
            conn.execute("INSERT INTO situation_cube_runs VALUES (?,?,?,'running',?,?,?,?,?,?)",
                         (dataset_id, run_id, cube_id, "incremental" if existing else "rebuild",
                          _json(parameters), "{}", now, None, None))
            specs = _unit_specs(cube_id, region_id, region_geometry, region_box,
                               nodes, start, end, task_type)
            active_ids = {item["unit_id"] for item in specs}
            old_ids = [row[0] for row in conn.execute(
                "SELECT unit_id FROM situation_units WHERE dataset_id=? AND cube_id=?",
                (dataset_id, cube_id))]
            for unit_id in old_ids:
                if unit_id not in active_ids:
                    conn.execute("DELETE FROM situation_units WHERE dataset_id=? AND unit_id=?",
                                 (dataset_id, unit_id))
            total_links = 0
            cube_modalities: set[str] = set()
            weather_nodes = [node for node in nodes if node["modality"] == "weather"]
            station_obs_for_cube: list[dict[str, Any]] = []
            for wn in weather_nodes:
                try:
                    attrs = json.loads(wn["attributes_json"] or "{}")
                except json.JSONDecodeError:
                    attrs = {}
                station_obs_for_cube.append({
                    "node_id": wn["node_id"],
                    "node": wn,
                    "lon": wn["bbox_min_lon"],
                    "lat": wn["bbox_min_lat"],
                    "location_id": attrs.get("location_id"),
                    "wind_speed_mps": attrs.get("wind_speed_mps"),
                    "wind_direction_deg": attrs.get("wind_direction_deg"),
                    "wind_gust_mps": attrs.get("wind_gust_mps"),
                    "air_pressure_hpa": attrs.get("air_pressure_hpa"),
                    "air_temperature_c": attrs.get("air_temperature_c"),
                    "relative_humidity_pct": attrs.get("relative_humidity_pct"),
                })
            for unit in specs:
                conn.execute("DELETE FROM situation_evidence_links WHERE dataset_id=? AND unit_id=?",
                             (dataset_id, unit["unit_id"]))
                conn.execute(
                    "INSERT INTO situation_units VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(dataset_id,unit_id) DO UPDATE SET geometry_json=excluded.geometry_json,"
                    "bbox_json=excluded.bbox_json,spatial_cell_id=excluded.spatial_cell_id,status='active',"
                    "updated_at_utc=excluded.updated_at_utc",
                    (dataset_id, unit["unit_id"], cube_id, region_id, start, end,
                     _json(unit["geometry"]), _json(unit["bbox"]), unit["spatial_cell_id"],
                     unit["spatial_level"], "native_query_interval", unit["creation_reason"],
                     task_type, "active", "[]", _json(list(MODALITIES)), now, now))
                available: set[str] = set()
                for node in nodes:
                    if _link_node(conn, dataset_id, region_id, unit, node, start, end, now):
                        available.add(node["modality"])
                        total_links += 1

                # 连续网格气象场升级：若该单元未直接落入离散测站，但本区域当前时窗内有气象站点数据，
                # 则利用 Barnes 客观分析高斯加权插值法合成该单元中心点连续气象，消除空间断崖
                if "weather" not in available and station_obs_for_cube:
                    c_lon, c_lat = _center(unit["bbox"])
                    interp = interpolate_point_weather(station_obs_for_cube, c_lon, c_lat)
                    if "error" not in interp and interp.get("spatial_confidence", 0.0) > 0.0:
                        nearest_node = None
                        min_d = float("inf")
                        for item in station_obs_for_cube:
                            d = _distance_m((c_lon, c_lat), (item["lon"], item["lat"]))
                            if d < min_d:
                                min_d = d
                                nearest_node = item["node"]
                        if nearest_node is not None:
                            spatial_score = float(interp["spatial_confidence"])
                            temporal_relation, delta, temporal_score = _time_relation(nearest_node, start, end)
                            source_score, schema_score, quality_details = _quality_scores(
                                conn, dataset_id, region_id, nearest_node)
                            final_conf = round(0.35 * temporal_score + 0.35 * spatial_score
                                               + 0.15 * source_score + 0.15 * schema_score, 6)
                            native_key = _node_key(nearest_node)
                            provenance = {
                                "method_version": METHOD_VERSION,
                                "weather_basis": "barnes_objective_wind_field",
                                "interpolated_weather": {
                                    "wind_speed_mps": interp.get("wind_speed_mps"),
                                    "wind_direction_deg": interp.get("wind_direction_deg"),
                                    "u_component_mps": interp.get("u_component_mps"),
                                    "v_component_mps": interp.get("v_component_mps"),
                                    "wind_gust_mps": interp.get("wind_gust_mps"),
                                    "air_pressure_hpa": interp.get("air_pressure_hpa"),
                                    "air_temperature_c": interp.get("air_temperature_c"),
                                    "relative_humidity_pct": interp.get("relative_humidity_pct"),
                                },
                                "spatial_confidence": spatial_score,
                                "nearest_station_km": interp.get("nearest_station_km"),
                                "contributions": interp.get("contributions", []),
                                "node_provenance": json.loads(nearest_node["provenance_json"] or "{}"),
                                "score_components": quality_details,
                                "native_time": {"start_utc": nearest_node["start_utc"],
                                                "end_utc": nearest_node["end_utc"],
                                                "time_precision": nearest_node["time_precision"]},
                                "no_hour_rounding": True
                            }
                            conn.execute(
                                "INSERT OR REPLACE INTO situation_evidence_links VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (dataset_id, unit["unit_id"], nearest_node["node_id"], "weather",
                                 nearest_node["native_table"], native_key, None,
                                 temporal_relation, "barnes_interpolated", delta, min_d,
                                 0.0, temporal_score, spatial_score, source_score, schema_score,
                                 final_conf, "barnes_objective_analysis", METHOD_VERSION,
                                 _json(provenance), now))
                            available.add("weather")
                            total_links += 1

                missing = [item for item in MODALITIES if item not in available]
                cube_modalities.update(available)
                conn.execute("UPDATE situation_units SET available_modalities_json=?,missing_modalities_json=? "
                             "WHERE dataset_id=? AND unit_id=?",
                             (_json(sorted(available)), _json(missing), dataset_id, unit["unit_id"]))
            cube_missing = [item for item in MODALITIES if item not in cube_modalities]
            summary = {"unit_count": len(specs), "evidence_link_count": total_links,
                       "available_modalities": sorted(cube_modalities),
                       "missing_modalities": cube_missing, "node_candidates": len(nodes)}
            conn.execute("UPDATE situation_cubes SET status='ready',available_modalities_json=?,"
                         "missing_modalities_json=?,updated_at_utc=? WHERE dataset_id=? AND cube_id=?",
                         (_json(sorted(cube_modalities)), _json(cube_missing), now, dataset_id, cube_id))
            conn.execute("UPDATE situation_cube_runs SET status='completed',summary_json=?,completed_at_utc=? "
                         "WHERE dataset_id=? AND run_id=?",
                         (_json(summary), now, dataset_id, run_id))
        return {"status": "ready", "dataset_id": dataset_id, "cube_id": cube_id,
                "region_id": region_id, "start_time_utc": start, "end_time_utc": end,
                "task_type": task_type, "build_mode": "incremental" if existing else "rebuild",
                "method_version": METHOD_VERSION, **summary}
    except Exception as exc:
        try:
            with conn:
                cube = locals().get("cube_id")
                if cube:
                    conn.execute("UPDATE situation_cubes SET status='failed',updated_at_utc=? "
                                 "WHERE dataset_id=? AND cube_id=?", (_now(), dataset_id, cube))
                    conn.execute("UPDATE situation_cube_runs SET status='failed',error_message=?,completed_at_utc=? "
                                 "WHERE dataset_id=? AND run_id=?", (str(exc)[:1000], _now(), dataset_id, run_id))
        finally:
            raise
    finally:
        conn.close()


def backfill_situation_cubes(db_path: Path, dataset_id: str, *,
                             refresh_fusion_index: bool = True) -> dict[str, Any]:
    """Safely build one native-range cube for every active/provisional region."""
    migrate_database(db_path)
    if refresh_fusion_index:
        _refresh_fusion(db_path, dataset_id)
    conn = sqlite3.connect(db_path)
    try:
        region_ids = [row[0] for row in conn.execute(
            "SELECT region_id FROM regions WHERE dataset_id=? AND status!='archived' ORDER BY region_id",
            (dataset_id,))]
    finally:
        conn.close()
    completed, pending = [], []
    for region_id in region_ids:
        try:
            completed.append(build_situation_cube(db_path, dataset_id, region_id,
                                                   refresh_fusion_index=False))
        except ValueError as exc:
            pending.append({"region_id": region_id, "reason": str(exc)})
    return {"dataset_id": dataset_id, "completed": completed, "pending_review": pending,
            "completed_count": len(completed), "pending_count": len(pending)}
