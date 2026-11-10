"""Rebuildable spatiotemporal index and deterministic relation layer.

The fusion layer never replaces modality-specific tables.  It stores stable
references to aligned records, plus versioned relations and their evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4
from zoneinfo import ZoneInfo

from .database import migrate_database


METHOD_VERSION = "fusion_rules_v1"
IDENTITY_METHOD_VERSION = "vessel_open_set_rules_v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = _json(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _day_bounds(day: str) -> tuple[str, str]:
    parsed = date.fromisoformat(day)
    return day + "T00:00:00Z", (parsed + timedelta(days=1)).isoformat() + "T00:00:00Z"


def _coordinate_pairs(value: Any) -> Iterable[tuple[float, float]]:
    if (isinstance(value, (list, tuple)) and len(value) >= 2
            and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float))):
        yield float(value[0]), float(value[1])
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _coordinate_pairs(item)


def geometry_bbox(geometry: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not geometry:
        return None
    points = list(_coordinate_pairs(geometry.get("coordinates")))
    if not points:
        return None
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def bbox_geometry(bbox: list[float] | tuple[float, float, float, float]) -> dict[str, Any]:
    west, south, east, north = (float(value) for value in bbox)
    return {"type": "Polygon", "coordinates": [[[west, south], [east, south],
                                                    [east, north], [west, north], [west, south]]]}


def _track_geometry(value: Any) -> dict[str, Any]:
    """Normalize both legacy coordinate arrays and canonical GeoJSON tracks."""
    if isinstance(value, dict) and value.get("type") and "coordinates" in value:
        return value
    if isinstance(value, list):
        return {"type": "LineString", "coordinates": value}
    raise ValueError("AIS 航迹 geometry_json 既不是 GeoJSON，也不是坐标数组")


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    previous = ring[-1]
    for current in ring:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if ((y1 > lat) != (y2 > lat)):
            crossing = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < crossing:
                inside = not inside
        previous = current
    return inside


def point_in_geometry(lon: float, lat: float, geometry: dict[str, Any]) -> bool:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    polygons = [coordinates] if kind == "Polygon" else coordinates if kind == "MultiPolygon" else []
    for polygon in polygons:
        if not polygon or not _point_in_ring(lon, lat, polygon[0]):
            continue
        if not any(_point_in_ring(lon, lat, hole) for hole in polygon[1:]):
            return True
    return False


def _bbox_intersection_ratio(left: tuple[float, float, float, float],
                             right: tuple[float, float, float, float]) -> float:
    west, south = max(left[0], right[0]), max(left[1], right[1])
    east, north = min(left[2], right[2]), min(left[3], right[3])
    if west > east or south > north:
        return 0.0
    intersection = max(east - west, 0.0) * max(north - south, 0.0)
    right_area = max(right[2] - right[0], 0.0) * max(right[3] - right[1], 0.0)
    if right_area == 0:
        return 1.0
    return min(1.0, intersection / right_area)


def _bbox_overlaps(left: tuple[float, float, float, float],
                   right: tuple[float, float, float, float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0]
                or left[3] < right[1] or right[3] < left[1])


def _haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*left, *right))
    value = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _dead_reckon_single(lon: float, lat: float, sog_kn: float | None,
                        cog_deg: float | None, delta_seconds: float,
                        rot_deg_min: float | None = None) -> tuple[float, float, str]:
    """Extrapolate position from a single AIS message along heading and speed (with ROT arc support)."""
    if sog_kn is None or cog_deg is None or sog_kn < 0.1:
        return lon, lat, "nearest_observed"
    speed_mps = float(sog_kn) * 0.514444
    distance_m = speed_mps * delta_seconds
    if abs(distance_m) < 0.1:
        return lon, lat, "nearest_observed"

    # 若具备有效且显著的转向率 ROT (|rot| >= 0.5 °/min)，使用等角速度圆弧轨迹积分外推
    if rot_deg_min is not None and abs(float(rot_deg_min)) >= 0.5:
        omega = math.radians(float(rot_deg_min) / 60.0)  # rad/s
        total_turn_rad = omega * delta_seconds
        r_turn = speed_mps / max(abs(omega), 1e-6)
        chord_len = 2.0 * r_turn * math.sin(abs(total_turn_rad) / 2.0)
        avg_cog = (float(cog_deg) + math.degrees(total_turn_rad) / 2.0) % 360.0

        radius_m = 6371008.8
        delta = chord_len / radius_m
        theta = math.radians(avg_cog)
        phi1 = math.radians(lat)
        lam1 = math.radians(lon)
        phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta))
        lam2 = lam1 + math.atan2(math.sin(theta) * math.sin(delta) * math.cos(phi1),
                                 math.cos(delta) - math.sin(phi1) * math.sin(phi2))
        lon2 = (math.degrees(lam2) + 540.0) % 360.0 - 180.0
        lat2 = math.degrees(phi2)
        return lon2, lat2, "circular_arc_dead_reckoning"

    radius_m = 6371008.8
    delta = distance_m / radius_m
    theta = math.radians(float(cog_deg))
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta))
    lam2 = lam1 + math.atan2(math.sin(theta) * math.sin(delta) * math.cos(phi1),
                             math.cos(delta) - math.sin(phi1) * math.sin(phi2))
    lon2 = (math.degrees(lam2) + 540.0) % 360.0 - 180.0
    lat2 = math.degrees(phi2)
    return lon2, lat2, "single_point_dead_reckoning"


def _restore_doppler_ground_position(
    det_lon: float,
    det_lat: float,
    sog_kn: float | None,
    cog_deg: float | None,
    orbit_heading_deg: float = 192.5,
    incidence_deg: float = 35.0,
    r_slant_m: float = 850000.0,
    v_sat_mps: float = 7125.0,
) -> tuple[float, float, float]:
    """针对星载 SAR 观测，计算船舶运动径向速度引起的多普勒频移，将切片回波坐标反偏拉回海面真实地理经纬度。"""
    if sog_kn is None or sog_kn < 0.5 or cog_deg is None:
        return det_lon, det_lat, 0.0
    speed_mps = float(sog_kn) * 0.514444
    rel_angle_rad = math.radians(float(cog_deg) - orbit_heading_deg)
    v_radial = speed_mps * math.sin(math.radians(incidence_deg)) * math.sin(rel_angle_rad)
    # 方位向几何假性位移 (沿卫星轨道飞行方向): delta_x_az = - (R / V_sat) * v_radial
    delta_x_az_m = - (r_slant_m / v_sat_mps) * v_radial
    # 地面反向拉回位移分量
    dlat = (-delta_x_az_m * math.cos(math.radians(orbit_heading_deg))) / 111132.954
    mid_lat = det_lat + dlat / 2.0
    dlon = (-delta_x_az_m * math.sin(math.radians(orbit_heading_deg))) / (
        111132.954 * max(0.01, math.cos(math.radians(mid_lat)))
    )
    return det_lon + dlon, det_lat + dlat, delta_x_az_m


def _estimate_vessel_position(v_points: list[dict[str, Any]],
                              observed: datetime,
                              max_interpolation_gap_seconds: int = 600
                              ) -> tuple[float, float, float, dict[str, Any], str, tuple[float, float]]:
    """Two-point interpolation preferred, single-point dead reckoning fallback."""
    before = [p for p in v_points if p["_dt"] <= observed]
    after = [p for p in v_points if p["_dt"] >= observed]

    # 1. Exact hit
    if before and after and before[-1]["_dt"] == after[0]["_dt"]:
        p = before[-1]
        return p["lon"], p["lat"], 0.0, p, "exact_observation", (p["lon"], p["lat"])

    # 2. Two-point linear interpolation
    if before and after:
        p1, p2 = before[-1], after[0]
        dt1, dt2 = p1["_dt"], p2["_dt"]
        span = (dt2 - dt1).total_seconds()
        if 0 < span <= max_interpolation_gap_seconds:
            alpha = (observed - dt1).total_seconds() / span
            lon_est = p1["lon"] + alpha * (p2["lon"] - p1["lon"])
            lat_est = p1["lat"] + alpha * (p2["lat"] - p1["lat"])
            ref = p1 if alpha <= 0.5 else p2
            time_delta = min(abs((observed - dt1).total_seconds()), abs((observed - dt2).total_seconds()))
            return lon_est, lat_est, time_delta, ref, "two_point_linear_interpolation", (ref["lon"], ref["lat"])

    # 3. Single-point dead reckoning fallback
    all_points = sorted(v_points, key=lambda p: abs((p["_dt"] - observed).total_seconds()))
    p_near = all_points[0]
    dt_signed = (observed - p_near["_dt"]).total_seconds()
    time_delta = abs(dt_signed)
    sog = p_near.get("_sog")
    cog = p_near.get("_cog")
    rot = p_near.get("_rot")
    lon_est, lat_est, method = _dead_reckon_single(p_near["lon"], p_near["lat"], sog, cog, dt_signed, rot_deg_min=rot)
    return lon_est, lat_est, time_delta, p_near, method, (p_near["lon"], p_near["lat"])


def _local_event_boundary(value: str | None, *, end_exclusive: bool) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    zone = ZoneInfo("America/Chicago")
    try:
        if len(text) == 7:
            year, month = map(int, text.split("-"))
            if end_exclusive:
                year, month = (year + 1, 1) if month == 12 else (year, month + 1)
            local = datetime(year, month, 1, tzinfo=zone)
        else:
            local = datetime.fromisoformat(text).replace(tzinfo=zone)
            if end_exclusive:
                local += timedelta(days=1)
    except ValueError:
        return None
    return local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _event_projection(row: sqlite3.Row) -> tuple[str | None, str | None, dict[str, Any] | None, dict[str, Any]]:
    """Use only explicit normalized fields from an event record; never geocode names implicitly."""
    raw = json.loads(row["record_json"] or "{}")
    start = raw.get("start_utc") or raw.get("valid_start_utc")
    end = raw.get("end_utc") or raw.get("valid_end_utc")
    time_basis = "explicit_utc"
    if not start:
        local_start = raw.get("valid_start_local") or row["valid_start_local"]
        published = raw.get("published_date") or row["published_date"]
        if raw.get("event_time_unknown") and not local_start:
            start = None
            time_basis = "unknown_event_time_publication_not_substituted"
        else:
            start = _local_event_boundary(local_start or published, end_exclusive=False)
            time_basis = ("valid_start_local_America_Chicago" if local_start
                          else "publication_day_America_Chicago" if published else "unknown")
    if not end:
        local_end = raw.get("valid_end_local") or row["valid_end_local"]
        end = _local_event_boundary(local_end, end_exclusive=True) if local_end else start
    geometry = raw.get("geometry") if isinstance(raw.get("geometry"), dict) else None
    if geometry is None and raw.get("lon") is not None and raw.get("lat") is not None:
        geometry = {"type": "Point", "coordinates": [float(raw["lon"]), float(raw["lat"])]}
    references = {key: raw.get(key) for key in ("region", "area", "location", "location_text",
                                                "gis_object_id", "mmsi")
                  if raw.get(key) not in (None, "")}
    references["time_normalization_basis"] = time_basis
    return start, end, geometry, references


def index_confirmed_events(db_path: Path, dataset_id: str) -> dict[str, Any]:
    """Incrementally normalize confirmed event nodes without rescanning AIS history."""
    migrate_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")
            count = 0
            for row in conn.execute("SELECT * FROM confirmed_events WHERE dataset_id=?", (dataset_id,)):
                start, end, geometry, references = _event_projection(row)
                _insert_node(conn, dataset_id, node_id=_stable_id("event", row["event_id"]),
                             node_kind="event", modality="event", object_type="confirmed_event",
                             native_table="confirmed_events", native_key={"event_id": row["event_id"]},
                             start_utc=start, end_utc=end, time_precision=row["time_precision"], geometry=geometry,
                             attributes={"event_name": row["event_name"],
                                         "valid_start_local": row["valid_start_local"],
                                         "valid_end_local": row["valid_end_local"],
                                         "published_date": row["published_date"], **references},
                             quality={"status": "confirmed", "utc_normalized": bool(start),
                                      "timezone_policy": "America/Chicago_for_local_dates"},
                             source_id=None, provenance={"native_table": "confirmed_events",
                                                        "normalization": references["time_normalization_basis"]})
                count += 1
        return {"dataset_id": dataset_id, "event_node_count": count,
                "method_version": "event_time_normalization_v2"}
    finally:
        conn.close()


def _insert_node(conn: sqlite3.Connection, dataset_id: str, *, node_id: str,
                 node_kind: str, modality: str, object_type: str, native_table: str | None,
                 native_key: dict[str, Any], start_utc: str | None, end_utc: str | None,
                 time_precision: str | None, geometry: dict[str, Any] | None,
                 attributes: dict[str, Any], quality: dict[str, Any], source_id: str | None,
                 provenance: dict[str, Any]) -> None:
    bbox = geometry_bbox(geometry)
    conn.execute(
        "INSERT OR REPLACE INTO fusion_nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (dataset_id, node_id, node_kind, modality, object_type, native_table, _json(native_key),
         start_utc, end_utc, time_precision, geometry.get("type") if geometry else None,
         _json(geometry) if geometry else None,
         *(bbox or (None, None, None, None)), _json(attributes), _json(quality), source_id,
         _json(provenance), _now()),
    )


def _track_node(conn: sqlite3.Connection, dataset_id: str, table: str, native_key: dict[str, Any],
                mmsi: str, start_utc: str, end_utc: str, geometry: dict[str, Any],
                attributes: dict[str, Any], quality: dict[str, Any], source_id: str | None) -> None:
    node_id = _stable_id("obs", "ais", table, native_key)
    _insert_node(conn, dataset_id, node_id=node_id, node_kind="observation", modality="ais",
                 object_type="vessel_tracklet", native_table=table, native_key=native_key,
                 start_utc=start_utc, end_utc=end_utc, time_precision="observation_range",
                 geometry=geometry, attributes={**attributes, "mmsi": mmsi}, quality=quality,
                 source_id=source_id, provenance={"native_table": table, "native_key": native_key})
    entity_id = _stable_id("entity", "vessel", "mmsi", mmsi)
    _insert_node(conn, dataset_id, node_id=entity_id, node_kind="entity", modality="fusion",
                 object_type="vessel", native_table="canonical_vessel", native_key={"mmsi": mmsi},
                 start_utc=None, end_utc=None, time_precision=None, geometry=None,
                 attributes={"mmsi": mmsi}, quality={"identity_basis": "reported_mmsi"},
                 source_id=None, provenance={"identity_key": "mmsi"})


def index_bound_source(db_path: Path, dataset_id: str, source_id: str) -> dict[str, Any]:
    """Incrementally project one region-bound intake source into fusion nodes."""
    migrate_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    counts = {"ais": 0, "gis": 0, "weather": 0, "remote_sensing": 0, "event": 0}
    try:
        binding = conn.execute(
            "SELECT source_kind,region_id FROM source_region_bindings WHERE dataset_id=? AND source_id=?",
            (dataset_id, source_id)).fetchone()
        if binding is None:
            raise ValueError("来源尚未绑定研究区域，不能增量建立融合索引")
        kind = binding["source_kind"]
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")
            # Re-indexing the same source is idempotent.  Shared vessel entities
            # are retained because they may be referenced by other sources.
            conn.execute("DELETE FROM fusion_nodes WHERE dataset_id=? AND source_id=? "
                         "AND node_kind!='entity'", (dataset_id, source_id))
            if kind == "ais_csv":
                rows = conn.execute(
                    "SELECT source_id,timestamp_utc,mmsi,lon,lat,vessel_name,vessel_class "
                    "FROM ingested_ais_positions WHERE source_id=? "
                    "ORDER BY mmsi,timestamp_utc,source_row_number", (source_id,)).fetchall()
                current: tuple[str, str] | None = None
                group: list[sqlite3.Row] = []

                def flush() -> None:
                    nonlocal group
                    if not group or current is None:
                        return
                    day, mmsi = current
                    coordinates = [[float(item["lon"]), float(item["lat"])] for item in group]
                    geometry = ({"type": "Point", "coordinates": coordinates[0]}
                                if len(coordinates) == 1 else
                                {"type": "LineString", "coordinates": coordinates})
                    _track_node(
                        conn, dataset_id, "ingested_ais_positions",
                        {"source_id": source_id, "date_utc": day, "mmsi": mmsi}, mmsi,
                        group[0]["timestamp_utc"], group[-1]["timestamp_utc"], geometry,
                        {"vessel_name": next((x["vessel_name"] for x in reversed(group) if x["vessel_name"]), None),
                         "vessel_class": next((x["vessel_class"] for x in reversed(group) if x["vessel_class"]), None),
                         "observation_count": len(group), "position_granularity": "message",
                         "first_observation_utc": group[0]["timestamp_utc"],
                         "last_observation_utc": group[-1]["timestamp_utc"]},
                        {"position_basis": "reported_ais_messages"}, source_id)
                    counts["ais"] += 1

                for row in rows:
                    key = (row["timestamp_utc"][:10], str(row["mmsi"]))
                    if current is not None and key != current:
                        flush()
                        group = []
                    current = key
                    group.append(row)
                flush()
            elif kind == "stac_json":
                for row in conn.execute("SELECT * FROM ingested_stac_scenes WHERE source_id=?", (source_id,)):
                    bbox = json.loads(row["bbox_json"])
                    _insert_node(conn, dataset_id,
                                 node_id=_stable_id("obs", "remote_sensing", "ingested_stac_scenes",
                                                    source_id, row["scene_id"]),
                                 node_kind="observation", modality="remote_sensing", object_type="scene",
                                 native_table="ingested_stac_scenes",
                                 native_key={"source_id": source_id, "scene_id": row["scene_id"]},
                                 start_utc=row["datetime_utc"], end_utc=row["datetime_utc"],
                                 time_precision="instant", geometry=bbox_geometry(bbox),
                                 attributes={"scene_id": row["scene_id"], "collection": row["collection"],
                                             "catalog_url": row["catalog_url"]},
                                 quality={"cloud_cover_pct": row["cloud_cover_pct"],
                                          "asset_status": "catalog_metadata_only"},
                                 source_id=source_id, provenance={"native_table": "ingested_stac_scenes"})
                    counts["remote_sensing"] += 1
            elif kind in {"gis_geojson", "noaa_enc_channels"}:
                for row in conn.execute("SELECT * FROM external_gis_features WHERE source_id=?", (source_id,)):
                    geometry = json.loads(row["geometry_json"])
                    _insert_node(conn, dataset_id,
                                 node_id=_stable_id("place", "external_gis_features", source_id, row["feature_id"]),
                                 node_kind="place", modality="gis", object_type=row["object_type"],
                                 native_table="external_gis_features",
                                 native_key={"source_id": source_id, "feature_id": row["feature_id"]},
                                 start_utc=None, end_utc=None, time_precision=None, geometry=geometry,
                                 attributes={"object_name": row["object_name"],
                                             "source_date": row["source_date"],
                                             "properties": json.loads(row["properties_json"] or "{}")},
                                 quality={"geometry_status": "aligned"}, source_id=source_id,
                                 provenance={"native_table": "external_gis_features"})
                    counts["gis"] += 1
            elif kind == "weather_csv":
                for row in conn.execute("SELECT * FROM weather_observations WHERE source_id=?", (source_id,)):
                    variables = {name: row[name] for name in (
                        "wind_speed_mps", "wind_direction_deg", "wind_gust_mps", "air_pressure_hpa",
                        "air_temperature_c", "precipitation_mm", "visibility_m", "relative_humidity_pct",
                        "cloud_cover_pct", "wave_height_m", "wave_period_s", "wave_direction_deg")
                        if row[name] is not None}
                    key = {"source_id": source_id, "source_row_number": row["source_row_number"]}
                    _insert_node(conn, dataset_id,
                                 node_id=_stable_id("context", "weather", key), node_kind="context",
                                 modality="weather", object_type="weather_observation",
                                 native_table="weather_observations", native_key=key,
                                 start_utc=row["valid_time_utc"], end_utc=row["valid_time_utc"],
                                 time_precision="observation",
                                 geometry={"type": "Point", "coordinates": [row["lon"], row["lat"]]},
                                 attributes={"location_id": row["location_id"], **variables},
                                 quality={"quality_flag": "accepted"}, source_id=source_id,
                                 provenance={"native_table": "weather_observations"})
                    counts["weather"] += 1
            elif kind == "remote_geotiff":
                source = conn.execute("SELECT scene_id FROM remote_intake_sources "
                                      "WHERE dataset_id=? AND source_id=?", (dataset_id, source_id)).fetchone()
                if source is None:
                    raise ValueError("找不到 GeoTIFF 来源记录")
                row = conn.execute("SELECT * FROM remote_sensing_aligned_scenes "
                                   "WHERE dataset_id=? AND scene_id=?",
                                   (dataset_id, source["scene_id"])).fetchone()
                if row is None:
                    raise ValueError("找不到 GeoTIFF 对齐场景")
                bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
                _insert_node(conn, dataset_id,
                             node_id=_stable_id("obs", "remote_sensing", "remote_sensing_aligned_scenes",
                                                row["scene_id"]),
                             node_kind="observation", modality="remote_sensing", object_type="scene",
                             native_table="remote_sensing_aligned_scenes",
                             native_key={"scene_id": row["scene_id"]}, start_utc=row["start_utc"],
                             end_utc=row["end_utc"] or row["start_utc"],
                             time_precision=row["temporal_precision"],
                             geometry=bbox_geometry(bbox) if bbox else None,
                             attributes={"scene_id": row["scene_id"], "collection": row["collection"],
                                         "asset_relative_path": row["source_relative_path"]},
                             quality={"quality_available": bool(row["quality_available"]),
                                      "asset_role": row["asset_role"]}, source_id=source_id,
                             provenance={"native_table": "remote_sensing_aligned_scenes",
                                         "sha256": row["sha256"]})
                counts["remote_sensing"] += 1
            elif kind == "event_text":
                rows = conn.execute(
                    "SELECT DISTINCT e.* FROM confirmed_events e "
                    "JOIN event_evidence_links l ON l.dataset_id=e.dataset_id AND l.event_id=e.event_id "
                    "WHERE e.dataset_id=? AND l.source_id=?",
                    (dataset_id, source_id)).fetchall()
                for row in rows:
                    start, end, geometry, references = _event_projection(row)
                    _insert_node(
                        conn, dataset_id, node_id=_stable_id("event", row["event_id"]),
                        node_kind="event", modality="event", object_type="confirmed_event",
                        native_table="confirmed_events", native_key={"event_id": row["event_id"]},
                        start_utc=start, end_utc=end, time_precision=row["time_precision"],
                        geometry=geometry,
                        attributes={"event_name": row["event_name"],
                                    "valid_start_local": row["valid_start_local"],
                                    "valid_end_local": row["valid_end_local"],
                                    "published_date": row["published_date"], **references},
                        quality={"status": "confirmed", "utc_normalized": bool(start),
                                 "evidence_source": "event_text_intake"},
                        source_id=source_id,
                        provenance={"native_table": "confirmed_events",
                                    "source_id": source_id,
                                    "normalization": references["time_normalization_basis"]})
                    counts["event"] += 1
            else:
                raise ValueError(f"暂不支持增量索引来源类型: {kind}")
        return {"dataset_id": dataset_id, "source_id": source_id,
                "source_kind": kind, "region_id": binding["region_id"],
                "counts_by_modality": {key: value for key, value in counts.items() if value},
                "node_count": sum(counts.values()), "method_version": "source_incremental_index_v1"}
    finally:
        conn.close()


def rebuild_fusion_index(db_path: Path, dataset_id: str, date_utc: str = "") -> dict[str, Any]:
    """Project aligned modality records into the common node layer.

    The derived layer for the selected dataset is rebuilt transactionally.  A
    date filter bounds dynamic observations; static GIS places remain indexed.
    """
    migrate_database(db_path)
    bounds = _day_bounds(date_utc) if date_utc else None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute("SELECT 1 FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone() is None:
            raise ValueError(f"数据库中找不到 dataset_id: {dataset_id}")
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM fusion_nodes WHERE dataset_id=?", (dataset_id,))

            def project_message_tracks(rows: Iterable[sqlite3.Row], native_table: str) -> None:
                current_key: tuple[str | None, str, str] | None = None
                group: list[sqlite3.Row] = []

                def flush() -> None:
                    if not group or current_key is None:
                        return
                    source_id, day, mmsi = current_key
                    coordinates = [[float(item["lon"]), float(item["lat"])] for item in group]
                    geometry = ({"type": "Point", "coordinates": coordinates[0]}
                                if len(coordinates) == 1 else
                                {"type": "LineString", "coordinates": coordinates})
                    vessel_name = next((item["vessel_name"] for item in reversed(group)
                                        if item["vessel_name"]), None)
                    vessel_class = next((item["vessel_class"] for item in reversed(group)
                                         if item["vessel_class"]), None)
                    native_key = {"date_utc": day, "mmsi": mmsi}
                    if source_id is not None:
                        native_key["source_id"] = source_id
                    _track_node(
                        conn, dataset_id, native_table, native_key, mmsi,
                        group[0]["timestamp_utc"], group[-1]["timestamp_utc"], geometry,
                        {"vessel_name": vessel_name, "vessel_class": vessel_class,
                         "observation_count": len(group), "position_granularity": "message",
                         "first_observation_utc": group[0]["timestamp_utc"],
                         "last_observation_utc": group[-1]["timestamp_utc"]},
                        {"position_basis": "reported_ais_messages"}, source_id)

                for row in rows:
                    key = (row["source_id"], row["timestamp_utc"][:10], str(row["mmsi"]))
                    if current_key is not None and key != current_key:
                        flush()
                        group = []
                    current_key = key
                    group.append(row)
                flush()

            base_sql = ("SELECT NULL AS source_id,timestamp_utc,mmsi,lon,lat,vessel_name,vessel_class "
                        "FROM ais_messages WHERE dataset_id=?")
            base_args: list[Any] = [dataset_id]
            if bounds:
                base_sql += " AND timestamp_utc>=? AND timestamp_utc<?"
                base_args.extend(bounds)
            base_sql += " ORDER BY mmsi,timestamp_utc,source_row_number"
            project_message_tracks(conn.execute(base_sql, base_args), "ais_messages")

            ingested_sql = ("SELECT source_id,timestamp_utc,mmsi,lon,lat,vessel_name,vessel_class "
                            "FROM ingested_ais_positions WHERE 1=1")
            ingested_args: list[Any] = []
            if bounds:
                ingested_sql += " AND timestamp_utc>=? AND timestamp_utc<?"
                ingested_args.extend(bounds)
            ingested_sql += " ORDER BY source_id,mmsi,timestamp_utc,source_row_number"
            project_message_tracks(conn.execute(ingested_sql, ingested_args), "ingested_ais_positions")

            for row in conn.execute("SELECT * FROM gis_objects WHERE dataset_id=?", (dataset_id,)):
                geometry = json.loads(row["geometry_json"])
                _insert_node(conn, dataset_id, node_id=_stable_id("place", "gis_objects", row["object_id"]),
                             node_kind="place", modality="gis", object_type=row["object_type"],
                             native_table="gis_objects", native_key={"object_id": row["object_id"]},
                             start_utc=None, end_utc=None, time_precision=None, geometry=geometry,
                             attributes={"object_name": row["object_name"], "source_date": row["source_date"]},
                             quality={"geometry_status": "aligned"}, source_id=row["source_dataset"],
                             provenance={"source_authority": row["source_authority"],
                                         "source_url": row["source_url"]})

            remote_queries = [
                ("remote_sensing_scenes", "datetime_utc", "scene_id", "bbox_json"),
                ("remote_sensing_aligned_scenes", "start_utc", "scene_id", "bbox_json"),
            ]
            for table, time_column, id_column, bbox_column in remote_queries:
                sql = f"SELECT * FROM {table} WHERE dataset_id=?"
                args: list[Any] = [dataset_id]
                if bounds:
                    sql += f" AND {time_column}>=? AND {time_column}<?"
                    args.extend(bounds)
                for row in conn.execute(sql, args):
                    bbox_value = json.loads(row[bbox_column]) if row[bbox_column] else None
                    geometry = bbox_geometry(bbox_value) if bbox_value else None
                    start = row[time_column]
                    end = row["end_utc"] if table == "remote_sensing_aligned_scenes" else start
                    collection = row["collection"]
                    quality = ({"cloud_cover_pct": row["cloud_cover_pct"]}
                               if table == "remote_sensing_scenes"
                               else {"quality_available": bool(row["quality_available"]),
                                     "asset_role": row["asset_role"]})
                    _insert_node(conn, dataset_id,
                                 node_id=_stable_id("obs", "remote_sensing", table, row[id_column]),
                                 node_kind="observation", modality="remote_sensing",
                                 object_type="scene", native_table=table,
                                 native_key={"scene_id": row[id_column]}, start_utc=start, end_utc=end,
                                 time_precision=(row["temporal_precision"] if table == "remote_sensing_aligned_scenes"
                                                 else "instant"), geometry=geometry,
                                 attributes={"scene_id": row[id_column], "collection": collection},
                                 quality=quality, source_id=collection,
                                 provenance={"native_table": table})

            detection_sql = "SELECT * FROM remote_sensing_detections WHERE dataset_id=?"
            detection_args: list[Any] = [dataset_id]
            if bounds:
                detection_sql += " AND observed_at_utc>=? AND observed_at_utc<?"
                detection_args.extend(bounds)
            for row in conn.execute(detection_sql, detection_args):
                attributes = json.loads(row["attributes_json"] or "{}")
                attributes.update({"detection_id": row["detection_id"], "scene_id": row["scene_id"],
                                   "length_m": row["length_m"], "width_m": row["width_m"]})
                _insert_node(conn, dataset_id, node_id=_stable_id("obs", "remote_detection", row["detection_id"]),
                             node_kind="observation", modality="remote_sensing",
                             object_type=row["object_type"], native_table="remote_sensing_detections",
                             native_key={"detection_id": row["detection_id"]},
                             start_utc=row["observed_at_utc"], end_utc=row["observed_at_utc"],
                             time_precision="instant",
                             geometry={"type": "Point", "coordinates": [row["lon"], row["lat"]]},
                             attributes=attributes,
                             quality={"detection_confidence": row["detection_confidence"]},
                             source_id=row["scene_id"], provenance=json.loads(row["provenance_json"] or "{}"))

            weather_sql = "SELECT * FROM weather_observations"
            weather_args: list[Any] = []
            if bounds:
                weather_sql += " WHERE valid_time_utc>=? AND valid_time_utc<?"
                weather_args.extend(bounds)
            for row in conn.execute(weather_sql, weather_args):
                geometry = {"type": "Point", "coordinates": [row["lon"], row["lat"]]}
                key = {"source_id": row["source_id"], "source_row_number": row["source_row_number"]}
                variables = {name: row[name] for name in (
                    "wind_speed_mps", "wind_direction_deg", "wind_gust_mps", "air_pressure_hpa",
                    "air_temperature_c", "precipitation_mm", "visibility_m", "relative_humidity_pct",
                    "cloud_cover_pct", "wave_height_m", "wave_period_s", "wave_direction_deg")
                    if row[name] is not None}
                _insert_node(conn, dataset_id, node_id=_stable_id("context", "weather", key),
                             node_kind="context", modality="weather", object_type="weather_observation",
                             native_table="weather_observations", native_key=key,
                             start_utc=row["valid_time_utc"], end_utc=row["valid_time_utc"],
                             time_precision="observation", geometry=geometry,
                             attributes={"location_id": row["location_id"], **variables},
                             quality={"quality_flag": "accepted"}, source_id=row["source_id"],
                             provenance={"native_table": "weather_observations"})

            for row in conn.execute("SELECT * FROM confirmed_events WHERE dataset_id=?", (dataset_id,)):
                start, end, geometry, references = _event_projection(row)
                _insert_node(conn, dataset_id, node_id=_stable_id("event", row["event_id"]),
                             node_kind="event", modality="event", object_type="confirmed_event",
                             native_table="confirmed_events", native_key={"event_id": row["event_id"]},
                             start_utc=start, end_utc=end, time_precision=row["time_precision"], geometry=geometry,
                             attributes={"event_name": row["event_name"],
                                         "valid_start_local": row["valid_start_local"],
                                         "valid_end_local": row["valid_end_local"],
                                         "published_date": row["published_date"], **references},
                             quality={"status": "confirmed", "utc_normalized": bool(start)}, source_id=None,
                             provenance={"native_table": "confirmed_events"})

        supplemental = [row[0] for row in conn.execute(
            "SELECT source_id FROM source_region_bindings WHERE dataset_id=? "
            "AND source_kind IN ('stac_json','gis_geojson','noaa_enc_channels','remote_geotiff')",
            (dataset_id,)).fetchall()]
        # The main projection transaction is complete.  Add source-specific
        # tables that are not part of the historical dataset schema.
        for source_id in supplemental:
            index_bound_source(db_path, dataset_id, source_id)
        counts = dict(conn.execute("SELECT modality,COUNT(*) FROM fusion_nodes WHERE dataset_id=? "
                                   "GROUP BY modality", (dataset_id,)).fetchall())
        return {"dataset_id": dataset_id, "date_utc": date_utc or None,
                "node_count": sum(counts.values()), "counts_by_modality": counts}
    finally:
        conn.close()


def index_remote_scene(db_path: Path, dataset_id: str, scene_id: str) -> dict[str, Any]:
    """Incrementally project one aligned remote scene without rebuilding AIS/GIS history."""
    migrate_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run_id = str(uuid4())
    try:
        row = conn.execute(
            "SELECT * FROM remote_sensing_aligned_scenes WHERE dataset_id=? AND scene_id=?",
            (dataset_id, scene_id)).fetchone()
        if row is None:
            raise ValueError(f"找不到已对齐遥感场景: {scene_id}")
        bbox_value = json.loads(row["bbox_json"])
        geometry = bbox_geometry(bbox_value)
        node_id = _stable_id("obs", "remote_sensing", "remote_sensing_aligned_scenes", scene_id)
        with conn:
            conn.execute("PRAGMA foreign_keys=ON")
            _insert_node(
                conn, dataset_id, node_id=node_id, node_kind="observation",
                modality="remote_sensing", object_type="scene",
                native_table="remote_sensing_aligned_scenes", native_key={"scene_id": scene_id},
                start_utc=row["start_utc"], end_utc=row["end_utc"],
                time_precision=row["temporal_precision"], geometry=geometry,
                attributes={"scene_id": scene_id, "collection": row["collection"]},
                quality={"quality_available": bool(row["quality_available"]),
                         "asset_role": row["asset_role"]},
                source_id=row["collection"],
                provenance={"native_table": "remote_sensing_aligned_scenes"})
            conn.execute("INSERT INTO fusion_runs VALUES (?,?,?,?,?,?,?,?,?)",
                         (dataset_id, run_id, _json(["COVERS"]), "remote_scene_incremental_v1",
                          _json({"scene_id": scene_id}), "running", _now(), None, _json({})))
            covers = 0
            scene_bbox = tuple(float(value) for value in bbox_value)
            places = conn.execute(
                "SELECT * FROM fusion_nodes WHERE dataset_id=? AND modality='gis' "
                "AND bbox_min_lon<=? AND bbox_max_lon>=? AND bbox_min_lat<=? AND bbox_max_lat>=?",
                (dataset_id, scene_bbox[2], scene_bbox[0], scene_bbox[3], scene_bbox[1])).fetchall()
            for place in places:
                place_bbox = (place["bbox_min_lon"], place["bbox_min_lat"],
                              place["bbox_max_lon"], place["bbox_max_lat"])
                if None in place_bbox:
                    continue
                ratio = _bbox_intersection_ratio(scene_bbox, place_bbox)
                if ratio <= 0:
                    continue
                _store_relation(conn, dataset_id, run_id, node_id, "COVERS", place["node_id"],
                                ratio, "bbox_intersection", {"gis_bbox_coverage_ratio": ratio},
                                {"rule": "scene_bbox_intersects_gis_bbox",
                                 "caveat": "bbox intersection is not pixel-level coverage"},
                                "remote_scene_incremental_v1")
                covers += 1
            summary = {"scene_nodes": 1, "COVERS": covers}
            conn.execute("UPDATE fusion_runs SET status='completed',completed_at_utc=?,summary_json=? "
                         "WHERE dataset_id=? AND run_id=?", (_now(), _json(summary), dataset_id, run_id))
        return {"dataset_id": dataset_id, "scene_id": scene_id, "run_id": run_id,
                "method_version": "remote_scene_incremental_v1", "summary": summary}
    finally:
        conn.close()


def _store_relation(conn: sqlite3.Connection, dataset_id: str, run_id: str, subject: str,
                    predicate: str, object_id: str, confidence: float, evidence_type: str,
                    evidence_value: dict[str, Any], details: dict[str, Any],
                    method_version: str = METHOD_VERSION) -> None:
    candidate_id = _stable_id("candidate", run_id, subject, predicate, object_id)
    relation_id = _stable_id("relation", run_id, subject, predicate, object_id)
    conn.execute("INSERT OR REPLACE INTO fusion_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (dataset_id, run_id, candidate_id, subject, predicate, object_id,
                  None, confidence, None, None, confidence, confidence, "accepted", _json(details)))
    conn.execute("INSERT INTO fusion_relations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (dataset_id, relation_id, run_id, subject, predicate, object_id,
                  confidence, "accepted", None, None, method_version, _now()))
    conn.execute("INSERT INTO fusion_evidence VALUES (?,?,?,?,?,?,?,?,?)",
                 (dataset_id, relation_id, _stable_id("evidence", relation_id, evidence_type),
                  evidence_type, _json(evidence_value), confidence, 1.0, "support", _json(details)))


def _store_scored_candidate(conn: sqlite3.Connection, dataset_id: str, run_id: str,
                            subject: str, predicate: str, object_id: str, scores: dict[str, float | None],
                            raw_score: float, decision: str, details: dict[str, Any]) -> None:
    candidate_id = _stable_id("candidate", run_id, subject, predicate, object_id)
    conn.execute("INSERT INTO fusion_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (dataset_id, run_id, candidate_id, subject, predicate, object_id,
                  scores.get("time"), scores.get("spatial"), scores.get("semantic"),
                  scores.get("behavior"), scores.get("quality"), raw_score, decision, _json(details)))


def build_open_set_associations(db_path: Path, dataset_id: str,
                                max_distance_km: float = 10.0,
                                accept_threshold: float = 0.75) -> dict[str, Any]:
    """Associate cross-source tracklets and remote detections, retaining rejections.

    Scores are transparent rule evidence, not calibrated probabilities.  A
    subject remains unresolved when no candidate reaches the threshold.
    """
    migrate_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run_id = str(uuid4())
    summary = {"SAME_VESSEL_AS": 0, "POSSIBLY_OBSERVES": 0, "unresolved": 0, "rejected": 0}
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("INSERT INTO fusion_runs VALUES (?,?,?,?,?,?,?,?,?)",
                         (dataset_id, run_id, _json(["SAME_VESSEL_AS", "POSSIBLY_OBSERVES"]),
                          IDENTITY_METHOD_VERSION,
                          _json({"max_distance_km": max_distance_km, "accept_threshold": accept_threshold,
                                 "confidence_kind": "uncalibrated_rule_score"}),
                          "running", _now(), None, _json({})))
            tracks = conn.execute("SELECT * FROM fusion_nodes WHERE dataset_id=? AND modality='ais' "
                                  "AND object_type='vessel_tracklet'", (dataset_id,)).fetchall()
            for index, left in enumerate(tracks):
                best: tuple[float, sqlite3.Row, dict[str, Any], dict[str, float]] | None = None
                margin = max_distance_km / 111.0
                for right in tracks[index + 1:]:
                    if (left["source_id"] or left["native_table"]) == (right["source_id"] or right["native_table"]):
                        continue
                    if left["end_utc"] < right["start_utc"] or right["end_utc"] < left["start_utc"]:
                        continue
                    if not _bbox_overlaps(
                            (left["bbox_min_lon"] - margin, left["bbox_min_lat"] - margin,
                             left["bbox_max_lon"] + margin, left["bbox_max_lat"] + margin),
                            (right["bbox_min_lon"], right["bbox_min_lat"],
                             right["bbox_max_lon"], right["bbox_max_lat"])):
                        continue
                    lp = list(_coordinate_pairs(json.loads(left["geometry_json"])["coordinates"]))
                    rp = list(_coordinate_pairs(json.loads(right["geometry_json"])["coordinates"]))
                    lp = lp[::max(1, len(lp) // 12)]
                    rp = rp[::max(1, len(rp) // 12)]
                    distance = min(_haversine_km(a, b) for a in lp for b in rp)
                    spatial = max(0.0, 1.0 - distance / max_distance_km)
                    la, ra = json.loads(left["attributes_json"]), json.loads(right["attributes_json"])
                    semantic_parts = []
                    for key in ("vessel_name", "vessel_class"):
                        if la.get(key) and ra.get(key):
                            semantic_parts.append(float(str(la[key]).casefold() == str(ra[key]).casefold()))
                    semantic = sum(semantic_parts) / len(semantic_parts) if semantic_parts else 0.5
                    raw = 0.75 * spatial + 0.25 * semantic
                    details = {"distance_km": distance, "confidence_kind": "uncalibrated_rule_score",
                               "mmsi_not_used_for_scoring": True}
                    decision = "candidate" if distance <= max_distance_km and raw >= accept_threshold else "rejected"
                    _store_scored_candidate(conn, dataset_id, run_id, left["node_id"], "SAME_VESSEL_AS",
                                            right["node_id"], {"time": 1.0, "spatial": spatial,
                                                               "semantic": semantic, "quality": 1.0},
                                            raw, decision, details)
                    if decision == "rejected":
                        summary["rejected"] += 1
                    elif best is None or raw > best[0]:
                        best = (raw, right, details, {"spatial": spatial, "semantic": semantic})
                if best:
                    raw, right, details, scores = best
                    _store_relation(conn, dataset_id, run_id, left["node_id"], "SAME_VESSEL_AS",
                                    right["node_id"], raw, "spatiotemporal_similarity",
                                    scores, details)
                    summary["SAME_VESSEL_AS"] += 1
                else:
                    summary["unresolved"] += 1

            # Remote detections are deliberately excluded here. They require exact AIS message
            # timestamps and are handled only by associate_remote_detections().
            conn.execute("UPDATE fusion_runs SET status='completed',completed_at_utc=?,summary_json=? "
                         "WHERE dataset_id=? AND run_id=?", (_now(), _json(summary), dataset_id, run_id))
        return {"dataset_id": dataset_id, "run_id": run_id, "method_version": IDENTITY_METHOD_VERSION,
                "confidence_kind": "uncalibrated_rule_score", "summary": summary}
    finally:
        conn.close()


def associate_remote_detections(db_path: Path, dataset_id: str, *,
                                time_window_minutes: int = 30,
                                max_distance_km: float = 5.0,
                                accept_threshold: float = 0.70,
                                base_match_gate_km: float = 0.50,
                                max_interpolation_gap_minutes: int = 10,
                                max_dead_reckoning_seconds: int = 180) -> dict[str, Any]:
    """Match instantaneous remote detections to real AIS messages, never hourly representatives."""
    migrate_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run_id = str(uuid4())
    method_version = "remote_ais_uncertainty_gate_v3"
    summary = {"DETECTED_IN": 0, "POSSIBLY_OBSERVES": 0, "candidate": 0,
               "rejected": 0, "unresolved": 0}
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys=ON")
            old_runs = [row[0] for row in conn.execute(
                "SELECT run_id FROM fusion_runs WHERE dataset_id=? AND method_version IN "
                "('remote_ais_nearest_hour_v1','remote_ais_nearest_message_v2',"
                "'remote_ais_uncertainty_gate_v3')", (dataset_id,))]
            for old_run in old_runs:
                conn.execute("DELETE FROM fusion_evidence WHERE dataset_id=? AND relation_id IN "
                             "(SELECT relation_id FROM fusion_relations WHERE dataset_id=? AND run_id=?)",
                             (dataset_id, dataset_id, old_run))
                conn.execute("DELETE FROM fusion_relations WHERE dataset_id=? AND run_id=?",
                             (dataset_id, old_run))
                conn.execute("DELETE FROM fusion_candidates WHERE dataset_id=? AND run_id=?",
                             (dataset_id, old_run))
                conn.execute("DELETE FROM fusion_runs WHERE dataset_id=? AND run_id=?",
                             (dataset_id, old_run))
            conn.execute("DELETE FROM fusion_nodes WHERE dataset_id=? AND native_table IN "
                         "('ais_messages','ingested_ais_positions') AND object_type='ais_message'",
                         (dataset_id,))
            conn.execute("DELETE FROM fusion_nodes WHERE dataset_id=? "
                         "AND native_table='remote_sensing_detections'", (dataset_id,))
            # A detection task may be rerun after its algorithm is upgraded.
            # Only the newest completed output of each job is authoritative;
            # otherwise stale edge artefacts from an older detector version
            # continue to participate in matching.  Manually imported
            # detections (with no run-output row) remain supported.
            detections = conn.execute(
                "SELECT d.* FROM remote_sensing_detections d WHERE d.dataset_id=? AND ("
                "NOT EXISTS (SELECT 1 FROM remote_detection_run_outputs o "
                "            WHERE o.dataset_id=d.dataset_id AND o.detection_id=d.detection_id) "
                "OR EXISTS (SELECT 1 FROM remote_detection_run_outputs o "
                "           JOIN remote_detection_runs r ON r.dataset_id=o.dataset_id AND r.run_id=o.run_id "
                "           WHERE o.dataset_id=d.dataset_id AND o.detection_id=d.detection_id "
                "             AND r.status='completed' AND r.run_id=("
                "               SELECT r2.run_id FROM remote_detection_runs r2 "
                "               WHERE r2.dataset_id=r.dataset_id AND r2.job_id=r.job_id "
                "                 AND r2.status='completed' "
                "               ORDER BY r2.completed_at_utc DESC, r2.started_at_utc DESC LIMIT 1)))",
                (dataset_id,)).fetchall()
            for row in detections:
                attributes = json.loads(row["attributes_json"] or "{}")
                attributes.update({"detection_id": row["detection_id"], "scene_id": row["scene_id"],
                                   "length_m": row["length_m"], "width_m": row["width_m"]})
                _insert_node(conn, dataset_id,
                             node_id=_stable_id("obs", "remote_detection", row["detection_id"]),
                             node_kind="observation", modality="remote_sensing",
                             object_type=row["object_type"], native_table="remote_sensing_detections",
                             native_key={"detection_id": row["detection_id"]},
                             start_utc=row["observed_at_utc"], end_utc=row["observed_at_utc"],
                             time_precision="instant",
                             geometry={"type": "Point", "coordinates": [row["lon"], row["lat"]]},
                             attributes=attributes,
                             quality={"detection_confidence": row["detection_confidence"]},
                             source_id=row["scene_id"], provenance=json.loads(row["provenance_json"] or "{}"))

            conn.execute("INSERT INTO fusion_runs VALUES (?,?,?,?,?,?,?,?,?)",
                         (dataset_id, run_id, _json(["DETECTED_IN", "POSSIBLY_OBSERVES"]),
                          method_version,
                          _json({"time_window_minutes": time_window_minutes,
                                 "max_distance_km": max_distance_km,
                                 "accept_threshold": accept_threshold,
                                 "base_match_gate_km": base_match_gate_km,
                                 "max_interpolation_gap_minutes": max_interpolation_gap_minutes,
                                 "max_dead_reckoning_seconds": max_dead_reckoning_seconds,
                                 "ais_granularity": "message",
                                 "confidence_kind": "uncalibrated_rule_score"}),
                          "running", _now(), None, _json({})))
            detection_nodes = conn.execute(
                "SELECT * FROM fusion_nodes WHERE dataset_id=? AND native_table='remote_sensing_detections'",
                (dataset_id,)).fetchall()
            scene_nodes = {}
            for scene in conn.execute("SELECT * FROM fusion_nodes WHERE dataset_id=? "
                                      "AND modality='remote_sensing' AND object_type='scene'", (dataset_id,)):
                scene_nodes[json.loads(scene["attributes_json"]).get("scene_id")] = scene
            window_seconds = time_window_minutes * 60
            point_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
            vessel_reckoned_cache: dict[tuple[str, str], dict[str, tuple[float, float, dict[str, Any], tuple[float, float], str, float]]] = {}
            pending_acceptances: list[tuple[float, str, str, str,
                                            dict[str, Any], dict[str, float]]] = []
            detection_node_ids: set[str] = set()
            inserted_observation_ids: set[str] = set()
            deg_lat_margin = max_distance_km / 110.0
            for detection in detection_nodes:
                detection_node_ids.add(detection["node_id"])
                attrs = json.loads(detection["attributes_json"])
                scene = scene_nodes.get(attrs.get("scene_id"))
                if scene:
                    quality = json.loads(detection["quality_json"]).get("detection_confidence")
                    _store_relation(conn, dataset_id, run_id, detection["node_id"], "DETECTED_IN",
                                    scene["node_id"], float(quality if quality is not None else 1.0),
                                    "scene_foreign_key", {"scene_id": attrs.get("scene_id")},
                                    {"rule": "explicit_scene_id"})
                    summary["DETECTED_IN"] += 1
                observed = datetime.fromisoformat(detection["start_utc"].replace("Z", "+00:00"))
                lower = (observed - timedelta(minutes=time_window_minutes)).isoformat().replace("+00:00", "Z")
                upper = (observed + timedelta(minutes=time_window_minutes)).isoformat().replace("+00:00", "Z")
                cache_key = (lower, upper)
                if cache_key not in point_cache:
                    points: list[dict[str, Any]] = []
                    for row in conn.execute(
                        "SELECT source_day_utc,source_row_number,timestamp_utc,mmsi,lon,lat,"
                        "vessel_name,vessel_class,sog_kn,cog_deg FROM ais_messages WHERE dataset_id=? "
                        "AND timestamp_utc>=? AND timestamp_utc<=?",
                        (dataset_id, lower, upper)):
                        d = dict(row)
                        points.append({**d, "native_table": "ais_messages", "source_id": None,
                                       "extra_attributes_json": "{}",
                                       "_dt": datetime.fromisoformat(d["timestamp_utc"].replace("Z", "+00:00")),
                                       "_sog": d.get("sog_kn"), "_cog": d.get("cog_deg"),
                                       "_rot": None, "_length": None, "_width": None})
                    for row in conn.execute(
                        "SELECT source_id,source_row_number,timestamp_utc,mmsi,lon,lat,vessel_name,"
                        "sog_knots,cog_degrees,extra_attributes_json FROM ingested_ais_positions WHERE timestamp_utc>=? "
                        "AND timestamp_utc<=?", (lower, upper)):
                        d = dict(row)
                        extra_obj = json.loads(d.get("extra_attributes_json") or "{}")
                        rot_val = extra_obj.get("rot") or extra_obj.get("rate_of_turn")
                        len_val = extra_obj.get("length")
                        if len_val is None and extra_obj.get("to_bow") and extra_obj.get("to_stern"):
                            try:
                                len_val = float(extra_obj["to_bow"]) + float(extra_obj["to_stern"])
                            except Exception:
                                pass
                        wid_val = extra_obj.get("width")
                        if wid_val is None and extra_obj.get("to_port") and extra_obj.get("to_starboard"):
                            try:
                                wid_val = float(extra_obj["to_port"]) + float(extra_obj["to_starboard"])
                            except Exception:
                                pass
                        points.append({**d, "native_table": "ingested_ais_positions",
                                       "source_day_utc": None, "vessel_class": None,
                                       "_dt": datetime.fromisoformat(d["timestamp_utc"].replace("Z", "+00:00")),
                                       "_sog": d.get("sog_knots"), "_cog": d.get("cog_degrees"),
                                       "_rot": float(rot_val) if rot_val is not None else None,
                                       "_length": float(len_val) if len_val is not None else None,
                                       "_width": float(wid_val) if wid_val is not None else None})
                    point_cache[cache_key] = points

                    # 性能优化：在时间窗口内对所有船舶预先推算船位并缓存，避免每个 detection 重复 30 万次排序和死推
                    by_v_temp: dict[str, list[dict[str, Any]]] = {}
                    for p in points:
                        by_v_temp.setdefault(str(p["mmsi"]), []).append(p)
                    reckoned_vessels: dict[str, tuple[float, float, dict[str, Any], tuple[float, float], str, float]] = {}
                    for v_key, v_pts in by_v_temp.items():
                        v_pts_sorted = sorted(v_pts, key=lambda item: item["_dt"])
                        r_lon, r_lat, dt_sec, ref_pt, meth, raw_p = _estimate_vessel_position(
                            v_pts_sorted, observed,
                            max_interpolation_gap_seconds=max_interpolation_gap_minutes * 60
                        )
                        reckoned_vessels[v_key] = (dt_sec, ref_pt, (r_lon, r_lat), meth, raw_p)
                    vessel_reckoned_cache[cache_key] = reckoned_vessels

                det_point = tuple(json.loads(detection["geometry_json"])["coordinates"])
                deg_lon_margin = max_distance_km / (111.0 * max(0.1, math.cos(math.radians(det_point[1]))))
                quality_value = json.loads(detection["quality_json"]).get("detection_confidence")
                quality = float(quality_value if quality_value is not None else 0.5)

                reckoned_map = vessel_reckoned_cache[cache_key]
                best_by_vessel: dict[str, tuple[float, float, dict[str, Any], tuple[float, float], str, float, float, tuple[float, float], float]] = {}
                for vessel_key, (delta_seconds, ref_point, reckoned_pos, method, raw_pos) in reckoned_map.items():
                    # 严格 15 分钟绝对硬时间窗口防御 (900秒)
                    if delta_seconds > 900.0:
                        continue
                    # 针对星载 SAR 观测，计算径向速度引起的多普勒频移，将切片回波坐标反偏拉回真实海面经纬度
                    sog_pt = ref_point.get("_sog")
                    cog_pt = ref_point.get("_cog")
                    det_corr_lon, det_corr_lat, doppler_shift_m = _restore_doppler_ground_position(
                        det_point[0], det_point[1], sog_pt, cog_pt
                    )
                    corr_det_point = (det_corr_lon, det_corr_lat)
                    if abs(corr_det_point[1] - reckoned_pos[1]) > deg_lat_margin or abs(corr_det_point[0] - reckoned_pos[0]) > deg_lon_margin:
                        continue
                    distance = _haversine_km(corr_det_point, reckoned_pos)
                    raw_distance = _haversine_km(corr_det_point, raw_pos)
                    uncorrected_det_dist = _haversine_km(det_point, reckoned_pos)
                    if distance > max_distance_km:
                        continue
                    current = best_by_vessel.get(vessel_key)
                    if current is None or (delta_seconds, distance) < (current[0], current[1]):
                        best_by_vessel[vessel_key] = (delta_seconds, distance, ref_point,
                                                     reckoned_pos, method, raw_distance,
                                                     doppler_shift_m, corr_det_point, uncorrected_det_dist)
                scored = []
                for vessel_key, (delta_seconds, distance, point, reckoned_pos, reckoning_method, raw_distance, doppler_shift_m, corr_det_point, uncorr_det_dist) in best_by_vessel.items():
                    is_prior_matched_target = bool(attrs.get("is_prior_match") and
                                                   str(attrs.get("matched_mmsi")) == str(vessel_key))
                    # 单点航位推算时间上限防线 (默认 180 秒)
                    if (reckoning_method in ("single_point_dead_reckoning", "circular_arc_dead_reckoning")
                            and delta_seconds > max_dead_reckoning_seconds):
                        continue

                    # 统一硬门限 (彻底移除针对 prior 的 0.80km 放宽与无脑豁免)
                    projection_allowance = (min(delta_seconds, max_dead_reckoning_seconds)
                                            * 0.0005
                                            if reckoning_method in ("single_point_dead_reckoning", "circular_arc_dead_reckoning") else 0.0)
                    match_gate_km = min(max_distance_km,
                                        base_match_gate_km + projection_allowance)
                    if distance > match_gate_km:
                        continue

                    # 船体物理尺度一致性校验 (若双方长度均 > 10m 且相差 3 倍以上，判定为严重物理背离)
                    det_len = attrs.get("length_m")
                    det_wid = attrs.get("width_m")
                    ais_len = point.get("_length")
                    ais_wid = point.get("_width")
                    scale_score = 1.0
                    if det_len is not None and ais_len is not None and det_len > 10.0 and ais_len > 10.0:
                        len_ratio = max(det_len, ais_len) / min(det_len, ais_len)
                        if len_ratio > 3.0:
                            continue
                        scale_score = max(0.2, 1.0 - (len_ratio - 1.0) / 2.0)

                    # 航向姿态物理一致性校验 (模 180° 主轴偏差若 > 65°，判定为严重物理背离)
                    det_orient = attrs.get("orientation_deg")
                    ais_cog = point.get("_cog")
                    ais_sog = point.get("_sog")
                    orient_score = 1.0
                    if det_orient is not None and ais_cog is not None and (ais_sog or 0.0) > 1.5:
                        angle_diff = abs(float(det_orient) - (float(ais_cog) % 180.0))
                        angle_diff = min(angle_diff, 180.0 - angle_diff)
                        if angle_diff > 65.0:
                            continue
                        orient_score = max(0.3, 1.0 - (angle_diff / 65.0))

                    # 物理协同综合打分 (空间 0.50, 时间 0.20, 雷达质量 0.10, 尺度 0.10, 姿态 0.10)
                    time_score = math.exp(-0.5 * (delta_seconds / 300.0) ** 2)
                    spatial_score = max(0.0, 1.0 - distance / max(match_gate_km, 1e-6))
                    raw = (0.50 * spatial_score + 0.20 * time_score + 0.10 * quality +
                           0.10 * scale_score + 0.10 * orient_score)
                    raw = min(0.99, max(0.01, round(raw, 4)))

                    extra = json.loads(point.get("extra_attributes_json") or "{}")
                    vessel_class = point.get("vessel_class") or extra.get("vessel_class") or extra.get("vessel_type")
                    native_key = ({"dataset_id": dataset_id, "source_day_utc": point["source_day_utc"],
                                   "source_row_number": point["source_row_number"]}
                                  if point["native_table"] == "ais_messages" else
                                  {"source_id": point["source_id"],
                                   "source_row_number": point["source_row_number"]})
                    observation_id = _stable_id("obs", "ais_message", point["native_table"], native_key)
                    if observation_id not in inserted_observation_ids:
                        inserted_observation_ids.add(observation_id)
                        if conn.execute("SELECT 1 FROM fusion_nodes WHERE dataset_id=? AND node_id=?",
                                        (dataset_id, observation_id)).fetchone() is None:
                            _insert_node(
                                conn, dataset_id, node_id=observation_id, node_kind="observation",
                                modality="ais", object_type="ais_message",
                                native_table=point["native_table"], native_key=native_key,
                                start_utc=point["timestamp_utc"], end_utc=point["timestamp_utc"],
                                time_precision="instant",
                                geometry={"type": "Point", "coordinates": [point["lon"], point["lat"]]},
                                attributes={"mmsi": str(point["mmsi"]),
                                            "vessel_name": point.get("vessel_name"),
                                            "vessel_class": vessel_class,
                                            "timestamp_utc": point["timestamp_utc"],
                                            "lon": point["lon"], "lat": point["lat"],
                                            "position_granularity": "message",
                                            "source_row_number": point["source_row_number"]},
                                quality={"position_basis": "reported_ais_message"},
                                source_id=point.get("source_id"),
                                provenance={"native_table": point["native_table"], "native_key": native_key})
                    details = {"distance_km": round(distance, 4), "raw_distance_km": round(raw_distance, 4),
                               "uncorrected_det_distance_km": round(uncorr_det_dist, 4),
                               "doppler_shift_m": round(doppler_shift_m, 1),
                               "match_gate_km": round(match_gate_km, 4),
                               "dead_reckoning_method": reckoning_method,
                               "dead_reckoning_position": [round(reckoned_pos[0], 6), round(reckoned_pos[1], 6)],
                               "restored_detection_point": [round(corr_det_point[0], 6), round(corr_det_point[1], 6)],
                               "time_delta_seconds": round(delta_seconds, 2),
                               "ais_timestamp_utc": point["timestamp_utc"],
                               "ais_granularity": "message",
                               "mmsi": str(point["mmsi"]),
                               "ais_native_table": point["native_table"],
                               "ais_source_row_number": point["source_row_number"],
                               "confidence_kind": "uncalibrated_rule_score", "is_prior_guided": is_prior_matched_target,
                               "physical_scale_score": round(scale_score, 3),
                               "physical_orient_score": round(orient_score, 3)}
                    scores = {"spatial": round(spatial_score, 4), "time": round(time_score, 4),
                              "quality": round(quality, 4), "scale": round(scale_score, 4),
                              "orient": round(orient_score, 4)}
                    scored.append((raw, observation_id, vessel_key, details, scores))
                accepted = []
                for raw, observation_id, vessel_key, details, scores in scored:
                    decision = "candidate" if raw >= accept_threshold else "rejected"
                    _store_scored_candidate(conn, dataset_id, run_id, detection["node_id"],
                                            "POSSIBLY_OBSERVES", observation_id, scores, raw,
                                            decision, details)
                    summary[decision] += 1
                    if decision == "candidate":
                        accepted.append((raw, observation_id, vessel_key, details, scores))
                if accepted:
                    pending_acceptances.extend(
                        (raw, detection["node_id"], observation_id, vessel_key, details, scores)
                        for raw, observation_id, vessel_key, details, scores in accepted)
            matched_detections: set[str] = set()
            matched_vessels: set[str] = set()
            for raw, detection_id, observation_id, vessel_key, details, scores in sorted(
                    pending_acceptances, key=lambda item: item[0], reverse=True):
                if detection_id in matched_detections or vessel_key in matched_vessels:
                    continue
                _store_relation(conn, dataset_id, run_id, detection_id,
                                "POSSIBLY_OBSERVES", observation_id, raw,
                                "uncertainty_gated_spatiotemporal_one_to_one",
                                {**scores, **details, "assignment": "greedy_one_to_one"},
                                {**details, "assignment": "greedy_one_to_one"}, method_version)
                matched_detections.add(detection_id)
                matched_vessels.add(vessel_key)
                summary["POSSIBLY_OBSERVES"] += 1
            summary["unresolved"] = len(detection_node_ids - matched_detections)
            conn.execute("UPDATE fusion_runs SET status='completed',completed_at_utc=?,summary_json=? "
                         "WHERE dataset_id=? AND run_id=?", (_now(), _json(summary), dataset_id, run_id))
        return {"dataset_id": dataset_id, "run_id": run_id,
                "method_version": method_version, "summary": summary}
    finally:
        conn.close()


def build_deterministic_relations(db_path: Path, dataset_id: str) -> dict[str, Any]:
    """Build explicit-identity and exact geometry relations from indexed nodes."""
    migrate_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run_id = str(uuid4())
    started = _now()
    summary = {"OBSERVES": 0, "LOCATED_IN": 0, "COVERS": 0, "OBSERVED_IN": 0,
               "OCCURS_IN": 0, "DETECTED_IN": 0}
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("INSERT INTO fusion_runs VALUES (?,?,?,?,?,?,?,?,?)",
                         (dataset_id, run_id, _json(list(summary)), METHOD_VERSION, _json({}),
                          "running", started, None, _json({})))

            tracks = conn.execute("SELECT * FROM fusion_nodes WHERE dataset_id=? AND modality='ais' "
                                  "AND object_type='vessel_tracklet'", (dataset_id,)).fetchall()
            places = conn.execute("SELECT * FROM fusion_nodes WHERE dataset_id=? AND modality='gis'",
                                  (dataset_id,)).fetchall()
            entity_by_mmsi = {json.loads(row["attributes_json"])["mmsi"]: row["node_id"] for row in
                              conn.execute("SELECT * FROM fusion_nodes WHERE dataset_id=? "
                                           "AND node_kind='entity' AND object_type='vessel'", (dataset_id,))}
            place_geometry = {row["node_id"]: json.loads(row["geometry_json"])
                              for row in places if row["geometry_json"]}

            # Assign polygon places to a small in-memory grid.  Testing every sampled AIS
            # message against every GIS bbox made a full historical rebuild unnecessarily
            # expensive.  Grid lookup changes only candidate generation; the final decision
            # is still the same exact point-in-polygon test on real AIS message positions.
            grid_size = 0.02
            place_grid: dict[tuple[int, int], list[sqlite3.Row]] = {}
            broad_places: list[sqlite3.Row] = []
            for place in places:
                if place["node_id"] not in place_geometry:
                    continue
                if place_geometry[place["node_id"]].get("type") not in {"Polygon", "MultiPolygon"}:
                    continue
                if None in (place["bbox_min_lon"], place["bbox_min_lat"],
                            place["bbox_max_lon"], place["bbox_max_lat"]):
                    continue
                min_x = math.floor(place["bbox_min_lon"] / grid_size)
                max_x = math.floor(place["bbox_max_lon"] / grid_size)
                min_y = math.floor(place["bbox_min_lat"] / grid_size)
                max_y = math.floor(place["bbox_max_lat"] / grid_size)
                cell_count = (max_x - min_x + 1) * (max_y - min_y + 1)
                if cell_count > 10_000:
                    broad_places.append(place)
                    continue
                for cell_x in range(min_x, max_x + 1):
                    for cell_y in range(min_y, max_y + 1):
                        place_grid.setdefault((cell_x, cell_y), []).append(place)

            def candidate_places(west: float, south: float, east: float, north: float) -> list[sqlite3.Row]:
                return conn.execute(
                    "SELECT * FROM fusion_nodes WHERE dataset_id=? AND modality='gis' "
                    "AND bbox_min_lon<=? AND bbox_max_lon>=? AND bbox_min_lat<=? AND bbox_max_lat>=?",
                    (dataset_id, east, west, north, south)).fetchall()

            for track in tracks:
                attributes = json.loads(track["attributes_json"])
                entity_id = entity_by_mmsi.get(attributes.get("mmsi"))
                if entity_id:
                    _store_relation(conn, dataset_id, run_id, track["node_id"], "OBSERVES", entity_id,
                                    1.0, "reported_identity", {"mmsi": attributes["mmsi"]},
                                    {"rule": "exact_reported_mmsi", "warning": "reported identity is not authenticated"})
                    summary["OBSERVES"] += 1
                geometry = json.loads(track["geometry_json"])
                points = list(_coordinate_pairs(geometry.get("coordinates")))
                total_observed_points = len(points)
                if len(points) > 512:
                    step = max(1, math.ceil(len(points) / 512))
                    evaluated_points = points[::step]
                    if evaluated_points[-1] != points[-1]:
                        evaluated_points.append(points[-1])
                else:
                    evaluated_points = points
                inside_by_place: dict[str, tuple[sqlite3.Row, int]] = {}
                for lon, lat in evaluated_points:
                    candidates = place_grid.get((math.floor(lon / grid_size),
                                                 math.floor(lat / grid_size)), [])
                    for place in [*candidates, *broad_places]:
                        if not (place["bbox_min_lon"] <= lon <= place["bbox_max_lon"]
                                and place["bbox_min_lat"] <= lat <= place["bbox_max_lat"]):
                            continue
                        if not point_in_geometry(lon, lat, place_geometry[place["node_id"]]):
                            continue
                        previous = inside_by_place.get(place["node_id"])
                        inside_by_place[place["node_id"]] = (place, 1 if previous is None else previous[1] + 1)
                for place, inside in inside_by_place.values():
                    confidence = inside / max(len(evaluated_points), 1)
                    _store_relation(conn, dataset_id, run_id, track["node_id"], "LOCATED_IN",
                                    place["node_id"], confidence, "point_in_polygon",
                                    {"inside_observed_messages": inside,
                                     "evaluated_observed_messages": len(evaluated_points),
                                     "total_observed_messages": total_observed_points,
                                     "position_granularity": "message"},
                                    {"rule": "observed_message_points_inside_polygon",
                                     "sampling": "uniform_real_messages_max_512"})
                    summary["LOCATED_IN"] += 1

            scenes = conn.execute("SELECT * FROM fusion_nodes WHERE dataset_id=? "
                                  "AND modality='remote_sensing' AND object_type='scene'", (dataset_id,)).fetchall()
            for scene in scenes:
                if scene["bbox_min_lon"] is None:
                    continue
                scene_bbox = (scene["bbox_min_lon"], scene["bbox_min_lat"],
                              scene["bbox_max_lon"], scene["bbox_max_lat"])
                for place in candidate_places(*scene_bbox):
                    if place["bbox_min_lon"] is None:
                        continue
                    place_bbox = (place["bbox_min_lon"], place["bbox_min_lat"],
                                  place["bbox_max_lon"], place["bbox_max_lat"])
                    ratio = _bbox_intersection_ratio(scene_bbox, place_bbox)
                    if ratio <= 0:
                        continue
                    _store_relation(conn, dataset_id, run_id, scene["node_id"], "COVERS",
                                    place["node_id"], ratio, "bbox_intersection",
                                    {"gis_bbox_coverage_ratio": ratio},
                                    {"rule": "scene_bbox_intersects_gis_bbox",
                                     "caveat": "bbox intersection is not pixel-level coverage"})
                    summary["COVERS"] += 1

            weather = conn.execute("SELECT * FROM fusion_nodes WHERE dataset_id=? AND modality='weather'",
                                   (dataset_id,)).fetchall()
            for item in weather:
                geometry = json.loads(item["geometry_json"])
                lon, lat = geometry["coordinates"]
                for place in candidate_places(lon, lat, lon, lat):
                    bbox = (place["bbox_min_lon"], place["bbox_min_lat"],
                            place["bbox_max_lon"], place["bbox_max_lat"])
                    if None in bbox or not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                        continue
                    if not point_in_geometry(lon, lat, place_geometry[place["node_id"]]):
                        continue
                    _store_relation(conn, dataset_id, run_id, item["node_id"], "OBSERVED_IN",
                                    place["node_id"], 1.0, "point_in_polygon",
                                    {"lon": lon, "lat": lat}, {"rule": "weather_station_inside_polygon"})
                    summary["OBSERVED_IN"] += 1

            events = conn.execute("SELECT * FROM fusion_nodes WHERE dataset_id=? AND modality='event'",
                                  (dataset_id,)).fetchall()
            for event in events:
                attributes = json.loads(event["attributes_json"])
                explicit_id = attributes.get("gis_object_id")
                candidates = places
                if explicit_id:
                    candidates = [place for place in places
                                  if json.loads(place["native_key_json"]).get("object_id") == explicit_id]
                elif event["geometry_json"] and event["geometry_type"] == "Point":
                    lon, lat = json.loads(event["geometry_json"])["coordinates"]
                    candidates = candidate_places(lon, lat, lon, lat)
                else:
                    continue
                for place in candidates:
                    supported = bool(explicit_id)
                    evidence = {"gis_object_id": explicit_id} if explicit_id else {}
                    if not supported and event["geometry_json"]:
                        lon, lat = json.loads(event["geometry_json"])["coordinates"]
                        supported = point_in_geometry(lon, lat, place_geometry[place["node_id"]])
                        evidence = {"lon": lon, "lat": lat}
                    if supported:
                        _store_relation(conn, dataset_id, run_id, event["node_id"], "OCCURS_IN",
                                        place["node_id"], 1.0, "explicit_event_location", evidence,
                                        {"rule": "explicit_object_reference_or_point_in_polygon"})
                        summary["OCCURS_IN"] += 1

            scenes_by_id = {json.loads(row["attributes_json"]).get("scene_id"): row for row in scenes}
            detections = conn.execute("SELECT * FROM fusion_nodes WHERE dataset_id=? "
                                      "AND native_table='remote_sensing_detections'", (dataset_id,)).fetchall()
            for detection in detections:
                scene_id = json.loads(detection["attributes_json"]).get("scene_id")
                scene = scenes_by_id.get(scene_id)
                if scene:
                    confidence = json.loads(detection["quality_json"]).get("detection_confidence")
                    _store_relation(conn, dataset_id, run_id, detection["node_id"], "DETECTED_IN",
                                    scene["node_id"], float(confidence if confidence is not None else 1.0),
                                    "scene_foreign_key", {"scene_id": scene_id},
                                    {"rule": "explicit_scene_id", "confidence_kind": "source_detection_score"})
                    summary["DETECTED_IN"] += 1

            conn.execute("UPDATE fusion_runs SET status='completed',completed_at_utc=?,summary_json=? "
                         "WHERE dataset_id=? AND run_id=?", (_now(), _json(summary), dataset_id, run_id))
        return {"dataset_id": dataset_id, "run_id": run_id, "method_version": METHOD_VERSION,
                "relation_count": sum(summary.values()), "relations_by_type": summary}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
