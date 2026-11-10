"""Validated, transactional import of the DS vessel-hour positions and daily tracks."""

from __future__ import annotations

import csv
import gzip
import json
import math
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .database import _initialize, _sha256, open_readonly
from .repository import _utc


TRACK_FILES = {
    "vessel_hour_positions": "ais_spatial/01_vessel_hour/ais_vessel_hour_positions.csv.gz",
    "vessel_daily_tracks": "ais_spatial/05_tracks/vessel_daily_hourly_tracks.geojson.gz",
}


def _coordinate(lon: Any, lat: Any) -> tuple[float, float]:
    try:
        x, y = float(lon), float(lat)
    except (TypeError, ValueError) as exc:
        raise ValueError("航迹包含无效经纬度") from exc
    if not (math.isfinite(x) and math.isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90):
        raise ValueError("航迹包含越界或非有限经纬度")
    return x, y


def _track_features(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError("航迹文件必须为 GeoJSON FeatureCollection")
    return data["features"]


def _assert_track_matches_hours(conn: sqlite3.Connection, dataset_id: str,
                                date_utc: str, mmsi: str, coordinates: list) -> None:
    # The exclusive end is a date string to preserve the indexed UTC text order.
    next_day = (date.fromisoformat(date_utc) + timedelta(days=1)).isoformat()
    hours = conn.execute(
        "SELECT representative_lon, representative_lat FROM vessel_hour_positions "
        "WHERE dataset_id=? AND mmsi=? AND timestamp_hour_utc>=? AND timestamp_hour_utc<? "
        "ORDER BY timestamp_hour_utc",
        (dataset_id, mmsi, date_utc + "T00:00:00Z", next_day + "T00:00:00Z"),
    ).fetchall()
    if len(hours) != len(coordinates):
        raise ValueError(f"日航迹与小时位置数量不一致: {date_utc} {mmsi}")
    for point, hour in zip(coordinates, hours):
        x, y = _coordinate(*point)
        if abs(x - hour[0]) > 1e-6 or abs(y - hour[1]) > 1e-6:
            raise ValueError(f"日航迹与小时位置坐标不一致: {date_utc} {mmsi}")


def import_tracks(source_root: Path, db_path: Path, dataset_id: str) -> dict[str, Any]:
    source_root, db_path = source_root.expanduser().resolve(), db_path.expanduser().resolve()
    paths = {name: source_root / relative for name, relative in TRACK_FILES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少航迹数据文件: {path}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    conn = sqlite3.connect(db_path)
    try:
        _initialize(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        dataset = conn.execute("SELECT source_root FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if dataset is None:
            raise ValueError(f"数据库中找不到 dataset_id: {dataset_id}；请先导入标准 DS")
        if Path(dataset[0]).resolve() != source_root:
            raise ValueError("航迹来源目录与该 dataset_id 已登记的源目录不同")
        stored = dict(conn.execute(
            "SELECT table_name, sha256 FROM import_files WHERE dataset_id=? AND table_name IN (?,?)",
            (dataset_id, *TRACK_FILES),
        ).fetchall())
        if stored:
            if stored == hashes:
                counts = dict(conn.execute(
                    "SELECT table_name, row_count FROM import_files WHERE dataset_id=? AND table_name IN (?,?)",
                    (dataset_id, *TRACK_FILES),
                ).fetchall())
                return {"status": "already_imported", "dataset_id": dataset_id,
                        "database": str(db_path), "row_counts": counts}
            raise ValueError("航迹已导入但源文件指纹不同；不会覆盖现有数据")

        counts = {name: 0 for name in TRACK_FILES}
        first, last = conn.execute(
            "SELECT MIN(timestamp_hour_utc), MAX(timestamp_hour_utc) FROM port_hourly WHERE dataset_id=?",
            (dataset_id,),
        ).fetchone()
        if not first:
            raise ValueError("该数据集没有港口小时覆盖范围")
        with conn:
            with gzip.open(paths["vessel_hour_positions"], "rt", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"timestamp_hour_utc", "mmsi", "representative_lon_deg",
                            "representative_lat_deg", "source_file", "vessel_class", "motion_state"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError("船舶小时位置缺少必需字段")
                for number, row in enumerate(reader, start=2):
                    if None in row:
                        raise ValueError(f"船舶小时位置第 {number} 行存在多余列")
                    hour = _utc(row["timestamp_hour_utc"]).isoformat().replace("+00:00", "Z")
                    if not first <= hour <= last:
                        raise ValueError(f"船舶小时位置第 {number} 行超出数据集时间范围")
                    mmsi = (row["mmsi"] or "").strip()
                    if not mmsi or not row["source_file"]:
                        raise ValueError(f"船舶小时位置第 {number} 行缺少 MMSI 或来源文件")
                    lon, lat = _coordinate(row["representative_lon_deg"], row["representative_lat_deg"])
                    conn.execute("INSERT INTO vessel_hour_positions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                 (dataset_id, hour, mmsi, row.get("vessel_name"), row.get("vessel_class"),
                                  row.get("motion_state"), lon, lat, row["source_file"], number,
                                  json.dumps(row, ensure_ascii=False, separators=(",", ":"))))
                    counts["vessel_hour_positions"] += 1
            features = _track_features(paths["vessel_daily_tracks"])
            for number, feature in enumerate(features, start=1):
                props, geometry = feature.get("properties"), feature.get("geometry")
                if not isinstance(props, dict) or not isinstance(geometry, dict) or geometry.get("type") != "LineString":
                    raise ValueError(f"第 {number} 个航迹要素不是有效 LineString")
                coords = geometry.get("coordinates")
                if not isinstance(coords, list) or not 2 <= len(coords) <= 24:
                    raise ValueError(f"第 {number} 个航迹点数无效")
                day, mmsi = str(props.get("date_utc", "")), str(props.get("mmsi", "")).strip()
                try:
                    date.fromisoformat(day)
                    point_count = int(props["hour_points"])
                    moving = int(props["moving_hours"])
                    stopped = int(props["stopped_hours"])
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"第 {number} 个航迹属性无效") from exc
                if not mmsi or point_count != len(coords) or min(moving, stopped) < 0:
                    raise ValueError(f"第 {number} 个航迹属性与坐标不一致")
                sources = props.get("source_files")
                if not isinstance(sources, list) or not sources or not all(isinstance(x, str) for x in sources):
                    raise ValueError(f"第 {number} 个航迹缺少来源文件")
                _assert_track_matches_hours(conn, dataset_id, day, mmsi, coords)
                conn.execute("INSERT INTO vessel_daily_tracks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                             (dataset_id, day, mmsi, props.get("vessel_name"), props.get("vessel_class"),
                              point_count, moving, stopped, json.dumps(sources, ensure_ascii=False),
                              json.dumps(coords, ensure_ascii=False, separators=(",", ":")), number))
                counts["vessel_daily_tracks"] += 1
            for name in TRACK_FILES:
                conn.execute("INSERT INTO import_files VALUES (?,?,?,?,?)",
                             (dataset_id, name, TRACK_FILES[name], hashes[name], counts[name]))
            if conn.execute("PRAGMA foreign_key_check").fetchone():
                raise ValueError("航迹导入后外键检查失败")
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("航迹导入后 SQLite 完整性检查失败")
        return {"status": "imported", "dataset_id": dataset_id,
                "database": str(db_path), "row_counts": counts}
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"航迹导入失败：重复船舶小时键或日航迹键: {exc}") from exc
    finally:
        conn.close()


def verify_tracks(source_root: Path, db_path: Path, dataset_id: str) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    with closing(open_readonly(db_path)) as conn:
        counts = {}
        for name, relative in TRACK_FILES.items():
            row = conn.execute("SELECT sha256, row_count FROM import_files WHERE dataset_id=? AND table_name=?",
                               (dataset_id, name)).fetchone()
            if row is None or row["sha256"] != _sha256(source_root / relative):
                raise ValueError(f"航迹来源指纹不一致: {name}")
            table_count = conn.execute(f"SELECT COUNT(*) FROM {name} WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
            if table_count != row["row_count"]:
                raise ValueError(f"航迹记录数不一致: {name}")
            counts[name] = table_count
        with gzip.open(source_root / TRACK_FILES["vessel_hour_positions"], "rt",
                       encoding="utf-8-sig", newline="") as stream:
            source_rows = csv.DictReader(stream)
            imported = conn.execute("SELECT record_json FROM vessel_hour_positions WHERE dataset_id=? "
                                    "ORDER BY source_row_number", (dataset_id,))
            for number, (source, saved) in enumerate(zip(source_rows, imported), start=2):
                if source != json.loads(saved[0]):
                    raise ValueError(f"船舶小时位置第 {number} 行与原文件不同")
        features = _track_features(source_root / TRACK_FILES["vessel_daily_tracks"])
        imported_tracks = conn.execute("SELECT date_utc,mmsi,vessel_name,vessel_class,hour_points,"
                                       "moving_hours,stopped_hours,source_files_json,geometry_json,source_feature_number "
                                       "FROM vessel_daily_tracks WHERE dataset_id=? ORDER BY source_feature_number",
                                       (dataset_id,)).fetchall()
        if len(features) != len(imported_tracks):
            raise ValueError("航迹 GeoJSON 与数据库数量不一致")
        for feature, saved in zip(features, imported_tracks):
            props = feature["properties"]
            if (props["date_utc"] != saved["date_utc"] or str(props["mmsi"]) != saved["mmsi"]
                    or props.get("vessel_name") != saved["vessel_name"]
                    or props.get("vessel_class") != saved["vessel_class"]
                    or int(props["hour_points"]) != saved["hour_points"]
                    or int(props["moving_hours"]) != saved["moving_hours"]
                    or int(props["stopped_hours"]) != saved["stopped_hours"]
                    or props["source_files"] != json.loads(saved["source_files_json"])
                    or feature["geometry"]["coordinates"] != json.loads(saved["geometry_json"])):
                raise ValueError(f"第 {saved['source_feature_number']} 个日航迹与原文件不同")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
        if integrity != "ok" or foreign_keys:
            raise ValueError("航迹数据库完整性检查失败")
    return {"status": "verified", "dataset_id": dataset_id, "row_counts": counts,
            "integrity": "ok"}
