"""Transactional import and verification of port GIS objects and AIS area hours."""

from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .database import _initialize, _sha256, open_readonly
from .repository import _utc


GIS_FILES = {
    "gis_objects": "port_gis/01_standardized/port_objects_all.geojson",
    "gis_area_hourly": "port_gis/03_ais_match/port_context_hourly.csv",
}


def _features(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError("港口 GIS 文件不是 GeoJSON FeatureCollection")
    return data["features"]


def _integer(row: dict, name: str, number: int) -> int | None:
    value = row.get(name, "")
    if value in (None, ""):
        return None
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"GIS 小时表第 {number} 行的 {name} 不是整数") from exc
    if result < 0:
        raise ValueError(f"GIS 小时表第 {number} 行的 {name} 为负数")
    return result


def import_gis(source_root: Path, db_path: Path, dataset_id: str) -> dict[str, Any]:
    source_root, db_path = source_root.expanduser().resolve(), db_path.expanduser().resolve()
    paths = {name: source_root / relative for name, relative in GIS_FILES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少 GIS 来源文件: {path}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    conn = sqlite3.connect(db_path)
    try:
        _initialize(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        dataset = conn.execute("SELECT source_root FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if dataset is None or Path(dataset[0]).resolve() != source_root:
            raise ValueError("数据集不存在，或 GIS 来源目录与数据集登记目录不同")
        stored = dict(conn.execute(
            "SELECT table_name, sha256 FROM import_files WHERE dataset_id=? AND table_name IN (?,?)",
            (dataset_id, *GIS_FILES)).fetchall())
        if stored:
            if stored == hashes:
                counts = dict(conn.execute(
                    "SELECT table_name,row_count FROM import_files WHERE dataset_id=? AND table_name IN (?,?)",
                    (dataset_id, *GIS_FILES)).fetchall())
                return {"status": "already_imported", "dataset_id": dataset_id, "row_counts": counts}
            raise ValueError("GIS 来源文件指纹已变化；不会覆盖现有数据")
        first, last = conn.execute(
            "SELECT MIN(timestamp_hour_utc),MAX(timestamp_hour_utc) FROM port_hourly WHERE dataset_id=?",
            (dataset_id,)).fetchone()
        if not first:
            raise ValueError("该数据集没有港口小时覆盖范围")
        counts = {name: 0 for name in GIS_FILES}
        with conn:
            for number, feature in enumerate(_features(paths["gis_objects"]), start=1):
                props, geometry = feature.get("properties"), feature.get("geometry")
                if not isinstance(props, dict) or not isinstance(geometry, dict):
                    raise ValueError(f"GIS 第 {number} 个对象无效")
                object_id = str(props.get("object_id") or "").strip()
                geom_type = geometry.get("type")
                if not object_id or geom_type not in {"Point", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}:
                    raise ValueError(f"GIS 第 {number} 个对象缺少 ID 或几何类型无效")
                conn.execute("INSERT INTO gis_objects VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (dataset_id, object_id, str(props.get("object_type") or ""),
                              props.get("object_name"), props.get("source_authority"),
                              props.get("source_dataset"), props.get("source_url"),
                              str(props.get("source_date") or ""), geom_type,
                              json.dumps(geometry, ensure_ascii=False, separators=(",", ":")),
                              json.dumps(props, ensure_ascii=False, separators=(",", ":")), number))
                counts["gis_objects"] += 1
            with paths["gis_area_hourly"].open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"timestamp_hour_utc", "primary_area_id", "primary_area_type",
                            "unique_mmsi", "vessel_hour_rows", "moving_mmsi", "stopped_mmsi"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError("GIS 小时表缺少必需字段")
                for number, row in enumerate(reader, start=2):
                    if None in row:
                        raise ValueError(f"GIS 小时表第 {number} 行有多余列")
                    hour = _utc(row["timestamp_hour_utc"]).isoformat().replace("+00:00", "Z")
                    if not first <= hour <= last or not row["primary_area_id"]:
                        raise ValueError(f"GIS 小时表第 {number} 行时间或对象 ID 无效")
                    conn.execute("INSERT INTO gis_area_hourly VALUES (?,?,?,?,?,?,?,?,?,?)",
                                 (dataset_id, hour, row["primary_area_id"], row["primary_area_type"],
                                  _integer(row, "unique_mmsi", number), _integer(row, "vessel_hour_rows", number),
                                  _integer(row, "moving_mmsi", number), _integer(row, "stopped_mmsi", number),
                                  number, json.dumps(row, ensure_ascii=False, separators=(",", ":"))))
                    counts["gis_area_hourly"] += 1
            for name in GIS_FILES:
                conn.execute("INSERT INTO import_files VALUES (?,?,?,?,?)",
                             (dataset_id, name, GIS_FILES[name], hashes[name], counts[name]))
            if conn.execute("PRAGMA foreign_key_check").fetchone():
                raise ValueError("GIS 导入外键检查失败")
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("GIS 导入 SQLite 完整性检查失败")
        return {"status": "imported", "dataset_id": dataset_id, "row_counts": counts}
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"GIS 导入失败，存在重复对象或小时键: {exc}") from exc
    finally:
        conn.close()


def verify_gis(source_root: Path, db_path: Path, dataset_id: str) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    with closing(open_readonly(db_path)) as conn:
        counts = {}
        for name, relative in GIS_FILES.items():
            expected = conn.execute("SELECT sha256,row_count FROM import_files WHERE dataset_id=? AND table_name=?",
                                    (dataset_id, name)).fetchone()
            if expected is None or expected["sha256"] != _sha256(source_root / relative):
                raise ValueError(f"GIS 来源指纹不一致: {name}")
            actual = conn.execute(f"SELECT COUNT(*) FROM {name} WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
            if actual != expected["row_count"]:
                raise ValueError(f"GIS 数据库行数不一致: {name}")
            counts[name] = actual
        imported = conn.execute("SELECT properties_json,geometry_json FROM gis_objects WHERE dataset_id=? "
                                "ORDER BY source_feature_number", (dataset_id,))
        for feature, saved in zip(_features(source_root / GIS_FILES["gis_objects"]), imported):
            if feature["properties"] != json.loads(saved[0]) or feature["geometry"] != json.loads(saved[1]):
                raise ValueError("GIS 对象与原始 GeoJSON 不一致")
        with (source_root / GIS_FILES["gis_area_hourly"]).open("r", encoding="utf-8-sig", newline="") as stream:
            saved_rows = conn.execute("SELECT record_json FROM gis_area_hourly WHERE dataset_id=? "
                                      "ORDER BY source_row_number", (dataset_id,))
            for source, saved in zip(csv.DictReader(stream), saved_rows):
                if source != json.loads(saved[0]):
                    raise ValueError("GIS 小时表与原 CSV 不一致")
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or conn.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("GIS 数据库完整性检查失败")
    return {"status": "verified", "dataset_id": dataset_id, "row_counts": counts, "integrity": "ok"}
