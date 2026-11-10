"""Import cleaned AIS observations without inventing positions between reports."""

from __future__ import annotations

import csv
import gzip
import math
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from .database import _initialize, _sha256, open_readonly


SOURCE_DIR = Path("ais_spatial/00_message_level/daily")
FIELDS = {"mmsi", "timestamp_utc", "lat_deg", "lon_deg", "sog_kn", "cog_deg",
          "vessel_name", "vessel_class"}
INSERT = ("INSERT INTO ais_messages(dataset_id,source_day_utc,source_row_number,mmsi,"
          "timestamp_utc,lon,lat,sog_kn,cog_deg,vessel_name,vessel_class) "
          "VALUES (?,?,?,?,?,?,?,?,?,?,?)")


def _optional_number(value: str | None, label: str) -> float | None:
    if value is None or not value.strip():
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} 不是有限数字")
    return number


def _sources(root: Path) -> list[tuple[str, Path, str]]:
    paths = sorted((root / SOURCE_DIR).glob("AIS_????_??_??_clean.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"没有找到逐报文 AIS 文件: {root / SOURCE_DIR}")
    found = []
    for path in paths:
        day = date.fromisoformat(path.name[4:14].replace("_", "-")).isoformat()
        found.append((day, path, path.relative_to(root).as_posix()))
    return found


def import_messages(source_root: Path, db_path: Path, dataset_id: str) -> dict:
    source_root, db_path = source_root.expanduser().resolve(), db_path.expanduser().resolve()
    sources = _sources(source_root)
    conn = sqlite3.connect(db_path)
    try:
        _initialize(conn)
        registered = conn.execute("SELECT source_root FROM datasets WHERE dataset_id=?",
                                  (dataset_id,)).fetchone()
        if registered is None or Path(registered[0]).resolve() != source_root:
            raise ValueError("数据集未登记，或 AIS 文件目录与数据集来源目录不一致")
        result = {}
        for day, path, relative in sources:
            key = f"ais_messages:{day}"
            digest = _sha256(path)
            saved = conn.execute("SELECT sha256,row_count FROM import_files WHERE dataset_id=? AND table_name=?",
                                 (dataset_id, key)).fetchone()
            if saved:
                if saved[0] != digest:
                    raise ValueError(f"已导入文件发生变化，不会覆盖: {relative}")
                result[day] = {"status": "already_imported", "rows": saved[1]}
                continue
            count = 0
            batch = []
            with conn:
                with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if not reader.fieldnames or not FIELDS.issubset(reader.fieldnames):
                        raise ValueError(f"AIS 文件缺少必需字段: {relative}")
                    for row_number, row in enumerate(reader, start=2):
                        if None in row:
                            raise ValueError(f"AIS 文件第 {row_number} 行存在多余列: {relative}")
                        stamp = row["timestamp_utc"]
                        try:
                            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                            lon = float(row["lon_deg"])
                            lat = float(row["lat_deg"])
                            sog = _optional_number(row["sog_kn"], "航速")
                            cog = _optional_number(row["cog_deg"], "航向")
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f"AIS 文件第 {row_number} 行数值或时间无效: {relative}") from exc
                        if (parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0
                                or parsed.date().isoformat() != day or not row["mmsi"].strip()
                                or not math.isfinite(lon) or not math.isfinite(lat)
                                or not -180 <= lon <= 180 or not -90 <= lat <= 90):
                            raise ValueError(f"AIS 文件第 {row_number} 行时间、MMSI 或坐标无效: {relative}")
                        batch.append((dataset_id, day, row_number, row["mmsi"].strip(),
                                      parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                                      lon, lat, sog, cog, row["vessel_name"] or None,
                                      row["vessel_class"] or None))
                        if len(batch) >= 5000:
                            conn.executemany(INSERT, batch)
                            count += len(batch)
                            batch.clear()
                    if batch:
                        conn.executemany(INSERT, batch)
                        count += len(batch)
                conn.execute("INSERT INTO import_files VALUES (?,?,?,?,?)",
                             (dataset_id, key, relative, digest, count))
            result[day] = {"status": "imported", "rows": count}
        if conn.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("逐报文 AIS 外键检查失败")
        return {"dataset_id": dataset_id, "database": str(db_path), "days": result,
                "total_rows": sum(item["rows"] for item in result.values())}
    finally:
        conn.close()


def verify_messages(source_root: Path, db_path: Path, dataset_id: str) -> dict:
    source_root = source_root.expanduser().resolve()
    conn = open_readonly(db_path)
    try:
        counts = {}
        for day, path, _ in _sources(source_root):
            saved = conn.execute("SELECT sha256,row_count FROM import_files WHERE dataset_id=? AND table_name=?",
                                 (dataset_id, f"ais_messages:{day}")).fetchone()
            actual = conn.execute("SELECT COUNT(*) FROM ais_messages WHERE dataset_id=? AND source_day_utc=?",
                                  (dataset_id, day)).fetchone()[0]
            if saved is None or saved[0] != _sha256(path) or saved[1] != actual:
                raise ValueError(f"逐报文 AIS 来源或记录数不一致: {day}")
            counts[day] = actual
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("逐报文 AIS 数据库完整性检查失败")
        return {"status": "verified", "row_counts": counts, "total_rows": sum(counts.values())}
    finally:
        conn.close()
