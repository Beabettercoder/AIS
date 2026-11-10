"""SQLite schema management and transactional DS CSV ingestion."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

from .repository import METRICS, TABLES, _number, _utc


DB_TABLES = {
    "port": "port_hourly",
    "zone": "zone_hourly",
    "vessel_class": "vessel_class_hourly",
    "objects": "object_phase_comparison",
    "events": "confirmed_events",
    "event_links": "event_evidence_links",
    "sources": "source_registry",
}
IMPORT_ORDER = ("sources", "events", "event_links", "port", "zone", "vessel_class", "objects")
SCHEMA_VERSION = 27


def open_readonly(db_path: Path) -> sqlite3.Connection:
    path = db_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite 数据库不存在: {path}")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 60000")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        conn.close()
        raise ValueError(f"数据库结构版本 {version} 不受支持，需要版本 {SCHEMA_VERSION}")
    return conn


def _initialize(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA busy_timeout = 60000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except Exception:
        pass
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, SCHEMA_VERSION):
        raise ValueError(f"数据库结构版本 {version} 不受支持")
    if version == 0:
        schema = files("maritime_agent").joinpath("migrations/001_init.sql").read_text(encoding="utf-8")
        conn.executescript(schema)
        conn.execute("PRAGMA user_version = 1")
        version = 1
    if version == 1:
        schema = files("maritime_agent").joinpath("migrations/002_tracks.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 2;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 2
    if version == 2:
        schema = files("maritime_agent").joinpath("migrations/003_gis.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 3;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 3
    if version == 3:
        schema = files("maritime_agent").joinpath("migrations/004_ais_messages.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 4;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 4
    if version == 4:
        schema = files("maritime_agent").joinpath("migrations/005_remote_sensing.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 5;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 5
    if version == 5:
        schema = files("maritime_agent").joinpath("migrations/006_data_intake.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 6;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 6
    if version == 6:
        schema = files("maritime_agent").joinpath("migrations/007_remote_ais_quality.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 7;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 7
    if version == 7:
        schema = files("maritime_agent").joinpath("migrations/008_intake_metrics.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 8;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 8
    if version == 8:
        schema = files("maritime_agent").joinpath("migrations/009_external_gis.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 9;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 9
    if version == 9:
        schema = files("maritime_agent").joinpath("migrations/010_batch_ais_intake.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 10;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 10
    if version == 10:
        schema = files("maritime_agent").joinpath("migrations/011_intake_quality.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 11;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 11
    if version == 11:
        schema = files("maritime_agent").joinpath("migrations/012_gis_intake_quality.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 12;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 12
    if version == 12:
        schema = files("maritime_agent").joinpath("migrations/013_weather_intake.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 13;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 13
    if version == 13:
        schema = files("maritime_agent").joinpath("migrations/014_weather_long_model.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 14;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 14
    if version == 14:
        schema = files("maritime_agent").joinpath("migrations/015_alignment_rules.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 15;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 15
    if version == 15:
        schema = files("maritime_agent").joinpath("migrations/016_ais_extra_attributes.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 16;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 16
    if version == 16:
        schema = files("maritime_agent").joinpath("migrations/017_remote_sensing_alignment.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 17;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 17
    if version == 17:
        schema = files("maritime_agent").joinpath("migrations/018_fusion_core.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 18;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 18
    if version == 18:
        schema = files("maritime_agent").joinpath("migrations/019_fusion_entities.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 19;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 19
    if version == 19:
        schema = files("maritime_agent").joinpath("migrations/020_remote_detection_pipeline.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 20;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 20
    if version == 20:
        schema = files("maritime_agent").joinpath("migrations/021_remote_raster_intake.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 21;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 21
    if version == 21:
        schema = files("maritime_agent").joinpath("migrations/022_ais_message_time_index.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 22;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 22
    if version == 22:
        schema = files("maritime_agent").joinpath("migrations/023_regions.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 23;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        from .region_registry import backfill_legacy_regions
        backfill_legacy_regions(conn)
        version = 23
    if version == 23:
        schema = files("maritime_agent").joinpath("migrations/024_situation_cube.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 24;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 24
    if version == 24:
        schema = files("maritime_agent").joinpath("migrations/025_event_text_intake.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 25;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 25
    if version == 25:
        schema = files("maritime_agent").joinpath("migrations/026_data_management.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 26;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
        version = 26
    if version == 26:
        schema = files("maritime_agent").joinpath("migrations/027_fusion_cascade_indexes.sql").read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + schema + "\nPRAGMA user_version = 27;\nCOMMIT;")
        except Exception:
            conn.rollback()
            raise
    conn.execute("PRAGMA foreign_keys = ON")


def migrate_database(db_path: Path) -> dict[str, int]:
    """Upgrade an existing database before read-only repository startup."""
    path = db_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite 数据库不存在: {path}")
    conn = sqlite3.connect(path)
    try:
        before = int(conn.execute("PRAGMA user_version").fetchone()[0])
        _initialize(conn)
        after = int(conn.execute("PRAGMA user_version").fetchone()[0])
        return {"before": before, "after": after}
    finally:
        conn.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV 缺少表头: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"CSV 中有多余列: {path}")
    return rows


def _required(row: dict[str, str], key: str, table: str, number: int) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{table} 第 {number} 行缺少 {key}")
    return value


def _insert_row(conn: sqlite3.Connection, table: str, row: dict[str, str],
                dataset_id: str, port_id: str, number: int) -> None:
    record = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    if table in METRICS:
        hour = _utc(_required(row, "timestamp_hour_utc", table, number)).isoformat().replace("+00:00", "Z")
        if table == "port":
            conn.execute("INSERT INTO port_hourly VALUES (?,?,?,?,?,?,?)",
                         (dataset_id, port_id, hour, _number(row.get("ais_source_file_available")),
                          _number(row.get("ais_no_reports")), number, record))
            selector = ""
        elif table == "zone":
            selector = _required(row, "macro_zone", table, number)
            conn.execute("INSERT INTO zone_hourly VALUES (?,?,?,?,?,?)",
                         (dataset_id, port_id, hour, selector, number, record))
        else:
            selector = _required(row, "vessel_class", table, number)
            conn.execute("INSERT INTO vessel_class_hourly VALUES (?,?,?,?,?,?)",
                         (dataset_id, port_id, hour, selector, number, record))
        for metric in METRICS[table]:
            value = _number(row.get(metric))
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError(f"{table} 第 {number} 行指标 {metric} 非有限数")
            conn.execute("INSERT INTO hourly_metric_values VALUES (?,?,?,?,?,?,?)",
                         (dataset_id, port_id, table, hour, selector, metric, value))
    elif table == "objects":
        conn.execute("INSERT INTO object_phase_comparison VALUES (?,?,?,?,?)",
                     (dataset_id, _required(row, "object_id", table, number), row.get("object_name"), number, record))
    elif table == "sources":
        conn.execute("INSERT INTO source_registry VALUES (?,?,?,?,?,?,?)",
                     (dataset_id, _required(row, "source_id", table, number), row.get("title"),
                      row.get("source_url"), row.get("publication_date"), number, record))
    elif table == "events":
        conn.execute("INSERT INTO confirmed_events VALUES (?,?,?,?,?,?,?,?,?)",
                     (dataset_id, _required(row, "event_id", table, number), row.get("event_name"),
                      row.get("valid_start_local"), row.get("valid_end_local"), row.get("time_precision"),
                      row.get("published_date"), number, record))
    elif table == "event_links":
        conn.execute("INSERT INTO event_evidence_links VALUES (?,?,?,?,?,?,?,?)",
                     (dataset_id, _required(row, "evidence_link_id", table, number), row.get("event_id") or None,
                      row.get("candidate_id") or None, _required(row, "source_id", table, number),
                      row.get("quote_verified"), number, record))


def import_dataset(source_root: Path, db_path: Path, dataset_id: str,
                   port_id: str, label: str) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    db_path = db_path.expanduser().resolve()
    dataset_id, port_id, label = dataset_id.strip(), port_id.strip(), label.strip()
    if not all((dataset_id, port_id, label)):
        raise ValueError("dataset_id、port_id、label 均不能为空")
    paths = {name: source_root / TABLES[name] for name in IMPORT_ORDER}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少导入文件: " + "; ".join(missing))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        _initialize(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        existing = conn.execute("SELECT dataset_id, port_id, label, source_root FROM datasets WHERE dataset_id=?",
                                (dataset_id,)).fetchone()
        if existing:
            stored = dict(conn.execute("SELECT table_name, sha256 FROM import_files WHERE dataset_id=?", (dataset_id,)).fetchall())
            if (all(stored.get(name) == digest for name, digest in hashes.items())
                    and existing[1] == port_id and existing[2] == label
                    and existing[3] == str(source_root)):
                counts = dict(conn.execute("SELECT table_name, row_count FROM import_files WHERE dataset_id=?",
                                           (dataset_id,)).fetchall())
                return {"status": "already_imported", "dataset_id": dataset_id,
                        "database": str(db_path), "row_counts": counts}
            raise ValueError("dataset_id 已存在且源文件或元数据不同；请使用新的 dataset_id，不会覆盖已有数据")
        counts: dict[str, int] = {}
        with conn:
            conn.execute("INSERT INTO datasets VALUES (?,?,?,?,?)",
                         (dataset_id, port_id, label, str(source_root),
                          datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")))
            for name in IMPORT_ORDER:
                rows = _rows(paths[name])
                for index, row in enumerate(rows, start=2):
                    _insert_row(conn, name, row, dataset_id, port_id, index)
                counts[name] = len(rows)
                conn.execute("INSERT INTO import_files VALUES (?,?,?,?,?)",
                             (dataset_id, name, TABLES[name], hashes[name], len(rows)))
            errors = conn.execute("PRAGMA foreign_key_check").fetchall()
            if errors:
                raise ValueError(f"外键检查失败: {errors[:3]}")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite 完整性检查失败: {integrity}")
        return {"status": "imported", "dataset_id": dataset_id, "port_id": port_id,
                "database": str(db_path), "row_counts": counts,
                "metric_values": conn.execute("SELECT COUNT(*) FROM hourly_metric_values WHERE dataset_id=?",
                                              (dataset_id,)).fetchone()[0]}
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"导入失败，发现重复主键或无效关联: {exc}") from exc
    finally:
        conn.close()
