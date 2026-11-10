"""Validate and import a public satellite scene catalog and local assets."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from .database import _initialize, _sha256, open_readonly
from .remote_sensing_alignment import ALIGNMENT_MANIFEST, load_internal_alignment

CATALOG = Path("remote_sensing/00_catalog/candidate_catalog.json")
MANIFEST = Path("remote_sensing/02_quality/asset_manifest.json")
QUALITY = Path("remote_sensing/02_quality/quality_probe.json")
IMPORT_KEY = "remote_sensing:catalog"
QUALITY_KEY = "remote_sensing:ais_quality"
ALIGNMENT_KEY = "remote_sensing:internal_alignment"


def _local(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"遥感资产不存在或越过数据根目录: {relative}")
    return path


def _records(root: Path):
    catalog_path = _local(root, CATALOG.as_posix())
    manifest_path = _local(root, MANIFEST.as_posix())
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = catalog.get("records")
    if not isinstance(records, list) or not records or catalog.get("record_count") != len(records):
        raise ValueError("遥感场景目录记录数无效")
    assets = {}
    for item in manifest:
        relative = item["file"]
        if relative in assets:
            raise ValueError("遥感资产清单有重复路径")
        path = _local(root, f"remote_sensing/{relative}")
        if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"遥感资产校验失败: {relative}")
        assets[relative] = item
    seen = set()
    normalized = []
    for record in records:
        scene_id = record.get("id", "")
        if not isinstance(scene_id, str) or not scene_id or scene_id in seen:
            raise ValueError("遥感场景 ID 缺失或重复")
        seen.add(scene_id)
        when = datetime.fromisoformat(record["datetime_utc"].replace("Z", "+00:00"))
        if when.utcoffset() is None or when.utcoffset().total_seconds() != 0:
            raise ValueError("遥感时间必须明确为 UTC")
        bbox = record["bbox_wgs84"]
        if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            raise ValueError("遥感场景 bbox 无效")
        cloud = record.get("scene_cloud_cover_pct")
        if cloud is not None and not 0 <= float(cloud) <= 100:
            raise ValueError("云量百分比超出范围")
        if urlparse(record["catalog_url"]).scheme != "https":
            raise ValueError("遥感目录链接必须使用 HTTPS")
        stac_relative = "remote_sensing/00_catalog/" + record["local_stac_item"]
        _local(root, stac_relative)
        record_assets = []
        for asset in record.get("downloaded_assets", []):
            key = Path(asset).name
            relative = "01_assets/" + key
            if relative not in assets:
                raise ValueError(f"场景资产未在清单登记: {asset}")
            record_assets.append((key, "remote_sensing/" + relative, assets[relative]))
        normalized.append((record, stac_relative, record_assets))
    return normalized, _sha256(catalog_path), assets


def _quality(root: Path):
    path = _local(root, QUALITY.as_posix())
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("遥感/AIS 质量报告格式无效")
    for day, item in data.items():
        datetime.fromisoformat(day)
        total = int(item["ais_vessel_hour_positions_17utc"])
        covered = int(item["covered_by_scl"])
        cloud = int(item["cloud_or_shadow_at_positions"])
        fraction = float(item["cloud_or_shadow_fraction_of_covered"])
        if not 0 <= cloud <= covered <= total or not 0 <= fraction <= 1:
            raise ValueError(f"{day} 遥感/AIS 质量计数无效")
        if covered and abs(cloud / covered - fraction) > 0.001:
            raise ValueError(f"{day} 遥感/AIS 云遮挡比例不匹配")
    return data, _sha256(path)


def import_remote_sensing(source_root: Path, db_path: Path, dataset_id: str,
                          refresh_alignment: bool = False) -> dict:
    root, db = source_root.expanduser().resolve(), db_path.expanduser().resolve()
    records, digest, assets = _records(root)
    quality, quality_digest = _quality(root)
    aligned_records, alignment_digest = load_internal_alignment(root)
    conn = sqlite3.connect(db)
    try:
        _initialize(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        dataset = conn.execute("SELECT source_root FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if dataset is None or Path(dataset[0]).resolve() != root:
            raise ValueError("数据集不存在或来源目录不匹配")
        stored = conn.execute("SELECT sha256,row_count FROM import_files WHERE dataset_id=? AND table_name=?",
                              (dataset_id, IMPORT_KEY)).fetchone()
        stored_quality = conn.execute("SELECT sha256,row_count FROM import_files WHERE dataset_id=? AND table_name=?",
                                      (dataset_id, QUALITY_KEY)).fetchone()
        stored_alignment = conn.execute(
            "SELECT sha256,row_count FROM import_files WHERE dataset_id=? AND table_name=?",
            (dataset_id, ALIGNMENT_KEY),
        ).fetchone()
        if stored:
            if stored[0] != digest or stored[1] != len(records):
                raise ValueError("遥感目录已发生变化；请审核后使用更新流程，不会自动覆盖")
        if stored_quality:
            if stored_quality[0] != quality_digest or stored_quality[1] != len(quality):
                raise ValueError("遥感/AIS 质量文件已发生变化；不会覆盖现有证据")
        if stored_alignment:
            if not alignment_digest:
                raise ValueError("遥感内部对齐清单已被删除；不会保留不可核验记录")
            if ((stored_alignment[0] != alignment_digest or stored_alignment[1] != len(aligned_records))
                    and not refresh_alignment):
                raise ValueError("遥感内部对齐清单已发生变化；请审核后使用更新流程")
        alignment_changed = bool(aligned_records and (not stored_alignment
                                 or stored_alignment[0] != alignment_digest
                                 or stored_alignment[1] != len(aligned_records)))
        if stored and stored_quality and (not aligned_records or stored_alignment) and not alignment_changed:
            return {"status": "already_imported", "scene_count": len(records),
                    "asset_count": len(assets), "quality_days": len(quality),
                    "aligned_scene_count": len(aligned_records),
                    "aligned_band_count": sum(len(item["bands"]) for item in aligned_records)}
        with conn:
            if alignment_changed and stored_alignment:
                conn.execute("DELETE FROM remote_sensing_aligned_scenes WHERE dataset_id=? "
                             "AND source_relative_path LIKE 'remote_sensing/01_assets/validation_samples/%'",
                             (dataset_id,))
                conn.execute("DELETE FROM import_files WHERE dataset_id=? AND table_name=?",
                             (dataset_id, ALIGNMENT_KEY))
                stored_alignment = None
            if not stored:
                for record, stac_relative, record_assets in records:
                    conn.execute("INSERT INTO remote_sensing_scenes VALUES (?,?,?,?,?,?,?,?,?,?)",
                                 (dataset_id, record["id"], record["collection"], record["datetime_utc"],
                                  json.dumps(record["bbox_wgs84"]), record.get("scene_cloud_cover_pct"),
                                  int(record["in_ais_window"]), record["catalog_url"], stac_relative,
                                  json.dumps(record, ensure_ascii=False)))
                    for key, relative, info in record_assets:
                        conn.execute("INSERT INTO remote_sensing_assets VALUES (?,?,?,?,?,?)",
                                     (dataset_id, record["id"], key, relative,
                                      info["sha256"], info["bytes"]))
                conn.execute("INSERT INTO import_files VALUES (?,?,?,?,?)",
                             (dataset_id, IMPORT_KEY, CATALOG.as_posix(), digest, len(records)))
            if not stored_quality:
                for day, item in quality.items():
                    conn.execute("INSERT INTO remote_sensing_ais_quality VALUES (?,?,?,?,?,?,?)",
                                 (dataset_id, day, int(item["ais_vessel_hour_positions_17utc"]),
                                  int(item["covered_by_scl"]), int(item["cloud_or_shadow_at_positions"]),
                                  float(item["cloud_or_shadow_fraction_of_covered"]),
                                  json.dumps(item, ensure_ascii=False)))
                conn.execute("INSERT INTO import_files VALUES (?,?,?,?,?)",
                             (dataset_id, QUALITY_KEY, QUALITY.as_posix(), quality_digest, len(quality)))
            if aligned_records and not stored_alignment:
                for record in aligned_records:
                    conn.execute(
                        "INSERT INTO remote_sensing_aligned_scenes VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (dataset_id, record["scene_id"], record["collection"], record["platform"],
                         record["sensor"], record["product_type"], record.get("processing_level"),
                         record["start_utc"], record.get("end_utc"), record.get("center_time_utc"),
                         record["temporal_precision"], json.dumps(record.get("bbox_wgs84")),
                         record["bbox_role"], record.get("source_crs"), record["analysis_crs"],
                         record.get("native_resolution_m"), record.get("grid_width"),
                         record.get("grid_height"), record.get("grid_pixel_size_x"),
                         record.get("grid_pixel_size_y"), record.get("grid_pixel_unit"),
                         record["asset_role"], record["file_format"], int(record["quality_available"]),
                         record["source_path"], record["sha256"], int(record["byte_count"]),
                         record.get("provenance_url"), json.dumps(record, ensure_ascii=False)))
                    for band in record["bands"]:
                        conn.execute(
                            "INSERT INTO remote_sensing_aligned_bands VALUES "
                            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (dataset_id, record["scene_id"], band["source_band_name"],
                             band["canonical_variable"], band.get("wavelength_um"), band["unit"],
                             band.get("data_type"), band.get("scale_factor"), band.get("add_offset"),
                             band.get("valid_min"), band.get("valid_max"),
                             None if band.get("nodata") is None else str(band["nodata"]),
                             band["band_role"], band.get("quality_band"),
                             json.dumps(band, ensure_ascii=False)))
                conn.execute("INSERT INTO import_files VALUES (?,?,?,?,?)",
                             (dataset_id, ALIGNMENT_KEY, ALIGNMENT_MANIFEST.as_posix(),
                              alignment_digest, len(aligned_records)))
        return {"status": "imported", "scene_count": len(records), "asset_count": len(assets),
                "quality_days": len(quality), "aligned_scene_count": len(aligned_records),
                "aligned_band_count": sum(len(item["bands"]) for item in aligned_records),
                "database": str(db)}
    finally:
        conn.close()


def verify_remote_sensing(source_root: Path, db_path: Path, dataset_id: str) -> dict:
    records, digest, assets = _records(source_root.expanduser().resolve())
    quality, quality_digest = _quality(source_root.expanduser().resolve())
    aligned_records, alignment_digest = load_internal_alignment(source_root.expanduser().resolve())
    conn = open_readonly(db_path)
    try:
        stored = conn.execute("SELECT sha256,row_count FROM import_files WHERE dataset_id=? AND table_name=?",
                              (dataset_id, IMPORT_KEY)).fetchone()
        count = conn.execute("SELECT COUNT(*) FROM remote_sensing_scenes WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
        asset_count = conn.execute("SELECT COUNT(*) FROM remote_sensing_assets WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
        quality_stored = conn.execute("SELECT sha256,row_count FROM import_files WHERE dataset_id=? AND table_name=?",
                                      (dataset_id, QUALITY_KEY)).fetchone()
        quality_count = conn.execute("SELECT COUNT(*) FROM remote_sensing_ais_quality WHERE dataset_id=?",
                                     (dataset_id,)).fetchone()[0]
        alignment_stored = conn.execute(
            "SELECT sha256,row_count FROM import_files WHERE dataset_id=? AND table_name=?",
            (dataset_id, ALIGNMENT_KEY),
        ).fetchone()
        aligned_count = conn.execute(
            "SELECT COUNT(*) FROM remote_sensing_aligned_scenes WHERE dataset_id=?", (dataset_id,)
        ).fetchone()[0]
        aligned_band_count = conn.execute(
            "SELECT COUNT(*) FROM remote_sensing_aligned_bands WHERE dataset_id=?", (dataset_id,)
        ).fetchone()[0]
        expected_band_count = sum(len(item["bands"]) for item in aligned_records)
        alignment_valid = ((not aligned_records and not alignment_stored and aligned_count == 0)
                           or bool(alignment_stored and alignment_digest
                                   and alignment_stored[0] == alignment_digest
                                   and alignment_stored[1] == aligned_count == len(aligned_records)
                                   and aligned_band_count == expected_band_count))
        valid = bool(stored and stored[0] == digest and stored[1] == count == len(records)
                     and asset_count == len(assets) and quality_stored
                     and quality_stored[0] == quality_digest
                     and quality_stored[1] == quality_count == len(quality)
                     and alignment_valid)
        return {"valid": valid, "scene_count": count, "asset_count": asset_count,
                "quality_days": quality_count, "catalog_sha256": digest,
                "aligned_scene_count": aligned_count, "aligned_band_count": aligned_band_count,
                "alignment_sha256": alignment_digest}
    finally:
        conn.close()
