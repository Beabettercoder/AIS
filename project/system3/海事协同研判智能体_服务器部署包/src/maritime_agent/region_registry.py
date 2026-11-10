"""Canonical study-region registry and source-to-region validation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import sqlite3
import unicodedata
import uuid
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_region_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def _bbox(value: Any) -> list[float] | None:
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
        west, south, east, north = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        return None
    return [west, south, east, north]


def bbox_polygon(bbox: list[float] | None) -> dict[str, Any] | None:
    if not bbox:
        return None
    west, south, east, north = bbox
    return {"type": "Polygon", "coordinates": [[[west, south], [east, south],
             [east, north], [west, north], [west, south]]]}


def bbox_overlap_ratio(region_bbox: list[float] | None,
                       observed_bbox: list[float] | None) -> float | None:
    """Return overlap divided by the smaller extent; handle point extents explicitly."""
    if not region_bbox or not observed_bbox:
        return None
    rw, rs, re, rn = region_bbox
    ow, os, oe, on = observed_bbox
    if ow == oe and os == on:
        return 1.0 if rw <= ow <= re and rs <= os <= rn else 0.0
    if rw == re and rs == rn:
        return 1.0 if ow <= rw <= oe and os <= rs <= on else 0.0
    width = max(0.0, min(re, oe) - max(rw, ow))
    height = max(0.0, min(rn, on) - max(rs, os))
    intersection = width * height
    r_area = max(0.0, (re - rw) * (rn - rs))
    o_area = max(0.0, (oe - ow) * (on - os))
    denominator = min(area for area in (r_area, o_area) if area > 0) if r_area > 0 and o_area > 0 else 0
    return intersection / denominator if denominator else 0.0


def _dataset_id(conn: sqlite3.Connection, requested: str | None) -> str:
    if requested:
        if conn.execute("SELECT 1 FROM datasets WHERE dataset_id=?", (requested,)).fetchone():
            return requested
        raise ValueError(f"数据集不存在: {requested}")
    rows = conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id").fetchall()
    if len(rows) != 1:
        raise ValueError("无法唯一确定研究区所属 dataset_id")
    return str(rows[0][0])


def ensure_region(conn: sqlite3.Connection, dataset_id: str | None, region_id: str | None,
                  label: str, observed_bbox: Any = None, *, strict: bool = True) -> dict[str, Any]:
    dataset = _dataset_id(conn, dataset_id)
    name = label.strip()[:120]
    if not name:
        raise ValueError("研究区域名称不能为空")
    bbox = _bbox(observed_bbox)
    row = None
    if region_id:
        row = conn.execute("SELECT * FROM regions WHERE dataset_id=? AND region_id=? AND status!='archived'",
                           (dataset, region_id)).fetchone()
        if row is None:
            raise ValueError("所选研究区域不存在或已经归档")
    else:
        normalized = normalize_region_name(name)
        if not normalized:
            raise ValueError("研究区域名称无效")
        row = conn.execute("SELECT * FROM regions WHERE dataset_id=? AND normalized_name=? AND status!='archived'",
                           (dataset, normalized)).fetchone()
        if row is None:
            now = _now()
            region_id = "region-" + uuid.uuid4().hex[:16]
            boundary = bbox_polygon(bbox)
            conn.execute(
                "INSERT INTO regions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (dataset, region_id, name, normalized, "[]",
                 json.dumps(boundary, ensure_ascii=False) if boundary else None,
                 json.dumps(bbox) if bbox else None, "observed_extent",
                 "provisional", now, now))
            row = conn.execute("SELECT * FROM regions WHERE dataset_id=? AND region_id=?",
                               (dataset, region_id)).fetchone()
    columns = [item[1] for item in conn.execute("PRAGMA table_info(regions)")]
    item = dict(zip(columns, row)) if not isinstance(row, sqlite3.Row) else dict(row)
    region_bbox = _bbox(item.get("bbox_json"))
    ratio = bbox_overlap_ratio(region_bbox, bbox)
    if ratio == 0.0 and strict:
        raise ValueError(f"数据空间范围与研究区域“{item['canonical_name']}”不相交，请重新选择研究区域")
    status = "unverified" if ratio is None else "verified" if ratio >= 0.5 else "partial"
    return {"dataset_id": dataset, "region_id": item["region_id"],
            "region_label": item["canonical_name"], "region_bbox_wgs84": region_bbox,
            "observed_bbox_wgs84": bbox, "overlap_ratio": ratio,
            "validation_status": status}


def bind_source(conn: sqlite3.Connection, region: dict[str, Any], source_kind: str,
                source_id: str) -> None:
    now = _now()
    details = {"method": "bbox_intersection_over_smaller_extent_v1",
               "region_bbox_wgs84": region.get("region_bbox_wgs84"),
               "observed_bbox_wgs84": region.get("observed_bbox_wgs84")}
    conn.execute(
        "INSERT OR REPLACE INTO source_region_bindings VALUES (?,?,?,?,?,?,?,?,?)",
        (region["dataset_id"], source_kind, source_id, region["region_id"],
         json.dumps(region.get("observed_bbox_wgs84")), region.get("overlap_ratio"),
         region["validation_status"], json.dumps(details, ensure_ascii=False), now))
    # 动态扩展海区边界为最大外接矩形 (Union Bounding Box)
    obs = region.get("observed_bbox_wgs84")
    reg = region.get("region_bbox_wgs84")
    if obs and reg:
        union_bbox = [min(reg[0], obs[0]), min(reg[1], obs[1]),
                      max(reg[2], obs[2]), max(reg[3], obs[3])]
        if union_bbox != reg:
            poly = bbox_polygon(union_bbox)
            conn.execute(
                "UPDATE regions SET bbox_json=?, boundary_geojson=?, updated_at_utc=? WHERE dataset_id=? AND region_id=?",
                (json.dumps(union_bbox), json.dumps(poly, ensure_ascii=False) if poly else None,
                 now, region["dataset_id"], region["region_id"]))


def promote_gis_boundary(conn: sqlite3.Connection, region: dict[str, Any],
                         observed_bbox: Any, source_id: str) -> dict[str, Any]:
    """Promote an uploaded GIS extent from provisional envelope to region boundary."""
    bbox = _bbox(observed_bbox)
    if not bbox:
        return region
    boundary = bbox_polygon(bbox)
    now = _now()
    conn.execute(
        "UPDATE regions SET bbox_json=?,boundary_geojson=?,boundary_source=?,status='active',updated_at_utc=? "
        "WHERE dataset_id=? AND region_id=?",
        (json.dumps(bbox), json.dumps(boundary, ensure_ascii=False),
         f"gis_geojson:{source_id}", now, region["dataset_id"], region["region_id"]))
    return {**region, "region_bbox_wgs84": bbox, "overlap_ratio": 1.0,
            "validation_status": "verified", "status": "active",
            "boundary_source": f"gis_geojson:{source_id}"}


def list_regions(conn: sqlite3.Connection, dataset_id: str | None) -> dict[str, Any]:
    dataset = _dataset_id(conn, dataset_id)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT r.*,COUNT(b.source_id) AS source_count FROM regions r "
        "LEFT JOIN source_region_bindings b ON b.dataset_id=r.dataset_id AND b.region_id=r.region_id "
        "WHERE r.dataset_id=? AND r.status!='archived' GROUP BY r.dataset_id,r.region_id "
        "ORDER BY r.canonical_name", (dataset,)).fetchall()
    return {"dataset_id": dataset, "regions": [
        {"region_id": row["region_id"], "canonical_name": row["canonical_name"],
         "aliases": json.loads(row["aliases_json"]), "bbox_wgs84": _bbox(row["bbox_json"]),
         "boundary_source": row["boundary_source"], "status": row["status"],
         "source_count": row["source_count"]} for row in rows]}


def backfill_legacy_regions(conn: sqlite3.Connection) -> None:
    """Bind pre-v23 sources when the database contains one unambiguous dataset."""
    datasets = conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id").fetchall()
    if len(datasets) != 1:
        return
    dataset = str(datasets[0][0])
    specs = [
        ("ingested_sources", "source_kind", "profile_json"),
        ("external_gis_sources", "source_kind", "bbox_json"),
        ("weather_sources", None, "bbox_json"),
        ("remote_intake_sources", None, "profile_json"),
    ]
    for table, kind_column, bbox_column in specs:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if not exists:
            continue
        kind_sql = kind_column if kind_column else "?"
        args: tuple[Any, ...] = () if kind_column else (("weather_csv" if table == "weather_sources" else "remote_geotiff"),)
        rows = conn.execute(f"SELECT source_id,{kind_sql} AS source_kind,region_label,{bbox_column} AS bbox_value FROM {table}", args).fetchall()
        for source_id, source_kind, label, bbox_value in rows:
            bbox = None
            try:
                parsed = json.loads(bbox_value or "null")
                bbox = parsed.get("observed_bbox_wgs84") if isinstance(parsed, dict) else parsed
            except (TypeError, json.JSONDecodeError):
                pass
            try:
                region = ensure_region(conn, dataset, None, label, bbox, strict=False)
            except ValueError:
                # Legacy free-text labels were never spatially enforced. Preserve them and
                # expose the mismatch for later review instead of blocking schema upgrade.
                region = ensure_region(conn, dataset, None, label, None, strict=False)
                region["observed_bbox_wgs84"] = bbox
                region["overlap_ratio"] = bbox_overlap_ratio(region.get("region_bbox_wgs84"), bbox)
            region["validation_status"] = "legacy"
            bind_source(conn, region, source_kind, source_id)
    conn.commit()
