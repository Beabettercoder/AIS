"""Plan missing maritime data from any registered source, without inventing coverage."""

from __future__ import annotations

import json
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

from .database import open_readonly
from .gis_sources import NOAA_LAYER_URL, _bbox


KINDS = {"ais", "gis", "remote_sensing", "weather", "events"}
# Bounded connector scope verified against NOAA; it is intentionally the port core,
# not a claim about the full spatial coverage of the base aggregate dataset.
HOUSTON_CONTEXT_BBOX = [-95.3, 29.68, -95.15, 29.78]


def _context(db_path: Path, source_id: str) -> dict[str, Any]:
    with closing(open_readonly(db_path)) as conn:
        def with_region(context: dict[str, Any], source_kind: str) -> dict[str, Any]:
            binding = conn.execute(
                "SELECT b.region_id,b.validation_status,r.canonical_name,r.bbox_json "
                "FROM source_region_bindings b JOIN regions r ON r.dataset_id=b.dataset_id "
                "AND r.region_id=b.region_id WHERE b.source_kind=? AND b.source_id=?",
                (source_kind, source_id)).fetchone()
            if binding:
                context.update({"region_id": binding["region_id"],
                                "region_label": binding["canonical_name"],
                                "region_validation_status": binding["validation_status"],
                                "region_bbox_wgs84": json.loads(binding["bbox_json"])
                                if binding["bbox_json"] else None})
            return context
        row = conn.execute("SELECT source_kind,region_label,first_utc,last_utc,profile_json "
                           "FROM ingested_sources WHERE source_id=?", (source_id,)).fetchone()
        if row:
            profile = json.loads(row["profile_json"])
            return with_region({"source_id": source_id,
                                "kind": "ais" if row["source_kind"] == "ais_csv" else "remote_sensing",
                                "region_label": row["region_label"],
                                "bbox_wgs84": profile.get("observed_bbox_wgs84"),
                                "first_utc": row["first_utc"], "last_utc": row["last_utc"]},
                               row["source_kind"])
        row = conn.execute("SELECT source_kind,region_label,bbox_json FROM external_gis_sources "
                           "WHERE source_id=?", (source_id,)).fetchone()
        if row:
            return with_region({"source_id": source_id, "kind": "gis", "region_label": row["region_label"],
                                "bbox_wgs84": json.loads(row["bbox_json"]), "first_utc": None, "last_utc": None},
                               row["source_kind"])
        row = conn.execute("SELECT source_name,region_label,first_utc,last_utc,bbox_json "
                           "FROM weather_sources WHERE source_id=?", (source_id,)).fetchone()
        if row:
            return with_region({"source_id": source_id, "kind": "weather",
                                "region_label": row["region_label"],
                                "bbox_wgs84": json.loads(row["bbox_json"]),
                                "first_utc": row["first_utc"], "last_utc": row["last_utc"]},
                               "weather_csv")
        row = conn.execute("SELECT region_label,observed_at_utc,profile_json FROM remote_intake_sources "
                           "WHERE source_id=?", (source_id,)).fetchone()
        if row:
            profile = json.loads(row["profile_json"])
            return with_region({"source_id": source_id, "kind": "remote_sensing",
                                "region_label": row["region_label"],
                                "bbox_wgs84": profile.get("observed_bbox_wgs84"),
                                "first_utc": row["observed_at_utc"],
                                "last_utc": row["observed_at_utc"]}, "remote_geotiff")
        row = conn.execute("SELECT dataset_id,port_id FROM datasets WHERE dataset_id=?",
                           (source_id,)).fetchone()
        if row and row["port_id"] == "houston":
            bounds = conn.execute("SELECT MIN(timestamp_hour_utc),MAX(timestamp_hour_utc) "
                                  "FROM port_hourly WHERE dataset_id=?", (source_id,)).fetchone()
            return {"source_id": source_id, "kind": "ais", "region_label": "houston",
                    "bbox_wgs84": HOUSTON_CONTEXT_BBOX, "bbox_note": "Houston 港核心区外部数据检索范围，不代表原数据完整覆盖",
                    "first_utc": bounds[0], "last_utc": bounds[1]}
    raise ValueError("找不到该来源，或其位置范围尚未登记")


def discover_complementary_data(db_path: Path, source_id: str, desired_kinds: list[str],
                                date_utc: str = "") -> dict[str, Any]:
    if not desired_kinds or len(desired_kinds) > len(KINDS) or any(kind not in KINDS for kind in desired_kinds):
        raise ValueError("desired_kinds 必须从 ais、gis、remote_sensing、weather、events 中选择")
    if date_utc:
        try:
            date.fromisoformat(date_utc)
        except ValueError as exc:
            raise ValueError("date_utc 必须是 YYYY-MM-DD") from exc
    context = _context(db_path, source_id)
    bbox = context.get("bbox_wgs84")
    if bbox:
        # Point-only or line-only uploads can have a zero-width envelope.
        bbox = [bbox[0] - (0.05 if bbox[0] == bbox[2] else 0),
                bbox[1] - (0.05 if bbox[1] == bbox[3] else 0),
                bbox[2] + (0.05 if bbox[0] == bbox[2] else 0),
                bbox[3] + (0.05 if bbox[1] == bbox[3] else 0)]
        try:
            bbox = _bbox(bbox)
        except ValueError:
            bbox = None
    candidates = []
    for kind in dict.fromkeys(desired_kinds):
        if kind == context["kind"]:
            candidates.append({"kind": kind, "status": "provided", "source_id": source_id})
        elif not bbox:
            candidates.append({"kind": kind, "status": "needs_spatial_scope",
                               "note": "来源未提供可用于外部检索的有界空间范围"})
        elif kind == "gis":
            us_candidate = -170 < bbox[0] < -60 and 15 < bbox[1] < 75
            candidates.append({"kind": kind, "status": "connector_available" if us_candidate else "no_connector",
                               "provider": "NOAA ENC Direct to GIS" if us_candidate else None,
                               "retrieval": "bounded_auto_fetch" if us_candidate else None,
                               "url": NOAA_LAYER_URL if us_candidate else None,
                               "note": "当前 NOAA 快照不能作为历史边界证明" if us_candidate else "需配置当地 GIS 来源或上传 GeoJSON"})
        elif kind == "ais":
            us_candidate = -170 < bbox[0] < -60 and 15 < bbox[1] < 75
            candidates.append({"kind": kind,
                               "status": ("needs_date" if not date_utc else "catalog_available") if us_candidate else "no_connector",
                               "provider": "NOAA AccessAIS" if us_candidate else None,
                               "retrieval": "interactive_download_then_upload" if us_candidate else None,
                               "url": "https://marinecadastre.gov/accessais/" if us_candidate else None,
                               "note": "目录/下载入口不是已取得的 AIS 点位；尚无自动下载连接器。" if us_candidate
                                       else "需配置当地 AIS 来源或上传 CSV"})
        elif kind == "remote_sensing":
            candidates.append({"kind": kind, "status": "needs_date" if not date_utc else "catalog_available",
                               "provider": "Copernicus Data Space STAC", "retrieval": "catalog_search_only",
                               "url": "https://stac.dataspace.copernicus.eu/v1/",
                               "note": "场景目录不代表影像已下载，也不代表船舶已被识别。"})
        else:
            candidates.append({"kind": kind, "status": "no_connector",
                               "note": "该类型的数据源连接器尚未配置；可先上传或配置可信来源。"})
    return {"input": context, "date_utc": date_utc or None, "search_bbox_wgs84": bbox,
            "candidates": candidates,
            "scope_note": "发现候选来源不等于取得数据；取得后仍须校验时间、空间、质量和使用条件。"}
