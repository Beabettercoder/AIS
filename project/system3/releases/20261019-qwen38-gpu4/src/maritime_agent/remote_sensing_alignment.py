"""Build and validate canonical internal alignment for local remote-sensing assets."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .alignment_rules import rules_for


VALIDATION_ROOT = Path("remote_sensing/01_assets/validation_samples")
GOES_MANIFEST = VALIDATION_ROOT / "validation_manifest.json"
GIBS_MANIFEST = VALIDATION_ROOT / "gibs_validation_manifest.json"
SENTINEL3_MANIFEST = VALIDATION_ROOT / "sentinel3/sentinel3_quicklook_manifest.json"
SENTINEL1_MANIFEST = VALIDATION_ROOT / "sentinel1/sentinel1_crop_manifest.json"
ALIGNMENT_MANIFEST = Path("remote_sensing/02_quality/internal_alignment.json")

AOI_BBOX = [-95.3, 29.25, -94.7, 30.0]
UTC = timezone.utc

GOES_BANDS = (
    ("C01", "blue_reflectance", 0.47, "1", 1000.0),
    ("C02", "red_reflectance", 0.64, "1", 500.0),
    ("C03", "vegetation_near_ir_reflectance", 0.86, "1", 1000.0),
    ("C04", "cirrus_reflectance", 1.37, "1", 2000.0),
    ("C05", "snow_ice_near_ir_reflectance", 1.61, "1", 1000.0),
    ("C06", "cloud_particle_shortwave_ir_reflectance", 2.24, "1", 2000.0),
    ("C07", "shortwave_ir_brightness_temperature", 3.90, "K", 2000.0),
    ("C08", "upper_water_vapor_brightness_temperature", 6.19, "K", 2000.0),
    ("C09", "mid_water_vapor_brightness_temperature", 6.93, "K", 2000.0),
    ("C10", "lower_water_vapor_brightness_temperature", 7.34, "K", 2000.0),
    ("C11", "cloud_top_phase_brightness_temperature", 8.44, "K", 2000.0),
    ("C12", "ozone_brightness_temperature", 9.61, "K", 2000.0),
    ("C13", "clean_ir_brightness_temperature", 10.33, "K", 2000.0),
    ("C14", "longwave_ir_brightness_temperature", 11.21, "K", 2000.0),
    ("C15", "dirty_ir_brightness_temperature", 12.29, "K", 2000.0),
    ("C16", "carbon_dioxide_brightness_temperature", 13.28, "K", 2000.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _asset(root: Path, value: str, expected_sha256: str, expected_bytes: int) -> tuple[Path, str]:
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"遥感对齐资产不存在或越过数据根目录: {value}")
    if expected_bytes and path.stat().st_size != int(expected_bytes):
        raise ValueError(f"遥感对齐资产大小不匹配: {path.name}")
    actual = _sha256(path)
    if expected_sha256 and actual != expected_sha256.lower():
        raise ValueError(f"遥感对齐资产 SHA-256 不匹配: {path.name}")
    return path, path.relative_to(root).as_posix()


def _utc_text(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError(f"遥感时间缺少时区: {value}")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _goes_stamp(value: str) -> str:
    # GOES filenames use YYYY + day-of-year + HHMMSS + tenths of a second.
    core = value[:13]
    year, day = int(core[:4]), int(core[4:7])
    hour, minute, second = int(core[7:9]), int(core[9:11]), int(core[11:13])
    when = datetime(year, 1, 1, hour, minute, second, tzinfo=UTC) + timedelta(days=day - 1)
    return when.isoformat().replace("+00:00", "Z")


def _jpeg_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            return None, None
        while True:
            byte = stream.read(1)
            if not byte:
                return None, None
            if byte != b"\xff":
                continue
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if marker in {bytes([code]) for code in range(0xC0, 0xC4)} | {bytes([code]) for code in range(0xC5, 0xC8)} | {bytes([code]) for code in range(0xC9, 0xCC)} | {bytes([code]) for code in range(0xCD, 0xD0)}:
                length = struct.unpack(">H", stream.read(2))[0]
                payload = stream.read(length - 2)
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height
            if marker in (b"\xd8", b"\xd9"):
                continue
            length_raw = stream.read(2)
            if len(length_raw) != 2:
                return None, None
            length = struct.unpack(">H", length_raw)[0]
            stream.seek(length - 2, 1)


def _display_bands() -> list[dict[str, Any]]:
    return [
        {"source_band_name": name, "canonical_variable": canonical, "wavelength_um": None,
         "unit": "display_dn", "data_type": "uint8", "scale_factor": 1.0,
         "add_offset": 0.0, "valid_min": 0.0, "valid_max": 255.0, "nodata": None,
         "band_role": "visual", "quality_band": None}
        for name, canonical in (("R", "display_red"), ("G", "display_green"), ("B", "display_blue"))
    ]


def _goes_records(root: Path) -> list[dict[str, Any]]:
    path = root / GOES_MANIFEST
    if not path.is_file():
        return []
    records = []
    for item in _load_json(path):
        if item.get("product") != "ABI-L2-MCMIPC" or item.get("status") not in {"downloaded", "existing"}:
            continue
        asset, relative = _asset(root, item["output_path"], item.get("sha256") or "",
                                 int(item.get("actual_bytes") or item.get("expected_bytes") or 0))
        match = re.search(r"_s(?P<start>\d{13,14})_e(?P<end>\d{13,14})_", asset.name)
        start = _goes_stamp(match.group("start")) if match else _utc_text(item["acquisition_utc"])
        end = _goes_stamp(match.group("end")) if match else None
        bands = []
        for name, canonical, wavelength, unit, resolution in GOES_BANDS:
            bands.append({"source_band_name": f"CMI_{name}", "canonical_variable": canonical,
                          "wavelength_um": wavelength, "unit": unit,
                          "data_type": "packed_numeric", "scale_factor": None, "add_offset": None,
                          "valid_min": None, "valid_max": None, "nodata": None,
                          "band_role": "scientific", "quality_band": f"DQF_{name}",
                          "native_resolution_m": resolution})
        records.append({
            "scene_id": asset.stem, "collection": "goes-16-abi-l2-mcmipc",
            "platform": "GOES-16", "sensor": "ABI", "product_type": "ABI-L2-MCMIPC",
            "processing_level": "L2+", "start_utc": start, "end_utc": end,
            "center_time_utc": _utc_text(item["acquisition_utc"]), "temporal_precision": "scan",
            "bbox_wgs84": AOI_BBOX, "bbox_role": "analysis_aoi_intersection",
            "source_crs": "GOES-R fixed grid", "analysis_crs": "EPSG:4326",
            "native_resolution_m": None, "grid_width": None, "grid_height": None,
            "grid_pixel_size_x": None, "grid_pixel_size_y": None, "grid_pixel_unit": None,
            "asset_role": "scientific", "file_format": "NetCDF4", "quality_available": True,
            "source_path": relative, "sha256": _sha256(asset), "byte_count": asset.stat().st_size,
            "provenance_url": item.get("url"), "bands": bands,
            "limitations": ["AOI bbox records the analysis intersection, not the full CONUS product footprint.",
                            "Per-variable scale_factor/add_offset remain in the NetCDF and are not invented in the registry."]})
    return records


def _gibs_records(root: Path) -> list[dict[str, Any]]:
    path = root / GIBS_MANIFEST
    if not path.is_file():
        return []
    records = []
    for item in _load_json(path):
        asset, relative = _asset(root, item["path"], item.get("sha256") or "", int(item["bytes"]))
        product = str(item["product"]).upper()
        sensor = product
        platform = str(item["platform"])
        day = datetime.fromisoformat(item["observation_date_utc"]).replace(tzinfo=UTC)
        native = 375.0 if product == "VIIRS" else 250.0
        bbox = [float(value) for value in item["aoi_bbox"]]
        width, height = int(item["width"]), int(item["height"])
        records.append({
            "scene_id": asset.stem,
            "collection": f"{product.lower()}-corrected-reflectance-true-color",
            "platform": platform, "sensor": sensor,
            "product_type": "corrected_reflectance_true_color",
            "processing_level": "visual_derived", "start_utc": day.isoformat().replace("+00:00", "Z"),
            "end_utc": (day + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
            "center_time_utc": None, "temporal_precision": "day",
            "bbox_wgs84": bbox, "bbox_role": "asset_extent",
            "source_crs": "EPSG:4326", "analysis_crs": "EPSG:4326",
            "native_resolution_m": native, "grid_width": width, "grid_height": height,
            "grid_pixel_size_x": (bbox[2] - bbox[0]) / width,
            "grid_pixel_size_y": (bbox[3] - bbox[1]) / height, "grid_pixel_unit": "degree",
            "asset_role": "visual", "file_format": "GeoTIFF", "quality_available": False,
            "source_path": relative, "sha256": _sha256(asset), "byte_count": asset.stat().st_size,
            "provenance_url": item.get("url"), "bands": _display_bands(),
            "limitations": ["Date-composite visualization; exact granule acquisition time is unavailable.",
                            "RGB display values are not physical surface reflectance.",
                            "Output grid spacing must not be reported as native sensor resolution."]})
    return records


def _sentinel3_records(root: Path) -> list[dict[str, Any]]:
    path = root / SENTINEL3_MANIFEST
    if not path.is_file():
        return []
    records = []
    for item in _load_json(path):
        asset, relative = _asset(root, item["path"], item.get("sha256") or "", int(item["bytes"]))
        product_name, sensor = item["product_name"], str(item["sensor"]).upper()
        platform = "Sentinel-3A" if product_name.startswith("S3A_") else "Sentinel-3B"
        match = re.search(r"_(?P<start>\d{8}T\d{6})_(?P<end>\d{8}T\d{6})_", product_name)
        start = datetime.strptime(match.group("start"), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        end = datetime.strptime(match.group("end"), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        width, height = _jpeg_size(asset)
        collection = "sentinel-3-olci-l1-quicklook" if sensor == "OLCI" else "sentinel-3-slstr-l1-quicklook"
        records.append({
            "scene_id": product_name, "collection": collection, "platform": platform,
            "sensor": sensor, "product_type": "OL_1_EFR" if sensor == "OLCI" else "SL_1_RBT",
            "processing_level": "L1", "start_utc": start.isoformat().replace("+00:00", "Z"),
            "end_utc": end.isoformat().replace("+00:00", "Z"),
            "center_time_utc": (start + (end - start) / 2).isoformat().replace("+00:00", "Z"),
            "temporal_precision": "scan", "bbox_wgs84": None,
            "bbox_role": "catalog_intersection_only", "source_crs": None,
            "analysis_crs": "EPSG:4326", "native_resolution_m": 300.0 if sensor == "OLCI" else None,
            "grid_width": width, "grid_height": height, "grid_pixel_size_x": None,
            "grid_pixel_size_y": None, "grid_pixel_unit": "display_pixel",
            "asset_role": "quicklook", "file_format": "JPEG", "quality_available": False,
            "source_path": relative, "sha256": _sha256(asset), "byte_count": asset.stat().st_size,
            "provenance_url": item.get("url"), "bands": _display_bands(),
            "limitations": ["Official quicklook only; it has no georeferenced scientific pixels.",
                            "Full OLCI/SLSTR bands and quality flags require the authenticated product."]})
    return records


def _sentinel1_records(root: Path) -> list[dict[str, Any]]:
    path = root / SENTINEL1_MANIFEST
    if not path.is_file():
        return []
    records = []
    for item in _load_json(path):
        asset, relative = _asset(root, item["path"], item.get("sha256") or "", int(item["bytes"]))
        bbox = [float(value) for value in item["bbox_wgs84"]]
        width, height = int(item["width"]), int(item["height"])
        polarization = str(item["polarization"]).upper()
        scene_id = f'{item["source_scene_id"]}_{polarization}_HOUSTON_CROP'
        records.append({
            "scene_id": scene_id, "collection": "sentinel-1-grd-houston-crop",
            "platform": "Sentinel-1A", "sensor": "C-SAR",
            "product_type": f"IW_GRDH_{polarization}_CROP", "processing_level": "L1",
            "start_utc": _utc_text(item["start_utc"]), "end_utc": _utc_text(item["end_utc"]),
            "center_time_utc": _utc_text(item["center_time_utc"]), "temporal_precision": "scan",
            "bbox_wgs84": bbox, "bbox_role": "cropped_asset_extent",
            "source_crs": "EPSG:4326", "analysis_crs": "EPSG:4326",
            "native_resolution_m": 10.0, "grid_width": width, "grid_height": height,
            "grid_pixel_size_x": (bbox[2] - bbox[0]) / width,
            "grid_pixel_size_y": (bbox[3] - bbox[1]) / height, "grid_pixel_unit": "degree",
            "asset_role": "scientific", "file_format": "GeoTIFF", "quality_available": False,
            "source_path": relative, "sha256": _sha256(asset), "byte_count": asset.stat().st_size,
            "provenance_url": item["provenance_url"],
            "bands": [{"source_band_name": polarization,
                       "canonical_variable": f"sar_{polarization.lower()}_amplitude",
                       "wavelength_um": None, "unit": "digital_number", "data_type": "uint16",
                       "scale_factor": 1.0, "add_offset": 0.0, "valid_min": 0.0,
                       "valid_max": 65535.0, "nodata": "0", "band_role": "scientific",
                       "quality_band": None}],
            "limitations": ["AOI crop from the public Sentinel-1 GRD COG, not the complete SAFE product.",
                            "Digital amplitude is not radiometrically calibrated sigma-nought."]})
    return records


def _validate(records: list[dict[str, Any]]) -> None:
    rules = rules_for("stac_json")
    required = set(rules["canonical_fields"])
    seen: set[str] = set()
    for record in records:
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"遥感对齐记录缺少字段: {', '.join(missing)}")
        if record["scene_id"] in seen:
            raise ValueError(f"遥感对齐场景重复: {record['scene_id']}")
        seen.add(record["scene_id"])
        _utc_text(record["start_utc"])
        if record.get("end_utc"):
            _utc_text(record["end_utc"])
        bbox = record.get("bbox_wgs84")
        if bbox is not None and (len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3])):
            raise ValueError(f"遥感对齐 bbox 无效: {record['scene_id']}")
        if record["asset_role"] not in {"scientific", "visual", "quicklook"}:
            raise ValueError(f"遥感资产角色无效: {record['scene_id']}")
        names = [band["source_band_name"] for band in record["bands"]]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"遥感波段缺失或重复: {record['scene_id']}")


def build_internal_alignment(source_root: Path) -> dict[str, Any]:
    root = source_root.expanduser().resolve()
    records = (_goes_records(root) + _gibs_records(root) + _sentinel3_records(root)
               + _sentinel1_records(root))
    if not records:
        raise ValueError("没有找到可对齐的遥感验证资产")
    _validate(records)
    rules = rules_for("stac_json")
    payload = {
        "schema_version": rules["schema_version"],
        "ruleset_version": rules["ruleset_version"],
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "record_count": len(records),
        "band_count": sum(len(record["bands"]) for record in records),
        "records": records,
    }
    output = root / ALIGNMENT_MANIFEST
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return {"status": "aligned", "scene_count": payload["record_count"],
            "band_count": payload["band_count"], "manifest": str(output)}


def load_internal_alignment(source_root: Path) -> tuple[list[dict[str, Any]], str] | tuple[list, None]:
    root = source_root.expanduser().resolve()
    path = root / ALIGNMENT_MANIFEST
    if not path.is_file():
        return [], None
    payload = _load_json(path)
    rules = rules_for("stac_json")
    if payload.get("schema_version") != rules["schema_version"] or payload.get("ruleset_version") != rules["ruleset_version"]:
        raise ValueError("遥感内部对齐清单版本不匹配，请重新生成")
    records = payload.get("records")
    if not isinstance(records, list) or payload.get("record_count") != len(records):
        raise ValueError("遥感内部对齐清单记录数无效")
    _validate(records)
    for record in records:
        asset = root / record["source_path"]
        _asset(root, str(asset), record["sha256"], int(record["byte_count"]))
    return records, _sha256(path)
