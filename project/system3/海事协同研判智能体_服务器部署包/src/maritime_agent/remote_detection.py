"""Configuration-driven remote target detection and normalized persistence."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .database import migrate_database
from .detectors import (
    DETECTOR_VERSION,
    detect_sar_targets,
    resolve_sar_geometry,
    compute_doppler_shifted_position,
)


DETECTORS = {"sar_cfar": (DETECTOR_VERSION, detect_sar_targets)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    def _default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: str) -> str:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("observed_at_utc 必须包含时区")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_detection_config(config_path: Path, data_root: Path) -> dict[str, Any]:
    path = config_path.expanduser().resolve()
    root = data_root.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    required = ("job_id", "region_id", "scene_id", "observed_at_utc", "detector", "assets")
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError("检测配置缺少字段: " + ", ".join(missing))
    detector = config["detector"]
    if not isinstance(detector, dict) or detector.get("name") not in DETECTORS:
        raise ValueError(f"不支持的检测器: {detector.get('name') if isinstance(detector, dict) else detector}")
    observed = _utc(config["observed_at_utc"])
    detector_parameters = dict(detector.get("parameters") or {})
    if detector_parameters.get("valid_mask_path"):
        mask_path = Path(detector_parameters["valid_mask_path"])
        if not mask_path.is_absolute():
            mask_path = root / mask_path
        mask_path = mask_path.resolve()
        if not mask_path.is_relative_to(root) or not mask_path.is_file():
            raise ValueError("valid_mask_path 不存在或越过数据根目录")
        detector_parameters["valid_mask_path"] = str(mask_path)
    normalized_assets = []
    seen = set()
    for raw in config["assets"]:
        asset_id = str(raw.get("asset_id") or raw.get("polarization") or "").strip()
        if not asset_id or asset_id in seen:
            raise ValueError("每个遥感资产必须具有唯一 asset_id 或 polarization")
        seen.add(asset_id)
        asset_path = Path(raw["path"])
        if not asset_path.is_absolute():
            asset_path = root / asset_path
        asset_path = asset_path.resolve()
        if not asset_path.is_relative_to(root) or not asset_path.is_file():
            raise ValueError(f"遥感检测资产不存在或越过数据根目录: {raw['path']}")
        actual_sha = _sha256(asset_path)
        if raw.get("sha256") and actual_sha != str(raw["sha256"]).lower():
            raise ValueError(f"遥感检测资产 SHA-256 不匹配: {asset_path.name}")
        normalized_assets.append({"asset_id": asset_id, "polarization": raw.get("polarization"),
                                  "path": str(asset_path), "sha256": actual_sha})
    output_dir = Path(config.get("output_dir") or "remote_sensing/03_detections")
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output_dir 必须位于数据根目录内")
    return {**config, "observed_at_utc": observed, "assets": normalized_assets,
            "detector": {**detector, "parameters": detector_parameters},
            "output_dir": str(output_dir), "config_path": str(path)}


def _export_results(config: dict[str, Any], run_id: str, detections: list[dict[str, Any]]) -> dict[str, str]:
    output = Path(config["output_dir"]) / config["job_id"]
    output.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": config["job_id"], "run_id": run_id, "scene_id": config["scene_id"],
               "observed_at_utc": config["observed_at_utc"], "detections": detections}
    json_path = output / "detections.json"
    geojson_path = output / "detections.geojson"
    json_tmp, geojson_tmp = json_path.with_suffix(".json.tmp"), geojson_path.with_suffix(".geojson.tmp")
    json_tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    features = [{"type": "Feature", "id": item["detection_id"],
                 "geometry": {"type": "Point", "coordinates": [item["lon"], item["lat"]]},
                 "properties": {key: value for key, value in item.items() if key not in {"lon", "lat"}}}
                for item in detections]
    geojson_tmp.write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    json_tmp.replace(json_path)
    geojson_tmp.replace(geojson_path)
    return {"json": str(json_path), "geojson": str(geojson_path)}


def run_detection(config_path: Path, data_root: Path, db_path: Path, dataset_id: str,
                  dry_run: bool = False) -> dict[str, Any]:
    config = load_detection_config(config_path, data_root)
    detector_name = config["detector"]["name"]
    detector_version, detector = DETECTORS[detector_name]
    if dry_run:
        return {"status": "validated", "job_id": config["job_id"], "detector": detector_name,
                "detector_version": detector_version, "asset_count": len(config["assets"]),
                "observed_at_utc": config["observed_at_utc"]}

    migrate_database(db_path)
    run_id = str(uuid4())
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        dataset = conn.execute("SELECT 1 FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if dataset is None:
            raise ValueError(f"数据库中找不到 dataset_id: {dataset_id}")
        scene = conn.execute(
            "SELECT 1 FROM remote_sensing_scenes WHERE dataset_id=? AND scene_id=? UNION ALL "
            "SELECT 1 FROM remote_sensing_aligned_scenes WHERE dataset_id=? AND scene_id=? LIMIT 1",
            (dataset_id, config["scene_id"], dataset_id, config["scene_id"])).fetchone()
        if scene is None:
            raise ValueError(f"数据库中找不到遥感场景: {config['scene_id']}")
        with conn:
            conn.execute(
                "INSERT INTO remote_detection_jobs VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(dataset_id,job_id) DO UPDATE SET scene_id=excluded.scene_id," 
                "region_id=excluded.region_id,detector_name=excluded.detector_name,config_json=excluded.config_json",
                (dataset_id, config["job_id"], config["scene_id"], config["region_id"], detector_name,
                 _json(config), _now()))
            conn.execute("INSERT INTO remote_detection_runs VALUES (?,?,?,?,?,?,?,?,?)",
                         (dataset_id, run_id, config["job_id"], detector_version, "running",
                          _now(), None, _json({}), None))
        parameters = dict(config["detector"].get("parameters") or {})
        enable_doppler = bool(parameters.get("enable_doppler_compensation", True))
        geometry = resolve_sar_geometry(parameters)
        parameters["sar_geometry"] = geometry.to_dict()
        parameters["enable_doppler_compensation"] = enable_doppler

        if "prior_targets" not in parameters and config.get("observed_at_utc"):
            try:
                obs_t = config["observed_at_utc"]
                obs_dt = datetime.fromisoformat(obs_t.replace("Z", "+00:00"))
                lower_utc = (obs_dt - timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
                upper_utc = (obs_dt + timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
                ais_rows = conn.execute(
                    "SELECT mmsi, vessel_name, timestamp_utc, lon, lat, sog_kn, cog_deg FROM ("
                    "  SELECT mmsi, vessel_name, timestamp_utc, lon, lat, sog_kn, cog_deg "
                    "  FROM ais_messages WHERE dataset_id=? AND timestamp_utc>=? AND timestamp_utc<=? "
                    "  UNION ALL "
                    "  SELECT mmsi, vessel_name, timestamp_utc, lon, lat, sog_knots as sog_kn, cog_degrees as cog_deg "
                    "  FROM ingested_ais_positions WHERE timestamp_utc>=? AND timestamp_utc<=?"
                    ") ORDER BY mmsi, timestamp_utc",
                    (dataset_id, lower_utc, upper_utc, lower_utc, upper_utc)
                ).fetchall()
                by_mmsi: dict[str, list[Any]] = {}
                for r in ais_rows:
                    by_mmsi.setdefault(str(r[0]), []).append(r)
                priors = []
                for mmsi_str, points in by_mmsi.items():
                    pts_sorted = sorted(points, key=lambda x: abs((datetime.fromisoformat(x[2].replace("Z", "+00:00")) - obs_dt).total_seconds()))
                    closest = pts_sorted[0]
                    p_dt = datetime.fromisoformat(closest[2].replace("Z", "+00:00"))
                    dt_sec = abs((p_dt - obs_dt).total_seconds())
                    reckoned_lon, reckoned_lat = float(closest[3]), float(closest[4])
                    sog, cog = closest[5], closest[6]
                    if dt_sec > 2.0 and sog is not None and sog > 0.5 and cog is not None:
                        travel_m = float(sog) * 0.514444 * dt_sec
                        direction = 1.0 if obs_dt > p_dt else -1.0
                        rad_cog = math.radians(float(cog))
                        dx_m = travel_m * math.sin(rad_cog) * direction
                        dy_m = travel_m * math.cos(rad_cog) * direction
                        d_lat = dy_m / 111320.0
                        d_lon = dx_m / (111320.0 * max(0.01, math.cos(math.radians(reckoned_lat))))
                        reckoned_lon += d_lon
                        reckoned_lat += d_lat

                    # 多普勒频移前向投影：解算雷达像面等效船位用于聚光灯开窗
                    if enable_doppler and sog is not None and sog > 0.5 and cog is not None:
                        dop = compute_doppler_shifted_position(
                            ground_lon=reckoned_lon,
                            ground_lat=reckoned_lat,
                            sog_kn=float(sog),
                            cog_deg=float(cog),
                            geometry=geometry,
                        )
                        image_spotlight_lon = dop["image_lon"]
                        image_spotlight_lat = dop["image_lat"]
                        azimuth_shift_m = dop["azimuth_shift_m"]
                        v_radial_slant = dop["v_radial_slant_mps"]
                        v_radial_ground = dop["v_radial_ground_mps"]
                    else:
                        image_spotlight_lon = reckoned_lon
                        image_spotlight_lat = reckoned_lat
                        azimuth_shift_m = 0.0
                        v_radial_slant = 0.0
                        v_radial_ground = 0.0

                    priors.append({
                        "mmsi": mmsi_str,
                        "vessel_name": closest[1],
                        "ground_lon": reckoned_lon,
                        "ground_lat": reckoned_lat,
                        "lon": image_spotlight_lon,
                        "lat": image_spotlight_lat,
                        "sog_kn": sog,
                        "cog_deg": cog,
                        "time_delta_seconds": round(dt_sec, 1),
                        "doppler_shift_m": azimuth_shift_m,
                        "v_radial_slant_mps": v_radial_slant,
                        "v_radial_ground_mps": v_radial_ground,
                    })
                parameters["prior_targets"] = priors
            except Exception:
                pass

        result = detector(config["assets"], parameters)
        normalized = []
        seen_coords = set()
        for item in result["detections"]:
            coord_key = (int(item["row"]), int(item["col"]))
            if coord_key in seen_coords:
                continue
            seen_coords.add(coord_key)
            identity = _json([dataset_id, config["job_id"], detector_version,
                              int(item["row"]), int(item["col"])])
            detection_id = "remote-det:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            normalized.append({**item, "detection_id": detection_id})
        output_paths = _export_results(config, run_id, normalized)
        summary = {**result["summary"], "detection_count": len(normalized),
                   "output_paths": output_paths, "confidence_kind": "uncalibrated_detector_score"}
        with conn:
            for asset, meta in zip(config["assets"], result["assets"]):
                conn.execute("INSERT INTO remote_detection_run_assets VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (dataset_id, run_id, asset["asset_id"], asset.get("polarization"),
                              asset["path"], asset["sha256"], meta["crs"], _json(meta["bbox"]),
                              meta["width"], meta["height"]))
            for item in normalized:
                attributes = {key: value for key, value in item.items()
                              if key not in {"detection_id", "lon", "lat", "detection_confidence",
                                             "component_length_m", "component_width_m"}}
                provenance = {"run_id": run_id, "job_id": config["job_id"],
                              "detector": detector_name, "detector_version": detector_version,
                              "config_path": config["config_path"]}
                conn.execute(
                    "INSERT INTO remote_sensing_detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(dataset_id,detection_id) DO UPDATE SET scene_id=excluded.scene_id," 
                    "observed_at_utc=excluded.observed_at_utc,lon=excluded.lon,lat=excluded.lat," 
                    "object_type=excluded.object_type,detection_confidence=excluded.detection_confidence," 
                    "attributes_json=excluded.attributes_json,provenance_json=excluded.provenance_json",
                    (dataset_id, item["detection_id"], config["scene_id"], config["observed_at_utc"],
                     item["lon"], item["lat"], "vessel_candidate", item["detection_confidence"],
                     item.get("component_length_m"), item.get("component_width_m"),
                     _json(attributes), _json(provenance)))
                conn.execute("INSERT OR REPLACE INTO remote_detection_run_outputs VALUES (?,?,?)",
                             (dataset_id, run_id, item["detection_id"]))
            conn.execute("UPDATE remote_detection_runs SET status='completed',completed_at_utc=?," 
                         "summary_json=? WHERE dataset_id=? AND run_id=?",
                         (_now(), _json(summary), dataset_id, run_id))
        return {"status": "completed", "dataset_id": dataset_id, "job_id": config["job_id"],
                    "run_id": run_id, "detector_version": detector_version, **summary}
    except Exception as exc:
        with conn:
            conn.execute("UPDATE remote_detection_runs SET status='failed',completed_at_utc=?,error_text=? "
                         "WHERE dataset_id=? AND run_id=?", (_now(), str(exc)[:2000], dataset_id, run_id))
        raise
    finally:
        conn.close()
