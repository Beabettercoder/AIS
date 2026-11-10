from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path

import test_repository
from maritime_agent.database import import_dataset
from maritime_agent import remote_detection
from maritime_agent.detectors.sar_cfar import detect_sar_targets


class RemoteDetectionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()
        self.root = self.fixture.repo.root
        self.db = self.root / "storage" / "detections.sqlite3"
        import_dataset(self.root, self.db, "sample", "houston", "Sample")
        self.asset = self.root / "remote_sensing" / "sample.tif"
        self.asset.parent.mkdir(parents=True)
        self.asset.write_bytes(b"fake-geotiff-for-detector-adapter-test")
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("INSERT INTO remote_sensing_scenes VALUES (?,?,?,?,?,?,?,?,?,?)",
                         ("sample", "SCENE", "sentinel-1-grd", "2024-07-06T12:23:21Z",
                          "[-95.2,29.5,-94.8,29.9]", None, 1, "https://example.com", "x.json", "{}"))
        self.config = self.root / "job.json"
        self.config.write_text(json.dumps({
            "job_id": "houston-s1-test", "region_id": "houston-port", "scene_id": "SCENE",
            "observed_at_utc": "2024-07-06T12:23:21Z",
            "detector": {"name": "fake", "parameters": {}},
            "assets": [{"asset_id": "VV", "polarization": "VV",
                        "path": "remote_sensing/sample.tif"}],
        }), encoding="utf-8")
        self.original = dict(remote_detection.DETECTORS)

        def fake_detector(assets, parameters):
            return {"detections": [{"rank": 1, "row": 10, "col": 20, "lon": -95.0,
                                     "lat": 29.7, "detection_confidence": 0.8,
                                     "cfar_z": 6.0, "amplitudes": {"VV": 123.0}}],
                    "assets": [{"asset_id": "VV", "polarization": "VV", "crs": "EPSG:4326",
                                "bbox": [-95.2, 29.5, -94.8, 29.9], "width": 100, "height": 100}],
                    "summary": {"candidate_pixels": 1, "peak_count": 1}}

        remote_detection.DETECTORS["fake"] = ("fake_v1", fake_detector)

    def tearDown(self):
        remote_detection.DETECTORS.clear()
        remote_detection.DETECTORS.update(self.original)
        self.fixture.tearDown()

    def test_configured_run_is_idempotent_and_traceable(self):
        first = remote_detection.run_detection(self.config, self.root, self.db, "sample")
        second = remote_detection.run_detection(self.config, self.root, self.db, "sample")
        self.assertEqual(first["detection_count"], 1)
        self.assertNotEqual(first["run_id"], second["run_id"])
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM remote_sensing_detections").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM remote_detection_runs").fetchone()[0], 2)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM remote_detection_run_outputs").fetchone()[0], 2)
            self.assertFalse(conn.execute("PRAGMA foreign_key_check").fetchall())

    def test_dry_run_validates_without_database_writes(self):
        result = remote_detection.run_detection(self.config, self.root, self.db, "sample", dry_run=True)
        self.assertEqual(result["status"], "validated")
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM remote_detection_runs").fetchone()[0], 0)

    def test_cfar_detector_finds_injected_bright_target(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        raster = self.root / "remote_sensing" / "synthetic.tif"
        values = np.random.default_rng(7).normal(100, 5, (128, 128)).clip(1).astype("uint16")
        values[64, 70] = 10000
        with rasterio.open(raster, "w", driver="GTiff", width=128, height=128, count=1,
                           dtype="uint16", crs="EPSG:4326",
                           transform=from_bounds(-95.2, 29.5, -94.8, 29.9, 128, 128), nodata=0) as dst:
            dst.write(values, 1)
        result = detect_sar_targets(
            [{"asset_id": "VV", "polarization": "VV", "path": str(raster)}],
            {"background_radius_px": 12, "guard_radius_px": 2, "threshold_z": 5,
             "min_percentile": 99.5, "cell_size_px": 4, "nms_radius_px": 6,
             "max_detections": 10})
        self.assertTrue(any(abs(item["row"] - 64) <= 1 and abs(item["col"] - 70) <= 1
                            for item in result["detections"]))
        target = min(result["detections"], key=lambda item: abs(item["row"] - 64)
                     + abs(item["col"] - 70))
        self.assertGreaterEqual(target["component_pixel_count"], 1)
        self.assertGreater(target["component_length_m"], 0)
        self.assertGreater(target["component_width_m"], 0)
        self.assertIn("component_orientation_deg", target)

    def test_cfar_detector_rejects_crop_edge_peak(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        raster = self.root / "remote_sensing" / "edge.tif"
        values = np.random.default_rng(9).normal(100, 5, (128, 128)).clip(1).astype("uint16")
        values[1, 64] = 10000
        values[64, 70] = 10000
        with rasterio.open(raster, "w", driver="GTiff", width=128, height=128, count=1,
                           dtype="uint16", crs="EPSG:4326",
                           transform=from_bounds(-95.2, 29.5, -94.8, 29.9, 128, 128), nodata=0) as dst:
            dst.write(values, 1)
        result = detect_sar_targets(
            [{"asset_id": "VV", "polarization": "VV", "path": str(raster)}],
            {"background_radius_px": 12, "guard_radius_px": 2, "threshold_z": 5,
             "min_percentile": 99.0, "cell_size_px": 4, "nms_radius_px": 6,
             "edge_margin_px": 16, "max_detections": 10})
        self.assertFalse(any(item["row"] < 16 for item in result["detections"]))
        self.assertTrue(any(abs(item["row"] - 64) <= 1 and abs(item["col"] - 70) <= 1
                            for item in result["detections"]))

    def test_cfar_detector_applies_aligned_valid_mask(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        raster = self.root / "remote_sensing" / "masked.tif"
        mask = self.root / "remote_sensing" / "water-mask.tif"
        transform = from_bounds(-95.2, 29.5, -94.8, 29.9, 128, 128)
        values = np.random.default_rng(11).normal(100, 5, (128, 128)).clip(1).astype("uint16")
        values[64, 40] = 10000
        values[64, 90] = 10000
        with rasterio.open(raster, "w", driver="GTiff", width=128, height=128, count=1,
                           dtype="uint16", crs="EPSG:4326", transform=transform, nodata=0) as dst:
            dst.write(values, 1)
        water = np.zeros((128, 128), dtype="uint8")
        water[:, :64] = 1
        with rasterio.open(mask, "w", driver="GTiff", width=128, height=128, count=1,
                           dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0) as dst:
            dst.write(water, 1)
        result = detect_sar_targets(
            [{"asset_id": "VV", "polarization": "VV", "path": str(raster)}],
            {"background_radius_px": 12, "guard_radius_px": 2, "threshold_z": 5,
             "min_percentile": 99.0, "cell_size_px": 4, "nms_radius_px": 6,
             "edge_margin_px": 16, "valid_mask_path": str(mask), "max_detections": 10})
        self.assertTrue(any(abs(item["col"] - 40) <= 1 for item in result["detections"]))
        self.assertFalse(any(item["col"] >= 64 for item in result["detections"]))


    def test_cfar_detector_prior_guided_spotlight_and_residual_blind_scan(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds, xy

        raster = self.root / "remote_sensing" / "prior_test.tif"
        transform = from_bounds(-95.2, 29.5, -94.8, 29.9, 128, 128)
        # 背景底噪
        values = np.random.default_rng(42).normal(100, 5, (128, 128)).clip(1).astype("uint16")
        # 强暗船目标（无 AIS 先验）：row 64, col 40
        values[64, 40] = 5000
        # 弱合规船目标（有 AIS 先验，但回波微弱）：row 64, col 80
        values[64, 80] = 118

        with rasterio.open(raster, "w", driver="GTiff", width=128, height=128, count=1,
                           dtype="uint16", crs="EPSG:4326", transform=transform, nodata=0) as dst:
            dst.write(values, 1)

        target_lon, target_lat = xy(transform, 64, 80, offset="center")

        # 1. 常规盲检测试（无先验）：弱目标被滤除
        blind_result = detect_sar_targets(
            [{"asset_id": "VV", "polarization": "VV", "path": str(raster)}],
            {"background_radius_px": 12, "guard_radius_px": 2, "threshold_z": 5.0,
             "min_percentile": 99.0, "cell_size_px": 4, "nms_radius_px": 6,
             "edge_margin_px": 16, "max_detections": 10})
        self.assertFalse(any(abs(item["col"] - 80) <= 2 for item in blind_result["detections"]),
                         "常规盲检应漏检弱散射小船")

        # 2. AIS 先验反向引导模式：弱目标被聚光灯高灵敏度捕获，强目标作为残差暗船捕获
        prior_targets = [{
            "mmsi": "413000001",
            "vessel_name": "TEST_VESSEL",
            "lon": float(target_lon),
            "lat": float(target_lat),
            "sog_kn": 10.5,
            "cog_deg": 90.0,
            "time_delta_seconds": 12.0
        }]
        guided_result = detect_sar_targets(
            [{"asset_id": "VV", "polarization": "VV", "path": str(raster)}],
            {"background_radius_px": 12, "guard_radius_px": 2, "threshold_z": 5.0,
             "min_percentile": 99.0, "cell_size_px": 4, "nms_radius_px": 6,
             "edge_margin_px": 16, "max_detections": 10,
             "prior_targets": prior_targets})

        self.assertEqual(guided_result["summary"]["prior_guided_count"], 1)
        prior_det = next((d for d in guided_result["detections"] if d.get("is_prior_match")), None)
        self.assertIsNotNone(prior_det)
        self.assertEqual(prior_det["matched_mmsi"], "413000001")
        self.assertAlmostEqual(prior_det["row"], 64, delta=2)
        self.assertAlmostEqual(prior_det["col"], 80, delta=2)

        dark_det = next((d for d in guided_result["detections"] if d.get("is_dark_vessel_candidate")), None)
        self.assertIsNotNone(dark_det)
        self.assertAlmostEqual(dark_det["row"], 64, delta=2)
        self.assertAlmostEqual(dark_det["col"], 40, delta=2)

    def test_cfar_detector_doppler_compensation_and_restoration(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds, xy, rowcol
        from maritime_agent.detectors.sar_geometry import resolve_sar_geometry
        from maritime_agent.detectors.doppler_shift import (
            compute_doppler_shifted_position,
            restore_ground_position_from_sar_echo,
            _geodesic_direct,
        )

        raster = self.root / "remote_sensing" / "doppler_test.tif"
        transform = from_bounds(5.20, 59.20, 5.30, 59.30, 1024, 1024)
        values = np.random.default_rng(99).normal(100, 5, (1024, 1024)).clip(1).astype("uint16")

        # 几何基准
        geometry = resolve_sar_geometry({"orbit_pass": "ASCENDING"})

        # 真实船舶海面位置
        vessel_ground_lon, vessel_ground_lat = 5.25, 59.25
        sog_kn = 14.0
        cog_deg = 80.0  # 正对右侧视雷达，产生最大负向方位位移

        # 多普勒等效成像位置
        dop_proj = compute_doppler_shifted_position(
            vessel_ground_lon, vessel_ground_lat, sog_kn, cog_deg, geometry
        )
        echo_lon = dop_proj["image_lon"]
        echo_lat = dop_proj["image_lat"]
        echo_row, echo_col = rowcol(transform, echo_lon, echo_lat)

        # 注入回波
        values[echo_row, echo_col] = 12000

        with rasterio.open(raster, "w", driver="GTiff", width=1024, height=1024, count=1,
                           dtype="uint16", crs="EPSG:4326", transform=transform, nodata=0) as dst:
            dst.write(values, 1)

        # 运行聚光灯检测（启用多普勒补偿）
        prior = {
            "mmsi": "538000002",
            "vessel_name": "FAST_CONTAINER",
            "ground_lon": vessel_ground_lon,
            "ground_lat": vessel_ground_lat,
            "lon": echo_lon,
            "lat": echo_lat,
            "sog_kn": sog_kn,
            "cog_deg": cog_deg,
        }
        res = detect_sar_targets(
            [{"asset_id": "VV", "polarization": "VV", "path": str(raster)}],
            {
                "background_radius_px": 12, "guard_radius_px": 2, "threshold_z": 5.0,
                "cell_size_px": 4, "nms_radius_px": 6, "edge_margin_px": 16,
                "prior_targets": [prior],
                "enable_doppler_compensation": True,
            }
        )

        matched = next((d for d in res["detections"] if d.get("is_prior_match")), None)
        self.assertIsNotNone(matched, "多普勒聚焦应准确捕获等效像面回波")
        self.assertTrue(matched["doppler_compensated"])
        self.assertAlmostEqual(matched["row"], echo_row, delta=2)
        self.assertAlmostEqual(matched["col"], echo_col, delta=2)
        # 逆向复原的地面坐标应精准回归物理真值位置
        self.assertAlmostEqual(matched["lon"], vessel_ground_lon, delta=0.001)
        self.assertAlmostEqual(matched["lat"], vessel_ground_lat, delta=0.001)
        # 地面物理误差应达到米级（通常 < 20米）
        self.assertLess(matched["ground_alignment_error_m"], 25.0)


if __name__ == "__main__":
    unittest.main()

