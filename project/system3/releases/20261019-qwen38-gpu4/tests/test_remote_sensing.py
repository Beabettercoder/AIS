from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import test_repository
from maritime_agent import api
from maritime_agent.config import Settings
from maritime_agent.database import import_dataset
from maritime_agent.remote_sensing_alignment import build_internal_alignment
from maritime_agent.remote_sensing_database import import_remote_sensing, verify_remote_sensing
from maritime_agent.sqlite_repository import SQLiteRepository


class RemoteSensingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()
        self.root = self.fixture.repo.root
        self.db = Path(self.fixture.tmp.name) / "remote.sqlite3"
        import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        base = self.root / "remote_sensing"
        (base / "00_catalog/items").mkdir(parents=True)
        (base / "01_assets").mkdir()
        (base / "02_quality").mkdir()
        self.asset = base / "01_assets/scene_thumbnail.jpg"
        self.asset.write_bytes(b"jpeg-example")
        record = {"id": "scene", "collection": "sentinel-2-c1-l2a",
                  "datetime_utc": "2024-07-09T17:07:53Z",
                  "bbox_wgs84": [-95.3, 29.25, -94.7, 30.0],
                  "scene_cloud_cover_pct": 49.0, "in_ais_window": True,
                  "catalog_url": "https://example.com/scene",
                  "local_stac_item": "items/scene.json",
                  "downloaded_assets": ["assets/scene_thumbnail.jpg"]}
        (base / "00_catalog/items/scene.json").write_text("{}", encoding="utf-8")
        (base / "00_catalog/candidate_catalog.json").write_text(
            json.dumps({"record_count": 1, "records": [record]}), encoding="utf-8")
        self.manifest = base / "02_quality/asset_manifest.json"
        self.manifest.write_text(json.dumps([{"file": "01_assets/scene_thumbnail.jpg",
                                              "bytes": self.asset.stat().st_size,
                                              "sha256": hashlib.sha256(self.asset.read_bytes()).hexdigest()}]),
                                 encoding="utf-8")
        (base / "02_quality/quality_probe.json").write_text(json.dumps({
            "2024-07-09": {"ais_vessel_hour_positions_17utc": 10, "covered_by_scl": 8,
                           "cloud_or_shadow_at_positions": 6,
                           "cloud_or_shadow_fraction_of_covered": 0.75,
                           "caveat": "hourly positions are not simultaneous imagery"}}), encoding="utf-8")

    def tearDown(self):
        self.fixture.tearDown()

    def test_import_query_preview_and_idempotence(self):
        result = import_remote_sensing(self.root, self.db, "sample")
        self.assertEqual(result["scene_count"], 1)
        self.assertEqual(import_remote_sensing(self.root, self.db, "sample")["status"], "already_imported")
        self.assertTrue(verify_remote_sensing(self.root, self.db, "sample")["valid"])
        repo = SQLiteRepository(self.db, "sample")
        self.assertEqual(repo.overview()["remote_sensing_scene_count"], 1)
        self.assertEqual(repo.search_remote_sensing("2024-07-09")["match_count"], 1)
        empty = repo.search_remote_sensing("2024-07-08")
        self.assertEqual(empty["match_count"], 0)
        self.assertTrue(empty["evidence_id"].startswith("remote_search:"))
        self.assertEqual(repo.assess_remote_ais_overlap("2024-07-09")["scl_covered_count"], 8)
        settings = Settings(self.root, "unused", "", None, 6, "sqlite", self.db, "sample")
        api.repository.cache_clear()
        with patch.object(Settings, "from_env", return_value=settings):
            client = TestClient(api.app)
            self.assertEqual(client.get("/remote-sensing/scene").json()["scene_id"], "scene")
            self.assertEqual(client.get("/remote-sensing/scene/preview").content, b"jpeg-example")

    def test_asset_tamper_rejected_before_write(self):
        self.asset.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            import_remote_sensing(self.root, self.db, "sample")
        self.assertEqual(SQLiteRepository(self.db, "sample").overview()["remote_sensing_scene_count"], 0)

    def test_internal_alignment_distinguishes_scientific_visual_and_quicklook(self):
        validation = self.root / "remote_sensing/01_assets/validation_samples"
        goes_dir = validation / "goes16"
        gibs_dir = validation / "gibs"
        sentinel_dir = validation / "sentinel3"
        goes_dir.mkdir(parents=True)
        gibs_dir.mkdir()
        sentinel_dir.mkdir()

        goes = goes_dir / "OR_ABI-L2-MCMIPC-M6_G16_s20241900901190_e20241900903575_c20241900904093.nc"
        goes.write_bytes(b"netcdf-test")
        gibs = gibs_dir / "viirs_2024-07-08.tif"
        gibs.write_bytes(b"tiff-test")
        # The aligner reads only the JPEG SOF marker to retain image dimensions.
        sentinel = sentinel_dir / "S3A_OL_1_EFR____20240709T160000_20240709T160300_quicklook.jpg"
        sentinel.write_bytes(
            b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xd9"
        )

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        (validation / "validation_manifest.json").write_text(json.dumps([{
            "product": "ABI-L2-MCMIPC", "status": "downloaded",
            "output_path": str(goes), "actual_bytes": goes.stat().st_size,
            "sha256": digest(goes), "acquisition_utc": "2024-07-08T09:02:38Z",
            "url": "https://example.com/goes.nc",
        }]), encoding="utf-8")
        (validation / "gibs_validation_manifest.json").write_text(json.dumps([{
            "path": str(gibs), "bytes": gibs.stat().st_size, "sha256": digest(gibs),
            "product": "VIIRS", "platform": "Suomi-NPP", "observation_date_utc": "2024-07-08",
            "aoi_bbox": [-95.3, 29.25, -94.7, 30.0], "width": 120, "height": 150,
            "url": "https://example.com/viirs.tif",
        }]), encoding="utf-8")
        (sentinel_dir / "sentinel3_quicklook_manifest.json").write_text(json.dumps([{
            "path": str(sentinel), "bytes": sentinel.stat().st_size, "sha256": digest(sentinel),
            "product_name": "S3A_OL_1_EFR____20240709T160000_20240709T160300_20240710T000000_0179_110_000_0000_MAR_O_NT_003.SEN3",
            "sensor": "OLCI", "url": "https://example.com/sentinel.jpg",
        }]), encoding="utf-8")

        aligned = build_internal_alignment(self.root)
        self.assertEqual(aligned["scene_count"], 3)
        self.assertEqual(aligned["band_count"], 22)
        result = import_remote_sensing(self.root, self.db, "sample")
        self.assertEqual(result["aligned_scene_count"], 3)
        self.assertEqual(result["aligned_band_count"], 22)
        self.assertTrue(verify_remote_sensing(self.root, self.db, "sample")["valid"])

        repo = SQLiteRepository(self.db, "sample")
        overview = repo.overview()
        self.assertEqual(overview["remote_sensing_catalog_scene_count"], 1)
        self.assertEqual(overview["remote_sensing_aligned_scene_count"], 3)
        self.assertEqual(overview["remote_sensing_scene_count"], 4)
        goes_scene = repo.get_remote_sensing_scene(goes.stem)
        self.assertEqual(goes_scene["asset_role"], "scientific")
        self.assertTrue(goes_scene["quality_available"])
        self.assertEqual(len(goes_scene["bands"]), 16)
        viirs_scene = repo.get_remote_sensing_scene(gibs.stem)
        self.assertEqual(viirs_scene["asset_role"], "visual")
        self.assertEqual({band["unit"] for band in viirs_scene["bands"]}, {"display_dn"})
        sentinel_scene = repo.get_remote_sensing_scene(
            "S3A_OL_1_EFR____20240709T160000_20240709T160300_20240710T000000_0179_110_000_0000_MAR_O_NT_003.SEN3"
        )
        self.assertEqual(sentinel_scene["asset_role"], "quicklook")
        self.assertIsNone(sentinel_scene["bbox_wgs84"])
