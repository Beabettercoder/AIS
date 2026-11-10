from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import test_repository
from maritime_agent import api
from maritime_agent.config import Settings
from maritime_agent.database import import_dataset
from maritime_agent.gis_database import GIS_FILES, import_gis, verify_gis
from maritime_agent.sqlite_repository import SQLiteRepository


class GisTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()
        self.root = self.fixture.repo.root
        self.db = Path(self.fixture.tmp.name) / "gis.sqlite3"
        import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        self.object_file = self.root / GIS_FILES["gis_objects"]
        self.object_file.parent.mkdir(parents=True, exist_ok=True)
        self.feature = {"type": "Feature", "properties": {
            "object_id": "AREA_1", "object_type": "named_water_area", "object_name": "Test Bay",
            "source_authority": "Test NOAA", "source_dataset": "Test ENC",
            "source_url": "https://example.com/gis", "source_date": "20230101"},
            "geometry": {"type": "Polygon", "coordinates": [[[-95.2, 29.4], [-95.0, 29.4],
                                                            [-95.0, 29.6], [-95.2, 29.4]]]}}
        self.write_feature()
        hour_file = self.root / GIS_FILES["gis_area_hourly"]
        hour_file.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"timestamp_hour_utc": f"2024-07-06T{hour:02}:00:00Z", "primary_area_id": "AREA_1",
                 "primary_area_type": "named_water_area", "unique_mmsi": str(2 + hour),
                 "vessel_hour_rows": str(2 + hour), "moving_mmsi": "1", "stopped_mmsi": str(1 + hour)}
                for hour in (0, 1)]
        with hour_file.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        self.fixture.tearDown()

    def write_feature(self):
        self.object_file.write_text(json.dumps({"type": "FeatureCollection", "features": [self.feature]}), encoding="utf-8")

    def test_import_search_visualization_api_and_reimport(self):
        self.assertEqual(import_gis(self.root, self.db, "sample")["row_counts"],
                         {"gis_objects": 1, "gis_area_hourly": 2})
        self.assertEqual(import_gis(self.root, self.db, "sample")["status"], "already_imported")
        self.assertEqual(verify_gis(self.root, self.db, "sample")["integrity"], "ok")
        repo = SQLiteRepository(self.db, "sample")
        self.assertEqual(repo.overview()["gis_object_count"], 1)
        found = repo.search_gis_objects("Test Bay", "2024-07-06")
        self.assertEqual(found["objects"][0]["ais_hourly"]["mean_unique_mmsi"], 2.5)
        self.assertEqual(found["objects"][0]["ais_hourly"]["observed_hours"], 2)
        self.assertEqual(repo.get_gis_object("AREA_1")["geometry"], self.feature["geometry"])
        settings = Settings(self.root, "unused", "", None, 6, "sqlite", self.db, "sample")
        api.repository.cache_clear()
        with patch.object(Settings, "from_env", return_value=settings):
            client = TestClient(api.app)
            self.assertEqual(client.get("/gis-objects/AREA_1").json()["properties"]["object_name"], "Test Bay")
            self.assertEqual(client.get("/assets/inline-visualizations.js").status_code, 200)

    def test_changed_gis_file_does_not_replace_import(self):
        import_gis(self.root, self.db, "sample")
        self.feature["properties"]["object_name"] = "Changed Bay"
        self.write_feature()
        with self.assertRaisesRegex(ValueError, "不会覆盖"):
            import_gis(self.root, self.db, "sample")
        with self.assertRaisesRegex(ValueError, "指纹不一致"):
            verify_gis(self.root, self.db, "sample")
