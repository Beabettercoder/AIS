from __future__ import annotations

import csv
import gzip
import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import test_repository
from maritime_agent import api
from maritime_agent.config import Settings
from maritime_agent.database import import_dataset, open_readonly
from maritime_agent.sqlite_repository import SQLiteRepository
from maritime_agent.track_database import TRACK_FILES, import_tracks, verify_tracks


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()
        self.root = self.fixture.repo.root
        self.db = Path(self.fixture.tmp.name) / "tracks.sqlite3"
        import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        rows = [
            {"timestamp_hour_utc": "2024-07-06T00:00:00Z", "mmsi": "123456789",
             "representative_lon_deg": "-95.1", "representative_lat_deg": "29.5",
             "vessel_name": "TEST", "vessel_class": "tug", "motion_state": "moving",
             "source_file": "ais_00.csv"},
            {"timestamp_hour_utc": "2024-07-06T01:00:00Z", "mmsi": "123456789",
             "representative_lon_deg": "-95.0", "representative_lat_deg": "29.6",
             "vessel_name": "TEST", "vessel_class": "tug", "motion_state": "stopped",
             "source_file": "ais_01.csv"},
        ]
        hour_path = self.root / TRACK_FILES["vessel_hour_positions"]
        hour_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(hour_path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.track_path = self.root / TRACK_FILES["vessel_daily_tracks"]
        self.track_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature = {"type": "Feature", "properties": {
            "date_utc": "2024-07-06", "mmsi": "123456789", "vessel_name": "TEST",
            "vessel_class": "tug", "hour_points": 2, "moving_hours": 1,
            "stopped_hours": 1, "source_files": ["ais_00.csv", "ais_01.csv"]},
            "geometry": {"type": "LineString", "coordinates": [[-95.1, 29.5], [-95.0, 29.6]]}}
        self.write_feature()

    def tearDown(self):
        self.fixture.tearDown()

    def write_feature(self):
        with gzip.open(self.track_path, "wt", encoding="utf-8") as stream:
            json.dump({"type": "FeatureCollection", "features": [self.feature]}, stream)

    def test_import_verify_query_and_idempotency(self):
        result = import_tracks(self.root, self.db, "sample")
        self.assertEqual(result["row_counts"], {"vessel_hour_positions": 2, "vessel_daily_tracks": 1})
        self.assertEqual(import_tracks(self.root, self.db, "sample")["status"], "already_imported")
        self.assertEqual(import_dataset(self.root, self.db, "sample", "houston", "Sample DS")["status"],
                         "already_imported")
        self.assertEqual(verify_tracks(self.root, self.db, "sample")["integrity"], "ok")
        repo = SQLiteRepository(self.db, "sample")
        self.assertEqual(repo.overview()["track_count"], 1)
        found = repo.search_tracks("2024-07-06")
        self.assertEqual(found["tracks"][0]["mmsi"], "123456789")
        detail = repo.get_track("2024-07-06", "123456789")
        self.assertEqual(len(detail["points"]), 2)
        self.assertEqual(detail["points"][1]["timestamp_hour_utc"], "2024-07-06T01:00:00Z")
        self.assertEqual(repo.map_tracks("2024-07-06")["features"][0]["geometry"],
                         self.feature["geometry"])
        settings = Settings(self.root, "unused", "", None, 6, "sqlite", self.db, "sample")
        api.repository.cache_clear()
        with patch.object(Settings, "from_env", return_value=settings):
            client = TestClient(api.app)
            self.assertEqual(client.get("/tracks?date_utc=2024-07-06").json()["returned"], 1)
            self.assertEqual(client.get("/tracks/2024-07-06/123456789").json()["hour_points"], 2)
            self.assertEqual(client.get("/tracks-map").status_code, 200)

    def test_bad_geometry_rolls_back_all_track_rows(self):
        self.feature["geometry"]["coordinates"][1] = [-94.9, 29.6]
        self.write_feature()
        with self.assertRaisesRegex(ValueError, "坐标不一致"):
            import_tracks(self.root, self.db, "sample")
        with closing(open_readonly(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM vessel_hour_positions").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM vessel_daily_tracks").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM import_files WHERE table_name='vessel_daily_tracks'").fetchone()[0], 0)

    def test_modified_source_does_not_replace_import(self):
        import_tracks(self.root, self.db, "sample")
        self.feature["properties"]["vessel_name"] = "CHANGED"
        self.write_feature()
        with self.assertRaisesRegex(ValueError, "不会覆盖"):
            import_tracks(self.root, self.db, "sample")
        with self.assertRaisesRegex(ValueError, "指纹不一致"):
            verify_tracks(self.root, self.db, "sample")
        self.assertEqual(SQLiteRepository(self.db, "sample").get_track("2024-07-06", "123456789")["vessel_name"], "TEST")
