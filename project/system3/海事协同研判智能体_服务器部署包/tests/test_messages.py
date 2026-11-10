from __future__ import annotations

import csv
import gzip
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import test_repository
from maritime_agent import api
from maritime_agent.config import Settings
from maritime_agent.database import import_dataset
from maritime_agent.message_database import SOURCE_DIR, import_messages, verify_messages
from maritime_agent.sqlite_repository import SQLiteRepository


class MessageTrackTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()
        self.root = self.fixture.repo.root
        self.db = Path(self.fixture.tmp.name) / "messages.sqlite3"
        import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        self.path = self.root / SOURCE_DIR / "AIS_2024_07_06_clean.csv.gz"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = [
            {"mmsi": "123456789", "timestamp_utc": "2024-07-06T00:00:01Z", "lat_deg": "29.5",
             "lon_deg": "-95.1", "sog_kn": "4.2", "cog_deg": "90", "vessel_name": "TEST", "vessel_class": "tug"},
            {"mmsi": "123456789", "timestamp_utc": "2024-07-06T00:05:00Z", "lat_deg": "29.51",
             "lon_deg": "-95.09", "sog_kn": "", "cog_deg": "", "vessel_name": "TEST", "vessel_class": "tug"},
            {"mmsi": "123456789", "timestamp_utc": "2024-07-06T00:30:00Z", "lat_deg": "29.52",
             "lon_deg": "-95.08", "sog_kn": "5", "cog_deg": "95", "vessel_name": "TEST", "vessel_class": "tug"},
        ]
        self.write_rows()

    def tearDown(self):
        self.fixture.tearDown()

    def write_rows(self):
        with gzip.open(self.path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_import_query_api_and_gap(self):
        result = import_messages(self.root, self.db, "sample")
        self.assertEqual(result["total_rows"], 3)
        self.assertEqual(import_messages(self.root, self.db, "sample")["days"]["2024-07-06"]["status"], "already_imported")
        self.assertEqual(verify_messages(self.root, self.db, "sample")["total_rows"], 3)
        repo = SQLiteRepository(self.db, "sample")
        self.assertEqual(repo.overview()["ais_message_position_count"], 3)
        detail = repo.get_message_track("2024-07-06", "123456789")
        self.assertEqual(detail["gap_segment_count"], 2)
        self.assertEqual(detail["observation_gap_count"], 1)
        self.assertEqual([point["segment"] for point in detail["points"]], [1, 1, 2])
        self.assertEqual(detail["points"][0]["sog_kn"], 4.2)
        self.assertIsNone(detail["points"][1]["sog_kn"])
        self.assertEqual(repo.get_message_track("2024-07-06", "123456789", 0)["points"], [])
        settings = Settings(self.root, "unused", "", None, 6, "sqlite", self.db, "sample")
        api.repository.cache_clear()
        with patch.object(Settings, "from_env", return_value=settings):
            client = TestClient(api.app)
            answer = client.get("/tracks/2024-07-06/123456789?granularity=message")
            self.assertEqual(answer.status_code, 200)
            self.assertEqual(answer.json()["observation_count"], 3)

    def test_invalid_row_rolls_back_and_changed_file_rejected(self):
        self.rows[-1]["lat_deg"] = "invalid"
        self.write_rows()
        with self.assertRaises(ValueError):
            import_messages(self.root, self.db, "sample")
        self.rows[-1]["lat_deg"] = "29.52"
        self.write_rows()
        import_messages(self.root, self.db, "sample")
        self.rows[-1]["lat_deg"] = "29.53"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "不会覆盖"):
            import_messages(self.root, self.db, "sample")


if __name__ == "__main__":
    unittest.main()
