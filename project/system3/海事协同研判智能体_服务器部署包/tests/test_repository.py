from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from maritime_agent.repository import DataRepository, TABLES


def write_table(root: Path, name: str, rows: list[dict[str, str]]) -> None:
    path = root / TABLES[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        write_table(root, "port", [
            {"timestamp_hour_utc": "2024-07-06T00:00:00Z", "unique_mmsi": "100", "moving_mmsi": "20",
             "ais_source_file_available": "1", "ais_no_reports": "0"},
            {"timestamp_hour_utc": "2024-07-06T01:00:00Z", "unique_mmsi": "80", "moving_mmsi": "10",
             "ais_source_file_available": "0", "ais_no_reports": "1"},
            {"timestamp_hour_utc": "2024-07-07T00:00:00Z", "unique_mmsi": "50", "moving_mmsi": "5",
             "ais_source_file_available": "1", "ais_no_reports": "0"},
        ])
        write_table(root, "zone", [{"timestamp_hour_utc": "2024-07-06T00:00:00Z", "macro_zone": "bayport", "unique_mmsi": "2"}])
        write_table(root, "vessel_class", [{"timestamp_hour_utc": "2024-07-06T00:00:00Z", "vessel_class": "tug", "unique_mmsi": "4"}])
        write_table(root, "events", [{"event_id": "EVT_001", "event_name": "closure", "time_precision": "day_range",
                                       "valid_start_local": "2024-07-08", "valid_end_local": "2024-07-09", "published_date": "2024-07-15"}])
        write_table(root, "event_links", [{"evidence_link_id": "EVID_001", "event_id": "EVT_001", "source_id": "SRC_001",
                                            "evidence_locator": "p.1", "evidence_quote": "Terminals closed", "quote_verified": "1"}])
        write_table(root, "sources", [{"source_id": "SRC_001", "title": "Port notice", "source_url": "https://example.com",
                                        "publication_date": "2024-07-15"}])
        write_table(root, "objects", [{"object_id": "OBJ_001", "object_name": "Test berth", "ranking_eligibility": "eligible"}])
        self.repo = DataRepository(root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hourly_mean_and_quality(self):
        result = self.repo.aggregate("port", "unique_mmsi", "2024-07-06T00:00:00Z", "2024-07-07T00:00:00Z")
        self.assertEqual(result["value"], 90)
        self.assertEqual(result["unit"], "艘/小时")
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["quality_flags"]["ais_missing_or_no_report_count"], 1)
        with self.assertRaises(ValueError):
            self.repo.aggregate("port", "unique_mmsi", "2024-07-06T00:00:00Z", "2024-07-07T00:00:00Z", statistic="sum")

    def test_comparison_and_utc_validation(self):
        result = self.repo.compare("port", "moving_mmsi", "2024-07-06T00:00:00Z", "2024-07-07T00:00:00Z",
                                   "2024-07-07T00:00:00Z", "2024-07-08T00:00:00Z")
        self.assertEqual(result["baseline"]["value"], 15)
        self.assertEqual(result["target"]["value"], 5)
        self.assertEqual(result["change_pct"], -66.67)
        with self.assertRaises(ValueError):
            self.repo.aggregate("port", "moving_mmsi", "2024-07-06", "2024-07-07")

    def test_event_precision_and_source(self):
        event = self.repo.event_evidence("EVT_001")
        self.assertEqual(event["event"]["time_precision"], "day_range")
        self.assertNotEqual(event["event"]["valid_start_local"], event["event"]["published_date"])
        self.assertEqual(event["evidence_links"][0]["source_title"], "Port notice")
        self.assertTrue(self.repo.search_events("missing")["evidence_id"].startswith("event_search:"))
        self.assertTrue(self.repo.object_comparisons("missing")["evidence_id"].startswith("object_search:"))

    def test_ais_daily_change_uses_previous_utc_day(self):
        result = self.repo.ais_daily_change("2024-07-07")
        self.assertEqual(result["baseline_day_utc"], "2024-07-06")
        self.assertEqual(result["target_day_utc"], "2024-07-07")
        self.assertEqual(result["metrics"]["unique_mmsi"]["baseline"]["value"], 90)
        self.assertEqual(result["metrics"]["unique_mmsi"]["target"]["value"], 50)


if __name__ == "__main__":
    unittest.main()
