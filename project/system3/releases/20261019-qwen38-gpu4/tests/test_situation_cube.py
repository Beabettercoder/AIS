from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing

import test_fusion
from maritime_agent.database import SCHEMA_VERSION, migrate_database
from maritime_agent.fusion import index_bound_source, index_confirmed_events
from maritime_agent.region_registry import bind_source, ensure_region
from maritime_agent.situation_cube import build_situation_cube
from maritime_agent.sqlite_repository import SQLiteRepository


class SituationCubeTests(unittest.TestCase):
    def setUp(self):
        self.fusion = test_fusion.FusionTests()
        self.fusion.setUp()
        self.db = self.fusion.db
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("UPDATE confirmed_events SET record_json=? WHERE dataset_id=? AND event_id=?",
                         (json.dumps({"start_utc": "2024-07-06T10:00:00Z",
                                      "valid_end_utc": "2024-07-06T10:30:00Z",
                                      "gis_object_id": "AREA_1"}), "sample", "EVT_001"))
            region = ensure_region(conn, "sample", None, "Houston test region",
                                   [-95.2, 28.8, -94.7, 29.3], strict=False)
            self.region_id = region["region_id"]

    def tearDown(self):
        self.fusion.tearDown()

    def test_builds_five_modality_cube_with_native_message_time(self):
        result = build_situation_cube(
            self.db, "sample", self.region_id,
            start_time_utc="2024-07-06T09:59:59Z",
            end_time_utc="2024-07-06T10:01:00Z",
            refresh_fusion_index=True,
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(set(result["available_modalities"]),
                         {"ais", "gis", "weather", "remote_sensing", "event"})
        self.assertFalse(result["missing_modalities"])
        repo = SQLiteRepository(self.db, "sample")
        units = repo.search_situation_units(self.region_id)
        region_unit = next(item for item in units["units"] if item["spatial_level"] == "region")
        detail = repo.get_situation_unit(region_unit["unit_id"])
        ais = next(item for item in detail["evidence"] if item["modality"] == "ais")
        self.assertEqual(ais["source_start_utc"], "2024-07-06T10:00:07Z")
        self.assertTrue(ais["provenance"]["no_hour_rounding"])
        self.assertEqual(ais["scores"]["kind"], "explainable_rule_score_not_probability")
        self.assertEqual(detail["unit"]["start_time_utc"], "2024-07-06T09:59:59Z")

    def test_repeated_build_is_idempotent_and_far_region_stays_empty(self):
        args = dict(start_time_utc="2024-07-06T09:59:59Z",
                    end_time_utc="2024-07-06T10:01:00Z")
        first = build_situation_cube(self.db, "sample", self.region_id, **args,
                                     refresh_fusion_index=True)
        second = build_situation_cube(self.db, "sample", self.region_id, **args)
        self.assertEqual(first["cube_id"], second["cube_id"])
        self.assertEqual(first["evidence_link_count"], second["evidence_link_count"])
        self.assertEqual(second["build_mode"], "incremental")
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.row_factory = sqlite3.Row
            far = ensure_region(conn, "sample", None, "Far region", [10, 10, 11, 11], strict=False)
        empty = build_situation_cube(self.db, "sample", far["region_id"], **args)
        self.assertEqual(empty["evidence_link_count"], 0)
        self.assertEqual(set(empty["missing_modalities"]),
                         {"ais", "gis", "weather", "remote_sensing", "event"})

    def test_v23_database_migrates_without_deleting_existing_rows(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            before = conn.execute("SELECT COUNT(*) FROM ais_messages").fetchone()[0]
            conn.execute("PRAGMA user_version = 23")
        migrated = migrate_database(self.db)
        self.assertEqual(migrated, {"before": 23, "after": SCHEMA_VERSION})
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ais_messages").fetchone()[0], before)
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_local_day_event_is_normalized_and_region_linked_without_fake_geometry(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("UPDATE confirmed_events SET valid_start_local=?,valid_end_local=?,record_json=? "
                         "WHERE dataset_id=? AND event_id=?",
                         ("2024-07-08", "2024-07-09", json.dumps({
                             "valid_start_local": "2024-07-08", "valid_end_local": "2024-07-09",
                             "published_date": "2024-07-08", "location_text": "Port Houston terminals"
                         }), "sample", "EVT_001"))
        indexed = index_confirmed_events(self.db, "sample")
        self.assertEqual(indexed["event_node_count"], 1)
        result = build_situation_cube(
            self.db, "sample", self.region_id,
            start_time_utc="2024-07-08T05:00:00Z",
            end_time_utc="2024-07-10T05:00:00Z")
        self.assertIn("event", result["available_modalities"])
        repo = SQLiteRepository(self.db, "sample")
        unit = next(item for item in repo.search_situation_units(
            self.region_id, "2024-07-08T05:00:00Z", "2024-07-10T05:00:00Z")["units"]
                    if item["spatial_level"] == "region")
        event = next(item for item in repo.get_situation_unit(unit["unit_id"])["evidence"]
                     if item["modality"] == "event")
        self.assertEqual(event["source_start_utc"], "2024-07-08T05:00:00Z")
        self.assertEqual(event["spatial_relation"], "region_text_reference")
        self.assertEqual(event["scores"]["spatial"], 0.35)

    def test_incremental_ais_source_indexes_only_new_source(self):
        source_id = "incremental-ais-test"
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(
                "INSERT INTO ingested_sources "
                "(source_id,source_kind,region_label,source_name,source_sha256,row_count,first_utc,last_utc,"
                "field_mapping_json,created_at_utc,profile_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (source_id, "ais_csv", "Houston test region", "new.csv", "sha-new", 2,
                 "2024-07-08T12:34:56Z", "2024-07-08T12:35:56Z", "{}",
                 "2024-07-09T00:00:00Z", "{}"))
            conn.executemany(
                "INSERT INTO ingested_ais_positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [(source_id, 1, "2024-07-08T12:34:56Z", "987654321", -95.0, 29.0,
                  5.0, 90.0, "NEW", 70, "cargo", "{}"),
                 (source_id, 2, "2024-07-08T12:35:56Z", "987654321", -94.99, 29.01,
                  5.2, 91.0, "NEW", 70, "cargo", "{}")])
            bind_source(conn, {"dataset_id": "sample", "region_id": self.region_id,
                               "observed_bbox_wgs84": [-95.0, 29.0, -94.99, 29.01],
                               "region_bbox_wgs84": [-95.2, 28.8, -94.7, 29.3],
                               "overlap_ratio": 1.0, "validation_status": "verified"},
                        "ais_csv", source_id)
        indexed = index_bound_source(self.db, "sample", source_id)
        self.assertEqual(indexed["counts_by_modality"], {"ais": 1})
        with closing(sqlite3.connect(self.db)) as conn:
            nodes = conn.execute("SELECT COUNT(*) FROM fusion_nodes WHERE source_id=?", (source_id,)).fetchone()[0]
        self.assertEqual(nodes, 1)
        cube = build_situation_cube(
            self.db, "sample", self.region_id,
            start_time_utc="2024-07-08T12:34:00Z", end_time_utc="2024-07-08T12:36:00Z",
            task_type="intake_incremental")
        self.assertIn("ais", cube["available_modalities"])


if __name__ == "__main__":
    unittest.main()
