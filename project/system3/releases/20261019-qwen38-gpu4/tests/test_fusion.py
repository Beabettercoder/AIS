from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path

import test_repository
from maritime_agent.database import SCHEMA_VERSION, import_dataset
from maritime_agent.fusion import (build_deterministic_relations, build_open_set_associations,
                                   associate_remote_detections, rebuild_fusion_index)
from maritime_agent.sqlite_repository import SQLiteRepository


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()
        self.db = Path(self.fixture.tmp.name) / "storage" / "fusion.sqlite3"
        import_dataset(self.fixture.repo.root, self.db, "sample", "houston", "Sample DS")
        polygon = {"type": "Polygon", "coordinates": [[
            [-95.1, 28.9], [-94.8, 28.9], [-94.8, 29.2], [-95.1, 29.2], [-95.1, 28.9]
        ]]}
        track = {"type": "LineString", "coordinates": [[-95.0, 29.0], [-94.9, 29.1]]}
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(
                "INSERT INTO vessel_daily_tracks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "2024-07-06", "123456789", "TEST", "cargo", 2, 2, 0,
                 "[]", json.dumps(track["coordinates"]), 1),
            )
            conn.execute(
                "INSERT INTO vessel_hour_positions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "2024-07-06T10:00:00Z", "123456789", "TEST", "cargo", "moving",
                 -95.0, 29.0, "test.csv", 1, "{}"),
            )
            conn.execute(
                "INSERT INTO ais_messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "2024-07-06", 1, "123456789", "2024-07-06T10:00:07Z",
                 -95.0, 29.0, 5.0, 90.0, "TEST", "cargo"),
            )
            conn.execute(
                "INSERT INTO gis_objects VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "AREA_1", "channel", "Test Channel", "test", "test-gis", None,
                 "2024-07-01", "Polygon", json.dumps(polygon), "{}", 1),
            )
            conn.execute(
                "INSERT INTO remote_sensing_scenes VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("sample", "SCENE_1", "sentinel-2-c1-l2a", "2024-07-06T10:00:00Z",
                 json.dumps([-95.2, 28.8, -94.7, 29.3]), 10.0, 1,
                 "https://example.com/scene", "scene.json", "{}"),
            )
            conn.execute(
                "INSERT INTO remote_sensing_detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "DET_1", "SCENE_1", "2024-07-06T10:00:00Z", -95.0, 29.0,
                 "vessel_detection", 0.9, 100.0, 20.0, "{}", "{}"),
            )
            conn.execute("UPDATE confirmed_events SET record_json=? WHERE dataset_id=? AND event_id=?",
                         (json.dumps({"start_utc": "2024-07-06T00:00:00Z",
                                      "gis_object_id": "AREA_1"}), "sample", "EVT_001"))
            conn.execute(
                "INSERT INTO weather_sources "
                "(source_id,source_name,region_label,data_mode,provider,source_sha256,first_utc,last_utc,"
                "bbox_json,field_mapping_json,unit_mapping_json,created_at_utc,row_count) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("weather-1", "weather.csv", "Houston", "observation", "test", "sha",
                 "2024-07-06T10:00:00Z", "2024-07-06T10:00:00Z", "[-95,29,-95,29]",
                 "{}", "{}", "2024-07-07T00:00:00Z", 1),
            )
            conn.execute(
                "INSERT INTO weather_observations "
                "(source_id,source_row_number,valid_time_utc,location_id,location_name,lon,lat,"
                "wind_speed_mps,properties_json) VALUES (?,?,?,?,?,?,?,?,?)",
                ("weather-1", 1, "2024-07-06T10:00:00Z", "station-1", "Station", -95.0, 29.0, 8.0, "{}"),
            )

    def tearDown(self):
        self.fixture.tearDown()

    def test_common_index_and_deterministic_relations(self):
        indexed = rebuild_fusion_index(self.db, "sample", "2024-07-06")
        self.assertEqual(indexed["counts_by_modality"]["ais"], 1)
        self.assertEqual(indexed["counts_by_modality"]["gis"], 1)
        self.assertEqual(indexed["counts_by_modality"]["remote_sensing"], 2)
        self.assertEqual(indexed["counts_by_modality"]["weather"], 1)
        self.assertEqual(indexed["counts_by_modality"]["event"], 1)
        self.assertEqual(indexed["counts_by_modality"]["fusion"], 1)

        related = build_deterministic_relations(self.db, "sample")
        self.assertEqual(related["relations_by_type"], {
            "OBSERVES": 1, "LOCATED_IN": 1, "COVERS": 1, "OBSERVED_IN": 1,
            "OCCURS_IN": 1, "DETECTED_IN": 1,
        })
        associated = build_open_set_associations(self.db, "sample")
        self.assertEqual(associated["confidence_kind"], "uncalibrated_rule_score")
        self.assertEqual(associated["summary"]["POSSIBLY_OBSERVES"], 0)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM fusion_relations").fetchone()[0], 6)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM fusion_evidence").fetchone()[0], 6)
            self.assertFalse(conn.execute("PRAGMA foreign_key_check").fetchall())
        repo = SQLiteRepository(self.db, "sample")
        overview = repo.fusion_overview()
        self.assertEqual(overview["accepted_relation_count"], 6)
        located = repo.search_fusion_relations(predicate="located_in")
        self.assertEqual(located["returned"], 1)
        self.assertEqual(located["relations"][0]["evidence"][0]["evidence_type"], "point_in_polygon")

    def test_rebuild_is_idempotent_and_date_is_validated(self):
        first = rebuild_fusion_index(self.db, "sample", "2024-07-06")
        second = rebuild_fusion_index(self.db, "sample", "2024-07-06")
        self.assertEqual(first["node_count"], second["node_count"])
        with self.assertRaises(ValueError):
            rebuild_fusion_index(self.db, "sample", "06/07/2024")

    def test_incremental_remote_association_uses_real_ais_message(self):
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        result = associate_remote_detections(self.db, "sample")
        self.assertEqual(result["method_version"], "remote_ais_uncertainty_gate_v3")
        self.assertEqual(result["summary"]["DETECTED_IN"], 1)
        self.assertEqual(result["summary"]["POSSIBLY_OBSERVES"], 1)
        self.assertEqual(result["summary"]["unresolved"], 0)
        relation = SQLiteRepository(self.db, "sample").search_fusion_relations(
            predicate="POSSIBLY_OBSERVES", limit=1)["relations"][0]
        self.assertEqual(relation["object"]["object_type"], "ais_message")
        self.assertEqual(relation["object"]["attributes"]["position_granularity"], "message")
        self.assertEqual(relation["object"]["attributes"]["timestamp_utc"],
                         "2024-07-06T10:00:07Z")
        self.assertEqual(relation["evidence"][0]["evidence_type"],
                         "uncertainty_gated_spatiotemporal_one_to_one")
        self.assertEqual(relation["evidence"][0]["evidence_value"]["time_delta_seconds"], 7.0)

    def test_remote_message_assignment_does_not_reuse_one_vessel(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(
                "INSERT INTO remote_sensing_detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "DET_2", "SCENE_1", "2024-07-06T10:00:00Z", -95.0001, 29.0001,
                 "vessel_detection", 0.9, None, None, "{}", "{}"),
            )
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        result = associate_remote_detections(self.db, "sample")
        self.assertEqual(result["summary"]["DETECTED_IN"], 2)
        self.assertEqual(result["summary"]["POSSIBLY_OBSERVES"], 1)
        self.assertEqual(result["summary"]["unresolved"], 1)

    def test_remote_target_600_metres_away_is_not_forced_to_match(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(
                "UPDATE remote_sensing_detections SET lat=? WHERE dataset_id=? AND detection_id=?",
                (29.006, "sample", "DET_1"),
            )
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        result = associate_remote_detections(self.db, "sample")
        self.assertEqual(result["summary"]["POSSIBLY_OBSERVES"], 0)
        self.assertEqual(result["summary"]["unresolved"], 1)

    def test_dead_reckoning_two_point_and_single_point(self):
        # 1. 单点航位推算测试 (只有单条报文时触发)
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        associate_remote_detections(self.db, "sample")
        repo = SQLiteRepository(self.db, "sample")
        rel = repo.search_fusion_relations(predicate="POSSIBLY_OBSERVES", limit=1)["relations"][0]
        evidence_details = rel["evidence"][0]["details"]
        self.assertEqual(evidence_details["dead_reckoning_method"], "single_point_dead_reckoning")
        self.assertIn("dead_reckoning_position", evidence_details)

        # 2. 双点线性内插测试 (成像时刻前后均有报文时触发)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(
                "INSERT INTO ais_messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "2024-07-06", 2, "123456789", "2024-07-06T09:59:50Z",
                 -95.0002, 29.0, 5.0, 90.0, "TEST", "cargo"),
            )
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        associate_remote_detections(self.db, "sample")
        rel2 = repo.search_fusion_relations(predicate="POSSIBLY_OBSERVES", limit=1)["relations"][0]
        evidence_details2 = rel2["evidence"][0]["details"]
        self.assertEqual(evidence_details2["dead_reckoning_method"], "two_point_linear_interpolation")
        self.assertIn("dead_reckoning_position", evidence_details2)

    def test_dead_reckoning_rot_circular_arc(self):
        """测试包含有效转向率 ROT 时触发等角速度圆弧轨迹积分外推。"""
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("DELETE FROM ais_messages WHERE dataset_id='sample'")
            conn.execute(
                "INSERT INTO ingested_ais_positions "
                "(source_id,source_row_number,timestamp_utc,mmsi,lon,lat,sog_knots,cog_degrees,vessel_name,vessel_type,vessel_class,extra_attributes_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("src-rot", 1, "2024-07-06T09:59:52Z", "987654321", -95.0, 29.0, 4.0, 90.0, "TURN_SHIP", "cargo", "cargo",
                 json.dumps({"rot": 12.0, "length": 100.0, "width": 20.0}))
            )
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        associate_remote_detections(self.db, "sample")
        repo = SQLiteRepository(self.db, "sample")
        rels = repo.search_fusion_relations(predicate="POSSIBLY_OBSERVES", limit=5)["relations"]
        self.assertTrue(len(rels) >= 1)
        details = rels[0]["evidence"][0]["details"]
        self.assertEqual(details["dead_reckoning_method"], "circular_arc_dead_reckoning")
        self.assertIn("doppler_shift_m", details)
        self.assertIn("restored_detection_point", details)

    def test_remote_association_scale_mismatch_blocks_match(self):
        """测试船体尺寸相差 3 倍以上时予以硬淘汰阻断。"""
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("DELETE FROM ais_messages WHERE dataset_id='sample'")
            # AIS 为 350 米超大油轮
            conn.execute(
                "INSERT INTO ingested_ais_positions "
                "(source_id,source_row_number,timestamp_utc,mmsi,lon,lat,sog_knots,cog_degrees,vessel_name,vessel_type,vessel_class,extra_attributes_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("src-tanker", 1, "2024-07-06T10:00:05Z", "111222333", -95.0, 29.0, 2.0, 90.0, "GIANT_VLCC", "tanker", "tanker",
                 json.dumps({"length": 350.0, "width": 60.0}))
            )
            # 雷达目标仅为 20 米小目标 (350 / 20 = 17.5 > 3.0)
            conn.execute(
                "UPDATE remote_sensing_detections SET length_m=20.0, width_m=6.0 WHERE dataset_id='sample' AND detection_id='DET_1'"
            )
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        result = associate_remote_detections(self.db, "sample")
        self.assertEqual(result["summary"]["POSSIBLY_OBSERVES"], 0)
        self.assertEqual(result["summary"]["unresolved"], 1)

    def test_remote_association_attitude_mismatch_blocks_match(self):
        """测试雷达船体主轴朝向与航向角夹角超过 65° 时予以硬淘汰阻断。"""
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("DELETE FROM ais_messages WHERE dataset_id='sample'")
            # 航向 90°
            conn.execute(
                "INSERT INTO ingested_ais_positions "
                "(source_id,source_row_number,timestamp_utc,mmsi,lon,lat,sog_knots,cog_degrees,vessel_name,vessel_type,vessel_class,extra_attributes_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("src-cross", 1, "2024-07-06T10:00:05Z", "444555666", -95.0, 29.0, 10.0, 90.0, "CROSS_SHIP", "cargo", "cargo",
                 json.dumps({"length": 100.0, "width": 20.0}))
            )
            # 雷达主轴 0° (夹角 90°，横漂冲突)
            conn.execute(
                "UPDATE remote_sensing_detections SET length_m=100.0, width_m=20.0, attributes_json=? WHERE dataset_id='sample' AND detection_id='DET_1'",
                (json.dumps({"orientation_deg": 0.0}),)
            )
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        result = associate_remote_detections(self.db, "sample")
        self.assertEqual(result["summary"]["POSSIBLY_OBSERVES"], 0)
        self.assertEqual(result["summary"]["unresolved"], 1)

    def test_remote_association_hard_time_window_blocks_match(self):
        """测试时差超过 15 分钟（如 24 小时跨天）时绝对硬阻断。"""
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("DELETE FROM ais_messages WHERE dataset_id='sample'")
            # 5 号的 AIS 数据 (时差 24 小时)
            conn.execute(
                "INSERT INTO ingested_ais_positions "
                "(source_id,source_row_number,timestamp_utc,mmsi,lon,lat,sog_knots,cog_degrees,vessel_name,vessel_type,vessel_class,extra_attributes_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("src-yesterday", 1, "2024-07-05T10:00:00Z", "777888999", -95.0, 29.0, 5.0, 90.0, "YESTERDAY_SHIP", "cargo", "cargo",
                 json.dumps({"length": 100.0, "width": 20.0}))
            )
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        result = associate_remote_detections(self.db, "sample")
        self.assertEqual(result["summary"]["POSSIBLY_OBSERVES"], 0)
        self.assertEqual(result["summary"]["unresolved"], 1)


    def test_search_fusion_relations_by_mmsi_and_tolerance(self):
        """测试按 MMSI 精确查询关联、智能 node_id 容错与 has_more 状态。"""
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        associate_remote_detections(self.db, "sample")
        repo = SQLiteRepository(self.db, "sample")

        # 1. 直接按 mmsi 查
        res1 = repo.search_fusion_relations(mmsi="123456789")
        self.assertEqual(res1["returned"], 1)
        self.assertEqual(res1["relations"][0]["predicate"], "POSSIBLY_OBSERVES")
        self.assertEqual(res1["has_more"], False)

        # 2. 容错：将纯数字 MMSI 作为 node_id 传入
        res2 = repo.search_fusion_relations(node_id="123456789")
        self.assertEqual(res2["returned"], 1)
        self.assertEqual(res2["relations"][0]["predicate"], "POSSIBLY_OBSERVES")

        # 3. 容错：带 MMSI: 前缀
        res3 = repo.search_fusion_relations(node_id="MMSI: 123456789")
        self.assertEqual(res3["returned"], 1)
        self.assertEqual(res3["relations"][0]["predicate"], "POSSIBLY_OBSERVES")

        # 4. 查不存在的 MMSI
        res_none = repo.search_fusion_relations(mmsi="999999999")
        self.assertEqual(res_none["returned"], 0)
        self.assertEqual(res_none["has_more"], False)

        # 5. 测试 limit 分页与 has_more 探针
        res_limit = repo.search_fusion_relations(limit=1)
        self.assertEqual(res_limit["returned"], 1)
        self.assertEqual(res_limit["has_more"], True)


if __name__ == "__main__":
    unittest.main()


