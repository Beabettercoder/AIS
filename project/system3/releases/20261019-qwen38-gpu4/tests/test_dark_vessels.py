from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path

import test_repository
from maritime_agent.database import import_dataset
from maritime_agent.fusion import associate_remote_detections, rebuild_fusion_index
from maritime_agent.sqlite_repository import SQLiteRepository
from maritime_agent.tools import build_tools


class DarkVesselTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()
        self.db = Path(self.fixture.tmp.name) / "storage" / "fusion.sqlite3"
        import_dataset(self.fixture.repo.root, self.db, "sample", "houston", "Sample DS")

        # 构造测试数据：场景 SCENE_1 在 2024-07-06T10:00:00Z
        with closing(sqlite3.connect(self.db)) as conn, conn:
            # 1. 遥感场景
            conn.execute(
                "INSERT INTO remote_sensing_scenes VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("sample", "SCENE_1", "sentinel-2-c1-l2a", "2024-07-06T10:00:00Z",
                 json.dumps([-95.2, 28.8, -94.7, 29.3]), 10.0, 1,
                 "https://example.com/scene", "scene.json", "{}"),
            )
            # 2. 正常合法已匹配目标 DET_LEGAL (位置 -95.0, 29.0)
            conn.execute(
                "INSERT INTO remote_sensing_detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "DET_LEGAL", "SCENE_1", "2024-07-06T10:00:00Z", -95.0, 29.0,
                 "vessel_detection", 0.95, 120.0, 22.0, "{}", "{}"),
            )
            conn.execute(
                "INSERT INTO ais_messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "2024-07-06", 1, "123456789", "2024-07-06T10:00:07Z",
                 -95.0, 29.0, 5.0, 90.0, "LEGAL_SHIP", "cargo"),
            )
            # 3. 靠近航标的虚警目标 DET_BUOY (位置 -94.85, 29.15)，在航标 50 米内
            conn.execute(
                "INSERT INTO gis_objects VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "BUOY_1", "navigation_aid", "Light Buoy No.1", "test", "test-gis", None,
                 "2024-07-01", "Point", json.dumps({"type": "Point", "coordinates": [-94.85, 29.15]}), "{}", 1),
            )
            conn.execute(
                "INSERT INTO remote_sensing_detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "DET_BUOY", "SCENE_1", "2024-07-06T10:00:00Z", -94.8502, 29.1501,
                 "vessel_detection", 0.85, 15.0, 8.0, "{}", "{}"),
            )
            # 4. 真正的高危暗船 DET_DARK (开阔海面 -94.72, 28.92，周围无任何合法船)
            conn.execute(
                "INSERT INTO remote_sensing_detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "DET_DARK", "SCENE_1", "2024-07-06T10:00:00Z", -94.72, 28.92,
                 "vessel_detection", 0.92, 95.0, 18.0, "{}", "{}"),
            )

    def tearDown(self):
        self.fixture.tearDown()

    def test_detect_dark_vessels_filters_legal_and_infrastructure(self):
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        associate_remote_detections(self.db, "sample")

        repo = SQLiteRepository(self.db, "sample")
        res = repo.detect_dark_vessels(date_utc="2024-07-06")

        self.assertIn("evidence_id", res)
        # 验证总排查目标数与过滤统计
        self.assertEqual(res["infrastructure_filtered_count"], 1)  # DET_BUOY 成功被排除
        # DET_LEGAL 已成功匹配合法船，不进入暗船列表
        suspect_ids = [s["detection_id"] for s in res["suspects"]]
        self.assertNotIn("DET_LEGAL", suspect_ids)
        self.assertNotIn("DET_BUOY", suspect_ids)
        self.assertIn("DET_DARK", suspect_ids)

        dark = next(s for s in res["suspects"] if s["detection_id"] == "DET_DARK")
        self.assertEqual(dark["classification"], "unregistered_ghost_vessel")
        self.assertEqual(dark["suspicion_level"], "critical")
        self.assertGreater(dark["min_active_ais_distance_km"], 10.0)

    def test_kinematic_reachability_cone_identifies_lost_contact_vessel(self):
        # 注入一艘在卫星成像前 30 分钟 (09:30:00Z) 突然断线的船舶
        # 航速 12 节，航向 90 度 (正东)，距离 DET_DARK (-94.72, 28.92) 约 11 公里西侧 (-94.83, 28.92)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(
                "INSERT INTO ais_messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "2024-07-06", 99, "987654321", "2024-07-06T09:30:00Z",
                 -94.83, 28.92, 12.0, 90.0, "SHADOW_RUNNER", "cargo"),
            )
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        associate_remote_detections(self.db, "sample")

        repo = SQLiteRepository(self.db, "sample")
        res = repo.detect_dark_vessels(date_utc="2024-07-06")

        dark = next(s for s in res["suspects"] if s["detection_id"] == "DET_DARK")
        # 成功通过时空断点回溯锁定该失联船舶
        self.assertEqual(dark["classification"], "suspected_deliberate_ais_shutoff")
        self.assertEqual(dark["suspicion_level"], "high")
        self.assertIsNotNone(dark["suspected_identity"])
        self.assertEqual(dark["suspected_identity"]["mmsi"], "987654321")
        self.assertEqual(dark["suspected_identity"]["vessel_name"], "SHADOW_RUNNER")
        self.assertAlmostEqual(dark["suspected_identity"]["time_gap_minutes"], 30.0, delta=1.0)

    def test_tool_execution_integration(self):
        rebuild_fusion_index(self.db, "sample", "2024-07-06")
        associate_remote_detections(self.db, "sample")

        repo = SQLiteRepository(self.db, "sample")
        tools = {t.name: t for t in build_tools(repo)}
        self.assertIn("detect_dark_vessels", tools)

        tool_result = json.loads(tools["detect_dark_vessels"].invoke({"date_utc": "2024-07-06"}))
        self.assertIn("evidence_id", tool_result)
        self.assertIn("suspects", tool_result)
        self.assertEqual(len(tool_result["suspects"]), 1)
        self.assertEqual(tool_result["suspects"][0]["detection_id"], "DET_DARK")


if __name__ == "__main__":
    unittest.main()
