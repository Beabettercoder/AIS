from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing

import test_fusion
from maritime_agent.region_registry import ensure_region
from maritime_agent.situation_cube import build_situation_cube
from maritime_agent.sqlite_repository import SQLiteRepository
from maritime_agent.tools import build_tools
from maritime_agent.weather_field import (
    decompose_wind,
    generate_grid_wind_field,
    haversine_distance_km,
    interpolate_point_weather,
    synthesize_wind,
)


class WeatherFieldTests(unittest.TestCase):
    def test_wind_vector_decomposition_and_synthesis_anti_inversion(self):
        """验证正交矢量分解与合成，彻底防止 350° 与 10° 标量平均导致的 180° 翻转失真。"""
        # 正北风 (0°): 向南吹，u=0, v=-speed
        u_n, v_n = decompose_wind(10.0, 0.0)
        self.assertAlmostEqual(u_n, 0.0, places=5)
        self.assertAlmostEqual(v_n, -10.0, places=5)
        sp_n, dir_n = synthesize_wind(u_n, v_n)
        self.assertEqual(sp_n, 10.0)
        self.assertEqual(dir_n, 0.0)

        # 正东风 (90°): 向西吹，u=-speed, v=0
        u_e, v_e = decompose_wind(10.0, 90.0)
        self.assertAlmostEqual(u_e, -10.0, places=5)
        self.assertAlmostEqual(v_e, 0.0, places=5)
        sp_e, dir_e = synthesize_wind(u_e, v_e)
        self.assertEqual(sp_e, 10.0)
        self.assertEqual(dir_e, 90.0)

        # 核心防翻转测试：350° 与 10°（两个相近的北偏风）
        # 传统标量平均: (350 + 10) / 2 = 180° (变成了正南风！严重物理错误)
        u1, v1 = decompose_wind(10.0, 350.0)
        u2, v2 = decompose_wind(10.0, 10.0)
        u_avg = (u1 + u2) / 2.0
        v_avg = (v1 + v2) / 2.0
        sp_synth, dir_synth = synthesize_wind(u_avg, v_avg)
        # 矢量合成应该严格为正北风 0°，风速约为 9.85 m/s
        self.assertEqual(dir_synth, 0.0)
        self.assertAlmostEqual(sp_synth, 9.85, places=2)

    def test_barnes_exact_station_match_and_origin_preservation(self):
        """验证在测站原点插值能够 100% 还原测站观测值，置信度严格为 1.0。"""
        obs = [
            {
                "lon": -95.0,
                "lat": 29.5,
                "location_id": "station-alpha",
                "wind_speed_mps": 12.5,
                "wind_direction_deg": 45.0,
                "wind_gust_mps": 18.2,
                "air_pressure_hpa": 1008.4,
                "air_temperature_c": 28.5,
            },
            {
                "lon": -94.5,
                "lat": 29.0,
                "location_id": "station-beta",
                "wind_speed_mps": 8.0,
                "wind_direction_deg": 120.0,
                "wind_gust_mps": 11.0,
                "air_pressure_hpa": 1012.0,
                "air_temperature_c": 30.0,
            },
        ]
        # 在 station-alpha 原点插值
        res = interpolate_point_weather(obs, -95.0, 29.5)
        self.assertEqual(res["interpolation_mode"], "exact_station_match")
        self.assertEqual(res["spatial_confidence"], 1.0)
        self.assertEqual(res["wind_speed_mps"], 12.5)
        self.assertEqual(res["wind_direction_deg"], 45.0)
        self.assertEqual(res["wind_gust_mps"], 18.2)
        self.assertEqual(res["air_pressure_hpa"], 1008.4)
        self.assertEqual(res["nearest_station_id"], "station-alpha")

    def test_spatial_confidence_distance_exponential_decay(self):
        """验证空间置信度随离站距离增加呈现物理平滑的单调指数衰减。"""
        obs = [
            {
                "lon": -95.0,
                "lat": 29.5,
                "location_id": "station-center",
                "wind_speed_mps": 10.0,
                "wind_direction_deg": 90.0,
                "air_pressure_hpa": 1010.0,
            }
        ]
        # 逐步远离测站
        p0 = interpolate_point_weather(obs, -95.0, 29.5)  # 0 km
        p1 = interpolate_point_weather(obs, -94.9, 29.5)  # ~9.7 km
        p2 = interpolate_point_weather(obs, -94.7, 29.5)  # ~29.1 km
        p3 = interpolate_point_weather(obs, -94.4, 29.5)  # ~58.2 km

        self.assertEqual(p0["spatial_confidence"], 1.0)
        self.assertTrue(1.0 > p1["spatial_confidence"] > p2["spatial_confidence"] > p3["spatial_confidence"] > 0.0)
        self.assertTrue(p1["nearest_station_km"] < p2["nearest_station_km"] < p3["nearest_station_km"])

    def test_grid_wind_field_generation(self):
        """验证规则网格连续风场矩阵生成及统计聚合。"""
        obs = [
            {"lon": -95.2, "lat": 29.2, "wind_speed_mps": 10.0, "wind_direction_deg": 100.0, "air_pressure_hpa": 1010.0},
            {"lon": -94.8, "lat": 29.8, "wind_speed_mps": 20.0, "wind_direction_deg": 120.0, "air_pressure_hpa": 1002.0},
        ]
        bbox = (-95.3, 29.1, -94.7, 29.9)
        grid = generate_grid_wind_field(obs, bbox, cols=4, rows=4)

        self.assertEqual(grid["grid_dimensions"]["total_cells"], 16)
        self.assertEqual(len(grid["cells"]), 16)
        self.assertTrue(10.0 <= grid["summary"]["max_wind_speed_mps"] <= 20.0)
        self.assertTrue(0.0 < grid["summary"]["min_spatial_confidence"] <= 1.0)

    def test_situation_cube_eliminates_weather_modality_gap(self):
        """验证态势立方体中连续风场自动为无测站落入的自适应网格补充气象证据，消除空间断崖。"""
        fusion_test = test_fusion.FusionTests()
        fusion_test.setUp()
        try:
            db = fusion_test.db
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.row_factory = sqlite3.Row
                # 建立一个较大的测试区域，使得许多自适应网格内部没有测站
                region = ensure_region(conn, "sample", None, "Houston Large Study Region",
                                       [-95.3, 28.7, -94.6, 29.4], strict=False)
                region_id = region["region_id"]

            result = build_situation_cube(
                db, "sample", region_id,
                start_time_utc="2024-07-06T09:59:59Z",
                end_time_utc="2024-07-06T10:01:00Z",
                refresh_fusion_index=True,
            )
            self.assertEqual(result["status"], "ready")
            self.assertIn("weather", result["available_modalities"])

            repo = SQLiteRepository(db, "sample")
            units = repo.search_situation_units(region_id)["units"]
            self.assertTrue(len(units) >= 2)

            # 检查每个 unit 的 available_modalities 均包含 weather（不再因为离散点而缺失）
            for u in units:
                detail = repo.get_situation_unit(u["unit_id"])
                available = detail["unit"]["available_modalities"]
                self.assertIn("weather", available, f"Unit {u['unit_id']} 应具备连续气象场支持")
                weather_evidence = [e for e in detail["evidence"] if e["modality"] == "weather"]
                self.assertTrue(len(weather_evidence) >= 1)
                # 检查连续插值关系
                has_barnes = any(e["spatial_relation"] == "barnes_interpolated" or e["spatial_relation"] == "contained"
                                 for e in weather_evidence)
                self.assertTrue(has_barnes)
        finally:
            fusion_test.tearDown()

    def test_repository_and_tool_query_wind_field(self):
        """验证仓储层与 Agent 暴露的 query_wind_field 工具调用。"""
        fusion_test = test_fusion.FusionTests()
        fusion_test.setUp()
        try:
            db = fusion_test.db
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.row_factory = sqlite3.Row
                region = ensure_region(conn, "sample", None, "Houston Test Area",
                                       [-95.2, 28.8, -94.7, 29.3], strict=False)
                region_id = region["region_id"]

            repo = SQLiteRepository(db, "sample")

            # 1. 单点查询
            point_res = repo.query_wind_field(
                region_id=region_id, valid_time_utc="2024-07-06T10:00:00Z", lon=-95.0, lat=29.0
            )
            self.assertIn("evidence_id", point_res)
            self.assertEqual(point_res["query_mode"], "point_interpolation")
            self.assertEqual(point_res["result"]["wind_speed_mps"], 8.0)
            self.assertEqual(point_res["result"]["spatial_confidence"], 1.0)

            # 2. 全域网格场查询
            grid_res = repo.query_wind_field(
                region_id=region_id, valid_time_utc="2024-07-06T10:00:00Z", grid_cols=4, grid_rows=4
            )
            self.assertIn("evidence_id", grid_res)
            self.assertEqual(grid_res["query_mode"], "continuous_grid_field")
            self.assertEqual(grid_res["grid_field"]["grid_dimensions"]["total_cells"], 16)

            # 3. Agent Tool 工具函数调用验证
            tools = build_tools(repo)
            wind_tool = next(t for t in tools if t.name == "query_wind_field")
            tool_output_json = wind_tool.invoke({
                "region_id": region_id,
                "valid_time_utc": "2024-07-06T10:00:00Z",
                "lon": -95.0,
                "lat": 29.0,
            })
            parsed = json.loads(tool_output_json)
            self.assertIn("evidence_id", parsed)
            self.assertEqual(parsed["result"]["wind_speed_mps"], 8.0)
        finally:
            fusion_test.tearDown()


if __name__ == "__main__":
    unittest.main()
