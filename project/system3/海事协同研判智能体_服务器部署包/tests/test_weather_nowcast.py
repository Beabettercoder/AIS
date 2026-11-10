from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing

import test_fusion
from maritime_agent.region_registry import ensure_region
from maritime_agent.sqlite_repository import SQLiteRepository
from maritime_agent.tools import build_tools
from maritime_agent.weather_field import estimate_weather_tendency, predict_weather_nowcast


class WeatherNowcastTests(unittest.TestCase):
    def test_estimate_weather_tendency_and_rapid_pressure_drop(self):
        """验证时间序列变率拟合与气压骤降特征研判。"""
        # 模拟台风迫近：1小时内气压从 1008 骤降至 1004 hPa（变率 -4.0 hPa/h），风速增强
        s1 = {
            "timestamp_seconds": 10000.0,
            "u_component_mps": -5.0,
            "v_component_mps": 5.0,
            "wind_speed_mps": 7.07,
            "air_pressure_hpa": 1008.0,
        }
        s2 = {
            "timestamp_seconds": 13600.0,  # +1 hour
            "u_component_mps": -10.0,
            "v_component_mps": 10.0,
            "wind_speed_mps": 14.14,
            "air_pressure_hpa": 1004.0,
        }
        tendency = estimate_weather_tendency([s1, s2])
        self.assertEqual(tendency["pressure_trend"], "rapidly_falling")
        self.assertAlmostEqual(tendency["pressure_rate_per_hour"], -4.0, places=2)
        self.assertEqual(tendency["wind_trend"], "rapidly_strengthening")
        self.assertTrue(tendency["speed_rate_per_hour"] > 0)

    def test_predict_weather_nowcast_zero_lead_preservation(self):
        """验证在 lead_hours = 0 时预测结果与基准观测完全一致，时间置信度严格为 1.0。"""
        s0 = {
            "timestamp_seconds": 1000.0,
            "u_component_mps": -8.0,
            "v_component_mps": 6.0,
            "wind_speed_mps": 10.0,
            "wind_direction_deg": 126.9,
            "air_pressure_hpa": 1010.0,
            "wind_gust_mps": 14.0,
            "spatial_confidence": 0.85,
        }
        res = predict_weather_nowcast([s0, s0], lead_hours=0.0)
        self.assertEqual(res["lead_hours"], 0.0)
        self.assertEqual(res["predicted_weather"]["wind_speed_mps"], 10.0)
        self.assertEqual(res["predicted_weather"]["air_pressure_hpa"], 1010.0)
        self.assertEqual(res["confidence"]["temporal_confidence"], 1.0)
        self.assertEqual(res["confidence"]["combined_confidence"], 0.85)

    def test_predict_weather_nowcast_damping_and_confidence_decay(self):
        """验证物理阻尼因子抑制外推线性发散，且时间置信度随前瞻时长单调连续衰减。"""
        s1 = {"timestamp_seconds": 0.0, "u_component_mps": -5.0, "v_component_mps": 5.0, "wind_speed_mps": 7.07, "air_pressure_hpa": 1012.0}
        s2 = {"timestamp_seconds": 3600.0, "u_component_mps": -10.0, "v_component_mps": 10.0, "wind_speed_mps": 14.14, "air_pressure_hpa": 1008.0}

        pred_1h = predict_weather_nowcast([s1, s2], lead_hours=1.0)
        pred_2h = predict_weather_nowcast([s1, s2], lead_hours=2.0)
        pred_3h = predict_weather_nowcast([s1, s2], lead_hours=3.0)

        # 检查时间置信度单调递减
        c1 = pred_1h["confidence"]["temporal_confidence"]
        c2 = pred_2h["confidence"]["temporal_confidence"]
        c3 = pred_3h["confidence"]["temporal_confidence"]
        self.assertTrue(1.0 > c1 > c2 > c3 > 0.0)

        # 检查阻尼系数使得增速放缓（第二小时增量小于第一小时增量）
        sp0 = s2["wind_speed_mps"]
        sp1 = pred_1h["predicted_weather"]["wind_speed_mps"]
        sp2 = pred_2h["predicted_weather"]["wind_speed_mps"]
        self.assertTrue((sp1 - sp0) > (sp2 - sp1))

    def test_warning_levels_evaluation(self):
        """验证大风预警等级根据风速和阵风准确划分。"""
        base = {"wind_speed_mps": 18.5, "wind_direction_deg": 120.0, "wind_gust_mps": 25.0, "air_pressure_hpa": 995.0}
        res = predict_weather_nowcast([base, base], lead_hours=1.0)
        self.assertEqual(res["warning_assessment"]["warning_level"], "gale_warning_yellow")
        self.assertIn("黄色大风预警", res["warning_assessment"]["warning_description"])

    def test_repository_and_tool_query_wind_nowcast(self):
        """验证 SQLite 数据仓储层与 Agent 暴露的 query_wind_nowcast 工具调用闭环。"""
        fusion_test = test_fusion.FusionTests()
        fusion_test.setUp()
        try:
            db = fusion_test.db
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.row_factory = sqlite3.Row
                region = ensure_region(conn, "sample", None, "Houston Test Area",
                                       [-95.2, 28.8, -94.7, 29.3], strict=False)
                region_id = region["region_id"]
                # 插入一个较早时刻的观测以形成时间序列
                conn.execute(
                    "INSERT INTO weather_observations "
                    "(source_id,source_row_number,valid_time_utc,location_id,location_name,lon,lat,"
                    "wind_speed_mps,air_pressure_hpa,properties_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    ("weather-1", 2, "2024-07-06T09:00:00Z", "station-1", "Station", -95.0, 29.0, 6.0, 1012.0, "{}"),
                )

            repo = SQLiteRepository(db, "sample")
            res = repo.query_wind_nowcast(
                region_id=region_id, base_time_utc="2024-07-06T10:00:00Z", lead_hours=1.5, lon=-95.0, lat=29.0
            )
            self.assertIn("evidence_id", res)
            self.assertEqual(res["forecast_lead_hours"], 1.5)
            self.assertIn("forecast_result", res)
            self.assertTrue(len(res["timeline_evolution"]) >= 3)
            self.assertIn("warning_level", res)

            # 测试智能体工具函数 build_tools
            tools = build_tools(repo)
            nowcast_tool = next(t for t in tools if t.name == "query_wind_nowcast")
            out_json = nowcast_tool.invoke({
                "region_id": region_id,
                "base_time_utc": "2024-07-06T10:00:00Z",
                "lead_hours": 1.0,
                "lon": -95.0,
                "lat": 29.0,
            })
            parsed = json.loads(out_json)
            self.assertIn("evidence_id", parsed)
            self.assertEqual(parsed["forecast_lead_hours"], 1.0)
            self.assertIn("forecast_result", parsed)
        finally:
            fusion_test.tearDown()


if __name__ == "__main__":
    unittest.main()
