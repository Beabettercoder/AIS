from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from maritime_agent.api import app, repository


class WindMapApiTests(unittest.TestCase):
    def setUp(self):
        repository.cache_clear()
        self.client = TestClient(app)

    def tearDown(self):
        repository.cache_clear()

    def test_wind_map_page_served_html(self):
        """测试 /wind-map 态势大屏页面返回正常 HTML 结构。"""
        response = self.client.get("/wind-map")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("海域态势工作台", response.text)
        self.assertIn('id="ai-copilot-panel"', response.text)
        self.assertIn('id="rail-analysis"', response.text)
        self.assertIn("wind-canvas", response.text)
        self.assertIn("sidebar-vessels", response.text)
        self.assertIn("layer-dock", response.text)
        self.assertIn("weather-picker-card", response.text)
        self.assertIn("vessel-card", response.text)
        self.assertIn("timeline-container", response.text)
        self.assertIn("gfw-timeline-panel", response.text)
        self.assertIn("gfw-histogram-track", response.text)
        self.assertIn('id="btn-open-briefing"', response.text)
        self.assertIn('id="briefing-drawer"', response.text)
        self.assertIn("态势研判简报", response.text)

    def test_wind_field_grid_api(self):
        """测试 /api/wind-field-grid 接口输出 Barnes 规则网格及测站数据。"""
        fake_grid = {
            "evidence_id": "wind_field_grid:test",
            "region_name": "Houston",
            "valid_time_utc": "2024-07-08T12:00:00Z",
            "grid_field": {
                "bbox": [-95.8, 28.8, -94.3, 30.1],
                "grid_dimensions": {"cols": 8, "rows": 8, "total_cells": 64},
                "summary": {
                    "max_wind_speed_mps": 24.5,
                    "avg_wind_speed_mps": 14.2,
                    "min_spatial_confidence": 0.85,
                    "station_count": 6,
                },
                "cells": [
                    {
                        "col": c,
                        "row": r,
                        "target_lon": -95.0 + c * 0.1,
                        "target_lat": 29.0 + r * 0.1,
                        "wind_speed_mps": 15.0,
                        "wind_direction_deg": 120.0,
                        "u_component_mps": -12.99,
                        "v_component_mps": 7.5,
                        "spatial_confidence": 0.92,
                    }
                    for r in range(8) for c in range(8)
                ],
            },
            "stations": [
                {
                    "station_id": "8770613",
                    "station_name": "Morgans Point",
                    "lon": -94.985,
                    "lat": 29.682,
                    "wind_speed_mps": 21.0,
                    "wind_direction_deg": 115.0,
                    "wind_gust_mps": 28.5,
                    "air_pressure_hpa": 988.2,
                    "air_temperature_c": 27.5,
                }
            ],
            "active_station_count": 1,
        }
        mock_repo = MagicMock()
        mock_repo.query_wind_field.return_value = fake_grid

        with patch("maritime_agent.api.repository", return_value=mock_repo):
            response = self.client.get("/api/wind-field-grid?valid_time_utc=2024-07-08T12:00:00Z&cols=8&rows=8")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("grid_field", data)
            self.assertIn("stations", data)
            self.assertEqual(data["grid_field"]["grid_dimensions"]["cols"], 8)
            self.assertEqual(data["grid_field"]["grid_dimensions"]["rows"], 8)
            self.assertEqual(len(data["grid_field"]["cells"]), 64)
            self.assertEqual(data["grid_field"]["summary"]["max_wind_speed_mps"], 24.5)
            self.assertEqual(data["stations"][0]["station_id"], "8770613")

    def test_vessels_snapshot_api(self):
        """测试 /api/vessels-snapshot 船舶实时态势快照接口。"""
        fake_vessels = {
            "valid_time_utc": "2024-07-08T12:00:00Z",
            "total_vessels": 2,
            "summary": {
                "total": 2, "cargo": 1, "tanker": 1, "tug": 0, "other": 0,
                "moving": 1, "moored": 1,
            },
            "vessels": [
                {
                    "mmsi": "205104000", "name": "FLANDERS PIONEER", "vessel_class": "tanker",
                    "motion_state": "stopped", "lon": -95.130, "lat": 29.738,
                    "sog_kn": 0.0, "cog_deg": 210.4, "heading_deg": 185.0,
                    "length_m": 230.0, "draft_m": 8.4, "storm_distance_km": 103.3,
                },
                {
                    "mmsi": "367123456", "name": "HOUSTON PILOT", "vessel_class": "cargo",
                    "motion_state": "moving", "lon": -94.850, "lat": 29.350,
                    "sog_kn": 8.5, "cog_deg": 120.0, "heading_deg": 122.0,
                    "length_m": 120.0, "draft_m": 6.2, "storm_distance_km": 85.0,
                }
            ]
        }
        mock_repo = MagicMock()
        mock_repo.query_vessels_snapshot.return_value = fake_vessels

        with patch("maritime_agent.api.repository", return_value=mock_repo):
            response = self.client.get("/api/vessels-snapshot?valid_time_utc=2024-07-08T12:00:00Z")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["total_vessels"], 2)
            self.assertEqual(data["summary"]["tanker"], 1)
            self.assertEqual(len(data["vessels"]), 2)
            self.assertEqual(data["vessels"][0]["name"], "FLANDERS PIONEER")

    def test_port_facilities_api(self):
        """测试 /api/port-facilities 航道与码头设施 GeoJSON 接口。"""
        fake_facilities = {
            "type": "FeatureCollection",
            "count": 1,
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-95.0, 29.7], [-94.9, 29.7], [-94.9, 29.8], [-95.0, 29.7]]]
                    },
                    "properties": {
                        "object_id": "NOAA_BERTH_01",
                        "object_name": "Pier 9",
                        "object_type": "berth"
                    }
                }
            ]
        }
        mock_repo = MagicMock()
        mock_repo.query_port_facilities.return_value = fake_facilities

        with patch("maritime_agent.api.repository", return_value=mock_repo):
            response = self.client.get("/api/port-facilities?limit=50")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["type"], "FeatureCollection")
            self.assertEqual(data["count"], 1)
            self.assertEqual(data["features"][0]["properties"]["object_name"], "Pier 9")

    def test_wind_timeline_api(self):
        """测试 /api/wind-timeline 接口演变序列。"""
        response = self.client.get("/api/wind-timeline?date_utc=2024-07-08")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["date_utc"], "2024-07-08")
        self.assertIsInstance(data["steps"], list)
        self.assertGreaterEqual(len(data["steps"]), 5)
        times = [s["time_utc"] for s in data["steps"]]
        self.assertIn("2024-07-08T12:00:00Z", times)

    def test_wind_assets_served(self):
        """测试风场相关 CSS、JS 静态资源路由。"""
        css_resp = self.client.get("/assets/wind-map.css")
        self.assertEqual(css_resp.status_code, 200)
        self.assertIn("text/css", css_resp.headers.get("content-type", ""))

        js_map = self.client.get("/assets/wind-map.js")
        self.assertEqual(js_map.status_code, 200)
        self.assertIn("javascript", js_map.headers.get("content-type", ""))

        js_engine = self.client.get("/assets/wind-particles.js")
        self.assertEqual(js_engine.status_code, 200)
        self.assertIn("javascript", js_engine.headers.get("content-type", ""))

    def test_timeline_histogram_api(self):
        """测试 /api/timeline-histogram GFW 全周期直方图接口。"""
        fake_hist = {
            "start_utc": "2024-07-06T00:00:00Z",
            "end_utc": "2024-07-10T23:00:00Z",
            "total_hours": 2,
            "items": [
                {
                    "time_utc": "2024-07-08T12:00:00Z",
                    "timestamp_epoch": 1720440000,
                    "max_wind_speed": 24.5,
                    "min_pressure": 988.0,
                    "obs_count": 48,
                    "typhoon_center": [-95.1, 29.35],
                    "stage_desc": "台风眼登陆演变"
                }
            ]
        }
        mock_repo = MagicMock()
        mock_repo.query_timeline_histogram.return_value = fake_hist

        with patch("maritime_agent.api.repository", return_value=mock_repo):
            response = self.client.get("/api/timeline-histogram")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["total_hours"], 2)
            self.assertEqual(data["items"][0]["max_wind_speed"], 24.5)

    def test_vessels_snapshot_with_custom_region(self):
        """测试 /api/vessels-snapshot 携带自定义海区 region_id 参数。"""
        fake_vessels = {
            "region_id": "region-hau",
            "region_name": "hau",
            "valid_time_utc": "2024-01-06T12:00:00Z",
            "total_vessels": 1,
            "summary": {"total": 1, "cargo": 0, "tanker": 0, "tug": 0, "other": 1, "moving": 0, "moored": 1},
            "vessels": [{"mmsi": "210350000", "name": "MMSI 210350000", "vessel_class": "other", "lon": 5.29, "lat": 59.36, "sog_kn": 0.0}]
        }
        mock_repo = MagicMock()
        mock_repo.query_vessels_snapshot.return_value = fake_vessels

        with patch("maritime_agent.api.repository", return_value=mock_repo):
            response = self.client.get("/api/vessels-snapshot?valid_time_utc=2024-01-06T12:00:00Z&region_id=region-hau")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["region_id"], "region-hau")
            self.assertEqual(data["total_vessels"], 1)
            mock_repo.query_vessels_snapshot.assert_called_once_with(
                valid_time_utc="2024-01-06T12:00:00Z", limit=800, region_id="region-hau"
            )

    def test_remote_sensing_overlays_api(self):
        """测试 /api/remote-sensing-overlays 遥感切片与 CFAR 检测要素接口。"""
        fake_overlays = {
            "region_id": "region-hau",
            "region_name": "hau",
            "scene_count": 1,
            "scenes": [{"scene_id": "INTAKE_01", "name": "Sentinel-1 切片", "bbox": [4.65, 58.95, 5.65, 59.65]}],
            "detection_count": 2,
            "detections": [
                {"detection_id": "det-1", "lon": 5.2, "lat": 59.3, "confidence": 0.98, "length_m": 85.0},
                {"detection_id": "det-2", "lon": 5.4, "lat": 59.4, "confidence": 0.95, "length_m": 120.0},
            ]
        }
        mock_repo = MagicMock()
        mock_repo.query_remote_sensing_overlays.return_value = fake_overlays

        with patch("maritime_agent.api.repository", return_value=mock_repo):
            response = self.client.get("/api/remote-sensing-overlays?region_id=region-hau")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["scene_count"], 1)
            self.assertEqual(data["detection_count"], 2)
            self.assertEqual(data["detections"][0]["confidence"], 0.98)
            mock_repo.query_remote_sensing_overlays.assert_called_once_with(region_id="region-hau")

    def test_track_detail_api_with_custom_date(self):
        """测试 /tracks/{date_utc}/{mmsi} 支持任意动态日期与 MMSI 查询。"""
        fake_track = {
            "evidence_id": "track:test",
            "date_utc": "2024-01-06",
            "mmsi": "241866000",
            "vessel_name": "TEST_VESSEL",
            "vessel_class": "cargo",
            "points": [{"representative_lat": 59.3, "representative_lon": 5.2}],
        }
        mock_repo = MagicMock()
        mock_repo.get_track.return_value = fake_track

        with patch("maritime_agent.api.repository", return_value=mock_repo):
            response = self.client.get("/tracks/2024-01-06/241866000?granularity=hourly")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["date_utc"], "2024-01-06")
            self.assertEqual(data["mmsi"], "241866000")
            mock_repo.get_track.assert_called_once_with("2024-01-06", "241866000")

    def test_timeline_histogram_api_with_region(self):
        """测试 /api/timeline-histogram 支持 region_id 参数。"""
        fake_hist = {
            "region_id": "region-hau",
            "region_name": "hau",
            "start_utc": "2024-01-05T00:00:00Z",
            "end_utc": "2024-01-07T23:00:00Z",
            "total_hours": 72,
            "items": [{"time_utc": "2024-01-05T00:00:00Z", "obs_count": 45}],
        }
        mock_repo = MagicMock()
        mock_repo.query_timeline_histogram.return_value = fake_hist

        with patch("maritime_agent.api.repository", return_value=mock_repo):
            response = self.client.get("/api/timeline-histogram?region_id=region-hau")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["region_id"], "region-hau")
            self.assertEqual(data["total_hours"], 72)
            mock_repo.query_timeline_histogram.assert_called_once_with(
                start_utc="2024-07-06T00:00:00Z", end_utc="2024-07-10T23:00:00Z", region_id="region-hau"
            )

    def test_vessel_track_history_api(self):
        """测试 /api/vessel-track-history/{mmsi} 接口全时段及指定日期轨迹查询。"""
        fake_history = {
            "mmsi": "538005263",
            "vessel_name": "STAR COURAGE",
            "vessel_class": "bulk_carrier",
            "mode": "full",
            "requested_date": "",
            "region_id": "region-hau",
            "point_count": 72,
            "available_dates": ["2024-01-05", "2024-01-06", "2024-01-07"],
            "points": [
                {"timestamp_utc": "2024-01-05T00:00:00Z", "representative_lat": 59.2, "representative_lon": 5.1},
                {"timestamp_utc": "2024-01-07T23:00:00Z", "representative_lat": 59.4, "representative_lon": 5.3},
            ],
        }
        mock_repo = MagicMock()
        mock_repo.get_vessel_track_history.return_value = fake_history

        with patch("maritime_agent.api.repository", return_value=mock_repo):
            # 1. 默认 full 模式
            response = self.client.get("/api/vessel-track-history/538005263?region_id=region-hau")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["mmsi"], "538005263")
            self.assertEqual(data["mode"], "full")
            self.assertEqual(data["point_count"], 72)
            self.assertEqual(len(data["available_dates"]), 3)
            mock_repo.get_vessel_track_history.assert_called_with(mmsi="538005263", date_utc="", region_id="region-hau")

            # 2. 指定日期模式
            response2 = self.client.get("/api/vessel-track-history/538005263?date_utc=2024-01-06&region_id=region-hau")
            self.assertEqual(response2.status_code, 200)
            mock_repo.get_vessel_track_history.assert_called_with(mmsi="538005263", date_utc="2024-01-06", region_id="region-hau")

    def test_situation_report_api(self):
        """测试 /api/situation-report 态势研判简报生成接口。"""
        fake_report = {
            "report_id": "RPT-20240708-TEST",
            "region_id": "region-hau",
            "region_name": "hau",
            "data_sufficiency": {
                "grade": "B",
                "grade_title": "B 级 · 航运环境完备（局部模态受限）",
                "confidence_level": "良好",
                "limitations": ["本时窗未捕获卫星遥感过顶切片，暂跳过雷达暗标交叉校验。"],
            },
            "kpi_metrics": {"total_vessels": 120, "underway_vessels": 15},
            "markdown": "# 态势研判简报\n测试内容",
        }
        mock_repo = MagicMock()
        mock_repo.db_path = Path("fake.sqlite3")

        with patch("maritime_agent.api.repository", return_value=mock_repo), \
             patch("maritime_agent.report_generator.generate_situation_briefing", return_value=fake_report) as mock_gen:
            response = self.client.get("/api/situation-report?region_id=region-hau")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["report_id"], "RPT-20240708-TEST")
            self.assertEqual(data["data_sufficiency"]["grade"], "B")
            self.assertIn("markdown", data)
            mock_gen.assert_called_once_with(repo=mock_repo, region_id="region-hau")



