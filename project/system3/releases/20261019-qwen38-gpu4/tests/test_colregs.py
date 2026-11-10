from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from maritime_agent.colregs import (
    EncounterType,
    calculate_collision_risk_index,
    calculate_dcpa_tcpa,
    classify_colregs_encounter,
    evaluate_traffic_conflict_matrix,
    solve_continuous_segment_cpa,
)
from maritime_agent.tools import build_tools


class ColregsTests(unittest.TestCase):
    def test_dcpa_tcpa_head_on(self):
        """测试对向行驶两船的相对运动学计算（DCPA 趋近于 0，TCPA 为正）。"""
        # 船 1 在南，向北行驶 (000°)，航速 12 节
        lat1, lon1, sog1, cog1 = 29.30, -94.80, 12.0, 0.0
        # 船 2 在北相距 3 海里，向南行驶 (180°)，航速 12 节
        lat2, lon2, sog2, cog2 = 29.35, -94.80, 12.0, 180.0

        dist, bearing, dcpa, tcpa = calculate_dcpa_tcpa(lat1, lon1, sog1, cog1, lat2, lon2, sog2, cog2)
        self.assertAlmostEqual(dist, 3.0, places=1)
        self.assertAlmostEqual(bearing, 0.0, places=1)
        self.assertAlmostEqual(dcpa, 0.0, places=1)
        # 相对速度 24 节，距离 3 海里，TCPA = 3 / 24 * 60 = 7.5 分钟
        self.assertAlmostEqual(tcpa, 7.5, delta=0.5)

    def test_dcpa_tcpa_diverging_safe(self):
        """测试相互远离的两船（TCPA 为负，CRI 为 0）。"""
        # 船 1 向北驶离
        lat1, lon1, sog1, cog1 = 29.35, -94.80, 12.0, 0.0
        # 船 2 向南驶离
        lat2, lon2, sog2, cog2 = 29.30, -94.80, 12.0, 180.0

        dist, bearing, dcpa, tcpa = calculate_dcpa_tcpa(lat1, lon1, sog1, cog1, lat2, lon2, sog2, cog2)
        self.assertTrue(tcpa < 0)
        cri = calculate_collision_risk_index(dist, dcpa, tcpa)
        self.assertEqual(cri, 0.0)

    def test_colregs_rule_14_head_on(self):
        """测试 Rule 14 对遇局面判定与避让义务。"""
        cog1, sog1 = 0.0, 12.0
        cog2, sog2 = 180.0, 12.0
        bearing12 = 0.5  # 他船在正前方偏 0.5°

        enc_type, desc, ob1, ob2 = classify_colregs_encounter(cog1, sog1, cog2, sog2, bearing12)
        self.assertEqual(enc_type, EncounterType.HEAD_ON)
        self.assertIn("Rule 14", desc)
        self.assertIn("向右转向", ob1)
        self.assertIn("向右转向", ob2)

    def test_colregs_rule_15_crossing_give_way(self):
        """测试 Rule 15 交叉相遇局面：他船在右舷，本船为让路船。"""
        # 本船向北 (000°)
        cog1, sog1 = 0.0, 10.0
        # 他船向西 (270°)，位于本船东北方 (方位角 045°) -> 右舷
        cog2, sog2 = 270.0, 10.0
        bearing12 = 45.0

        enc_type, desc, ob1, ob2 = classify_colregs_encounter(cog1, sog1, cog2, sog2, bearing12)
        self.assertEqual(enc_type, EncounterType.CROSSING_GIVE_WAY)
        self.assertIn("让路船", ob1)
        self.assertIn("直航船", ob2)
        self.assertIn("右转避让", ob1)

    def test_colregs_rule_15_crossing_stand_on(self):
        """测试 Rule 15 交叉相遇局面：他船在左舷，本船为直航船。"""
        # 本船向北 (000°)
        cog1, sog1 = 0.0, 10.0
        # 他船向东 (090°)，位于本船西北方 (方位角 315°) -> 左舷
        cog2, sog2 = 90.0, 10.0
        bearing12 = 315.0

        enc_type, desc, ob1, ob2 = classify_colregs_encounter(cog1, sog1, cog2, sog2, bearing12)
        self.assertEqual(enc_type, EncounterType.CROSSING_STAND_ON)
        self.assertIn("直航船", ob1)
        self.assertIn("让路船", ob2)

    def test_colregs_rule_13_overtaking(self):
        """测试 Rule 13 追越局面：后船航速更快追越前船。"""
        # 船 1 在后方，航速 15 节追赶；船 2 在前方，航速 8 节；同向北 (000°)
        cog1, sog1 = 0.0, 15.0
        cog2, sog2 = 0.0, 8.0
        bearing12 = 0.0  # 船 2 在船 1 前方，船 1 在船 2 正后方 (180°)

        enc_type, desc, ob1, ob2 = classify_colregs_encounter(cog1, sog1, cog2, sog2, bearing12)
        self.assertEqual(enc_type, EncounterType.OVERTAKING_GIVE_WAY)
        self.assertIn("Rule 13", desc)
        self.assertIn("追越船（让路船）", ob1)
        self.assertIn("被追越船（直航船）", ob2)

    def test_colregs_rule_18_anchor_dragging(self):
        """测试 Rule 18 责任条款：走锚失控船无机动能力，在航船承担全部避让责任。"""
        cog1, sog1 = 312.0, 1.8  # 顺风漂移
        cog2, sog2 = 045.0, 8.0   # 进港航行
        bearing12 = 120.0

        enc_type, desc, ob1, ob2 = classify_colregs_encounter(
            cog1, sog1, cog2, sog2, bearing12, is_anchor_drag1=True, is_anchor_drag2=False
        )
        self.assertEqual(enc_type, EncounterType.ANCHOR_DRIFT_NUC)
        self.assertIn("Rule 18", desc)
        self.assertIn("走锚失控", ob1)
        self.assertIn("全部让路义务", ob2)

    def test_evaluate_traffic_conflict_matrix(self):
        """测试多船交通流成对碰撞风险矩阵评估与高风险排序。"""
        vessels = [
            # 船 A：向北行驶
            {"mmsi": "111111111", "name": "Vessel A", "lat": 29.30, "lon": -94.80, "sog_kn": 12.0, "cog_deg": 0.0, "motion_state": "moving"},
            # 船 B：向南行驶，与 A 对遇紧迫局面
            {"mmsi": "222222222", "name": "Vessel B", "lat": 29.33, "lon": -94.80, "sog_kn": 12.0, "cog_deg": 180.0, "motion_state": "moving"},
            # 船 C：远在 10 海里外
            {"mmsi": "333333333", "name": "Vessel C", "lat": 29.50, "lon": -94.60, "sog_kn": 10.0, "cog_deg": 90.0, "motion_state": "moving"},
        ]

        # 全域扫描
        records = evaluate_traffic_conflict_matrix(vessels, min_cri=0.3, max_distance_nm=4.0)
        self.assertTrue(len(records) >= 1)
        top = records[0]
        self.assertIn(top.vessel1_mmsi, ("111111111", "222222222"))
        self.assertIn(top.vessel2_mmsi, ("111111111", "222222222"))
        self.assertGreater(top.cri, 0.7)
        self.assertTrue(top.is_urgent)

        # 指定目标 MMSI 过滤
        target_records = evaluate_traffic_conflict_matrix(vessels, target_mmsi="111111111", min_cri=0.3)
        self.assertEqual(len(target_records), 1)
        self.assertEqual(target_records[0].vessel2_mmsi, "222222222")

    def test_assess_collision_risk_tool_integration(self):
        """测试 assess_collision_risk 工具通过 Mock 仓储正常运行。"""
        mock_repo = MagicMock()
        mock_repo.query_vessels_snapshot.return_value = {
            "vessels": [
                {"mmsi": "367425000", "name": "Tanker 367425000", "lat": 29.30, "lon": -94.80, "sog_kn": 2.0, "cog_deg": 315.0, "motion_state": "走锚漂移"},
                {"mmsi": "413000000", "name": "Cargo Ship", "lat": 29.31, "lon": -94.79, "sog_kn": 8.0, "cog_deg": 270.0, "motion_state": "moving"},
            ]
        }

        tools = {t.name: t for t in build_tools(mock_repo)}
        self.assertIn("assess_collision_risk", tools)

        tool_res_str = tools["assess_collision_risk"].invoke({"target_mmsi": "367425000"})
        data = json.loads(tool_res_str)
        self.assertIn("evidence_id", data)
        self.assertIn("conflicts", data)
        self.assertTrue(len(data["conflicts"]) >= 1)
        conflict = data["conflicts"][0]
        self.assertIn("dcpa_m", conflict)
        self.assertIn("tcpa_minutes", conflict)
        self.assertIn("cri", conflict)
        self.assertIn("Rule 18", conflict["encounter_desc"])

    def test_solve_continuous_segment_cpa(self):
        """测试连续时间切平面相对运动学微分极值导数解析求解 (Continuous CPA)。"""
        # 1. 对遇闭式解析求解：两船在同一经线上南北相向，初距约 1000 米，以 10 节相对航行
        lat1, lon1, sog1, cog1 = 59.00, 5.00, 10.0, 0.0   # 向北
        lat2 = 59.00 + (1000.0 / 111139.0)
        lon2 = 5.00
        sog2, cog2 = 10.0, 180.0                           # 向南

        min_m, min_nm, tau_sec, init_m = solve_continuous_segment_cpa(
            lat1, lon1, sog1, cog1,
            lat2, lon2, sog2, cog2,
            duration_sec=3600.0
        )
        self.assertAlmostEqual(init_m, 1000.0, delta=5.0)
        self.assertAlmostEqual(min_m, 0.0, delta=1.0)
        self.assertAlmostEqual(min_nm, 0.0, delta=0.01)
        # 相对速度 20 节 = 10.288 m/s，1000m 极值发生时间 tau = 1000 / 10.288 ≈ 97.2 秒
        self.assertAlmostEqual(tau_sec, 97.2, delta=1.0)

        # 2. 远离两船（背向航行）：极值发生在此前，闭区间截断为 tau=0，极值即为初距
        min_m_div, _, tau_div, init_div = solve_continuous_segment_cpa(
            lat1, lon1, 10.0, 180.0,
            lat2, lon2, 10.0, 0.0,
            duration_sec=3600.0
        )
        self.assertEqual(tau_div, 0.0)
        self.assertAlmostEqual(min_m_div, init_div, delta=0.1)

        # 3. 两船相对静止 (vr2 < 1e-4)
        min_m_stat, _, tau_stat, init_stat = solve_continuous_segment_cpa(
            lat1, lon1, 0.0, 0.0,
            lat2, lon2, 0.0, 0.0,
            duration_sec=3600.0
        )
        self.assertEqual(tau_stat, 0.0)
        self.assertAlmostEqual(min_m_stat, init_stat, delta=0.1)


if __name__ == "__main__":
    unittest.main()
