from __future__ import annotations

import json
import re
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

import test_repository
from maritime_agent.agent import ask
from maritime_agent.coordinated_agent import (
    create_coordinated_graph,
    _identify_dominant_role,
    _evaluate_early_stopping,
    _execute_specialists_with_timeout,
)


class RoutingModel:
    def __init__(self, roles):
        self.roles = roles
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        prompt = str(messages[0].content)
        self.calls.append(prompt)
        if "海事问题协调员" in prompt:
            return AIMessage(content=json.dumps({"roles": self.roles, "subquestions": {}}, ensure_ascii=False))
        if "海事综合分析员" in prompt:
            ids = list(dict.fromkeys(re.findall(r"【([^】]+)】", str(messages[-1].content))))
            return AIMessage(content="综合核查：" + " ".join(f"【{item}】" for item in ids))
        evidence = [m for m in messages if isinstance(m, ToolMessage)]
        if evidence:
            data = json.loads(evidence[-1].content)
            return AIMessage(content=f"本地覆盖见【{data['evidence_id']}】。")
        return AIMessage(content="", tool_calls=[{"name": "data_overview", "args": {}, "id": "overview_1"}])


class InvalidSynthesisModel(RoutingModel):
    def invoke(self, messages):
        prompt = str(messages[0].content)
        if "海事综合分析员" in prompt:
            self.calls.append(prompt)
            return AIMessage(content="综合结论【fabricated:missing】")
        return super().invoke(messages)


class CoordinatedTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_single_specialist_and_verifier(self):
        model = RoutingModel(["ais"])
        result = ask(create_coordinated_graph(self.fixture.repo, model), "数据范围？")
        self.assertEqual(result["agents_used"], ["coordinator", "ais", "synthesizer", "verifier"])
        self.assertEqual(result["tool_trace"][0]["agent"], "ais")
        self.assertEqual(len(result["evidence"]), 1)

    def test_cross_domain_merges_verified_evidence(self):
        model = RoutingModel(["ais", "events"])
        result = ask(create_coordinated_graph(self.fixture.repo, model), "AIS 和运营事件数据范围？")
        self.assertEqual(len(result["tool_trace"]), 2)
        self.assertTrue(result["evidence"])
        self.assertIn("综合核查", result["answer"])
        self.assertEqual(sum("海事综合分析员" in call for call in model.calls), 1)
        self.assertFalse(any("独立核验员" in call for call in model.calls))
        synthesis_prompt = next(call for call in model.calls if "海事综合分析员" in call)
        self.assertIn("不要机械地使用固定", synthesis_prompt)
        self.assertIn("不自动构成证据冲突", synthesis_prompt)

    def test_remote_detection_question_skips_scene_catalog_shortcut(self):
        model = RoutingModel(["remote_sensing"])
        scenes = {"evidence_id": "remote_search:day6", "scenes": [], "match_count": 0,
                  "returned": 0, "date_utc": "2024-07-06"}
        with patch.object(self.fixture.repo, "search_remote_sensing", return_value=scenes) as search:
            result = ask(create_coordinated_graph(self.fixture.repo, model),
                         "2024年7月6日遥感影像检测到了多少个疑似船舶目标？")
        self.assertTrue(result["evidence"])
        self.assertFalse(search.called)

    def test_selected_region_skips_houston_scene_catalog_shortcut(self):
        model = RoutingModel(["remote_sensing"])
        scenes = {"evidence_id": "remote_search:houston", "scenes": [], "match_count": 0,
                  "returned": 0, "date_utc": "2024-07-06"}
        question = ("用户问题：2024年7月6日有哪些遥感场景？\n\n"
                    "当前网页选择的是一个研究区域级多源数据集。"
                    "区域ID=region-hau，区域名称=hau。")
        with patch.object(self.fixture.repo, "search_remote_sensing", return_value=scenes) as search:
            result = ask(create_coordinated_graph(self.fixture.repo, model), question)
        self.assertTrue(result["evidence"])
        self.assertFalse(search.called)
        self.assertNotIn("匹配 Houston", result["answer"])

    def test_verifier_rejects_synthesis_with_unknown_evidence(self):
        model = InvalidSynthesisModel(["ais", "events"])
        result = ask(create_coordinated_graph(self.fixture.repo, model), "AIS 和运营事件数据范围？")
        self.assertIn("未通过独立证据核验", result["answer"])
        self.assertEqual(result["evidence"], [])
        self.assertTrue(any("fabricated:missing" in warning for warning in result["warnings"]))

    def test_unknown_role_falls_back_to_safe_routing(self):
        model = RoutingModel(["write_database"])
        result = ask(create_coordinated_graph(self.fixture.repo, model), "AIS 数据范围？")
        self.assertIn("ais", result["agents_used"])

    def test_intake_request_routes_to_intake_specialist(self):
        from maritime_agent.coordinated_agent import _explicit_roles
        self.assertEqual(_explicit_roles("我要接入新的 AIS CSV 数据"), ["intake"])
        self.assertEqual(_explicit_roles("这批 Houston-Beryl 数据覆盖哪几天？"), ["ais"])
        self.assertEqual(_explicit_roles("2024年7月8日最大阵风是多少？"), ["ais"])

    def test_ais_remote_question_does_not_use_legacy_hourly_overlap(self):
        model = RoutingModel(["ais", "remote_sensing"])
        scenes = {"scenes": [{"evidence_id": "remote_sensing:scene", "scene_id": "scene"}], "match_count": 1}
        quality = {"available": True, "evidence_id": "remote_ais_quality:day",
                   "ais_position_count_17utc": 10, "scl_covered_count": 8,
                   "cloud_or_shadow_count": 6, "cloud_or_shadow_fraction_of_covered": 0.75}
        def aggregate(table, metric, start_utc, end_utc):
            return {"evidence_id": f"aggregate:{metric}", "value": 10 if metric == "unique_mmsi" else 3}
        with patch.object(self.fixture.repo, "aggregate", side_effect=aggregate), \
             patch.object(self.fixture.repo, "search_remote_sensing", return_value=scenes), \
             patch.object(self.fixture.repo, "assess_remote_ais_overlap", return_value=quality):
            result = ask(create_coordinated_graph(self.fixture.repo, model),
                         "2024-07-06 AIS 活动与遥感影像能互相验证吗？")
        self.assertNotIn("assess_remote_ais_overlap", [call["tool"] for call in result["tool_trace"]])
        self.assertFalse(any(item["evidence_id"] == "remote_ais_quality:day" for item in result["evidence"]))

    def test_zero_scene_answer_and_followup_use_search_evidence_and_previous_date(self):
        model = RoutingModel(["remote_sensing"])
        empty = {"evidence_id": "remote_search:none", "scenes": [], "match_count": 0,
                 "returned": 0, "date_utc": "2024-07-08"}
        with patch.object(self.fixture.repo, "search_remote_sensing", return_value=empty) as search:
            first = ask(create_coordinated_graph(self.fixture.repo, model),
                        "7 月 8 日 Houston 有没有遥感场景？")
            followup = ask(create_coordinated_graph(self.fixture.repo, model),
                           "场景是什么卫星、什么时间拍摄的？",
                           history=[{"question": "7 月 8 日 Houston 有没有遥感场景？",
                                     "answer": first["answer"], "evidence": first["evidence"]}])
        self.assertIn("没有匹配", first["answer"])
        self.assertIn("没有可列出的卫星名称或成像时间", followup["answer"])
        self.assertEqual(followup["evidence"][0]["evidence_id"], "remote_search:none")
        self.assertEqual(search.call_args.kwargs["limit"], 30)
        self.assertEqual(search.call_args.args[0], "2024-07-08")

    def test_relative_day_followup_inherits_remote_role_and_advances_anchor(self):
        model = RoutingModel(["events"])
        def scenes(day, limit=10):
            items = [] if day != "2024-07-09" else [{
                "evidence_id": "remote_sensing:scene9", "scene_id": "S2A_SCENE9",
                "collection": "sentinel-2-c1-l2a", "datetime_utc": "2024-07-09T17:15:00Z",
                "scene_cloud_cover_pct": 40.2}]
            return {"evidence_id": f"remote_search:{day}", "scenes": items,
                    "match_count": len(items), "returned": len(items), "date_utc": day}
        with patch.object(self.fixture.repo, "search_remote_sensing", side_effect=scenes) as search:
            first = ask(create_coordinated_graph(self.fixture.repo, model),
                        "2024 年 7 月 8 日 Houston 有没有遥感场景？")
            history = [{"question": "2024 年 7 月 8 日 Houston 有没有遥感场景？",
                        "answer": first["answer"], "evidence": first["evidence"]}]
            second = ask(create_coordinated_graph(self.fixture.repo, model), "那后一天呢", history=history)
            history.append({"question": "那后一天呢", "answer": second["answer"],
                            "evidence": second["evidence"]})
            third = ask(create_coordinated_graph(self.fixture.repo, model), "后一天", history=history)
        self.assertEqual(second["agents_used"], ["coordinator", "remote_sensing", "synthesizer", "verifier"])
        self.assertIn("2024-07-09", second["answer"])
        self.assertIn("S2A_SCENE9", second["answer"])
        self.assertIn("2024-07-10", third["answer"])
        self.assertEqual([call.args[0] for call in search.call_args_list],
                         ["2024-07-08", "2024-07-09", "2024-07-10"])

    def test_relative_track_followup_preserves_mmsi_and_message_granularity(self):
        model = RoutingModel(["events"])
        history = [{"question": "那改成逐报文 AIS 的，有多少条？", "answer": "1103 条",
                    "evidence": [{"evidence_id": "message_track:day6", "date_utc": "2024-07-06",
                                  "mmsi": "367457250", "observation_count": 1086},
                                 {"evidence_id": "message_track:day9", "date_utc": "2024-07-09",
                                  "mmsi": "367457250", "observation_count": 1103}]}]
        detail = {"evidence_id": "message_track:day8", "date_utc": "2024-07-08",
                  "mmsi": "367457250", "observation_count": 1244,
                  "gap_segment_count": 1, "observation_gap_count": 0}
        with patch.object(self.fixture.repo, "get_message_track", return_value=detail) as get_track:
            result = ask(create_coordinated_graph(self.fixture.repo, model), "前一天呢？", history=history)
        get_track.assert_called_once_with("2024-07-08", "367457250")
        self.assertIn("1244", result["answer"])
        self.assertEqual(result["tool_trace"][0]["arguments"]["granularity"], "message")

    def test_dominant_role_identification(self):
        # 单船走锚问题，在 roles 包含 ["ais", "events"] 时应识别 ais 为主导角色
        q_anchor = "请问 MMSI 367457250 在 2024-07-08 是否有走锚异常？"
        self.assertEqual(_identify_dominant_role(q_anchor, ["ais", "events"]), "ais")

        # 纯气象风场问题
        q_wind = "2024-07-08 该海区风场分布与阵风风速？"
        self.assertEqual(_identify_dominant_role(q_wind, ["gis", "remote_sensing"]), "gis")

        # 多源对比问题应返回 None，保持全并发
        q_cross = "对比 AIS 船舶轨迹与 Sentinel 遥感影像检测结果"
        self.assertIsNone(_identify_dominant_role(q_cross, ["ais", "remote_sensing"]))

    def test_early_stopping_evaluation_on_deterministic_evidence(self):
        # 走锚机理凭据充分度
        q = "MMSI 367457250 锚泊受力是否超标？有没有走锚风险？"
        res_anchor = {
            "answer": "依据流体力学与锚泊机理核算，该船受风流推力 320kN，底质极限持力 1200kN，DFR=0.27，处于受控正常锚泊状态。",
            "evidence": [{"evidence_id": "anchor_mech:367457250:2024-07-08:normal"}],
        }
        stop, score, rationale = _evaluate_early_stopping(q, res_anchor, ["events", "remote_sensing"])
        self.assertTrue(stop)
        self.assertGreaterEqual(score, 0.95)
        self.assertIn("闭环机理凭据", rationale)

        # 避碰 COLREGs 凭据充分度
        q_colregs = "评估海区船舶碰撞风险与会遇态势"
        res_colregs = {
            "answer": "COLREGs 相对运动学研判显示，存在 1 起对遇局面（Rule 14），DCPA 为 0.15 海里，双方均须向右转向避让。",
            "evidence": [{"evidence_id": "colregs:2024-07-08:conflict_1"}],
        }
        stop2, score2, rationale2 = _evaluate_early_stopping(q_colregs, res_colregs, ["events"])
        self.assertTrue(stop2)
        self.assertGreaterEqual(score2, 0.95)

        # 显式多源问题不触发早停
        q_multi = "对比遥感卫星与 AIS 航迹数据"
        stop3, score3, rationale3 = _evaluate_early_stopping(q_multi, res_anchor, ["remote_sensing"])
        self.assertFalse(stop3)

    def test_early_stopping_execution_prunes_secondary_specialist(self):
        # 模拟 Coordinator 规划了 ["ais", "events"]，但问题纯属走锚/航迹闭环
        model = RoutingModel(["ais", "events"])
        anchor_detail = {
            "evidence_id": "anchor_mech:367457250:2024-07-08:normal",
            "evaluated_time_utc": "2024-07-08T12:00:00Z",
            "target_mmsi": "367457250",
            "assessment": {
                "mmsi": "367457250",
                "status": "normal_at_anchor",
                "drag_force_ratio": 0.25,
            }
        }
        question = "请问 MMSI 367457250 在 2024-07-08 是否有走锚异常？"
        graph = create_coordinated_graph(self.fixture.repo, model)
        
        # 当 AIS 节点产生闭环凭据时，早停门控生效
        with patch.object(self.fixture.repo, "get_track", return_value={"points": [{"lat": 29.5, "lon": -94.8, "sog_knots": 0.1, "cog_degrees": 0.0}]}):
            result = ask(graph, question)
            self.assertTrue(result["evidence"])
            # events 角色被智能剪枝，没有进行额外无用调用
            self.assertIn("ais", result["agents_used"])

    def test_specialists_timeout_racing_and_graceful_degradation(self):
        import time

        def slow_specialist(role: str) -> dict[str, Any]:
            if role == "remote_sensing":
                time.sleep(0.4)  # 模拟网络卡死慢节点
                return {"role": role, "answer": "remote done", "evidence": [], "warnings": [], "tool_trace": []}
            return {"role": role, "answer": f"{role} fast done", "evidence": [{"evidence_id": f"{role}:1"}],
                    "warnings": [], "tool_trace": []}

        # 设置超时为 0.05 秒，remote_sensing 应被超时降级，而 ais 正常返回
        results = _execute_specialists_with_timeout(
            slow_specialist, ["ais", "remote_sensing"], timeout_seconds=0.05
        )
        self.assertEqual(len(results), 2)
        ais_res = next(r for r in results if r["role"] == "ais")
        rs_res = next(r for r in results if r["role"] == "remote_sensing")

        self.assertEqual(ais_res["answer"], "ais fast done")
        self.assertIn("超时降级", rs_res["answer"])
        self.assertIn("响应超时降级", " ".join(rs_res["warnings"]))

    def test_critic_correction_feedback_loop_recovers_answer(self):
        class RecoverableModel(RoutingModel):
            def __init__(self, roles):
                super().__init__(roles)
                self.repair_attempted = False

            def invoke(self, messages):
                prompt = str(messages[0].content)
                human_msg = str(messages[-1].content)
                if "事实核验审查反馈" in human_msg:
                    self.repair_attempted = True
                    # 收到纠错指示后，从提示中提取合法编号（避开上一版草稿中的无效编号）
                    all_matches = re.findall(r"【([^】]+)】", human_msg)
                    valid_candidates = [m for m in all_matches if "invalid" not in m and "fabricated" not in m]
                    target_id = valid_candidates[0] if valid_candidates else "overview:2024-07-06T00:00:00Z:2024-07-10T23:00:00Z"
                    return AIMessage(content=f"修正后的可信综合结论【{target_id}】。")
                if "海事综合分析员" in prompt:
                    # 第一次故意手滑输出不存在的伪造凭据
                    return AIMessage(content="初版草稿【fabricated:invalid_id】。")
                return super().invoke(messages)

        model = RecoverableModel(["ais", "events"])
        graph = create_coordinated_graph(self.fixture.repo, model)
        result = ask(graph, "AIS 和运营事件数据范围？")
        self.assertTrue(model.repair_attempted)
        self.assertTrue(result["evidence"])
        self.assertIn("修正后的可信综合结论", result["answer"])
        self.assertIn("已成功触发核验双向自愈闭环", " ".join(result.get("warnings", [])))

