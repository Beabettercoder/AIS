from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from maritime_agent.agent import ask, clean_user_answer, create_graph, validate_answer
import test_repository


class FakeModel:
    def __init__(self):
        self.system_prompt = ""
        self.seen_messages = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.system_prompt = messages[0].content
        self.seen_messages = messages
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_messages:
            return AIMessage(content="", tool_calls=[{"name": "data_overview", "args": {}, "id": "call_1"}])
        data = json.loads(tool_messages[-1].content)
        return AIMessage(content=f"数据覆盖范围见【{data['evidence_id']}】。")


class PersistentToolModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], HumanMessage) and "预算已用完" in messages[-1].content:
            tool_message = next(m for m in messages if isinstance(m, ToolMessage))
            evidence_id = json.loads(tool_message.content)["evidence_id"]
            return AIMessage(content=f"已取得的数据范围见【{evidence_id}】。")
        count = sum(isinstance(m, ToolMessage) for m in messages)
        return AIMessage(content="", tool_calls=[{"name": "data_overview", "args": {}, "id": f"call_{count}"}])


class UncitedRefusalModel:
    def __init__(self, overview_id):
        self.overview_id = overview_id

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], HumanMessage) and "上条回答缺少可核查的证据" in messages[-1].content:
            return AIMessage(content=f"当前数据库只有小时汇总，没有逐船 MMSI 和轨迹坐标，无法列出 10 艘船的航迹【{self.overview_id}】。")
        return AIMessage(content="无法列出 10 艘船的航迹。")


class OldCitationModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="沿用上一轮结论【aggregate:old】")


class AgentTests(unittest.TestCase):
    def test_dynamic_answer_policy_and_workflow_prefix_cleanup(self):
        self.assertIn("不要机械地使用固定", create_graph.__globals__["SYSTEM_PROMPT"])
        self.assertIn("问题复杂度", create_graph.__globals__["SYSTEM_PROMPT"])
        answer = clean_user_answer("I now have the evidence needed. Let me summarize the result.\n\n实际结论【overview:abc】")
        self.assertEqual(answer, "实际结论【overview:abc】")

    def test_graph_tool_loop_and_citation(self):
        fixture = test_repository.RepositoryTests()
        fixture.setUp()
        try:
            model = FakeModel()
            result = create_graph(fixture.repo, model).invoke({"messages": [HumanMessage(content="数据范围？")], "rounds": 0})
            self.assertIn("数据覆盖范围", result["answer"])
            self.assertEqual(len(result["evidence"]), 1)
            self.assertEqual(result["rounds"], 1)
            self.assertIn("2024-07-06", model.system_prompt)
            self.assertIn("只说月日时，必须采用数据覆盖的 2024 年", model.system_prompt)
        finally:
            fixture.tearDown()

    def test_invalid_citation_is_rejected(self):
        answer, evidence, warnings = validate_answer([
            ToolMessage(content=json.dumps({"evidence_id": "overview:abc"}), tool_call_id="c1"),
            AIMessage(content="伪造结论【overview:fake】")])
        self.assertIn("证据引用校验", answer)
        self.assertFalse(evidence)
        self.assertTrue(warnings)

    def test_plain_current_evidence_id_is_normalized_to_citation(self):
        answer, evidence, warnings = validate_answer([
            ToolMessage(content=json.dumps({"evidence_id": "remote_search:abc", "match_count": 0}),
                        tool_call_id="c1"),
            AIMessage(content="本地目录没有匹配，查询证据 remote_search:abc。")])
        self.assertIn("【remote_search:abc】", answer)
        self.assertEqual(evidence[0]["evidence_id"], "remote_search:abc")
        self.assertFalse(warnings)

    def test_tool_limit_summarizes_existing_evidence(self):
        fixture = test_repository.RepositoryTests()
        fixture.setUp()
        try:
            result = ask(create_graph(fixture.repo, PersistentToolModel(), max_tool_rounds=1), "数据范围？")
            self.assertIn("已取得的数据范围", result["answer"])
            self.assertEqual(len(result["tool_trace"]), 1)
            self.assertEqual(len(result["evidence"]), 1)
            self.assertTrue(result["warnings"])
        finally:
            fixture.tearDown()

    def test_uncited_refusal_is_repaired_with_data_scope(self):
        fixture = test_repository.RepositoryTests()
        fixture.setUp()
        try:
            model = UncitedRefusalModel(fixture.repo.overview()["evidence_id"])
            result = ask(create_graph(fixture.repo, model), "列出 10 艘船的 AIS 轨迹")
            self.assertIn("没有逐船 MMSI 和轨迹坐标", result["answer"])
            self.assertEqual(len(result["evidence"]), 1)
            self.assertEqual(result["tool_trace"], [{"tool": "data_overview", "arguments": {}}])
            self.assertFalse(result["warnings"])
        finally:
            fixture.tearDown()

    def test_history_is_context_but_previous_evidence_is_not_current_evidence(self):
        fixture = test_repository.RepositoryTests()
        fixture.setUp()
        try:
            history = [{"question": "7 月 8 日移动船舶数？",
                        "answer": "3 艘/小时【aggregate:old】",
                        "evidence": [{"evidence_id": "aggregate:old", "value": 3,
                                      "scope": {"table": "port", "metric": "moving_mmsi",
                                                "start_utc": "2024-07-08T00:00:00Z",
                                                "end_utc_exclusive": "2024-07-09T00:00:00Z"}}]}]
            model = FakeModel()
            result = ask(create_graph(fixture.repo, model), "那前一天呢？", history=history)
            self.assertEqual(result["tool_trace"][0]["tool"], "data_overview")
            self.assertIn("7 月 8 日移动船舶数？", [m.content for m in model.seen_messages])
            self.assertNotIn("aggregate:old", str(model.seen_messages))
            self.assertNotIn("3 艘/小时", str(model.seen_messages))
            self.assertIn("2024-07-08T00:00:00Z", str(model.seen_messages))

            rejected = ask(create_graph(fixture.repo, OldCitationModel()),
                           "那前一天呢？", history=history)
            self.assertFalse(rejected["evidence"])
            self.assertIn("证据引用校验", rejected["answer"])
        finally:
            fixture.tearDown()

    def test_comparison_history_identifies_target_day_without_old_values(self):
        fixture = test_repository.RepositoryTests()
        fixture.setUp()
        try:
            history = [{"question": "7 月 8 日相较前一天 AIS 有什么变化？",
                        "answer": "AIS 报文数为 18180.17【comparison:old】",
                        "evidence": [{"evidence_id": "comparison:old", "target_day_utc": "2024-07-08",
                                      "baseline_day_utc": "2024-07-07",
                                      "metrics": {"ais_message_count": {"value": 18180.17}}}]}]
            model = FakeModel()
            ask(create_graph(fixture.repo, model), "后一天呢？", history=history)
            prompt = str(model.seen_messages)
            self.assertIn("上轮目标日期(UTC)：2024-07-08", prompt)
            self.assertIn("上轮对比基准日期(UTC)：2024-07-07", prompt)
            self.assertIn("ais_message_count", prompt)
            self.assertNotIn("18180.17", prompt)
            self.assertNotIn("comparison:old", prompt)
        finally:
            fixture.tearDown()

    def test_track_followup_keeps_selected_ship_and_excluded_date(self):
        fixture = test_repository.RepositoryTests()
        fixture.setUp()
        try:
            history = [{"question": "随机只船的航迹是怎么样的", "answer": "24 点【track:old】",
                        "evidence": [{"evidence_id": "track:old", "date_utc": "2024-07-08",
                                      "mmsi": "219382000", "vessel_name": "MAERSK CYPRUS",
                                      "hour_points": 24}]}]
            model = FakeModel()
            ask(create_graph(fixture.repo, model), "不要8号的", history=history)
            seen = str(model.seen_messages)
            self.assertIn("上轮航迹日期(UTC)：2024-07-08", seen)
            self.assertIn("上轮船舶MMSI：219382000", seen)
            self.assertIn("上轮船名：MAERSK CYPRUS", seen)
            self.assertIn("优先选择数据覆盖范围内的下一 UTC 日", model.system_prompt)
            self.assertNotIn("track:old", seen)
            self.assertNotIn("24 点", seen)
        finally:
            fixture.tearDown()

    def test_ten_recent_turns_and_older_summary_reach_model(self):
        fixture = test_repository.RepositoryTests()
        fixture.setUp()
        try:
            history = [{"question": f"第 {index} 问", "answer": "已回答", "evidence": []}
                       for index in range(12)]
            model = FakeModel()
            ask(create_graph(fixture.repo, model), "继续", history=history,
                summary="更早讨论过 Bayport 分区。")
            seen = str(model.seen_messages)
            self.assertIn("第 11 问", seen)
            self.assertNotIn("第 0 问", seen)
            self.assertNotIn("第 1 问", seen)
            self.assertIn("更早讨论过 Bayport 分区", seen)
        finally:
            fixture.tearDown()

    def test_token_guard_and_context_budget(self):
        from maritime_agent.tools import sanitize_tool_payload
        from maritime_agent.agent import _prune_messages_for_budget, _evidence_items

        mock_evidence = [
            {
                "evidence_id": f"ev_{i}",
                "modality": "ais",
                "object_type": "vessel",
                "provenance": {"huge_data": "x" * 500},
                "geometry": {"coordinates": [[0, 0], [1, 1]] * 50}
            }
            for i in range(40)
        ]
        raw_payload = {"evidence_id": "root_ev", "evidence": mock_evidence}
        sanitized = sanitize_tool_payload(raw_payload)

        # All evidence_ids must be retained for citation ledger
        all_ev = _evidence_items(sanitized)
        ev_ids = {e["evidence_id"] for e in all_ev}
        self.assertIn("root_ev", ev_ids)
        self.assertEqual(len(ev_ids), 41)

        # Payload must be dramatically compressed
        raw_len = len(json.dumps(raw_payload))
        san_len = len(json.dumps(sanitized))
        self.assertLess(san_len, raw_len * 0.2)

        # Budget pruner must compact older messages
        messages = [
            HumanMessage(content="Question 1"),
            AIMessage(content="", tool_calls=[{"name": "t1", "args": {}, "id": "c1"}]),
            ToolMessage(content=json.dumps(sanitized), tool_call_id="c1"),
            HumanMessage(content="Question 2"),
        ]
        pruned = _prune_messages_for_budget(messages, max_total_chars=1000)
        self.assertEqual(len(pruned), len(messages))
        self.assertLess(sum(len(str(m.content)) for m in pruned), sum(len(str(m.content)) for m in messages))

    def test_argument_normalizer_and_spatial_sandbox(self):
        from maritime_agent.tools import (
            normalize_date_arg, normalize_mmsi_arg, normalize_region_arg,
            set_active_region, get_active_region, build_tools
        )
        # Normalization
        self.assertEqual(normalize_date_arg("2024-01-06T17:11:00Z"), "2024-01-06")
        self.assertEqual(normalize_date_arg("2024/7/8"), "2024-07-08")
        self.assertEqual(normalize_mmsi_arg("MMSI: 258041000"), "258041000")
        self.assertEqual(normalize_region_arg("hau"), "region-d3529b060d6d4d65")
        self.assertEqual(normalize_region_arg("mobile"), "region-0a589e4f05844f47")

        # Hard Spatial Sandbox
        fixture = test_repository.RepositoryTests()
        fixture.setUp()
        try:
            repo = fixture.repo
            tools = {t.name: t for t in build_tools(repo)}

            # Sandboxed mode for Mobile
            set_active_region("region-0a589e4f05844f47")
            self.assertEqual(get_active_region(), "region-0a589e4f05844f47")

            # Calling data_overview while in Mobile sandbox must redirect to situation_cube_overview
            if hasattr(repo, "situation_cube_overview"):
                res = json.loads(tools["data_overview"].invoke({}))
                self.assertEqual(res.get("region_id"), "region-0a589e4f05844f47")

            set_active_region("")
        finally:
            fixture.tearDown()

    def test_tool_oscillation_dedup_and_breaker(self):
        class OscillationModel:
            def __init__(self):
                self.calls = 0

            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                self.calls += 1
                tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
                if len(tool_messages) == 0:
                    # 第 1 轮发起调用
                    return AIMessage(content="", tool_calls=[{"name": "data_overview", "args": {}, "id": "call_1"}])
                elif len(tool_messages) == 1:
                    # 第 2 轮故意机械重复发起一模一样的调用
                    return AIMessage(content="", tool_calls=[{"name": "data_overview", "args": {}, "id": "call_2"}])
                else:
                    # 第 3 轮收到自愈干预后给出回答，动态引用真实 evidence_id
                    last_tool = tool_messages[-1]
                    self.last_tool_content = str(last_tool.content)
                    ev_tool = next(m for m in tool_messages if "evidence_id" in str(m.content))
                    ev_id = json.loads(ev_tool.content)["evidence_id"]
                    return AIMessage(content=f"检测到重复，依据目录已知数据完成说明【{ev_id}】。")

        model = OscillationModel()
        fixture = test_repository.RepositoryTests()
        fixture.setUp()
        try:
            graph = create_graph(fixture.repo, model)
            result = ask(graph, "查询数据覆盖？")
            self.assertTrue(result["evidence"])
            self.assertIn("检测到工具重复调用振荡", " ".join(result.get("warnings", [])))
            self.assertIn("系统自愈干预与去重熔断", model.last_tool_content)
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()

