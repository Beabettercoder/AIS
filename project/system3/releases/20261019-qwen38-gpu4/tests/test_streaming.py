from __future__ import annotations

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

import test_repository
from maritime_agent.api import app
from maritime_agent.conversation_store import ConversationStore
from maritime_agent.coordinated_agent import create_coordinated_graph


class StreamingTestModel:
    def __init__(self, roles):
        self.roles = roles
        self.repair_attempted = False

    def bind_tools(self, tools):
        return self

    def stream(self, messages):
        prompt = str(messages[0].content)
        if "海事综合分析员" in prompt:
            ids = list(dict.fromkeys(re.findall(r"【([^】]+)】", str(messages[-1].content))))
            text = "综合研判报告：" + " ".join(f"【{item}】" for item in ids)
            for char in text:
                yield AIMessage(content=char)
            return
        yield from self.invoke(messages)

    def invoke(self, messages):
        prompt = str(messages[0].content)
        human_msg = str(messages[-1].content)
        if "事实核验审查反馈" in human_msg:
            self.repair_attempted = True
            all_matches = re.findall(r"【([^】]+)】", human_msg)
            valid_candidates = [m for m in all_matches if "invalid" not in m and "fabricated" not in m]
            target_id = valid_candidates[0] if valid_candidates else "overview:2024-07-06T00:00:00Z:2024-07-10T23:00:00Z"
            return AIMessage(content=f"自愈修正后的报告【{target_id}】。")
        if "海事问题协调员" in prompt:
            return AIMessage(content=json.dumps({"roles": self.roles, "subquestions": {}}, ensure_ascii=False))
        if "海事综合分析员" in prompt:
            ids = list(dict.fromkeys(re.findall(r"【([^】]+)】", human_msg)))
            return AIMessage(content="综合研判报告：" + " ".join(f"【{item}】" for item in ids))
        evidence = [m for m in messages if isinstance(m, ToolMessage)]
        if evidence:
            data = json.loads(evidence[-1].content)
            return AIMessage(content=f"本地覆盖见【{data['evidence_id']}】。")
        return AIMessage(content="", tool_calls=[{"name": "data_overview", "args": {}, "id": "overview_1"}])


class StreamingEngineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_stream_coordinated_event_sequence_and_tokens(self):
        model = StreamingTestModel(["ais", "gis"])
        graph = create_coordinated_graph(self.fixture.repo, model)

        events = list(graph.stream_coordinated("Houston 港 AIS 和空间边界概况？", history=[], summary=""))
        self.assertTrue(len(events) >= 5)

        event_types = [e["event"] for e in events]
        self.assertIn("routing", event_types)
        self.assertIn("status", event_types)
        self.assertIn("token", event_types)
        self.assertIn("verifier", event_types)
        self.assertIn("done", event_types)

        # 检查首包 routing
        routing_ev = next(e for e in events if e["event"] == "routing")
        self.assertEqual(routing_ev["data"]["stage"], "routing")
        self.assertIn("ais", routing_ev["data"]["roles"])

        # 检查 token 拼接
        tokens = [e["data"]["delta"] for e in events if e["event"] == "token"]
        full_streamed = "".join(tokens)
        self.assertIn("综合研判报告", full_streamed)

        # 检查 done 帧
        done_ev = next(e for e in events if e["event"] == "done")
        final_result = done_ev["data"]
        self.assertTrue(final_result["evidence"])
        self.assertIn("综合研判报告", final_result["answer"])
        self.assertTrue(final_result["thought"])

    def test_stream_coordinated_with_critic_correction_feedback(self):
        class FlawedStreamModel(StreamingTestModel):
            def stream(self, messages):
                # 首次综合故意输出不存在的证据
                yield AIMessage(content="初版草稿【fabricated:invalid_id】。")

        model = FlawedStreamModel(["ais", "events"])
        graph = create_coordinated_graph(self.fixture.repo, model)

        events = list(graph.stream_coordinated("AIS 和运营事件数据范围？", history=[], summary=""))
        self.assertTrue(model.repair_attempted)

        # 验证发射了 healing 和 healed 的 verifier 状态
        verifier_statuses = [
            e["data"].get("status")
            for e in events
            if e["event"] == "verifier"
        ]
        self.assertIn("healing", verifier_statuses)
        self.assertIn("healed", verifier_statuses)

        done_ev = next(e for e in events if e["event"] == "done")
        self.assertIn("自愈修正后的报告", done_ev["data"]["answer"])
        self.assertIn("已成功触发核验双向自愈闭环", " ".join(done_ev["data"].get("warnings", [])))


class ApiStreamingTests(unittest.TestCase):
    def test_api_ask_stream_sse_greeting_interception(self):
        with TemporaryDirectory() as temp:
            store = ConversationStore(Path(temp) / "history.sqlite3")
            with patch("maritime_agent.api.conversation_store", return_value=store):
                client = TestClient(app)
                response = client.post("/ask/stream", json={"question": "你好，请问你是谁？"})
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])

                # 解析 SSE 流
                raw_text = response.text
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                events = []
                current_event = None
                for line in lines:
                    if line.startswith("event:"):
                        current_event = line.split("event:")[1].strip()
                    elif line.startswith("data:") and current_event:
                        data_val = json.loads(line.split("data:")[1].strip())
                        events.append((current_event, data_val))
                        current_event = None

                event_names = [e[0] for e in events]
                self.assertIn("routing", event_names)
                self.assertIn("thought", event_names)
                self.assertIn("token", event_names)
                self.assertIn("done", event_names)

                done_data = next(d for name, d in events if name == "done")
                self.assertIn("海事态势调查智能体", done_data["answer"])
                self.assertTrue(done_data.get("conversation_id"))

    def test_api_ask_stream_sse_normal_agent_flow(self):
        with TemporaryDirectory() as temp:
            store = ConversationStore(Path(temp) / "history.sqlite3")

            def fake_stream_ask(_graph, question, history=None, summary="", cognitive_context=""):
                yield {"event": "routing", "data": {"stage": "routing", "roles": ["ais"]}}
                yield {"event": "status", "data": {"stage": "specialists_done", "message": "已调取航迹"}}
                yield {"event": "token", "data": {"stage": "synthesis", "delta": "正在推演..."}}
                yield {"event": "token", "data": {"stage": "synthesis", "delta": "【track:1】"}}
                yield {"event": "verifier", "data": {"stage": "verification", "status": "passed"}}
                yield {"event": "done", "data": {
                    "answer": "正在推演...【track:1】",
                    "evidence": [{"evidence_id": "track:1"}],
                    "warnings": [],
                    "tool_trace": [{"tool": "get_vessel_track"}],
                    "thought": "研判推演完成"
                }}

            with patch("maritime_agent.api.conversation_store", return_value=store), \
                 patch("maritime_agent.api.graph", return_value=object()), \
                 patch("maritime_agent.api.stream_ask", side_effect=fake_stream_ask):
                client = TestClient(app)
                response = client.post("/ask/stream", json={"question": "查询 MMSI 258041000 航迹"})
                self.assertEqual(response.status_code, 200)

                raw_text = response.text
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                events = []
                current_event = None
                for line in lines:
                    if line.startswith("event:"):
                        current_event = line.split("event:")[1].strip()
                    elif line.startswith("data:") and current_event:
                        data_val = json.loads(line.split("data:")[1].strip())
                        events.append((current_event, data_val))
                        current_event = None

                event_names = [e[0] for e in events]
                self.assertEqual(event_names, ["routing", "status", "token", "token", "verifier", "done"])

                done_data = next(d for name, d in events if name == "done")
                cid = done_data["conversation_id"]
                self.assertTrue(cid)
                self.assertEqual(done_data["evidence"], [{"evidence_id": "track:1"}])

                # 验证会话已正确持久化写入 store
                recent, _ = store.context(cid, 10)
                self.assertEqual(len(recent), 1)
                self.assertEqual(recent[0]["question"], "查询 MMSI 258041000 航迹")
                self.assertEqual(recent[0]["answer"], "正在推演...【track:1】")
