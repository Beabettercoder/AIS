from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from maritime_agent.api import app
from maritime_agent.conversation_store import ConversationStore


class MemoryApiTests(unittest.TestCase):
    def test_dataset_scope_excludes_turns_from_other_regions(self):
        with TemporaryDirectory() as temp:
            store = ConversationStore(Path(temp) / "history.sqlite3")
            seen = []

            class Intake:
                def regions(self):
                    return {"regions": [{"region_id": "region-hau", "canonical_name": "hau"},
                                        {"region_id": "region-other", "canonical_name": "other"}]}

            class Repo:
                def search_ingested_data(self, limit=100):
                    return {"sources": [
                        {"source_id": "ais-hau", "region_id": "region-hau", "kind": "ais_csv"},
                        {"source_id": "sar-hau", "region_id": "region-hau", "kind": "remote_geotiff"},
                    ]}

            def fake_ask(_graph, question, history=None, summary=""):
                seen.append((question, list(history or []), summary))
                return {"answer": "已核查", "evidence": [], "warnings": [], "tool_trace": []}

            with patch("maritime_agent.api.conversation_store", return_value=store), \
                 patch("maritime_agent.api.intake_agent", return_value=Intake()), \
                 patch("maritime_agent.api.repository", return_value=Repo()), \
                 patch("maritime_agent.api.graph", return_value=object()), \
                 patch("maritime_agent.api.ask", side_effect=fake_ask):
                client = TestClient(app)
                first = client.post("/ask", json={"question": "Houston 有什么数据？"})
                conversation_id = first.json()["conversation_id"]
                second = client.post("/ask", json={"question": "这里有什么遥感数据？",
                                                    "conversation_id": conversation_id,
                                                    "region_id": "region-hau"})
                self.assertEqual(second.status_code, 200)
                self.assertEqual(seen[1][1], [])
                self.assertEqual(seen[1][2], "")
                self.assertIn("区域ID=region-hau", seen[1][0])
                third = client.post("/ask", json={"question": "那 AIS 呢？",
                                                   "conversation_id": conversation_id,
                                                   "region_id": "region-hau"})
                self.assertEqual(third.status_code, 200)
                self.assertEqual([turn["question"] for turn in seen[2][1]],
                                 ["这里有什么遥感数据？"])
                self.assertEqual(seen[2][1][0]["selected_region_id"], "region-hau")

    def test_follow_up_uses_only_its_conversation_and_can_be_reloaded(self):
        with TemporaryDirectory() as temp:
            store = ConversationStore(Path(temp) / "history.sqlite3")
            seen = []

            def fake_ask(_graph, question, history=None, summary=""):
                seen.append((question, list(history or []), summary))
                return {"answer": f"收到第 {len(history or []) + 1} 问", "evidence": [],
                        "warnings": [], "tool_trace": []}

            with patch("maritime_agent.api.conversation_store", return_value=store), \
                 patch("maritime_agent.api.graph", return_value=object()), \
                 patch("maritime_agent.api.ask", side_effect=fake_ask):
                client = TestClient(app)
                first = client.post("/ask", json={"question": "7 月 8 日呢？"})
                self.assertEqual(first.status_code, 200)
                conversation_id = first.json()["conversation_id"]
                second = client.post("/ask", json={"question": "那前一天呢？",
                                                   "conversation_id": conversation_id})
                self.assertEqual(second.status_code, 200)
                self.assertEqual(seen[1][1][0]["question"], "7 月 8 日呢？")
                self.assertEqual(seen[1][1][0]["answer"], "收到第 1 问")

                separate = client.post("/ask", json={"question": "新问题"})
                self.assertEqual(separate.status_code, 200)
                self.assertNotEqual(separate.json()["conversation_id"], conversation_id)
                self.assertEqual(seen[2][1], [])

                listed = client.get("/conversations")
                self.assertEqual(listed.status_code, 200)
                self.assertEqual(listed.json()["total"], 2)
                self.assertEqual(listed.json()["conversations"][0]["title"], "新问题")
                self.assertEqual(client.get("/conversations?limit=1&offset=1").json()["conversations"][0]["conversation_id"], conversation_id)

                loaded = client.get(f"/conversations/{conversation_id}")
                self.assertEqual([turn["question"] for turn in loaded.json()["turns"]],
                                 ["7 月 8 日呢？", "那前一天呢？"])
                self.assertEqual(client.get(
                    "/conversations/00000000-0000-4000-8000-000000000000").status_code, 404)

    def test_api_passes_older_summary_after_ten_turns(self):
        with TemporaryDirectory() as temp:
            store = ConversationStore(Path(temp) / "history.sqlite3")
            seen = []

            def fake_ask(_graph, question, history=None, summary=""):
                seen.append((len(history or []), summary))
                return {"answer": "已核查", "evidence": [], "warnings": [], "tool_trace": []}

            with patch("maritime_agent.api.conversation_store", return_value=store), \
                 patch("maritime_agent.api.graph", return_value=object()), \
                 patch("maritime_agent.api.ask", side_effect=fake_ask):
                client = TestClient(app)
                conversation_id = None
                for index in range(12):
                    payload = {"question": f"第 {index + 1} 个问题"}
                    if conversation_id:
                        payload["conversation_id"] = conversation_id
                    result = client.post("/ask", json=payload)
                    self.assertEqual(result.status_code, 200)
                    conversation_id = result.json()["conversation_id"]
            self.assertEqual(seen[10], (10, ""))
            self.assertEqual(seen[11][0], 10)
            self.assertIn("更早的 1 轮", seen[11][1])
            self.assertIn("第 1 个问题", seen[11][1])


if __name__ == "__main__":
    unittest.main()
