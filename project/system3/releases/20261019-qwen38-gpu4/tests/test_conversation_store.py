from __future__ import annotations

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sqlite3
import unittest

from maritime_agent.conversation_store import ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def test_turns_persist_and_conversations_are_isolated(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "conversations.sqlite3"
            store = ConversationStore(path)
            with store.session(None) as first:
                store.append(first, "7 月 8 日如何？", {"answer": "每小时 3 艘", "evidence": []})
                store.append(first, "那前一天呢？", {"answer": "每小时 46 艘", "evidence": []})
            with store.session(None) as second:
                store.append(second, "数据范围？", {"answer": "2024 年", "evidence": []})

            reopened = ConversationStore(path)
            self.assertEqual([turn["question"] for turn in reopened.turns(first)],
                             ["7 月 8 日如何？", "那前一天呢？"])
            self.assertEqual(reopened.turns(first, limit=1)[0]["answer"], "每小时 46 艘")
            self.assertEqual(len(reopened.turns(second)), 1)
            self.assertNotEqual(first, second)
            with self.assertRaises(KeyError):
                with reopened.session("00000000-0000-4000-8000-000000000000"):
                    pass

    def test_older_turns_are_summarized_without_old_values(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "conversations.sqlite3"
            store = ConversationStore(path)
            with store.session(None) as conversation_id:
                for day in range(1, 13):
                    store.append(conversation_id, f"7 月 {day} 日 Bayport AIS 如何？",
                                 {"answer": f"数值 {day * 100}【aggregate:old{day}】",
                                  "evidence": [{"evidence_id": f"aggregate:old{day}", "value": day * 100,
                                                "scope": {"table": "zone", "selector": "bayport",
                                                          "metric": "unique_mmsi",
                                                          "start_utc": f"2024-07-{day:02d}T00:00:00Z"}}]})
            reopened = ConversationStore(path)
            recent, summary = reopened.context(conversation_id, recent_limit=10)
            self.assertEqual(len(recent), 10)
            self.assertEqual(recent[0]["question"], "7 月 3 日 Bayport AIS 如何？")
            self.assertIn("更早的 2 轮", summary)
            self.assertIn("bayport", summary)
            self.assertIn("2024-07-01", summary)
            self.assertNotIn("aggregate:old", summary)
            self.assertNotIn("数值 100", summary)
            listed, total = reopened.list_conversations()
            self.assertEqual(total, 1)
            self.assertEqual(listed[0]["turn_count"], 12)

    def test_existing_history_backfills_topic_notes(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "old.sqlite3"
            conversation_id = "00000000-0000-4000-8000-000000000001"
            with closing(sqlite3.connect(path)) as conn:
                conn.executescript("""
                    CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL);
                    CREATE TABLE turns (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id), question TEXT NOT NULL,
                        response_json TEXT NOT NULL, created_at_utc TEXT NOT NULL);
                """)
                conn.execute("INSERT INTO conversations VALUES (?, ?, ?)",
                             (conversation_id, "2024-01-01", "2024-01-01"))
                for question in ("7 月 8 日 AIS 变化？", "后一天呢？"):
                    conn.execute("INSERT INTO turns (conversation_id, question, response_json, created_at_utc) "
                                 "VALUES (?, ?, ?, ?)",
                                 (conversation_id, question, json.dumps({"answer": "已回答", "evidence": []}),
                                  "2024-01-01"))
                conn.commit()
            store = ConversationStore(path)
            recent, summary = store.context(conversation_id, recent_limit=1)
            self.assertEqual(recent[0]["question"], "后一天呢？")
            self.assertIn("7 月 8 日 AIS 变化？", summary)


if __name__ == "__main__":
    unittest.main()
