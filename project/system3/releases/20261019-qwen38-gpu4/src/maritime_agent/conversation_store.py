"""Durable, local conversation history kept separate from the read-only DS."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from threading import Lock
from typing import Iterator
from uuid import UUID, uuid4

from .reasoning_memory import (
    CognitiveBlackboard,
    EvolvedCritique,
    ReasoningSnapshot,
    detect_and_evolve_critique,
    distill_tool_failure_critique,
    distill_verifier_warning_critique,
    extract_reasoning_snapshot,
    update_blackboard_state,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _turn_note(question: str, response: dict) -> dict:
    """Keep topic and scope for old turns, without answer values or citations."""
    note = {"question": question[:180], "dates": [], "metrics": [], "selectors": []}

    def add(key: str, value: object) -> None:
        if value is not None and str(value) and str(value) not in note[key]:
            note[key].append(str(value))

    def scope_fields(scope: object) -> None:
        if not isinstance(scope, dict):
            return
        add("dates", str(scope.get("start_utc", ""))[:10])
        add("metrics", scope.get("metric"))
        add("selectors", scope.get("selector"))
        add("selectors", scope.get("table"))

    for item in response.get("evidence", []):
        if not isinstance(item, dict):
            continue
        add("dates", item.get("target_day_utc"))
        add("dates", item.get("baseline_day_utc"))
        if item.get("date_utc") and item.get("mmsi"):
            add("dates", item.get("date_utc"))
            add("selectors", "MMSI " + str(item["mmsi"])[:20])
            add("selectors", item.get("vessel_name"))
        if isinstance(item.get("metrics"), dict):
            for metric in item["metrics"]:
                add("metrics", metric)
        scope_fields(item.get("scope"))
        for part in ("target", "baseline"):
            nested = item.get(part)
            if isinstance(nested, dict):
                scope_fields(nested.get("scope"))
    return note


def _memory_summary(notes: list[dict]) -> str:
    if not notes:
        return ""
    parts = [f"更早的 {len(notes)} 轮仅保留主题线索。"]
    for key, label, limit in (("dates", "涉及日期", 12), ("metrics", "涉及指标", 12),
                              ("selectors", "涉及对象", 10)):
        values = list(dict.fromkeys(str(value) for note in notes for value in note.get(key, []) if value))
        if values:
            parts.append(label + "：" + "、".join(values[-limit:]))
    questions = [str(note.get("question", ""))[:120] for note in notes[-6:] if note.get("question")]
    if questions:
        parts.append("最近的较早问题：" + "；".join(questions))
    corrections = [str(note.get("question", ""))[:120] for note in notes
                   if any(word in str(note.get("question", "")) for word in ("不是", "改成", "只看", "不要", "更正"))]
    if corrections:
        parts.append("较早的限定或纠正：" + "；".join(corrections[-4:]))
    return "。".join(parts)[:2400]


class ConversationStore:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, Lock] = {}
        self._locks_guard = Lock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    question TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_conversation
                    ON turns(conversation_id, id);
                CREATE TABLE IF NOT EXISTS turn_notes (
                    turn_id INTEGER PRIMARY KEY REFERENCES turns(id),
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    note_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turn_notes_conversation
                    ON turn_notes(conversation_id, turn_id);
                CREATE TABLE IF NOT EXISTS conversation_reasoning_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    turn_index INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reasoning_snapshots_conv
                    ON conversation_reasoning_snapshots(conversation_id, turn_index);
                CREATE TABLE IF NOT EXISTS conversation_blackboards (
                    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id),
                    blackboard_json TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_evolved_critiques (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    critique_id TEXT UNIQUE NOT NULL,
                    domain TEXT NOT NULL,
                    trigger_pattern TEXT NOT NULL,
                    failed_path TEXT NOT NULL,
                    corrected_lesson TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at_utc TEXT NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'user_correction',
                    updated_at_utc TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_critiques_domain
                    ON agent_evolved_critiques(domain);
            """)

            # Migration: Ensure columns exist in agent_evolved_critiques
            col_info = conn.execute("PRAGMA table_info(agent_evolved_critiques)").fetchall()
            existing_cols = {r[1] for r in col_info}
            if "hit_count" not in existing_cols:
                conn.execute("ALTER TABLE agent_evolved_critiques ADD COLUMN hit_count INTEGER DEFAULT 0")
            if "success_count" not in existing_cols:
                conn.execute("ALTER TABLE agent_evolved_critiques ADD COLUMN success_count INTEGER DEFAULT 0")
            if "source" not in existing_cols:
                conn.execute("ALTER TABLE agent_evolved_critiques ADD COLUMN source TEXT DEFAULT 'user_correction'")
            if "updated_at_utc" not in existing_cols:
                conn.execute("ALTER TABLE agent_evolved_critiques ADD COLUMN updated_at_utc TEXT DEFAULT ''")

            missing = conn.execute("SELECT t.id, t.conversation_id, t.question, t.response_json "
                                   "FROM turns t LEFT JOIN turn_notes n ON n.turn_id=t.id "
                                   "WHERE n.turn_id IS NULL").fetchall()
            for turn_id, conversation_id, question, payload in missing:
                note = _turn_note(question, json.loads(payload))
                conn.execute("INSERT INTO turn_notes VALUES (?, ?, ?)",
                             (turn_id, conversation_id, json.dumps(note, ensure_ascii=False)))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create(self) -> str:
        conversation_id = str(uuid4())
        now = _now()
        with self._connect() as conn:
            conn.execute("INSERT INTO conversations VALUES (?, ?, ?)",
                         (conversation_id, now, now))
        return conversation_id

    def exists(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return row is not None

    @contextmanager
    def session(self, conversation_id: str | None) -> Iterator[str]:
        if conversation_id is None:
            conversation_id = self.create()
        else:
            conversation_id = str(UUID(conversation_id))
        with self._locks_guard:
            lock = self._locks.setdefault(conversation_id, Lock())
        with lock:
            if not self.exists(conversation_id):
                raise KeyError("会话不存在")
            yield conversation_id

    def append(self, conversation_id: str, question: str, response: dict) -> None:
        now = _now()
        payload = json.dumps(response, ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute("INSERT INTO turns (conversation_id, question, response_json, created_at_utc) "
                                  "VALUES (?, ?, ?, ?)", (conversation_id, question, payload, now))
            turn_id = cursor.lastrowid
            note = _turn_note(question, response)
            conn.execute("INSERT INTO turn_notes VALUES (?, ?, ?)",
                         (turn_id, conversation_id, json.dumps(note, ensure_ascii=False)))
            conn.execute("UPDATE conversations SET updated_at_utc=? WHERE id=?", (now, conversation_id))

            # 1. 自动抽取并沉淀本轮思维链快照 (Reasoning Checkpoint)
            active_scope = {
                "region_id": response.get("selected_region_id"),
                "source_id": response.get("selected_source_id"),
            }
            snapshot = extract_reasoning_snapshot(
                turn_index=turn_id,
                question=question,
                answer=str(response.get("answer") or response.get("response") or ""),
                evidence=response.get("evidence"),
                active_scope=active_scope,
            )
            conn.execute(
                "INSERT INTO conversation_reasoning_snapshots (conversation_id, turn_index, snapshot_json, created_at_utc) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, turn_id, json.dumps(snapshot.to_dict(), ensure_ascii=False), now)
            )

            # 2. 自动更新当前会话的态势认知黑板 (Cognitive Blackboard State Machine)
            bb = self._get_blackboard_with_conn(conn, conversation_id)
            bb = update_blackboard_state(bb, snapshot, response)
            conn.execute(
                "INSERT INTO conversation_blackboards (conversation_id, blackboard_json, updated_at_utc) "
                "VALUES (?, ?, ?) ON CONFLICT(conversation_id) DO UPDATE SET "
                "blackboard_json=excluded.blackboard_json, updated_at_utc=excluded.updated_at_utc",
                (conversation_id, json.dumps(bb.to_dict(), ensure_ascii=False), now)
            )

            # 3. 自进化反思多通道提炼 (Autonomous Multi-Channel Reflexion)
            critiques_to_save: list[EvolvedCritique] = []

            # 通道 A: 用户明确纠错提炼
            critique_user = detect_and_evolve_critique(question, history=[])
            if critique_user:
                critiques_to_save.append(critique_user)

            # 通道 B: 事实核验器告警反思提炼
            critique_warn = distill_verifier_warning_critique(response.get("warnings", []), question)
            if critique_warn:
                critiques_to_save.append(critique_warn)

            # 通道 C: 工具调用空查/失败自省提炼
            critique_tool = distill_tool_failure_critique(
                response.get("tool_trace", []),
                response.get("evidence", [])
            )
            if critique_tool:
                critiques_to_save.append(critique_tool)

            for critique in critiques_to_save:
                conn.execute(
                    "INSERT INTO agent_evolved_critiques (critique_id, domain, trigger_pattern, failed_path, corrected_lesson, confidence, created_at_utc, hit_count, success_count, source, updated_at_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(critique_id) DO UPDATE SET "
                    "corrected_lesson=excluded.corrected_lesson, confidence=excluded.confidence, updated_at_utc=excluded.updated_at_utc",
                    (critique.critique_id, critique.domain, critique.trigger_pattern, critique.failed_path,
                     critique.corrected_lesson, critique.confidence, critique.created_at_utc,
                     critique.hit_count, critique.success_count, critique.source, now)
                )

    def _get_blackboard_with_conn(self, conn: sqlite3.Connection, conversation_id: str) -> CognitiveBlackboard:
        row = conn.execute("SELECT blackboard_json FROM conversation_blackboards WHERE conversation_id=?",
                           (conversation_id,)).fetchone()
        if not row:
            return CognitiveBlackboard(conversation_id=conversation_id)
        data = json.loads(row[0])
        return CognitiveBlackboard.from_dict(data)

    def get_cognitive_blackboard(self, conversation_id: str) -> CognitiveBlackboard:
        with self._connect() as conn:
            return self._get_blackboard_with_conn(conn, conversation_id)

    def get_latest_reasoning_snapshot(self, conversation_id: str) -> ReasoningSnapshot | None:
        with self._connect() as conn:
            row = conn.execute("SELECT snapshot_json FROM conversation_reasoning_snapshots "
                               "WHERE conversation_id=? ORDER BY turn_index DESC LIMIT 1",
                               (conversation_id,)).fetchone()
            if not row:
                return None
            return ReasoningSnapshot.from_dict(json.loads(row[0]))

    def get_reasoning_snapshots(self, conversation_id: str, limit: int = 5) -> list[ReasoningSnapshot]:
        with self._connect() as conn:
            rows = conn.execute("SELECT snapshot_json FROM conversation_reasoning_snapshots "
                                "WHERE conversation_id=? ORDER BY turn_index DESC LIMIT ?",
                                (conversation_id, limit)).fetchall()
            return [ReasoningSnapshot.from_dict(json.loads(r[0])) for r in rows[::-1]]

    def save_evolved_critique(self, critique: EvolvedCritique) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO agent_evolved_critiques (critique_id, domain, trigger_pattern, failed_path, corrected_lesson, confidence, created_at_utc, hit_count, success_count, source, updated_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(critique_id) DO UPDATE SET "
                "corrected_lesson=excluded.corrected_lesson, confidence=excluded.confidence, updated_at_utc=excluded.updated_at_utc",
                (critique.critique_id, critique.domain, critique.trigger_pattern, critique.failed_path,
                 critique.corrected_lesson, critique.confidence, critique.created_at_utc,
                 critique.hit_count, critique.success_count, critique.source, now)
            )

    def get_relevant_critiques(
        self,
        query: str = "",
        domain: str | None = None,
        limit: int = 3,
        min_confidence: float = 0.5,
    ) -> list[EvolvedCritique]:
        with self._connect() as conn:
            cols = "critique_id, domain, trigger_pattern, failed_path, corrected_lesson, confidence, created_at_utc, hit_count, success_count, source, updated_at_utc"
            if query:
                tokens = [t.lower() for t in re.split(r"[\s,;，。！？]+", query) if len(t) >= 2]
                all_rows = conn.execute(
                    f"SELECT {cols} FROM agent_evolved_critiques WHERE confidence >= ? ORDER BY confidence DESC, id DESC LIMIT 50",
                    (min_confidence,)
                ).fetchall()
                scored = []
                for r in all_rows:
                    text = f"{r[1]} {r[2]} {r[4]}".lower()
                    score = sum(1 for t in tokens if t in text)
                    if domain and r[1] == domain:
                        score += 2.0
                    scored.append((score, r))
                scored.sort(key=lambda x: (x[0], x[1][5]), reverse=True)
                rows = [x[1] for x in scored[:limit]]
            else:
                rows = conn.execute(
                    f"SELECT {cols} FROM agent_evolved_critiques WHERE confidence >= ? ORDER BY id DESC LIMIT ?",
                    (min_confidence, limit)
                ).fetchall()
            return [EvolvedCritique(
                critique_id=r[0], domain=r[1], trigger_pattern=r[2], failed_path=r[3],
                corrected_lesson=r[4], confidence=r[5], created_at_utc=r[6],
                hit_count=r[7] if len(r) > 7 and r[7] is not None else 0,
                success_count=r[8] if len(r) > 8 and r[8] is not None else 0,
                source=r[9] if len(r) > 9 and r[9] is not None else "user_correction",
                updated_at_utc=r[10] if len(r) > 10 and r[10] is not None else "",
            ) for r in rows]

    def delete_evolved_critique(self, critique_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM agent_evolved_critiques WHERE critique_id=?", (critique_id,))
            return cursor.rowcount > 0

    def adjust_critique_confidence(self, critique_id: str, delta: float) -> float | None:
        now = _now()
        with self._connect() as conn:
            row = conn.execute("SELECT confidence, success_count FROM agent_evolved_critiques WHERE critique_id=?", (critique_id,)).fetchone()
            if not row:
                return None
            new_conf = max(0.0, min(2.0, round(float(row[0]) + delta, 2)))
            succ_delta = 1 if delta > 0 else 0
            conn.execute("UPDATE agent_evolved_critiques SET confidence=?, success_count=success_count+?, updated_at_utc=? WHERE critique_id=?",
                         (new_conf, succ_delta, now, critique_id))
            return new_conf

    def context(self, conversation_id: str, recent_limit: int = 10) -> tuple[list[dict], str]:
        with self._connect() as conn:
            recent_rows = conn.execute("SELECT id, question, response_json FROM turns "
                                       "WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                                       (conversation_id, recent_limit)).fetchall()[::-1]
            cutoff = recent_rows[0][0] if recent_rows else 0
            older_rows = conn.execute("SELECT note_json FROM turn_notes WHERE conversation_id=? "
                                      "AND turn_id<? ORDER BY turn_id", (conversation_id, cutoff)).fetchall()
        recent = [{"question": question, **json.loads(payload)} for _, question, payload in recent_rows]
        summary = _memory_summary([json.loads(payload) for (payload,) in older_rows])
        return recent, summary

    def list_conversations(self, limit: int = 40, offset: int = 0) -> tuple[list[dict], int]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM conversations c WHERE EXISTS "
                                 "(SELECT 1 FROM turns t WHERE t.conversation_id=c.id)").fetchone()[0]
            rows = conn.execute("SELECT c.id, c.created_at_utc, c.updated_at_utc, "
                                "(SELECT question FROM turns t WHERE t.conversation_id=c.id ORDER BY t.id LIMIT 1), "
                                "(SELECT COUNT(*) FROM turns t WHERE t.conversation_id=c.id) "
                                "FROM conversations c WHERE EXISTS "
                                "(SELECT 1 FROM turns t WHERE t.conversation_id=c.id) "
                                "ORDER BY c.updated_at_utc DESC, c.id DESC LIMIT ? OFFSET ?",
                                (limit, offset)).fetchall()
        return ([{"conversation_id": row[0], "created_at_utc": row[1],
                  "updated_at_utc": row[2], "title": row[3][:80], "turn_count": row[4]}
                 for row in rows], total)

    def turns(self, conversation_id: str, limit: int | None = None) -> list[dict]:
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute("SELECT question, response_json FROM turns "
                                     "WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
            else:
                rows = conn.execute("SELECT question, response_json FROM turns "
                                     "WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                                     (conversation_id, limit)).fetchall()[::-1]
        return [{"question": question, **json.loads(payload)} for question, payload in rows]
