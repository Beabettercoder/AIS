"""Append-only manual review audit, isolated from the host application databases."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
import hashlib
import math
from pathlib import Path
import sqlite3
import statistics


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=15000")
        db.execute("""CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT NOT NULL, record_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('priority','normal','signal')),
            elapsed_seconds REAL NOT NULL CHECK(elapsed_seconds >= 0 AND elapsed_seconds <= 86400),
            note TEXT NOT NULL, created_at_utc TEXT NOT NULL, evidence_json TEXT NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS patrol_windows (
            dataset_id TEXT NOT NULL, window_id TEXT NOT NULL, mmsi TEXT NOT NULL,
            behavior_type TEXT NOT NULL, far_budget REAL NOT NULL,
            start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
            start_epoch REAL NOT NULL, end_epoch REAL NOT NULL,
            score REAL NOT NULL, first_seen_utc TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(dataset_id,window_id,behavior_type,far_budget)
        )""")
        if "evidence_json" not in {row[1] for row in db.execute("PRAGMA table_info(patrol_windows)")}:
            db.execute("ALTER TABLE patrol_windows ADD COLUMN evidence_json TEXT NOT NULL DEFAULT '{}'")
        db.commit()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def append(self, *, scan_id: str, record_id: str, dataset_id: str, status: str,
               elapsed_seconds: float, note: str, evidence: dict) -> dict:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as db:
            cursor = db.execute("INSERT INTO reviews (scan_id,record_id,dataset_id,status,elapsed_seconds,note,created_at_utc,evidence_json) VALUES (?,?,?,?,?,?,?,?)",
                                (scan_id, record_id, dataset_id, status, elapsed_seconds, note, stamp, json.dumps(evidence, ensure_ascii=False, allow_nan=False)))
            row = db.execute("SELECT * FROM reviews WHERE id=?", (cursor.lastrowid,)).fetchone()
            return self._public(row)

    @staticmethod
    def _public(row) -> dict:
        return {key: row[key] for key in row.keys() if key != "evidence_json"}

    def history(self, scan_id: str, record_id: str) -> list[dict]:
        with self._connect() as db:
            return [self._public(row) for row in db.execute("SELECT * FROM reviews WHERE scan_id=? AND record_id=? ORDER BY id", (scan_id, record_id))]

    def stats(self) -> dict:
        with self._connect() as db:
            rows = list(db.execute("SELECT id,dataset_id,record_id,status,elapsed_seconds FROM reviews ORDER BY id"))
        durations = [r["elapsed_seconds"] for r in rows]
        latest = {(r["dataset_id"], r["record_id"]): r for r in rows}
        return {"total_reviews": len(rows),
                **{status: sum(r["status"] == status for r in rows) for status in ("priority", "normal", "signal")},
                "unique_records": len(latest),
                "latest_status_counts": {status: sum(r["status"] == status for r in latest.values()) for status in ("priority", "normal", "signal")},
                "total_seconds": round(sum(durations), 2),
                "average_seconds": round(statistics.mean(durations), 2) if durations else None,
                "median_seconds": round(statistics.median(durations), 2) if durations else None,
                "time_source": "analyst_browser_timer",
                "notice": "统计仅汇总人工复核操作和浏览器计时，不等同于检出率、误报率或省时率；重复复核按审计操作分别计数。"}

    def merge_patrol(self, records: list[dict], dataset_id: str, far_budget: float) -> tuple[list[dict], list[dict]]:
        """Merge only identified, UTC-timed observations; preserve original window IDs.

        A continuity gap of at most 120 seconds is explicit. Long observation
        gaps start a new timeline instead of inventing continuous behavior.
        """
        def epoch(value):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                raise ValueError("Patrol timestamps must identify their timezone")
            return dt.timestamp()
        valid, others = [], []
        for record in records:
            vessel = str(record.get("vessel_id", ""))
            if not (len(vessel) == 9 and vessel.isdigit() and record.get("start_utc") and record.get("end_utc")):
                others.append(record)
                continue
            try:
                start, end = epoch(record["start_utc"]), epoch(record["end_utc"])
                if end < start or not all(math.isfinite(x) for x in (start, end)):
                    raise ValueError()
            except (ValueError, TypeError):
                others.append(record)
                continue
            valid.append((record, vessel, start, end))
        if not valid:
            return records, []
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as db:
            existing = {(row["window_id"], row["behavior_type"]) for row in db.execute(
                "SELECT window_id,behavior_type FROM patrol_windows WHERE dataset_id=? AND far_budget=?", (dataset_id, far_budget))}
            for record, vessel, start, end in valid:
                db.execute("""INSERT OR IGNORE INTO patrol_windows
                    (dataset_id,window_id,mmsi,behavior_type,far_budget,start_utc,end_utc,start_epoch,end_epoch,score,first_seen_utc,evidence_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (dataset_id, str(record.get("window_id") or record.get("id")), vessel, record.get("type", "unknown"),
                     far_budget, record["start_utc"], record["end_utc"], start, end, float(record.get("score") or 0), stamp,
                     json.dumps(record, ensure_ascii=False, allow_nan=False)))
            rows = list(db.execute("SELECT * FROM patrol_windows WHERE dataset_id=? AND far_budget=? ORDER BY mmsi,behavior_type,start_epoch,end_epoch", (dataset_id, far_budget)))
        events, current = [], None
        for row in rows:
            if current is None or current["mmsi"] != row["mmsi"] or current["type"] != row["behavior_type"] or row["start_epoch"] > current["_end_epoch"] + 120:
                current = {"mmsi": row["mmsi"], "type": row["behavior_type"], "start_utc": row["start_utc"],
                           "end_utc": row["end_utc"], "window_ids": [], "max_score": row["score"],
                           "continuity_gap_seconds": 120, "_end_epoch": row["end_epoch"]}
                current["event_id"] = "EV-" + hashlib.sha256(f"{dataset_id}|{row['mmsi']}|{row['behavior_type']}|{far_budget}|{row['window_id']}".encode()).hexdigest()[:16]
                events.append(current)
            current["window_ids"].append(row["window_id"])
            if row["end_epoch"] > current["_end_epoch"]:
                current["end_utc"] = row["end_utc"]
                current["_end_epoch"] = row["end_epoch"]
            current["max_score"] = max(current["max_score"], row["score"])
        by_window = {(event["mmsi"], event["type"], window): event for event in events for window in event["window_ids"]}
        representatives = {}
        selected_events = {}
        for record, vessel, _, _ in valid:
            window = str(record.get("window_id") or record.get("id"))
            source = by_window[(vessel, record.get("type", "unknown"), window)]
            timeline = {key: value for key, value in source.items() if not key.startswith("_")}
            timeline.update(window_count=len(timeline["window_ids"]), source_kind="observed_patrol",
                            repeat_window=(window, record.get("type", "unknown")) in existing,
                            continuing_event=any((prior, record.get("type", "unknown")) in existing for prior in timeline["window_ids"]))
            event_id = timeline["event_id"]
            annotated = {**record, "event_timeline": timeline}
            if event_id not in representatives or float(record.get("score") or 0) > float(representatives[event_id].get("score") or 0):
                representatives[event_id] = annotated
            selected_events[event_id] = timeline
        result = list(representatives.values()) + others
        result.sort(key=lambda record: float(record.get("score") or 0), reverse=True)
        return result, list(selected_events.values())

    def patrol_evidence(self, dataset_id: str, far_budget: float, timeline: dict) -> list[dict]:
        windows = timeline.get("window_ids", [])
        if not windows:
            return []
        with self._connect() as db:
            rows = db.execute("SELECT window_id,evidence_json FROM patrol_windows WHERE dataset_id=? AND far_budget=? AND mmsi=? AND behavior_type=? ORDER BY start_epoch",
                              (dataset_id, far_budget, timeline.get("mmsi", ""), timeline.get("type", "")))
            allowed = set(windows)
            return [json.loads(row["evidence_json"]) for row in rows if row["window_id"] in allowed]
