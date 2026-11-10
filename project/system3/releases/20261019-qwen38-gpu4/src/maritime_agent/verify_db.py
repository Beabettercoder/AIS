"""Compare every imported row against source CSV and verify SQLite integrity."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import closing
from pathlib import Path

from .config import load_dotenv
from .database import DB_TABLES, _sha256, open_readonly
from .repository import DataRepository, TABLES
from .sqlite_repository import SQLiteRepository


def verify(source_root: Path, db_path: Path, dataset_id: str) -> dict:
    csv_repo = DataRepository(source_root)
    sql_repo = SQLiteRepository(db_path, dataset_id)
    counts = {}
    for table in DB_TABLES:
        source = csv_repo.rows(table)
        imported = sql_repo.rows(table)
        if source != imported:
            raise ValueError(f"CSV 与数据库内容不一致: {table}")
        counts[table] = len(imported)
    with closing(open_readonly(db_path)) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
        metric_count = conn.execute("SELECT COUNT(*) FROM hourly_metric_values WHERE dataset_id=?",
                                    (dataset_id,)).fetchone()[0]
        fingerprints = dict(conn.execute("SELECT table_name, sha256 FROM import_files WHERE dataset_id=?",
                                         (dataset_id,)).fetchall())
    for table in DB_TABLES:
        if fingerprints.get(table) != _sha256(source_root / TABLES[table]):
            raise ValueError(f"CSV 来源文件 SHA-256 与导入记录不一致: {table}")
    if integrity != "ok" or foreign_keys:
        raise ValueError(f"SQLite 完整性检查失败: integrity={integrity}, foreign_keys={foreign_keys[:3]}")
    checks = [("moving_mmsi", "2024-07-06T00:00:00Z", "2024-07-07T00:00:00Z"),
              ("moving_mmsi", "2024-07-08T00:00:00Z", "2024-07-09T00:00:00Z")]
    comparisons = []
    for metric, start, end in checks:
        old = csv_repo.aggregate("port", metric, start, end)
        new = sql_repo.aggregate("port", metric, start, end)
        if old != new:
            raise ValueError(f"CSV 与数据库统计不一致: {metric} {start}")
        comparisons.append({"metric": metric, "start_utc": start, "value": new["value"]})
    return {"status": "verified", "dataset_id": dataset_id, "row_counts": counts,
            "total_source_rows": sum(counts.values()), "metric_values": metric_count,
            "integrity": integrity, "sample_aggregates": comparisons}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="逐行核对 SQLite 与 DS CSV")
    parser.add_argument("--data-root", type=Path, default=os.getenv("MARITIME_DATA_ROOT"))
    parser.add_argument("--db", type=Path, default=os.getenv("MARITIME_DB_PATH", "storage/maritime.sqlite3"))
    parser.add_argument("--dataset-id", default=os.getenv("MARITIME_DATASET_ID", "houston-beryl-2024"))
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("需要 --data-root 或 MARITIME_DATA_ROOT")
    print(json.dumps(verify(args.data_root, args.db, args.dataset_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
