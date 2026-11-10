"""Run a configured remote-sensing target detector."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_dotenv
from .remote_detection import run_detection
from .fusion import associate_remote_detections


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="运行可复用遥感目标检测任务")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=os.getenv("MARITIME_DATA_ROOT"))
    parser.add_argument("--db", type=Path, default=os.getenv("MARITIME_DB_PATH", "storage/maritime.sqlite3"))
    parser.add_argument("--dataset-id", default=os.getenv("MARITIME_DATASET_ID", "houston-beryl-2024"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--associate", action="store_true",
                        help="检测入库后，按拍摄时刻附近AIS位置增量建立候选关系")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("需要 --data-root 或 MARITIME_DATA_ROOT")
    result = run_detection(args.config, args.data_root, args.db, args.dataset_id, args.dry_run)
    if args.associate and not args.dry_run and result.get("status") == "completed":
        result["association"] = associate_remote_detections(args.db, args.dataset_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
