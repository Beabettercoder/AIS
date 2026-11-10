"""Import and verify port GIS objects and matched AIS area-hour statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_dotenv
from .gis_database import import_gis, verify_gis


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="导入或核对港口 GIS 对象与 AIS 空间匹配小时表")
    parser.add_argument("--data-root", type=Path, default=os.getenv("MARITIME_DATA_ROOT"))
    parser.add_argument("--db", type=Path, default=os.getenv("MARITIME_DB_PATH", "storage/maritime.sqlite3"))
    parser.add_argument("--dataset-id", default=os.getenv("MARITIME_DATASET_ID", "houston-beryl-2024"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("需要 --data-root 或 MARITIME_DATA_ROOT")
    fn = verify_gis if args.verify else import_gis
    print(json.dumps(fn(args.data_root, args.db, args.dataset_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
