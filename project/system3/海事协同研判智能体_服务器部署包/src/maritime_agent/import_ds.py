"""Import one standardized DS dataset into SQLite without model/API access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_dotenv
from .database import import_dataset


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="将一批标准 DS CSV 事务性导入 SQLite")
    parser.add_argument("--data-root", type=Path, default=os.getenv("MARITIME_DATA_ROOT"))
    parser.add_argument("--db", type=Path, default=os.getenv("MARITIME_DB_PATH", "storage/maritime.sqlite3"))
    parser.add_argument("--dataset-id", default=os.getenv("MARITIME_DATASET_ID", "houston-beryl-2024"))
    parser.add_argument("--port-id", default="houston")
    parser.add_argument("--label", default="Houston-Beryl DS")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("需要 --data-root 或 MARITIME_DATA_ROOT")
    result = import_dataset(args.data_root, args.db, args.dataset_id, args.port_id, args.label)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
