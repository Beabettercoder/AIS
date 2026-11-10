"""Import or verify remote-sensing scenes in the local SQLite database."""

import argparse
import json
import os
from pathlib import Path

from .config import load_dotenv
from .remote_sensing_database import import_remote_sensing, verify_remote_sensing


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=os.getenv("MARITIME_DATA_ROOT"))
    parser.add_argument("--db", type=Path, default=os.getenv("MARITIME_DB_PATH", "storage/maritime.sqlite3"))
    parser.add_argument("--dataset-id", default=os.getenv("MARITIME_DATASET_ID", "houston-beryl-2024"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--refresh-alignment", action="store_true",
                        help="审核新清单后，事务性重建遥感内部对齐派生表")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("需要 --data-root 或 MARITIME_DATA_ROOT")
    if args.verify:
        result = verify_remote_sensing(args.data_root, args.db, args.dataset_id)
    else:
        result = import_remote_sensing(args.data_root, args.db, args.dataset_id,
                                       refresh_alignment=args.refresh_alignment)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
