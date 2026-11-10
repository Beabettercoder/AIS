"""CLI for importing and verifying the DS vessel-hour and daily track layers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_dotenv
from .track_database import import_tracks, verify_tracks


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="事务性导入或核对 DS 船舶小时位置和日航迹")
    parser.add_argument("--data-root", type=Path, default=os.getenv("MARITIME_DATA_ROOT"))
    parser.add_argument("--db", type=Path, default=os.getenv("MARITIME_DB_PATH", "storage/maritime.sqlite3"))
    parser.add_argument("--dataset-id", default=os.getenv("MARITIME_DATASET_ID", "houston-beryl-2024"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("需要 --data-root 或 MARITIME_DATA_ROOT")
    fn = verify_tracks if args.verify else import_tracks
    print(json.dumps(fn(args.data_root, args.db, args.dataset_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
