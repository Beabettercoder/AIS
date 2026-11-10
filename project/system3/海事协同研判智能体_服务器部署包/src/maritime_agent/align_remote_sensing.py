"""Generate the canonical internal-alignment manifest for downloaded imagery."""

import argparse
import json
import os
from pathlib import Path

from .config import load_dotenv
from .remote_sensing_alignment import build_internal_alignment


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=os.getenv("MARITIME_DATA_ROOT"))
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("需要 --data-root 或 MARITIME_DATA_ROOT")
    print(json.dumps(build_internal_alignment(args.data_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
