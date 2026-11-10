from __future__ import annotations

import argparse

from .config import Settings
from .situation_cube import backfill_situation_cubes, build_situation_cube


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task-adaptive maritime situation cubes")
    parser.add_argument("--region-id", default="", help="Stable region ID; omit with --all-regions")
    parser.add_argument("--start-utc", default="")
    parser.add_argument("--end-utc", default="")
    parser.add_argument("--task-type", default="general_situation")
    parser.add_argument("--refresh-fusion-index", action="store_true")
    parser.add_argument("--all-regions", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    if settings.db_path is None:
        raise SystemExit("MARITIME_DB_PATH is required")
    if args.all_regions:
        print(backfill_situation_cubes(settings.db_path, settings.dataset_id,
                                       refresh_fusion_index=args.refresh_fusion_index))
        return
    if not args.region_id:
        raise SystemExit("--region-id is required unless --all-regions is used")
    print(build_situation_cube(settings.db_path, settings.dataset_id, args.region_id,
                               start_time_utc=args.start_utc, end_time_utc=args.end_utc,
                               task_type=args.task_type,
                               refresh_fusion_index=args.refresh_fusion_index))


if __name__ == "__main__":
    main()
