"""Import an already harmonized weather CSV with v2 alignment provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_intake_agent import _record_alignment_metadata, _source_profile
from .weather_intake import parse_weather, store_weather


def import_weather_file(path: Path, db_path: Path, region_label: str,
                        source_type: str = "official_station_bundle") -> dict:
    source = path.expanduser().resolve()
    data = source.read_bytes()
    rows, report = parse_weather(data, source.name, "observation", source_type=source_type)
    profile = _source_profile("weather_csv", rows, report, region_label, False)
    result = store_weather(db_path.expanduser().resolve(), data, source.name, region_label,
                           "observation", False, None, set(), source_type=source_type)
    _record_alignment_metadata(db_path.expanduser().resolve(), result["source_id"],
                               "weather_csv", profile, report)
    return {"status": result["status"], "source_id": result["source_id"],
            "schema_version": report["schema_version"],
            "ruleset_version": report["ruleset_version"],
            "row_count": report["row_count"], "first_utc": report["first_utc"],
            "last_utc": report["last_utc"], "bbox_wgs84": report["bbox_wgs84"],
            "quality_status": report["quality_report"]["status"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--region", default="Houston-Galveston")
    parser.add_argument("--source-type", default="official_station_bundle")
    args = parser.parse_args()
    print(json.dumps(import_weather_file(args.file, args.db, args.region, args.source_type),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
