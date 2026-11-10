"""Build the derived common observation and deterministic relation layers."""

from __future__ import annotations

import argparse
import json

from .config import Settings
from .fusion import build_deterministic_relations, build_open_set_associations, rebuild_fusion_index


def main() -> None:
    parser = argparse.ArgumentParser(description="构建统一时空实体关联底座")
    parser.add_argument("--date-utc", default="", help="可选 YYYY-MM-DD；为空时索引全部动态观测")
    parser.add_argument("--index-only", action="store_true", help="只重建统一节点，不生成关系")
    args = parser.parse_args()
    settings = Settings.from_env()
    if settings.storage_backend != "sqlite" or settings.db_path is None:
        raise ValueError("统一关联底座只支持 SQLite 模式")
    indexed = rebuild_fusion_index(settings.db_path, settings.dataset_id, args.date_utc)
    output = {"index": indexed}
    if not args.index_only:
        output["relations"] = build_deterministic_relations(settings.db_path, settings.dataset_id)
        output["associations"] = build_open_set_associations(settings.db_path, settings.dataset_id)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
