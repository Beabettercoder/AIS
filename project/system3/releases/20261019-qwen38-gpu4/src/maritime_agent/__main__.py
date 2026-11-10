"""CLI for local inspection or a single natural-language question."""

from __future__ import annotations

import argparse
import json

from .agent import ask, create_graph, create_model
from .config import Settings
from .database import migrate_database
from .storage import open_repository


def main() -> None:
    parser = argparse.ArgumentParser(description="海事态势调查智能体")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inspect", action="store_true", help="只读检查当前数据库及数据集，无需模型密钥")
    group.add_argument("--migrate", action="store_true", help="将现有 SQLite 数据库迁移到当前结构版本")
    group.add_argument("--ask", metavar="QUESTION", help="提问，需要 MODEL_API_KEY")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.migrate:
        if settings.storage_backend != "sqlite" or settings.db_path is None:
            raise ValueError("数据库迁移只支持 SQLite 模式")
        print(json.dumps(migrate_database(settings.db_path), ensure_ascii=False, indent=2))
        return
    repo = open_repository(settings)
    if args.inspect:
        output = repo.overview()
    else:
        model = create_model(settings)
        if settings.agent_mode == "coordinated":
            from .coordinated_agent import create_coordinated_graph
            graph = create_coordinated_graph(repo, model, settings.max_tool_rounds,
                                             settings.memory_recent_turns)
        else:
            graph = create_graph(repo, model, settings.max_tool_rounds, settings.memory_recent_turns)
        output = ask(graph, args.ask)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
