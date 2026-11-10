"""Configuration and local .env loading without an extra dependency."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    path = path or Path.cwd() / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


@dataclass(frozen=True)
class Settings:
    data_root: Path | None
    model_name: str
    model_api_key: str
    model_base_url: str | None
    max_tool_rounds: int
    storage_backend: str = "sqlite"
    db_path: Path | None = None
    dataset_id: str = "houston-beryl-2024"
    conversation_db_path: Path = Path("storage/conversations.sqlite3")
    memory_recent_turns: int = 10
    agent_mode: str = "single"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        root = os.getenv("MARITIME_DATA_ROOT", "").strip()
        backend = os.getenv("MARITIME_STORAGE_BACKEND", "sqlite").strip().lower()
        if backend not in {"csv", "sqlite"}:
            raise ValueError("MARITIME_STORAGE_BACKEND 只能是 csv 或 sqlite")
        if backend == "csv" and not root:
            raise ValueError("CSV 模式需要 MARITIME_DATA_ROOT；请填写 DS 数据目录")
        try:
            rounds = int(os.getenv("MAX_TOOL_ROUNDS", "6"))
        except ValueError as exc:
            raise ValueError("MAX_TOOL_ROUNDS 必须是整数") from exc
        if not 1 <= rounds <= 12:
            raise ValueError("MAX_TOOL_ROUNDS 必须在 1 到 12 之间")
        db_path = os.getenv("MARITIME_DB_PATH", "storage/maritime.sqlite3").strip()
        if backend == "sqlite" and not db_path:
            raise ValueError("SQLite 模式需要 MARITIME_DB_PATH")
        dataset_id = os.getenv("MARITIME_DATASET_ID", "houston-beryl-2024").strip()
        if not dataset_id:
            raise ValueError("MARITIME_DATASET_ID 不能为空")
        conversation_db_path = os.getenv("MARITIME_CONVERSATION_DB_PATH", "storage/conversations.sqlite3").strip()
        if not conversation_db_path:
            raise ValueError("MARITIME_CONVERSATION_DB_PATH 不能为空")
        try:
            memory_recent_turns = int(os.getenv("MEMORY_RECENT_TURNS", "10"))
        except ValueError as exc:
            raise ValueError("MEMORY_RECENT_TURNS 必须是整数") from exc
        if not 1 <= memory_recent_turns <= 30:
            raise ValueError("MEMORY_RECENT_TURNS 必须在 1 到 30 之间")
        agent_mode = os.getenv("MARITIME_AGENT_MODE", "single").strip().lower()
        if agent_mode not in {"single", "coordinated"}:
            raise ValueError("MARITIME_AGENT_MODE 只能是 single 或 coordinated")
        return cls(
            data_root=Path(root).expanduser().resolve() if root else None,
            model_name=os.getenv("MODEL_NAME", "gpt-4.1-mini").strip(),
            model_api_key=(os.getenv("MODEL_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip(),
            model_base_url=os.getenv("MODEL_BASE_URL", "").strip() or None,
            max_tool_rounds=rounds,
            storage_backend=backend,
            db_path=Path(db_path).expanduser().resolve() if db_path else None,
            dataset_id=dataset_id,
            conversation_db_path=Path(conversation_db_path).expanduser().resolve(),
            memory_recent_turns=memory_recent_turns,
            agent_mode=agent_mode,
        )
