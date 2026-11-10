"""Select one read-only data adapter for API, CLI and evaluations."""

from __future__ import annotations

from .config import Settings
from .repository import DataRepository
from .sqlite_repository import SQLiteRepository


def open_repository(settings: Settings) -> DataRepository:
    if settings.storage_backend == "sqlite":
        if settings.db_path is None:
            raise ValueError("MARITIME_DB_PATH 未配置")
        return SQLiteRepository(settings.db_path, settings.dataset_id)
    if settings.data_root is None:
        raise ValueError("CSV 模式需要 MARITIME_DATA_ROOT")
    return DataRepository(settings.data_root)
