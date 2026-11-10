from __future__ import annotations

import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import test_repository
from maritime_agent import api
from maritime_agent.config import Settings
from maritime_agent.database import DB_TABLES, SCHEMA_VERSION, import_dataset, migrate_database, open_readonly
from maritime_agent.repository import TABLES
from maritime_agent.sqlite_repository import SQLiteRepository
from maritime_agent.storage import open_repository


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()
        self.root = self.fixture.repo.root
        self.db = Path(self.fixture.tmp.name) / "storage" / "test.sqlite3"

    def tearDown(self):
        self.fixture.tearDown()

    def test_import_is_idempotent_and_rows_match(self):
        first = import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        self.assertEqual(first["status"], "imported")
        self.assertEqual(sum(first["row_counts"].values()), 9)
        second = import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        self.assertEqual(second["status"], "already_imported")
        sql_repo = SQLiteRepository(self.db, "sample")
        for table in DB_TABLES:
            self.assertEqual(sql_repo.rows(table), self.fixture.repo.rows(table), table)
        args = ("port", "unique_mmsi", "2024-07-06T00:00:00Z", "2024-07-07T00:00:00Z")
        self.assertEqual(sql_repo.aggregate(*args), self.fixture.repo.aggregate(*args))
        self.assertEqual(sql_repo.aggregate("zone", "unique_mmsi", "2024-07-06T00:00:00Z",
                                            "2024-07-07T00:00:00Z", selector="bayport"),
                         self.fixture.repo.aggregate("zone", "unique_mmsi", "2024-07-06T00:00:00Z",
                                                     "2024-07-07T00:00:00Z", selector="bayport"))
        self.assertEqual(sql_repo.event_evidence("EVT_001"), self.fixture.repo.event_evidence("EVT_001"))

    def test_explicit_migration_is_idempotent(self):
        import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        result = migrate_database(self.db)
        self.assertEqual(result, {"before": SCHEMA_VERSION, "after": SCHEMA_VERSION})
        settings = Settings(self.root, "unused", "", None, 6, "sqlite", self.db, "sample")
        self.assertIsInstance(open_repository(settings), SQLiteRepository)
        with closing(open_readonly(self.db)) as conn:
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertFalse(conn.execute("PRAGMA foreign_key_check").fetchall())
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM datasets")

    def test_duplicate_key_rolls_back_dataset(self):
        from test_repository import write_table
        rows = list(self.fixture.repo.rows("port"))
        write_table(self.root, "port", rows + [rows[0]])
        with self.assertRaises(ValueError):
            import_dataset(self.root, self.db, "bad", "houston", "Bad DS")
        with closing(open_readonly(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0], 0)

    def test_existing_dataset_cannot_be_overwritten(self):
        import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        with self.assertRaises(ValueError):
            import_dataset(self.root, self.db, "sample", "houston", "Changed label")

    def test_runtime_uses_database_without_source_csv(self):
        import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        for relative_path in TABLES.values():
            (self.root / relative_path).unlink()
        settings = Settings(None, "unused", "", None, 6, "sqlite", self.db, "sample")
        repo = open_repository(settings)
        self.assertEqual(repo.overview()["port_hour_rows"], 3)
        self.assertEqual(repo.aggregate("port", "unique_mmsi", "2024-07-06T00:00:00Z",
                                        "2024-07-07T00:00:00Z")["value"], 90)
        with closing(sqlite3.connect(self.db)) as conn:
            with conn:
                conn.execute("UPDATE hourly_metric_values SET value=120 WHERE dataset_id='sample' "
                             "AND table_name='port' AND metric='unique_mmsi' "
                             "AND timestamp_hour_utc='2024-07-06T00:00:00Z'")
        self.assertEqual(repo.aggregate("port", "unique_mmsi", "2024-07-06T00:00:00Z",
                                        "2024-07-07T00:00:00Z")["value"], 100)
        self.assertEqual(repo.event_evidence("EVT_001")["link_count"], 1)
        api.repository.cache_clear()
        with patch.object(Settings, "from_env", return_value=settings):
            response = TestClient(api.app).get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["data_backend"], "sqlite")
            self.assertEqual(response.json()["dataset_id"], "sample")
            self.assertTrue(response.json()["database_ready"])
            self.assertEqual(TestClient(api.app).get("/catalog").json()["port_hour_rows"], 3)
        api.repository.cache_clear()

    def test_sqlite_is_default_runtime_backend(self):
        with patch("maritime_agent.config.load_dotenv"), patch.dict(
            "os.environ", {"MARITIME_DB_PATH": str(self.db)}, clear=True
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.storage_backend, "sqlite")
        self.assertIsNone(settings.data_root)
