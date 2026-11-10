from __future__ import annotations

import json
import gzip
import io
import sqlite3
import unittest
import zipfile
from contextlib import closing
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import test_repository
from maritime_agent import api
from maritime_agent.alignment_rules import canonical_profile, rules_for
from maritime_agent.config import Settings
from maritime_agent.data_intake_agent import DataIntakeAgent
from maritime_agent.database import SCHEMA_VERSION, _initialize, import_dataset
from maritime_agent.event_intake import extract_event_text
from maritime_agent.fusion import index_bound_source
from maritime_agent.schema_matcher import build_ais_candidates, validate_ais_suggestion
from maritime_agent.sqlite_repository import SQLiteRepository


class DataIntakeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()
        self.root = self.fixture.repo.root
        self.db = Path(self.fixture.tmp.name) / "intake.sqlite3"
        import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        self.agent = DataIntakeAgent(self.db, Path(self.fixture.tmp.name) / "staging")

    def tearDown(self):
        self.fixture.tearDown()

    def test_ais_field_harmonization_and_query(self):
        content = ("BaseDateTime,MMSI,LON,LAT,SOG,COG,VesselName\n"
                   "2025-01-01T10:00:00Z,123456789,-95.1,29.5,4.2,120,TEST\n"
                   "2025-01-01T10:05:00Z,123456789,-95.0,29.6,4.3,121,TEST\n").encode()
        preview = self.agent.preview(content, "ais_csv", "Galveston", "points.csv")
        self.assertEqual(preview["mapping"]["timestamp_utc"], "BaseDateTime")
        self.assertEqual(preview["row_count"], 2)
        self.assertEqual(preview["profile"]["observed_bbox_wgs84"], [-95.1, 29.5, -95.0, 29.6])
        result = self.agent.commit(preview["job_id"])
        self.assertEqual(result["status"], "imported")
        self.assertEqual(self.agent.commit(preview["job_id"])["status"], "already_imported")
        found = SQLiteRepository(self.db, "sample").search_ingested_data("2025-01-01", "Galveston", "ais_csv")
        self.assertEqual(found["sources"][0]["matched_observations"], 2)
        self.assertEqual(found["sources"][0]["unique_mmsi"], 1)
        hourly = SQLiteRepository(self.db, "sample").query_data_hourly(
            result["source_id"], "unique_mmsi", "2025-01-01T10:00:00Z", "2025-01-01T12:00:00Z")
        self.assertEqual(hourly["points"], [{"timestamp_hour_utc": "2025-01-01T10:00:00Z", "value": 1}])
        self.assertEqual(hourly["missing_hours"], 1)
        self.assertEqual(hourly["profile"]["granularity"], "ais_observation")
        self.assertEqual(SQLiteRepository(self.db, "sample").query_data_hourly(
            result["source_id"], "ais_message_count", "2025-01-01T10:00:00Z",
            "2025-01-01T11:00:00Z")["points"][0]["value"], 2)
        base = SQLiteRepository(self.db, "sample").query_data_hourly(
            "sample", "unique_mmsi", "2024-07-08T00:00:00Z", "2024-07-08T01:00:00Z")
        self.assertEqual(base["source_kind"], "base_ds_port")
        self.assertEqual(base["metric"], "unique_mmsi")
        self.assertEqual(base["expected_hours"], 1)
        moving = SQLiteRepository(self.db, "sample").query_data_hourly(
            result["source_id"], "moving_mmsi", "2025-01-01T10:00:00Z", "2025-01-01T11:00:00Z")
        self.assertEqual(moving["points"][0]["value"], 1)

    def test_event_text_preview_commit_evidence_and_incremental_index(self):
        class Response:
            content = json.dumps({"events": [{
                "event_type": "port_status_change", "event_subtype": "closure",
                "title": "Port Houston terminals closed", "summary": "Terminals closed for Beryl.",
                "valid_start_original": "2024-07-08", "valid_end_original": "2024-07-09",
                "published_at_original": "2024-07-07", "location_text": "Port Houston",
                "entities": [{"type": "port", "name": "Port Houston", "identifier": None}],
                "severity": "high", "status": "ended",
                "evidence_quote": "Port Houston terminals will close on July 8 because of Hurricane Beryl.",
                "model_score": 0.91,
            }]})

        class Model:
            def invoke(self, messages):
                return Response()

        agent = DataIntakeAgent(self.db, Path(self.fixture.tmp.name) / "event-staging",
                                Model(), dataset_id="sample")
        text = ("Marine notice. Port Houston terminals will close on July 8 because of "
                "Hurricane Beryl. Operations are expected to resume July 10.").encode()
        preview = agent.preview(text, "event_text", "Houston", "notice.txt",
                                source_url="https://example.test/notice", published_at="2024-07-07")
        self.assertEqual(preview["status"], "pending_review")
        self.assertEqual(preview["row_count"], 1)
        self.assertTrue(preview["event_candidates"][0]["quote_verified"])
        result = agent.commit(preview["job_id"])
        self.assertEqual(result["status"], "imported")
        self.assertEqual(result["inserted_event_count"], 1)
        indexed = index_bound_source(self.db, "sample", result["source_id"])
        self.assertEqual(indexed["counts_by_modality"], {"event": 1})
        repo = SQLiteRepository(self.db, "sample")
        sources = repo.search_ingested_data("", "Houston", "event_text")
        self.assertEqual(sources["sources"][0]["source_id"], result["source_id"])
        event = repo.search_events("Port Houston terminals closed")["events"][0]
        evidence = repo.event_evidence(event["event_id"])
        self.assertEqual(evidence["link_count"], 1)
        self.assertEqual(evidence["evidence_links"][0]["quote_verified"], "yes")
        self.assertEqual(agent.commit(preview["job_id"])["status"], "already_imported")
        impact = agent.source_delete_impact(result["source_id"])
        self.assertEqual(impact["counts"]["events_deleted_without_other_evidence"], 1)
        deleted = agent.delete_source(result["source_id"])
        self.assertEqual(len(deleted["recoverable_files"]), 1)
        self.assertTrue(Path(deleted["recoverable_files"][0]).is_file())
        self.assertEqual(repo.search_events("Port Houston terminals closed")["match_count"], 0)

    def test_event_text_rejects_hallucinated_quote(self):
        class Response:
            content = json.dumps({"events": [{
                "event_type": "navigation_restriction", "title": "Invented closure",
                "evidence_quote": "This sentence is not present in the source.",
            }]})

        class Model:
            def invoke(self, messages):
                return Response()

        agent = DataIntakeAgent(self.db, Path(self.fixture.tmp.name) / "event-staging-bad",
                                Model(), dataset_id="sample")
        preview = agent.preview(b"The channel is open under normal conditions.", "event_text",
                                "Houston", "notice.txt")
        self.assertEqual(preview["status"], "needs_repair")
        self.assertEqual(preview["row_count"], 0)
        self.assertGreater(preview["quality_report"]["summary"]["blockers"], 0)

    def test_event_pdf_text_extraction(self):
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length 61 >>\nstream\nBT /F1 12 Tf 72 720 Td (Port Houston closed on July 8 2024.) Tj ET\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        pdf = bytearray(b"%PDF-1.4\n")
        offsets = []
        for number, obj in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
        xref = len(pdf)
        pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
        for offset in offsets:
            pdf.extend(f"{offset:010d} 00000 n \n".encode())
        pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        text, media_type, method = extract_event_text(bytes(pdf), "notice.pdf")
        self.assertIn("Port Houston closed", text)
        self.assertEqual(media_type, "application/pdf")
        self.assertEqual(method, "pypdf_text_v1")

    def test_region_registry_groups_sources_and_blocks_spatial_mismatch(self):
        first = ("timestamp,mmsi,lon,lat\n"
                 "2025-01-01T10:00:00Z,123456789,-95.2,29.4\n"
                 "2025-01-01T10:00:30Z,123456789,-94.9,29.8\n").encode()
        preview = self.agent.preview(first, "ais_csv", "Houston-Galveston", "first.csv")
        imported = self.agent.commit(preview["job_id"])
        region_id = imported["region_id"]
        self.assertTrue(region_id.startswith("region-"))

        second = ("timestamp,mmsi,lon,lat\n"
                  "2025-01-01T10:01:00Z,987654321,-95.05,29.55\n").encode()
        second_preview = self.agent.preview(
            second, "ais_csv", "任意别名", "second.csv", region_id=region_id)
        second_result = self.agent.commit(second_preview["job_id"])
        self.assertEqual(second_result["region_id"], region_id)
        self.assertEqual(second_result["region_label"], "Houston-Galveston")

        regions = self.agent.regions()["regions"]
        selected = next(item for item in regions if item["region_id"] == region_id)
        self.assertEqual(selected["source_count"], 2)

        far = ("timestamp,mmsi,lon,lat\n"
               "2025-01-01T10:02:00Z,111222333,121.5,31.2\n").encode()
        with self.assertRaisesRegex(ValueError, "不相交"):
            self.agent.preview(far, "ais_csv", "Houston-Galveston", "far.csv",
                               region_id=region_id)

    def test_remote_geotiff_preview_commit_and_deduplicate(self):
        import numpy as np
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds

        array = np.arange(400, dtype="uint16").reshape(20, 20)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=20, height=20, count=1,
                             dtype="uint16", crs="EPSG:4326",
                             transform=from_bounds(-95.3, 29.4, -94.9, 29.8, 20, 20)) as raster:
                raster.write(array, 1)
            content = memory.read()
        preview = self.agent.preview(
            content, "remote_geotiff", "Houston", "S1A_TEST_VV.tif",
            observed_at_utc="2025-01-01T10:00:00Z")
        self.assertEqual(preview["row_count"], 1)
        self.assertEqual(preview["remote_raster"]["source_crs"], "EPSG:4326")
        self.assertEqual(preview["remote_raster"]["polarization"], "VV")
        self.assertTrue(preview["remote_raster"]["detector_supported"])
        result = self.agent.commit(preview["job_id"], auto_process=False)
        self.assertEqual(result["status"], "imported")
        self.assertEqual(result["automation"]["status"], "not_requested")
        duplicate = self.agent.commit(preview["job_id"], auto_process=False)
        self.assertEqual(duplicate["status"], "already_imported")

        found = SQLiteRepository(self.db, "sample").search_ingested_data(
            "2025-01-01", "Houston", "remote_geotiff")
        self.assertEqual(found["sources"][0]["scene_id"], result["scene_id"])
        asset = self.root / found["sources"][0]["asset_relative_path"]
        self.assertTrue(asset.is_file())
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM remote_intake_sources").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM remote_sensing_aligned_scenes WHERE scene_id=?",
                                          (result["scene_id"],)).fetchone()[0], 1)
            self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])
        impact = self.agent.source_delete_impact(result["source_id"])
        self.assertEqual(impact["counts"]["remote_bands"], 1)
        deleted = self.agent.delete_source(result["source_id"])
        self.assertEqual(len(deleted["recoverable_files"]), 1)
        self.assertFalse(asset.exists())
        self.assertTrue(Path(deleted["recoverable_files"][0]).is_file())

    def test_remote_geotiff_accepts_explicit_sensor_metadata(self):
        import numpy as np
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds

        array = np.arange(400, dtype="float32").reshape(20, 20)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=20, height=20, count=1,
                             dtype="float32", crs="EPSG:4326",
                             transform=from_bounds(4.65, 58.95, 5.65, 59.65, 20, 20)) as raster:
                raster.write(array, 1)
            content = memory.read()
        preview = self.agent.preview(
            content, "remote_geotiff", "Houston", "vv.tif",
            observed_at_utc="2024-01-05T21:53:05Z",
            platform="Sentinel-1A", sensor="C-SAR", collection="sentinel-1-grd")
        remote = preview["remote_raster"]
        self.assertEqual(remote["platform"], "Sentinel-1A")
        self.assertEqual(remote["sensor"], "C-SAR")
        self.assertEqual(remote["polarization"], "VV")
        self.assertTrue(remote["detector_supported"])

    def test_remote_geotiff_extracts_time_from_filename_timestamp(self):
        import numpy as np
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds

        array = np.arange(400, dtype="uint16").reshape(20, 20)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=20, height=20, count=1,
                             dtype="uint16", crs="EPSG:4326",
                             transform=from_bounds(-95.3, 29.4, -94.9, 29.8, 20, 20)) as raster:
                raster.write(array, 1)
            content = memory.read()
        preview = self.agent.preview(
            content, "remote_geotiff", "Houston",
            "S1A_IW_GRDH_1SDV_20240106T171108_20240106T171133_TEST.tif"
        )
        self.assertEqual(preview["first_utc"], "2024-01-06T17:11:08Z")
        self.assertEqual(preview["remote_raster"]["temporal_source"], "filename_regex")

    def test_remote_geotiff_extracts_time_from_filename_date(self):
        import numpy as np
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds

        array = np.arange(400, dtype="uint16").reshape(20, 20)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=20, height=20, count=1,
                             dtype="uint16", crs="EPSG:4326",
                             transform=from_bounds(-95.3, 29.4, -94.9, 29.8, 20, 20)) as raster:
                raster.write(array, 1)
            content = memory.read()
        preview = self.agent.preview(
            content, "remote_geotiff", "Houston", "vv_2024-01-06.tif"
        )
        self.assertEqual(preview["first_utc"], "2024-01-06T00:00:00Z")
        self.assertEqual(preview["remote_raster"]["temporal_source"], "filename_regex")

    def test_remote_geotiff_extracts_time_from_tiff_tags(self):
        import numpy as np
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds

        array = np.arange(400, dtype="uint16").reshape(20, 20)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=20, height=20, count=1,
                             dtype="uint16", crs="EPSG:4326",
                             transform=from_bounds(-95.3, 29.4, -94.9, 29.8, 20, 20)) as raster:
                raster.write(array, 1)
                raster.update_tags(TIFFTAG_DATETIME="2024:01:06 17:11:08")
            content = memory.read()
        preview = self.agent.preview(
            content, "remote_geotiff", "Houston", "pure_image.tif"
        )
        self.assertEqual(preview["first_utc"], "2024-01-06T17:11:08Z")
        self.assertEqual(preview["remote_raster"]["temporal_source"], "tiff_tag")

    def test_remote_geotiff_falls_back_when_no_time_without_error(self):
        import numpy as np
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds

        array = np.arange(400, dtype="uint16").reshape(20, 20)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=20, height=20, count=1,
                             dtype="uint16", crs="EPSG:4326",
                             transform=from_bounds(-95.3, 29.4, -94.9, 29.8, 20, 20)) as raster:
                raster.write(array, 1)
            content = memory.read()
        preview = self.agent.preview(
            content, "remote_geotiff", "Houston", "vv.tif"
        )
        self.assertTrue(preview["first_utc"].endswith("Z"))
        self.assertEqual(preview["remote_raster"]["temporal_source"], "fallback_now")
        issues = preview["quality_report"]["issues"]
        self.assertTrue(any(item["code"] == "observed_time_inferred_fallback" for item in issues))

    def test_remote_geotiff_prefers_user_explicit_time(self):
        import numpy as np
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds

        array = np.arange(400, dtype="uint16").reshape(20, 20)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=20, height=20, count=1,
                             dtype="uint16", crs="EPSG:4326",
                             transform=from_bounds(-95.3, 29.4, -94.9, 29.8, 20, 20)) as raster:
                raster.write(array, 1)
            content = memory.read()
        preview = self.agent.preview(
            content, "remote_geotiff", "Houston",
            "S1A_IW_GRDH_1SDV_20240106T171108.tif",
            observed_at_utc="2025-05-05T12:00:00Z"
        )
        self.assertEqual(preview["first_utc"], "2025-05-05T12:00:00Z")
        self.assertEqual(preview["remote_raster"]["temporal_source"], "user_input")

    def test_xlsx_ais_preserves_unmapped_columns(self):
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "AIS"
        sheet.append(["timestamp", "mmsi", "latitude", "longitude", "sog_kn",
                      "cog_deg", "name", "ship_type", "nav_status", "imo", "destination"])
        sheet.append([datetime(2025, 1, 1, 10, 0), 123456789, 29.5, -95.1, 4.2,
                      120, "TEST", 70, "under_way", 9876543, "Houston"])
        content = io.BytesIO()
        workbook.save(content)

        preview = self.agent.preview(content.getvalue(), "ais_csv", "Houston", "sample.xlsx",
                                     assume_utc=True)
        self.assertEqual(preview["mapping"]["sog_knots"], "sog_kn")
        self.assertEqual(preview["mapping"]["cog_degrees"], "cog_deg")
        self.assertEqual(preview["mapping"]["vessel_name"], "name")
        self.assertEqual(preview["extra_columns"], [])
        result = self.agent.commit(preview["job_id"])
        with closing(sqlite3.connect(self.db)) as conn:
            stored = conn.execute(
                "SELECT mmsi,extra_attributes_json FROM ingested_ais_positions WHERE source_id=?",
                (result["source_id"],),
            ).fetchone()
        self.assertEqual(stored[0], "123456789")
        self.assertEqual(json.loads(stored[1]), {
            "nav_status": "under_way", "imo": "9876543", "destination": "Houston"
        })

    def test_legacy_import_is_backfilled_on_schema_upgrade(self):
        db = Path(self.fixture.tmp.name) / "legacy.sqlite3"
        with closing(sqlite3.connect(db)) as conn:
            for number in range(1, 8):
                name = ("001_init", "002_tracks", "003_gis", "004_ais_messages",
                        "005_remote_sensing", "006_data_intake", "007_remote_ais_quality")[number - 1]
                conn.executescript(files("maritime_agent").joinpath(f"migrations/{name}.sql").read_text())
            conn.execute("PRAGMA user_version=7")
            conn.execute("INSERT INTO ingested_sources VALUES (?,?,?,?,?,?,?,?,?,?)",
                         ("old", "ais_csv", "Other", "old.csv", "sha", 2,
                          "2025-01-01T10:00:00Z", "2025-01-01T10:01:00Z", "{}", "2025-01-02T00:00:00Z"))
            conn.executemany("INSERT INTO ingested_ais_positions VALUES (?,?,?,?,?,?,?,?,?)", [
                ("old", 2, "2025-01-01T10:00:00Z", "123456789", -95, 29, None, None, None),
                ("old", 3, "2025-01-01T10:01:00Z", "123456789", -95, 29, None, None, None)])
            _initialize(conn)
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(conn.execute("SELECT ais_message_count,unique_mmsi FROM ingested_ais_hourly "
                                          "WHERE source_id='old'").fetchone(), (2, 1))
            self.assertEqual(conn.execute("SELECT profile_json FROM ingested_sources WHERE source_id='old'")
                             .fetchone()[0], "{}")

    def test_gzip_batch_builds_metrics_tracks_and_file_manifest(self):
        header = "BaseDateTime,MMSI,LON,LAT,SOG,COG,VesselName,VesselType\n"
        day1 = (header +
                "2024-07-06T10:00:00,123456789,-88.10,30.60,0.0,10,STOPPED,80\n"
                "2024-07-06T10:05:00,987654321,-88.11,30.61,5.0,20,MOVING,70\n").encode()
        day2 = (header +
                "2024-07-07T11:00:00,123456789,-88.12,30.62,2.0,30,STOPPED,80\n").encode()
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("AIS_2024_07_06_mobile.csv.gz", gzip.compress(day1))
            archive.writestr("AIS_2024_07_07_mobile.csv.gz", gzip.compress(day2))
        preview = self.agent.preview(bundle.getvalue(), "ais_csv", "Mobile", "mobile.zip", assume_utc=True)
        self.assertEqual(preview["file_count"], 2)
        self.assertEqual(preview["row_count"], 3)
        self.assertEqual(preview["mapping"]["vessel_type"], "VesselType")
        result = self.agent.commit(preview["job_id"])
        source_id = result["source_id"]
        repo = SQLiteRepository(self.db, "sample")
        verified = repo.verify_ingested_source(source_id)
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(len(verified["files"]), 2)
        moving = repo.query_data_hourly(source_id, "moving_mmsi", "2024-07-06T10:00:00Z", "2024-07-06T11:00:00Z")
        stopped = repo.query_data_hourly(source_id, "stopped_mmsi", "2024-07-06T10:00:00Z", "2024-07-06T11:00:00Z")
        self.assertEqual(moving["points"][0]["value"], 1)
        self.assertEqual(stopped["points"][0]["value"], 1)
        tracks = repo.search_ingested_tracks(source_id, "2024-07-06", limit=10)
        self.assertEqual(tracks["returned"], 2)
        detail = repo.get_ingested_track(source_id, "2024-07-06", "987654321", "message")
        self.assertEqual(detail["observation_count"], 1)
        self.assertEqual(detail["points"][0]["vessel_class"], "cargo")

    def test_heterogeneous_ais_sources_align_to_versioned_schema(self):
        first = ("BaseDateTime,MMSI,LON,LAT,SOG,VesselType\n"
                 "2025-01-01T10:00:00Z,123456789,-95.1000,29.5000,10,Cargo\n").encode()
        second = ("time,ship_id,x,y,speed_kmh,ship_type\n"
                  "2025-01-01T10:00:03Z,123456789,-95.0995,29.5000,18.52,货船\n").encode()
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("provider-a.csv", first)
            archive.writestr("provider-b.csv", second)

        preview = self.agent.preview(bundle.getvalue(), "ais_csv", "Houston", "sources.zip")
        self.assertEqual(preview["schema_version"], "ais_schema_v2")
        self.assertEqual(preview["ruleset_version"], "ais_rules_v2")
        self.assertEqual(preview["file_count"], 2)
        self.assertEqual(len(preview["file_mappings"]), 2)
        self.assertEqual(preview["file_mappings"][0]["timestamp_utc"], "BaseDateTime")
        self.assertEqual(preview["file_mappings"][1]["timestamp_utc"], "time")
        self.assertEqual(preview["near_duplicate_candidates"], 1)
        self.assertTrue(any(item["code"] == "AIS_CROSS_SOURCE_DUPLICATE_CANDIDATE"
                            for item in preview["quality_report"]["issues"]))

        result = self.agent.commit(preview["job_id"])
        self.assertEqual(result["profile"]["schema_version"], "ais_schema_v2")
        with closing(sqlite3.connect(self.db)) as conn:
            rows = conn.execute(
                "SELECT sog_knots,vessel_class FROM ingested_ais_positions WHERE source_id=? "
                "ORDER BY timestamp_utc", (result["source_id"],)).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[1][0], 10.0, places=5)
            self.assertEqual([row[1] for row in rows], ["cargo", "cargo"])
            files_rows = conn.execute(
                "SELECT schema_version,field_mapping_json,normalization_json "
                "FROM ingested_source_files WHERE source_id=? ORDER BY file_index",
                (result["source_id"],)).fetchall()
            self.assertEqual([row[0] for row in files_rows], ["ais_schema_v2", "ais_schema_v2"])
            self.assertEqual(json.loads(files_rows[1][1])["sog_knots"], "speed_kmh")
            self.assertEqual(json.loads(files_rows[1][2])[0]["source_unit"], "km/h")
            metadata = conn.execute(
                "SELECT schema_version,ruleset_version,normalization_json "
                "FROM source_alignment_metadata WHERE source_id=?", (result["source_id"],)).fetchone()
            self.assertEqual(metadata[:2], ("ais_schema_v2", "ais_rules_v2"))
            self.assertEqual(json.loads(metadata[2])["near_duplicate_candidates"], 1)

    def test_alignment_rule_registry_has_common_contract(self):
        self.assertIn("timestamp_utc", rules_for("ais_csv")["canonical_fields"])
        self.assertEqual(len(rules_for("ais_csv")["canonical_fields"]), 36)
        self.assertEqual(len(rules_for("gis_geojson")["canonical_fields"]), 25)
        self.assertEqual(rules_for("gis_geojson")["schema_version"], "gis_schema_v2")
        profile = canonical_profile("weather_csv")
        self.assertEqual(profile["schema_version"], "weather_schema_v2")
        self.assertEqual(len(rules_for("weather_csv")["canonical_fields"]), 20)
        self.assertEqual(profile["canonical_crs"], "EPSG:4326")

    def test_weather_v2_normalizes_units_missing_values_and_preserves_raw(self):
        content = (
            "observation_time,station_id,longitude,latitude,wind_speed_knots,temp_f,"
            "pressure_pa,visibility_km,relative_humidity_pct,vendor_flag\n"
            "2024-08-18T00:00:00Z,ST01,-95.1,29.5,10,68,101300,10,-9999,keep-me\n"
        ).encode()
        preview = self.agent.preview(content, "weather_csv", "Houston", "weather.csv")
        self.assertEqual(preview["schema_version"], "weather_schema_v2")
        self.assertEqual(preview["mapping"]["wind_speed_mps"], "wind_speed_knots")
        self.assertEqual(preview["mapping"]["air_temperature_c"], "temp_f")
        self.assertIn("vendor_flag", preview["extra_columns"])
        self.assertEqual(preview["quality_report"]["summary"]["blockers"], 0)
        self.assertTrue(any(item["field"] == "wind_speed_mps" for item in preview["normalizations"]))
        result = self.agent.commit(preview["job_id"])
        with closing(sqlite3.connect(self.db)) as conn:
            row = conn.execute(
                "SELECT wind_speed_mps,air_pressure_hpa,air_temperature_c,visibility_m,"
                "relative_humidity_pct,properties_json FROM weather_observations WHERE source_id=?",
                (result["source_id"],)).fetchone()
            metadata = conn.execute(
                "SELECT schema_version,ruleset_version,normalization_json FROM source_alignment_metadata "
                "WHERE source_id=?", (result["source_id"],)).fetchone()
        self.assertAlmostEqual(row[0], 5.14444, places=5)
        self.assertAlmostEqual(row[1], 1013.0, places=3)
        self.assertAlmostEqual(row[2], 20.0, places=3)
        self.assertEqual(row[3], 10000.0)
        self.assertIsNone(row[4])
        props = json.loads(row[5])
        self.assertEqual(props["vendor_flag"], "keep-me")
        self.assertEqual(props["_raw_properties"]["wind_speed_knots"], "10")
        self.assertEqual(metadata[:2], ("weather_schema_v2", "weather_rules_v2"))
        self.assertTrue(json.loads(metadata[2])["normalizations"])
        impact = self.agent.source_delete_impact(result["source_id"])
        self.assertEqual(impact["counts"]["weather_observations"], 1)
        self.agent.delete_source(result["source_id"])
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM weather_observations WHERE source_id=?",
                                          (result["source_id"],)).fetchone()[0], 0)

    def test_grouped_model_maps_unknown_weather_metric_with_confidence_gate(self):
        class MappingModel:
            def invoke(self, messages):
                class Response:
                    content = ('{"mappings":['
                               '{"source_column":"wind_velocity_kts_x",'
                               '"target_field":"wind_speed_mps","model_score":0.99}]}')
                return Response()

        agent = DataIntakeAgent(self.db, Path(self.fixture.tmp.name) / "staging-weather-model",
                                MappingModel())
        content = (
            "valid_time_utc,location_id,lon,lat,wind_velocity_kts_x\n"
            "2024-08-18T00:00:00Z,ST01,-95.1,29.5,10\n"
        ).encode()
        preview = agent.preview(content, "weather_csv", "Houston", "smart-weather.csv")
        self.assertEqual(preview["mapping_source"], "grouped_model_suggestion")
        self.assertEqual(preview["mapping"]["wind_speed_mps"], "wind_velocity_kts_x")
        proposal = next(item for item in preview["mapping_proposals"]
                        if item["source_column"] == "wind_velocity_kts_x")
        self.assertEqual(proposal["field_group"], "wind")
        self.assertEqual(proposal["validation"]["decision"], "auto_map")
        result = agent.commit(preview["job_id"])
        with closing(sqlite3.connect(self.db)) as conn:
            speed = conn.execute("SELECT wind_speed_mps FROM weather_observations WHERE source_id=?",
                                 (result["source_id"],)).fetchone()[0]
        self.assertAlmostEqual(speed, 5.14444, places=5)

    def test_weather_forecast_temporal_and_consistency_warnings(self):
        content = (
            "valid_time_utc,issue_time_utc,station_id,lon,lat,wind_speed_mps,wind_gust_mps\n"
            "2024-08-18T00:00:00Z,2024-08-18T01:00:00Z,ST01,-95.1,29.5,10,5\n"
        ).encode()
        preview = self.agent.preview(content, "weather_csv", "Houston", "forecast.csv",
                                     data_mode="forecast")
        codes = {item["code"] for item in preview["quality_report"]["issues"]}
        self.assertIn("WEATHER_ISSUE_AFTER_VALID", codes)
        self.assertIn("WEATHER_GUST_BELOW_MEAN", codes)
        self.assertEqual(preview["quality_report"]["summary"]["blockers"], 0)

    def test_ais_v2_normalizes_common_optional_fields_and_units(self):
        content = (
            "timestamp_source,received_at,MMSI,Longitude,Latitude,SOG,COG,Heading,"
            "ROT,Navigation Status,Position Accuracy,RAIM,Special Maneuver,IMO,Call Sign,"
            "Name,Ship Type,Cargo Type,Type of mobile,length_ft,width_ft,A,B,C,D,"
            "Type of position fixing device,draught_ft,Destination,ETA,DTE,MSG_TYPE,"
            "Repeat Indicator,AIS Version,Receiver,AIS Channel,Data source type,new_vendor_field\n"
            "18/08/2024 00:00:01,2024-08-18T00:00:02Z,123456789,-95.1,29.5,10,120,121,"
            "3.5,Under way using engine,High,true,1,IMO9876543,CALL1,TEST,70,70,Class A,"
            "328.084,65.6168,80,20,8,12,GPS,32.8084,HOUSTON,2024-08-19T12:30:00Z,available,"
            "1,0,2,RX-1,AIS1,Terrestrial,keep-me\n"
        ).encode()
        preview = self.agent.preview(content, "ais_csv", "Houston", "provider.csv", assume_utc=True)
        self.assertEqual(preview["schema_version"], "ais_schema_v2")
        self.assertEqual(preview["extra_columns"], ["new_vendor_field"])
        result = self.agent.commit(preview["job_id"])
        with closing(sqlite3.connect(self.db)) as conn:
            payload = conn.execute(
                "SELECT extra_attributes_json FROM ingested_ais_positions WHERE source_id=?",
                (result["source_id"],),
            ).fetchone()[0]
        attributes = json.loads(payload)
        self.assertEqual(attributes["received_at_utc"], "2024-08-18T00:00:02Z")
        self.assertEqual(attributes["heading_degrees"], 121.0)
        self.assertEqual(attributes["nav_status"], "under_way_using_engine")
        self.assertEqual(attributes["position_accuracy"], 1)
        self.assertEqual(attributes["raim_flag"], 1)
        self.assertEqual(attributes["imo"], "9876543")
        self.assertEqual(attributes["ais_class"], "A")
        self.assertAlmostEqual(attributes["length_m"], 100.0, places=3)
        self.assertAlmostEqual(attributes["width_m"], 20.0, places=3)
        self.assertAlmostEqual(attributes["draught_m"], 10.0, places=3)
        self.assertEqual(attributes["eta_utc"], "2024-08-19T12:30:00Z")
        self.assertEqual(attributes["dte"], 0)
        self.assertEqual(attributes["message_type"], 1)
        self.assertEqual(attributes["new_vendor_field"], "keep-me")

    def test_missing_mapping_can_be_repaired_before_commit(self):
        content = b"when,identifier,x,y\n2025-01-01T10:00:00Z,123456789,-95.1,29.5\n"
        preview = self.agent.preview(content, "ais_csv", "Other Port", "unknown.csv")
        self.assertEqual(preview["status"], "needs_mapping")
        mapping = {"timestamp_utc": "when", "mmsi": "identifier", "lon": "x", "lat": "y"}
        report = self.agent.validate(preview["job_id"], mapping)
        self.assertEqual(report["row_count"], 1)
        self.assertEqual(self.agent.commit(preview["job_id"], mapping)["row_count"], 1)

    def test_model_mapping_is_checked_against_real_columns(self):
        class MappingModel:
            def invoke(self, messages):
                class Response:
                    content = '{"timestamp_utc":"when","mmsi":"identifier","lon":"x","lat":"y","fake":"ignored"}'
                return Response()
        agent = DataIntakeAgent(self.db, Path(self.fixture.tmp.name) / "staging-model", MappingModel())
        preview = agent.preview(b"when,identifier,x,y\n2025-01-01T10:00:00Z,123456789,-95.1,29.5\n",
                                "ais_csv", "Other Port", "unknown.csv")
        self.assertEqual(preview["status"], "pending_review")
        self.assertEqual(preview["mapping_source"], "grouped_model_suggestion")
        self.assertNotIn("fake", preview["mapping"])
        self.assertTrue(preview["mapping_proposals"])
        self.assertEqual(preview["mapping_proposals"][0]["validation"]["confidence_kind"],
                         "uncalibrated_evidence_score")

    def test_schema_candidates_use_names_and_real_values(self):
        aliases = rules_for("ais_csv")["aliases"]
        candidates = build_ais_candidates(
            ["ShipIdentifier", "ObservedAt"],
            [{"ShipIdentifier": "123456789", "ObservedAt": "2025-01-01T10:00:00Z"}],
            aliases,
            field_groups=rules_for("ais_csv")["field_groups"],
        )
        self.assertEqual(candidates["ShipIdentifier"][0]["target_field"], "mmsi")
        self.assertEqual(candidates["ShipIdentifier"][0]["value_score"], 1.0)
        self.assertEqual(candidates["ObservedAt"][0]["target_field"], "timestamp_utc")
        self.assertEqual(candidates["ObservedAt"][0]["field_group"], "time_position_identity")

    def test_grouped_model_maps_unknown_optional_field_and_keeps_low_confidence_unknown(self):
        class MappingModel:
            def invoke(self, messages):
                class Response:
                    content = ('{"mappings":['
                               '{"source_column":"hdg_true_value","target_field":"heading_degrees","model_score":0.99},'
                               '{"source_column":"arrival_plan_x","target_field":"eta_utc","model_score":0.1}]}')
                return Response()

        agent = DataIntakeAgent(self.db, Path(self.fixture.tmp.name) / "staging-grouped", MappingModel())
        content = ("timestamp_utc,mmsi,lon,lat,hdg_true_value,arrival_plan_x\n"
                   "2025-01-01T10:00:00Z,123456789,-95.1,29.5,121,2025-01-02T12:00:00Z\n").encode()
        preview = agent.preview(content, "ais_csv", "Houston", "grouped.csv")
        self.assertEqual(preview["mapping"]["heading_degrees"], "hdg_true_value")
        self.assertEqual(preview["mapping_source"], "grouped_model_suggestion")
        route = next(item for item in preview["group_routing"]
                     if item["source_column"] == "hdg_true_value")
        self.assertEqual(route["field_group"], "dynamic_navigation")
        self.assertEqual(route["decision"], "auto_map")
        low = next(item for item in preview["mapping_proposals"]
                   if item["source_column"] == "arrival_plan_x")
        self.assertNotEqual(low["validation"]["decision"], "auto_map")
        self.assertIn("arrival_plan_x", preview["extra_columns"])

        result = agent.commit(preview["job_id"])
        with closing(sqlite3.connect(self.db)) as conn:
            payload = conn.execute(
                "SELECT extra_attributes_json FROM ingested_ais_positions WHERE source_id=?",
                (result["source_id"],),
            ).fetchone()[0]
        attributes = json.loads(payload)
        self.assertEqual(attributes["heading_degrees"], 121.0)
        self.assertEqual(attributes["arrival_plan_x"], "2025-01-02T12:00:00Z")

    def test_schema_validation_blocks_bad_mmsi_and_reused_column(self):
        candidate = {"name_score": 0.9, "unit_score": 1.0}
        invalid = validate_ais_suggestion(
            "ShipIdentifier", "mmsi", ["ABC", "123"], candidate, 0.99,
            {"mmsi": "ShipIdentifier"},
        )
        self.assertEqual(invalid["status"], "block")
        self.assertEqual(invalid["raw_confidence"], 0.0)
        conflict = validate_ais_suggestion(
            "identifier", "mmsi", ["123456789"], candidate, 0.99,
            {"mmsi": "identifier", "vessel_name": "identifier"},
        )
        self.assertEqual(conflict["decision"], "blocked")

    def test_model_can_abstain_from_unknown_mapping(self):
        class MappingModel:
            def invoke(self, messages):
                class Response:
                    content = ('{"mappings":['
                               '{"source_column":"when","target_field":"timestamp_utc","model_score":0.9},'
                               '{"source_column":"identifier","target_field":"none_of_above","model_score":0.8}]}')
                return Response()
        agent = DataIntakeAgent(self.db, Path(self.fixture.tmp.name) / "staging-abstain", MappingModel())
        preview = agent.preview(
            b"when,identifier,x,y\n2025-01-01T10:00:00Z,unknown,-95.1,29.5\n",
            "ais_csv", "Other Port", "unknown.csv",
        )
        self.assertEqual(preview["status"], "needs_mapping")
        abstained = next(item for item in preview["mapping_proposals"]
                         if item["source_column"] == "identifier")
        self.assertEqual(abstained["suggested_target"], "none_of_above")
        self.assertEqual(abstained["validation"]["decision"], "needs_mapping")

    def test_bad_coordinates_rejected_before_write(self):
        content = b"timestamp_utc,MMSI,lon,lat\n2025-01-01T10:00:00Z,123456789,-195,29.5\n"
        with self.assertRaises(ValueError):
            self.agent.preview(content, "ais_csv", "Port", "bad.csv")
        self.assertEqual(SQLiteRepository(self.db, "sample").overview()["additional_source_count"], 0)

    def test_delete_imported_source_cascades_and_protects_base_dataset(self):
        content = ("BaseDateTime,MMSI,LON,LAT,SOG\n"
                   "2025-01-01T10:00:00Z,123456789,-95.1,29.5,2.0\n").encode()
        preview = self.agent.preview(content, "ais_csv", "Disposable", "points.csv")
        result = self.agent.commit(preview["job_id"])
        impact = self.agent.source_delete_impact(result["source_id"])
        self.assertEqual(impact["counts"]["ais_positions"], 1)
        self.assertEqual(impact["confirmation_required"], result["source_id"])
        deleted = self.agent.delete_source(result["source_id"])
        self.assertEqual(deleted["status"], "deleted")
        self.assertEqual(deleted["deleted_records"]["ais_positions"], 1)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ingested_sources WHERE source_id=?",
                                          (result["source_id"],)).fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ingested_ais_positions WHERE source_id=?",
                                          (result["source_id"],)).fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ingested_source_files WHERE source_id=?",
                                          (result["source_id"],)).fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM source_alignment_metadata WHERE source_id=?",
                                          (result["source_id"],)).fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM data_management_audit WHERE target_id=?",
                                          (result["source_id"],)).fetchone()[0], 1)
        with self.assertRaises(ValueError):
            self.agent.delete_source("sample")

    def test_data_management_page_requires_exact_source_confirmation(self):
        content = ("BaseDateTime,MMSI,LON,LAT,SOG\n"
                   "2025-01-01T10:00:00Z,123456789,-95.1,29.5,2.0\n").encode()
        result = self.agent.commit(self.agent.preview(
            content, "ais_csv", "Disposable", "managed.csv")["job_id"])
        settings = Settings(self.root, "unused", "", None, 6, "sqlite", self.db, "sample")
        api.intake_agent.cache_clear()
        api.repository.cache_clear()
        with patch.object(Settings, "from_env", return_value=settings):
            client = TestClient(api.app)
            self.assertEqual(client.get("/data-management").status_code, 200)
            management_page = client.get("/data-management").text
            self.assertIn('id="stat-source-count"', management_page)
            self.assertIn('class="source-grid"', management_page)
            home = client.get("/")
            self.assertIn("选择研究区域数据集", home.text)
            home_script = client.get("/assets/ui.js").text
            self.assertIn('fetch("/regions")', home_script)
            self.assertIn("region_id: selectedRegionId", home_script)
            self.assertNotIn('/data-sources?kind=ais_csv', home_script)
            impact = client.get(f"/data-sources/{result['source_id']}/delete-impact")
            self.assertEqual(impact.status_code, 200)
            self.assertEqual(client.delete(f"/data-sources/{result['source_id']}").status_code, 409)
            deleted = client.delete(
                f"/data-sources/{result['source_id']}",
                params={"confirmation": result["source_id"]})
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(deleted.json()["status"], "deleted")

    def test_naive_us_ais_time_requires_explicit_utc_choice(self):
        content = b"BaseDateTime,MMSI,LON,LAT\n7/9/2024 17:15:00,123456789,-95.1,29.5\n"
        with self.assertRaises(ValueError):
            self.agent.preview(content, "ais_csv", "Houston", "us.csv")
        report = self.agent.preview(content, "ais_csv", "Houston", "us.csv", assume_utc=True)
        self.assertEqual(report["first_utc"], "2024-07-09T17:15:00Z")

    def test_timestamp_source_and_dmy_time_are_inferred(self):
        content = ("timestamp_source,mmsi,longitude,latitude\n"
                   "18/08/2024 00:00:01,123456789,-95.1,29.5\n"
                   "19/08/2024 00:00:02,123456789,-95.0,29.6\n").encode()
        report = self.agent.preview(content, "ais_csv", "Houston", "provider.csv",
                                    assume_utc=True)
        self.assertEqual(report["mapping"]["timestamp_utc"], "timestamp_source")
        self.assertEqual(report["first_utc"], "2024-08-18T00:00:01Z")
        timestamp_rule = next(item for item in report["files"][0]["normalizations"]
                              if item["field"] == "timestamp_utc")
        self.assertEqual(timestamp_rule["date_order"], "DMY")
        self.assertEqual(timestamp_rule["date_order_status"], "inferred")

    def test_stac_catalog_and_http_flow(self):
        item = {"type": "Feature", "id": "sat-1", "collection": "sentinel-2-c1-l2a",
                "properties": {"datetime": "2025-01-02T12:00:00Z", "eo:cloud_cover": 42},
                "bbox": [-96, 29, -94, 31],
                "links": [{"rel": "self", "href": "https://example.com/stac/sat-1"}]}
        content = json.dumps(item).encode()
        settings = Settings(self.root, "unused", "", None, 6, "sqlite", self.db, "sample")
        api.intake_agent.cache_clear()
        api.repository.cache_clear()
        with patch.object(Settings, "from_env", return_value=settings):
            client = TestClient(api.app)
            page = client.get("/data-intake")
            self.assertEqual(page.status_code, 200)
            self.assertIn('id="file" type="file" multiple', page.text)
            self.assertNotIn('id="mapping-fields"', page.text)
            self.assertNotIn("按当前映射重新校验", page.text)
            script = client.get("/assets/data-intake.js").text
            self.assertNotIn("currentMapping", script)
            self.assertIn("batchItems", script)
            self.assertIn("确认入库（", script)
            preview = client.post("/data-intake/preview?kind=stac_json&region_label=NewPort&source_name=scene.json",
                                  content=content).json()
            self.assertEqual(preview["row_count"], 1)
            result = client.post(f"/data-intake/{preview['job_id']}/commit", json={}).json()
            self.assertEqual(result["status"], "imported")
            found = client.get("/data-sources?date_utc=2025-01-02&region=NewPort").json()
            self.assertEqual(found["sources"][0]["scenes"][0]["scene_id"], "sat-1")
            self.assertEqual(found["sources"][0]["profile"]["granularity"], "satellite_scene_metadata")
            self.assertEqual(client.get("/data/metrics/hourly", params={"source_id": result["source_id"],
                             "metric": "unique_mmsi", "start_utc": "2025-01-02T12:00:00Z",
                             "end_utc": "2025-01-02T13:00:00Z"}).status_code, 400)
            blocked = client.post("/data-intake/preview?kind=stac_json&region_label=X&source_name=x.json",
                                  content=content, headers={"Origin": "https://evil.example"})
            self.assertEqual(blocked.status_code, 403)

    def test_ais_csv_gbk_encoding_intake(self):
        csv_text = "定位时间,九位码,经度,纬度,航速节,航向角\n2025-01-01 10:00:00,123456789,-95.1,29.5,10.5,180.0\n"
        gbk_bytes = csv_text.encode("gb18030")
        preview = self.agent.preview(
            gbk_bytes, "ais_csv", "Houston", "chinese_ais.csv", assume_utc=True
        )
        self.assertEqual(preview["row_count"], 1)
        self.assertEqual(preview["status"], "pending_review")

    def test_weather_csv_gbk_encoding_intake(self):
        csv_text = "时间,站点编号,经度,纬度,风速,气温\n2025-01-01 10:00:00,ST_01,-95.1,29.5,5.2,22.4\n"
        gbk_bytes = csv_text.encode("gb18030")
        preview = self.agent.preview(
            gbk_bytes, "weather_csv", "Houston", "chinese_weather.csv", assume_utc=True
        )
        self.assertEqual(preview["row_count"], 1)
        self.assertEqual(preview["status"], "pending_review")

    def test_gis_geojson_single_feature_auto_promoted(self):
        geojson_obj = {
            "type": "Feature",
            "properties": {"name": "Test Zone", "category": "port"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-95.3, 29.4], [-94.9, 29.4], [-94.9, 29.8], [-95.3, 29.8], [-95.3, 29.4]]]
            }
        }
        data = json.dumps(geojson_obj).encode("utf-8")
        preview = self.agent.preview(data, "gis_geojson", "Houston", "single_feature.geojson")
        self.assertEqual(preview["row_count"], 1)
        self.assertEqual(preview["status"], "pending_review")

    def test_gis_geojson_geometry_auto_promoted(self):
        geojson_geom = {
            "type": "Polygon",
            "coordinates": [[[-95.3, 29.4], [-94.9, 29.4], [-94.9, 29.8], [-95.3, 29.8], [-95.3, 29.4]]]
        }
        data = json.dumps(geojson_geom).encode("utf-8")
        preview = self.agent.preview(data, "gis_geojson", "Houston", "pure_polygon.geojson")
        self.assertEqual(preview["row_count"], 1)
        self.assertEqual(preview["status"], "pending_review")

    def test_stac_json_with_assume_utc(self):
        stac_obj = {
            "type": "Feature",
            "id": "scene-assume-utc",
            "collection": "test-collection",
            "bbox": [-95.3, 29.4, -94.9, 29.8],
            "properties": {
                "datetime": "2025-01-01 12:00:00"
            }
        }
        data = json.dumps(stac_obj).encode("utf-8")
        preview = self.agent.preview(data, "stac_json", "Houston", "scene.json", assume_utc=True)
        self.assertEqual(preview["row_count"], 1)
        self.assertEqual(preview["first_utc"], "2025-01-01T12:00:00Z")
