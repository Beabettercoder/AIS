from __future__ import annotations

import json
import math
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import test_repository
from maritime_agent import api
from maritime_agent.config import Settings
from maritime_agent.data_intake_agent import DataIntakeAgent
from maritime_agent.database import SCHEMA_VERSION, import_dataset
from maritime_agent.gis_sources import ensure_noaa_channels, parse_geojson
from maritime_agent.sqlite_repository import SQLiteRepository


GEOJSON = json.dumps({"type": "FeatureCollection", "features": [{
    "type": "Feature", "id": "channel-1", "properties": {"name": "Test Channel"},
    "geometry": {"type": "Polygon", "coordinates": [[[-95.2, 29.5], [-95.1, 29.5],
                                                     [-95.1, 29.6], [-95.2, 29.5]]]}
}]}).encode()


class DataDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_repository.RepositoryTests()
        self.fixture.setUp()
        self.root = self.fixture.repo.root
        self.db = Path(self.fixture.tmp.name) / "discovery.sqlite3"
        import_dataset(self.root, self.db, "sample", "houston", "Sample DS")
        self.agent = DataIntakeAgent(self.db, Path(self.fixture.tmp.name) / "staging-discovery")
        self.repo = SQLiteRepository(self.db, "sample")

    def tearDown(self):
        self.fixture.tearDown()

    def test_gis_upload_discovery_and_feature_api(self):
        preview = self.agent.preview(GEOJSON, "gis_geojson", "Houston", "channel.geojson",
                                     use_as_region_boundary=True)
        self.assertEqual(preview["row_count"], 1)
        result = self.agent.commit(preview["job_id"])
        self.assertEqual(result["region_validation"]["status"], "active")
        self.assertEqual(self.agent.regions()["regions"][0]["status"], "active")
        self.assertEqual(self.agent.commit(preview["job_id"])["status"], "already_imported")
        sources = self.repo.search_ingested_data(kind="gis_geojson")
        self.assertEqual(sources["sources"][0]["source_id"], result["source_id"])
        found = self.repo.search_external_gis("Test Channel", result["source_id"])
        self.assertEqual(found["match_count"], 1)
        obj = found["objects"][0]
        self.assertEqual(self.repo.get_external_gis_feature(
            obj["external_source_id"], obj["external_feature_id"])["geometry"]["type"], "Polygon")
        plan = self.repo.discover_complementary_data(result["source_id"],
                                                      ["gis", "ais", "remote_sensing", "weather"], "2024-07-08")
        self.assertEqual([item["status"] for item in plan["candidates"]],
                         ["provided", "catalog_available", "catalog_available", "no_connector"])
        self.assertEqual(self.repo.discover_complementary_data("sample", ["gis"])["candidates"][0]["status"],
                         "connector_available")

        settings = Settings(self.root, "unused", "", None, 6, "sqlite", self.db, "sample")
        api.repository.cache_clear()
        with patch.object(Settings, "from_env", return_value=settings):
            client = TestClient(api.app)
            response = client.get("/data/discover", params={"source_id": result["source_id"],
                                                           "desired_kinds": "ais,gis", "date_utc": "2024-07-08"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(client.get("/gis/external/search", params={"query": "Test Channel"}).status_code, 200)
            self.assertEqual(client.get(f"/gis/external/{obj['external_source_id']}/{obj['external_feature_id']}").status_code, 200)

        impact = self.agent.source_delete_impact(result["source_id"])
        self.assertEqual(impact["counts"]["gis_features"], 1)
        self.agent.delete_source(result["source_id"])
        self.assertEqual(self.repo.search_external_gis("Test Channel", result["source_id"])["match_count"], 0)

    def test_noaa_cache_is_bounded_and_idempotent(self):
        bbox = [-95.3, 29.4, -95.0, 29.7]
        with patch("maritime_agent.gis_sources.fetch_noaa_channels", return_value=GEOJSON) as fetch:
            first = ensure_noaa_channels(self.db, bbox, "Houston")
            second = ensure_noaa_channels(self.db, bbox, "Houston")
        self.assertEqual(first["status"], "imported")
        self.assertEqual(second["status"], "cached")
        fetch.assert_called_once()
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM external_gis_features").fetchone()[0], 1)

    def test_bad_geojson_and_unregistered_source_are_rejected(self):
        bad = json.dumps({"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [-200, 30]}}
        ]}).encode()
        with self.assertRaises(ValueError):
            parse_geojson(bad)
        with self.assertRaises(ValueError):
            self.repo.discover_complementary_data("missing", ["ais"])

    def test_gis_v2_aligns_properties_categories_units_and_preserves_raw(self):
        data = json.dumps({"type": "FeatureCollection", "features": [{
            "type": "Feature", "id": "anchor-1",
            "properties": {"OBJNAM": "Test Anchorage", "category": "锚地",
                           "SORDAT": "20240818", "depth_ft": 32.8084,
                           "area_km2": 0.5, "vendor_flag": "keep-me"},
            "geometry": {"type": "Polygon", "coordinates": [[[-95.2, 29.5], [-95.1, 29.5],
                                                                 [-95.1, 29.6], [-95.2, 29.5]]]}
        }]}).encode()
        preview = self.agent.preview(data, "gis_geojson", "Houston", "anchorage.geojson")
        self.assertEqual(preview["mapping"]["object_name"], "OBJNAM")
        self.assertEqual(preview["sample"][0]["object_type"], "anchorage")
        self.assertEqual(preview["schema_version"], "gis_schema_v2")
        self.assertIn("vendor_flag", preview["extra_columns"])
        self.assertTrue(any(item["field"] == "depth_m" for item in preview["normalizations"]))
        result = self.agent.commit(preview["job_id"])
        with closing(sqlite3.connect(self.db)) as conn:
            row = conn.execute("SELECT object_type,source_date,properties_json FROM external_gis_features "
                               "WHERE source_id=?", (result["source_id"],)).fetchone()
        props = json.loads(row[2])
        self.assertEqual(row[:2], ("anchorage", "2024-08-18"))
        self.assertAlmostEqual(props["depth_m"], 10.0, places=3)
        self.assertEqual(props["area_m2"], 500000.0)
        self.assertEqual(props["vendor_flag"], "keep-me")
        self.assertEqual(props["_raw_properties"]["depth_ft"], 32.8084)

    def test_gis_v2_converts_web_mercator_to_wgs84(self):
        lon, lat = -95.1, 29.5
        x = lon * 20037508.342789244 / 180.0
        y = 6378137.0 * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        data = json.dumps({"type": "FeatureCollection",
                           "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
                           "features": [{"type": "Feature", "properties": {"name": "Berth"},
                                         "geometry": {"type": "Point", "coordinates": [x, y]}}]}).encode()
        preview = self.agent.preview(data, "gis_geojson", "Houston", "mercator.geojson")
        self.assertAlmostEqual(preview["bbox_wgs84"][0], lon, places=5)
        self.assertAlmostEqual(preview["bbox_wgs84"][1], lat, places=5)
        self.assertEqual(preview["quality_report"]["source_crs"], "EPSG:3857")
        self.assertEqual(preview["normalizations"][0]["target_crs"], "EPSG:4326")

    def test_grouped_model_maps_unknown_gis_property_with_confidence_gate(self):
        class MappingModel:
            def invoke(self, messages):
                class Response:
                    content = ('{"mappings":['
                               '{"source_column":"depth_ft_custom","target_field":"depth_m",'
                               '"model_score":0.99}]}')
                return Response()

        agent = DataIntakeAgent(self.db, Path(self.fixture.tmp.name) / "staging-gis-model", MappingModel())
        data = json.dumps({"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {"name": "Test", "depth_ft_custom": 32.8084},
            "geometry": {"type": "Point", "coordinates": [-95.1, 29.5]}
        }]}).encode()
        preview = agent.preview(data, "gis_geojson", "Houston", "smart.geojson")
        self.assertEqual(preview["mapping_source"], "grouped_model_suggestion")
        self.assertEqual(preview["mapping"]["depth_m"], "depth_ft_custom")
        proposal = next(item for item in preview["mapping_proposals"]
                        if item["source_column"] == "depth_ft_custom")
        self.assertEqual(proposal["field_group"], "physical_dimensions")
        self.assertEqual(proposal["validation"]["decision"], "auto_map")
        result = agent.commit(preview["job_id"])
        with closing(sqlite3.connect(self.db)) as conn:
            props = json.loads(conn.execute(
                "SELECT properties_json FROM external_gis_features WHERE source_id=?",
                (result["source_id"],)).fetchone()[0])
        self.assertAlmostEqual(props["depth_m"], 10.0, places=3)


if __name__ == "__main__":
    unittest.main()
