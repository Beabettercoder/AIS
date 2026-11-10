"""Actual catalog, source isolation, physical windows and future imports."""
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime,timezone
import hashlib,math,sqlite3,unittest
import numpy as np
from anomaly_workbench import sqlite_stream as stream

def make_db(path):
    db=sqlite3.connect(path)
    db.executescript('''CREATE TABLE ingested_sources(source_id TEXT PRIMARY KEY,source_kind TEXT,region_label TEXT,source_name TEXT,row_count INTEGER,first_utc TEXT,last_utc TEXT,created_at_utc TEXT);
    CREATE TABLE source_region_bindings(source_id TEXT,source_kind TEXT,region_id TEXT,dataset_id TEXT);
    CREATE TABLE regions(dataset_id TEXT,region_id TEXT,canonical_name TEXT,bbox_json TEXT);
    CREATE TABLE ingested_ais_positions(source_id TEXT,timestamp_utc TEXT,mmsi TEXT,lon REAL,lat REAL,sog_knots REAL,cog_degrees REAL,extra_attributes_json TEXT);
    CREATE INDEX source_time ON ingested_ais_positions(source_id,timestamp_utc);''')
    db.execute("INSERT INTO ingested_sources VALUES ('s1','ais_csv','danmai','actual.csv',120,'2024-08-16T00:00:00Z','2024-08-16T00:19:00Z','2026-01-01')")
    db.execute("INSERT INTO source_region_bindings VALUES ('s1','ais_csv','r1','host')")
    db.execute("INSERT INTO regions VALUES ('host','r1','danmai','[10.7,55.0,11.5,55.6]')")
    start=datetime(2024,8,16,tzinfo=timezone.utc).timestamp()
    for vessel in range(6):
        for k in range(20):
            lon=11.+(vessel%3)*.01+k*60*5*.514444/(111195*math.cos(math.radians(55.3)))
            lat=55.3+(vessel//3)*.012
            if vessel==0 and k==10:lat+=.1
            db.execute('INSERT INTO ingested_ais_positions VALUES (?,?,?,?,?,?,?,?)',('s1',stream.utc(start+k*60),str(219000000+vessel),lon,lat,5.,90.,'{"heading_degrees":90}'))
    db.commit();db.close();return start

class SQLiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.path=Path(self.tmp.name)/'host.sqlite3';self.start=make_db(self.path)
    def tearDown(self):self.tmp.cleanup()
    def test_catalog_matches_actual_rows_and_geometry(self):
        entry=stream.dataset_metadata(stream.catalog(self.path,True)['datasets'][0]['id'],self.path)
        self.assertEqual(entry['name'],'danmai');self.assertEqual(entry['row_count'],120);self.assertEqual(entry['unique_mmsi'],6)
        self.assertEqual(entry['start_utc'],'2024-08-16T00:00:00Z');self.assertGreater(entry['bbox'][2],11.)
    def test_new_import_appears_without_code_change(self):
        stream.catalog(self.path,True)
        db=sqlite3.connect(self.path);db.execute("INSERT INTO ingested_sources VALUES ('s2','ais_csv','future','future.csv',1,'2025-01-01T00:00:00Z','2025-01-01T00:00:00Z','2026-02-01')")
        db.execute("INSERT INTO ingested_ais_positions VALUES ('s2','2025-01-01T00:00:00Z','123456789',8,54,3,80,'{}')");db.commit();db.close()
        self.assertIn('future',[d['name'] for d in stream.catalog(self.path,True)['datasets']])
    def test_read_only_connection_and_unchanged_source(self):
        before=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with stream.connect(self.path) as db:
            with self.assertRaises(sqlite3.OperationalError):db.execute('DELETE FROM ingested_sources')
        stream.catalog(self.path,True);self.assertEqual(before,hashlib.sha256(self.path.read_bytes()).hexdigest())
    def test_jump_and_leave_one_vessel_out(self):
        entry=stream.catalog(self.path,True)['datasets'][0];w=stream.build_window(entry,self.start,self.start+1200,self.path)
        self.assertEqual(w['arrays']['raw'].shape,(6,20,7));self.assertEqual(w['point_count'],120)
        self.assertLess(w['arrays']['w_self'][0,10],.3);self.assertGreater(w['arrays']['e_kin'][0,10],5)
        self.assertEqual(w['metadata'][0]['mmsi'],'219000000')
        self.assertTrue(all(n['mmsi']!='219000000' for n in w['metadata'][0]['neighbor_tracks']))
    def test_source_and_time_bounds(self):
        entry=stream.catalog(self.path,True)['datasets'][0]
        self.assertEqual(len(stream.read_points(entry,self.start,self.start+60,self.path)['timestamp']),12)
        self.assertIsNone(stream.read_points(entry,self.start+90000,self.start+91200,self.path))

if __name__=='__main__':unittest.main()
