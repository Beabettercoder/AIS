from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
import numpy as np
import sqlite3
from anomaly_workbench.tests.test_sqlite_pipeline import make_db
from anomaly_workbench import sqlite_stream as stream,engine

class LiveModelTests(unittest.TestCase):
    def test_missing_report_fields_are_not_fabricated_as_position_jumps(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'host.sqlite';start=make_db(path)
            db=sqlite3.connect(path);db.execute("UPDATE ingested_ais_positions SET cog_degrees=NULL WHERE mmsi='219000001'");db.commit();db.close()
            entry=stream.catalog(path,True)['datasets'][0];w=stream.build_window(entry,start,start+1200,path)
            r=engine.evaluate_window(w,.01)['records'][1]
            self.assertIsNone(r['e_kin_mps']);self.assertFalse(r['has_quality_issue']);self.assertFalse(r['is_candidate'])
            self.assertIn('缺少有效报告',r['why_flagged'])
    def test_actual_sqlite_physics_and_model_diagnostics(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'host.sqlite';start=make_db(path);entry=stream.catalog(path,True)['datasets'][0]
            w=stream.build_window(entry,start,start+1200,path);result=engine.evaluate_window(w,.01)
            self.assertEqual(len(result['records']),6);self.assertEqual(result['summary']['quality_count'],1)
            target=result['records'][0]
            self.assertTrue(target['has_quality_issue']);self.assertFalse(target['is_candidate'])
            raw=np.asarray(target['raw_xy']);self.assertAlmostEqual(raw[0,0],0.)
            self.assertEqual(len(target['repaired_latlon']),20)
            diagnostics=engine.diagnose(w,target['id'])
            self.assertEqual(len(diagnostics['pg_probability']),20)
            self.assertTrue(all(0<=v<=1 for v in diagnostics['pg_probability']))
            self.assertGreater(diagnostics['e_kin_series'][10],5)

if __name__=='__main__':unittest.main()
