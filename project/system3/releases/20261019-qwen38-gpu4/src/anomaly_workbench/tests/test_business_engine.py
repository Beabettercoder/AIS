"""Integration of maritime gates with inference and the live record payload."""
import unittest
from unittest.mock import patch
from anomaly_workbench import engine,sqlite_stream as stream
from anomaly_workbench.tests.test_neighbors import point,decoded,BASE


def window(points):
    data=decoded(points)
    with patch.object(stream,'read_points',return_value=data):
        return stream.build_window({'id':'business-fixture','name':'business'},BASE,BASE+1200,neighbor_revision=3)


class BusinessEngineTests(unittest.TestCase):
    def test_stationary_never_enters_model_or_topk(self):
        w=window([point('367359250',i*60,x=(i%3)*2,sog=0,cog=i*17%360) for i in range(20)])
        with patch.object(engine,'_predict',side_effect=AssertionError('stationary must skip inference')):
            result=engine.evaluate_window(w,.05)
        r=result['records'][0]
        self.assertTrue(r['maritime_policy']['stationary']);self.assertFalse(r['is_candidate'])
        self.assertEqual(r['score'],0);self.assertIsNone(r['model_raw_score'])
        self.assertEqual(result['summary']['stationary_excluded_count'],1)

    def test_low_median_and_small_real_movement_blocks_isolated_speed_spike(self):
        points=[point('367574880',i*60,x=i*9,sog=4.1 if i==10 else .1,cog=90) for i in range(20)]
        with patch.object(engine,'_predict',side_effect=AssertionError('stationary must skip inference')):
            r=engine.evaluate_window(window(points),.05)['records'][0]
        self.assertTrue(r['maritime_policy']['stationary']);self.assertFalse(r['is_candidate'])
        self.assertLess(r['maritime_policy']['path_length_m'],300);self.assertEqual(r['score'],0)

    def test_actual_cruise_then_stop_with_sparse_neighbors_is_self_only(self):
        points=[];x=0.;previous=8.
        for i in range(40):
            speed=8. if i<12 else max(0.,8*(24-i)/12)
            if i:x+=(speed+previous)/2*.514444*30
            points.append(point('367359250',i*30,x=x,sog=speed,cog=90));previous=speed
        with patch.object(engine,'_predict',side_effect=AssertionError('sparse priority must not use model')):
            r=engine.evaluate_window(window(points),.05)['records'][0]
        self.assertFalse(r['maritime_policy']['stationary']);self.assertTrue(r['is_candidate'])
        self.assertGreater(r['maritime_policy']['path_length_m'],500)
        self.assertGreater(r['score'],.5);self.assertLessEqual(r['score'],.99)
        self.assertEqual(r['score_source'],'self_kinematics');self.assertIsNone(r['model_raw_score'])
        self.assertEqual(r['maritime_policy']['source_observations'],40)

    def test_constant_low_speed_and_stationary_course_noise_cannot_be_turnbacks(self):
        points=[point('static',i*60,x=i*.1,sog=.2,cog=0 if i%2 else 180) for i in range(20)]
        r=engine.evaluate_window(window(points),.05)['records'][0]
        self.assertFalse(r['is_candidate']);self.assertLess(r['score'],.5)

if __name__=='__main__':unittest.main()
