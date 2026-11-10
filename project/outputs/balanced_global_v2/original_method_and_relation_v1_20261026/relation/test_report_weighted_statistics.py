import unittest
import numpy as np
from test_observed_statistics_v2 import fixture
from report_weighted_statistics import score_report_weighted,weighted_segment

class Tests(unittest.TestCase):
    def test_clean_constant_quartet(self):
        o=fixture();r=score_report_weighted(o)
        np.testing.assert_allclose([x['relative_report_weighted'] for x in r],[0,.3,.3,0],atol=1e-12)
        for x in r:self.assertEqual(len(x['usable_members']),4)
    def test_zero_weight_segment_falls_back(self):
        m,f,w=weighted_segment(np.array([3.,5.,7.]),np.zeros(3));self.assertEqual(m,5);self.assertTrue(f);self.assertEqual(w,0)
    def test_latest_multiplicity_preserved(self):
        m,f,w=weighted_segment(np.array([10.,10.,20.]),np.array([1.,1.,.1]));self.assertAlmostEqual(m,22/2.1);self.assertFalse(f)
    def test_corrupted_report_downweighted_before_ratio(self):
        o=fixture();o['neighbor_sog_kn'][:,10:12]*=4
        r=score_report_weighted(o)
        self.assertLess(r[0]['relative_report_weighted'],r[0]['original_relative_mean'])
        self.assertLess(r[0]['members'][0]['suffix_weighted_mean_kn'],r[0]['members'][0]['suffix_plain_mean_kn'])
    def test_no_input_mutation(self):
        o=fixture();copy={k:x.copy() for k,x in o.items()};score_report_weighted(o)
        for k in o:np.testing.assert_array_equal(o[k],copy[k])
    def test_oracle_rejected(self):
        o=fixture();o['meta_cohort_factor']=np.array([0,1,0,1])
        with self.assertRaises(ValueError):score_report_weighted(o)

if __name__=='__main__':unittest.main()
