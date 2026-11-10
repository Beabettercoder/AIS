import unittest
import numpy as np
from test_observed_statistics_v2 import fixture
from build_speed_stress import stress_one,select_members

class Tests(unittest.TestCase):
    def data(self):
        x=fixture();x['cohort_mmsi']=np.array(['200000001','200000002','200000003','200000004']);return x
    def test_same_quartet_operator_and_target_unchanged(self):
        x=self.data();out,p=stress_one(x,'source-a')
        np.testing.assert_array_equal(out['target_sog_kn'],x['target_sog_kn'])
        np.testing.assert_array_equal(out['neighbor_sog_kn'][0],out['neighbor_sog_kn'][2])
        np.testing.assert_array_equal(out['neighbor_sog_kn'][1],out['neighbor_sog_kn'][3])
        self.assertEqual(p['selected_member_count'],1)
        for m in p['members']:
            self.assertTrue(m['applied']);self.assertTrue(all(t>p['anchor_time_s'] for t in m['times_s']));self.assertEqual(len(m['flat_indices']),2);self.assertTrue(3<=m['factor']<=5)
    def test_deterministic_and_no_input_mutation(self):
        x=self.data();before={k:v.copy() for k,v in x.items()};a,ma=stress_one(x,'source-b');b,mb=stress_one(x,'source-b')
        self.assertEqual(ma,mb)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        for k in x:np.testing.assert_array_equal(x[k],before[k])
    def test_member_selection_independent_of_ordered_label_values(self):
        x=self.data();a,p=stress_one(x,'source-c');x['y']=np.array([1,0,0,1]);b,q=stress_one(x,'source-c')
        self.assertEqual(p,q)
        for k in a:np.testing.assert_array_equal(a[k],b[k])

if __name__=='__main__':unittest.main()
