import unittest
import numpy as np
from audit_neighborhood import Source,kin
from build_relation_smoke import slowdown,innovation,build_one

class Check(unittest.TestCase):
 def test_delta_integral_preserves_source_innovation(self):
  t=np.array([0.,71.,145.,220.,310.,400.,500.]);s=np.array([10.,11.,9.,12.,11.,10.,9.]);c=np.array([80.,85.,90.,94.,95.,97.,99.]);p=np.column_stack((t*4+np.arange(len(t))**2,t*.7))
  p2,s2,c2=slowdown(t,p,s,c,146.)
  np.testing.assert_allclose(innovation(t,p2,s2,c2),innovation(t,p,s,c),atol=1e-12)
 def fixture(self):
  src=Source.__new__(Source);t=np.arange(12)*70.;ids=[str(100+i) for i in range(5)];src.ids=np.repeat(ids,12);src.t=np.tile(t,5);src.s=np.ones(60)*10;src.c=np.ones(60)*90
  src.p=np.column_stack((src.t*10*1852/3600,np.repeat(np.arange(5)*300.,12)));src.ship={m:np.arange(i*12,(i+1)*12) for i,m in enumerate(ids)};src.lookup={(m,float(tt)):i for i,(m,tt) in enumerate(zip(src.ids,src.t))}
  src.good=np.ones(60,bool)
  row={'mmsi':'100','source_timestamps':t.tolist(),'anchor_step':5,'anchor_time_s':350.,'window_id':'fixture','base_window_id':'fixture_base','split':'train','region':'synthetic','cohort_mmsi':ids[1:],'cohort_report_time_s':[350.]*4}
  return src,row
 def test_quartet_marginal_balance_and_fixed_membership(self):
  src,row=self.fixture();got,why=build_one(src,row,set(src.ids));self.assertIsNone(why);a,p=got
  np.testing.assert_array_equal(a['y'],[0,1,1,0]);np.testing.assert_array_equal(a['target_pos'][0],a['target_pos'][1]);np.testing.assert_array_equal(a['target_pos'][2],a['target_pos'][3]);np.testing.assert_array_equal(a['neighbor_pos'][0],a['neighbor_pos'][2]);np.testing.assert_array_equal(a['neighbor_pos'][1],a['neighbor_pos'][3])
  self.assertEqual(len(p['fixed_cohort_mmsi']),4);self.assertLess(p['max_innovation_error_mps'],1e-8)
 def test_future_failure_rejects_whole_quartet_not_member(self):
  src,row=self.fixture();src.good[src.ship['101'][8:]]=False
  got,why=build_one(src,row,set(src.ids));self.assertIsNone(got);self.assertEqual(why,'cohort_insufficient_observed_future')

if __name__=='__main__':unittest.main()
