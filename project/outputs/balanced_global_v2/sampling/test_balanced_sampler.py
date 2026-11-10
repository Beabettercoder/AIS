import hashlib,json,unittest
import numpy as np
import pandas as pd
from balanced_sampler import quota,pick_windows,audit

def fixture(region='r',offset=0):
 rows=[];windows=[]
 for m,n in zip(range(200000001,200000007),[1,2,10,25,5,8]):
  for k in range(n):
   t=1700000000+offset+k*1500+np.array([0,60,180,300,420,540,600,660])
   ix=list(range(len(rows),len(rows)+8));rows.extend([{'mmsi':m,'timestamp':int(x)} for x in t])
   h=hashlib.sha256((str(m)+':'+','.join(map(str,t))).encode()).hexdigest()[:24]
   windows.append({'region':region,'mmsi':m,'window_id':h,'n_obs':8,'duration_s':660,'start_utc_s':int(t[0]),'end_utc_s':int(t[-1]),'timestamps':json.dumps(t.tolist()),'row_indices':json.dumps(ix)})
 return pd.DataFrame(rows),pd.DataFrame(windows)

class SamplerContracts(unittest.TestCase):
 def setUp(self):
  self.raw,self.c=fixture()
  self.s={'train':[200000001,200000002,200000003,200000004],'val':[200000005],'test':[200000006]}
  self.shipq=np.array([4,1,1]);self.winq=np.array([28,4,6])
  self.p,self.a=pick_windows(self.c,self.s,self.winq,20261025,'r')
 def test_window_caps_and_time_disjointness(self):
  self.assertTrue(self.p.groupby('mmsi').size().between(1,20).all())
  self.assertEqual(self.p.groupby('split').size().to_dict(),{'train':28,'val':4,'test':6})
  self.assertTrue(audit({'r':self.p},{'r':self.raw},{},self.shipq,self.winq)['all_checks_passed'])
 def test_seed_reproducibility(self):
  other,_=pick_windows(self.c,self.s,self.winq,20261025,'r')
  pd.testing.assert_frame_equal(self.p,other)
 def test_actual_report_mismatch_rejected(self):
  altered=self.raw.copy();index=json.loads(self.p.iloc[0].row_indices)[0];altered.loc[index,'timestamp']+=1
  with self.assertRaises(AssertionError):audit({'r':self.p},{'r':altered},{},self.shipq,self.winq)
 def test_repeated_source_window_rejected(self):
  doubled=self.p.copy();doubled.iloc[1]=doubled.iloc[0]
  with self.assertRaises(AssertionError):audit({'r':doubled},{'r':self.raw},{},self.shipq,self.winq)
 def test_global_identity_cannot_switch_split(self):
  raw2,c2=fixture('s',offset=1000000)
  swapped={**self.s,'val':self.s['test'],'test':self.s['val']}
  # Test group has only5 original windows: use4 each for this collision test.
  q=np.array([28,4,4]);p1,_=pick_windows(self.c,self.s,q,1,'r');p2,_=pick_windows(c2,swapped,q,1,'s')
  with self.assertRaises(AssertionError):audit({'r':p1,'s':p2},{'r':self.raw,'s':raw2},{},self.shipq,q)
 def test_rounded_counts_preserve_total(self):
  for n in range(1,1000):self.assertEqual(int(quota(n).sum()),n)

if __name__=='__main__':unittest.main(verbosity=2)
