import unittest
from types import SimpleNamespace
import numpy as np
from build_labeled_features import (target_dict,noise_spec,apply_target_noise,
  augment_background,shared_deceleration,BackgroundIndex,ContextConfig)

def source(identity='200000001',x=0.,y=0.):
 t=np.arange(20,dtype=float)*60;speed=10.;v=speed*1852/3600
 pos=np.column_stack((np.full(20,x),y+t*v))
 return target_dict(t,pos,np.full(20,speed),np.zeros(20),identity)

def scene(ids=('200000002','200000003','200000004','200000005','200000006')):
 parts=[source(m,(-1000,-500,500,1000,1500)[i],(500,-500,1000,-1000,500)[i]) for i,m in enumerate(ids)]
 return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}

def row(kind='teleport',scope='both',y=0):
 return SimpleNamespace(observation_noise=kind!='none',noise_kind=kind,noise_scope=scope,
                        noise_seed=123,group_seed=456,onset_utc_s=400.,y=y)

class LabelFeatureContracts(unittest.TestCase):
 def test_target_noise_plan_is_label_and_protocol_independent(self):
  original=source()
  for kind in ['teleport','speed_spike','drift']:
   a=noise_spec(original,row(kind,'both',0));b=noise_spec(original,row(kind,'both',1))
   self.assertEqual(a,b)
   x,_=apply_target_noise(original,a);y,_=apply_target_noise(original,b)
   for k in original:np.testing.assert_array_equal(x[k],y[k])
 def test_no_neighbors_never_causes_target_fallback(self):
  original=source();idx=BackgroundIndex(original);spec=noise_spec(original,row(scope='neighbors_only'))
  newer,g,n=augment_background(idx,original,spec,row(scope='neighbors_only'),ContextConfig())
  self.assertIs(newer,idx);self.assertFalse(n['effective']);self.assertEqual(n['candidate_mmsi'],[])
  target,md=apply_target_noise(original,spec)
  for k in original:np.testing.assert_array_equal(target[k],original[k])
  self.assertFalse(md['observations_changed'])
 def test_target_noise_effectiveness_is_measured_from_actual_arrays(self):
  original=source()
  for kind in ['teleport','speed_spike','drift']:
   spec=noise_spec(original,row(kind,'target_only'))
   changed,metadata=apply_target_noise(original,spec)
   self.assertTrue(metadata['observations_changed'])
   self.assertTrue(any(not np.array_equal(changed[k],original[k]) for k in ['x_m','y_m','sog_kn','cog_deg']))
 def test_background_pollution_reindexes_and_hits_a_query_latest_report(self):
  original=source();idx=BackgroundIndex(scene());cfg=ContextConfig()
  for kind in ['teleport','speed_spike','drift']:
   spec=noise_spec(original,row(kind));newer,g,n=augment_background(idx,original,spec,row(kind),cfg)
   self.assertIsNot(newer,idx);self.assertNotEqual(newer.fingerprint,idx.fingerprint);self.assertTrue(n['effective'])
   self.assertGreaterEqual(len(n['selected_mmsi']),1)
   for change in n['changes']:
    self.assertIn(change['original_latest_query_report_s'],change['affected_time_s'])
    self.assertLessEqual(max(change['affected_time_s']),original['time_s'][-1])
 def test_strict_background_pollution_never_accesses_other_identities(self):
  original=source();allowed={'200000002','200000003'};full=scene()
  use=np.isin(full['mmsi'],list(allowed));idx=BackgroundIndex({k:v[use] for k,v in full.items()})
  spec=noise_spec(original,row());newer,g,n=augment_background(idx,original,spec,row(),ContextConfig())
  self.assertLessEqual(set(newer.data['mmsi']),allowed)
  self.assertLessEqual(set(n['candidate_mmsi']),allowed)
 def test_shared_group_changes_target_and_neighbors_with_same_factor(self):
  original=source();idx=BackgroundIndex(scene());cfg=ContextConfig();r=row('none','none')
  spec=noise_spec(original,r);newer,g,n=augment_background(idx,original,spec,r,cfg,group=True)
  target,md=shared_deceleration(original,g['onset_s'],g['factor'])
  self.assertEqual(g['duration_s'],300.);self.assertGreater(len(g['controlled_mmsi']),0)
  self.assertAlmostEqual(target['sog_kn'][-1]/original['sog_kn'][-1],g['factor'])
  for identity in g['controlled_mmsi']:
   latest=newer.latest(original['time_s'][-1],original['mmsi'][0],None)
   latest=latest[newer.data['mmsi'][latest]==identity]
   self.assertEqual(len(latest),1);self.assertAlmostEqual(newer.data['sog_kn'][latest[0]]/10,g['factor'])
  self.assertLess(md['innovation_preservation_max_mps'],1e-10)
 def test_inputs_and_cached_index_remain_immutable(self):
  original=source();snapshot={k:v.copy() for k,v in original.items()};idx=BackgroundIndex(scene());fingerprint=idx.fingerprint
  spec=noise_spec(original,row());augment_background(idx,original,spec,row(),ContextConfig(),group=True);apply_target_noise(original,spec)
  for k in original:np.testing.assert_array_equal(original[k],snapshot[k])
  self.assertEqual(idx.fingerprint,fingerprint)

if __name__=='__main__':unittest.main(verbosity=2)
