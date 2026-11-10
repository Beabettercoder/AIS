import unittest
import numpy as np
from build_historical_inputs import gate,target_bridge,KNOT

def fixture():
    m=np.array([[1,1,1,0,0]],bool);p=np.array([[[100,200],[110,200],[130,200],[0,0],[0,0]]],float)
    s=np.array([[4,5,6,0,0]],float);c=np.array([[90,90,90,0,0]],float)
    z=dict(mask=m,target_pos=p,target_sog_kn=s,target_cog_deg=c,time_s=np.array([[0,10,20,0,0.]],float),
           self_gate_weight=np.array([[1,.1,1,0,0]],np.float32),residual_mps=np.array([[[3,4],[0,0],[0,2],[0,0],[0,0]]],np.float32),
           background_field_mps=np.array([[[3,0],[5,0],[5,0],[0,0],[0,0]]],np.float32),divergence_per_s=np.zeros((1,5)),curl_per_s=np.zeros((1,5)),
           raw=np.zeros((1,5,7),np.float32),y=np.array([1]),behavior_type=np.array(['speed_drop']),
           window_id=np.array(['w']),base_window_id=np.array(['b']),mmsi=np.array(['123456789']),region=np.array(['example']),
           noise_type=np.array(['none']),noise_scope=np.array(['none']),observation_noise=np.array([False]),
           observation_noise_effective=np.array([False]),target_noise_effective=np.array([False]),neighbor_noise_effective=np.array([False]),
           kinematic_valid=m.copy(),context_valid=m.copy(),plain_context_valid=m.copy(),boundary_context_valid=m.copy())
    return z

class TestHistoricalInputBridge(unittest.TestCase):
    def test_physical_meanings(self):
        o=target_bridge(fixture());np.testing.assert_allclose(o['raw_legacy7'][0,:3,0],[0,10,30])
        np.testing.assert_allclose(o['raw_legacy7'][0,:3,2],np.array([4,5,6])*KNOT,rtol=1e-6)
        np.testing.assert_array_equal(o['raw_legacy7'][0,:3,4],[4,5,6])
        self.assertEqual(o['label_class'][0],5)
    def test_legacy_and_masked_padding_are_distinct(self):
        o=target_bridge(fixture());g=o['kg_compat8']
        np.testing.assert_allclose(g[0,:,0],[0,1,21,21,21])
        np.testing.assert_allclose(g[0,:,4],[4,4,6,6,6])
        np.testing.assert_array_equal(o['kglf_masked9'][0,3:],0)
    def test_scalar_field_and_controller_units(self):
        o=target_bridge(fixture());np.testing.assert_array_equal(o['rf'][0],[5,0,2,0,0])
        np.testing.assert_allclose(o['ep'][0,:,1],[0,.2,0,0,0])
        np.testing.assert_array_equal(o['kglf_compat9'][...,7],o['rf'])
    def test_full_length_gate_paths_agree(self):
        z=fixture();z['mask'][:]=True;z['self_gate_weight'][0,3:]=1
        o=target_bridge(z);np.testing.assert_array_equal(o['kglf_compat9'],o['kglf_masked9'])
    def test_reject_label_mismatch(self):
        z=fixture();z['y'][:]=0
        with self.assertRaises(AssertionError):target_bridge(z)
    def test_reject_holey_mask(self):
        z=fixture();z['mask'][0]=[1,0,1,0,0]
        with self.assertRaises(ValueError):target_bridge(z)

if __name__=='__main__':unittest.main()
