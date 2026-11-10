import tempfile
from pathlib import Path
import unittest
import numpy as np
import build_component_features as f


def fixture():
    n, t = 2, 20
    mask = np.zeros((n, t), bool)
    mask[:, :8] = True
    raw = np.zeros((n, t, 7), np.float32)
    raw[mask] = np.array([.1, .2, 2., 1., .05, .6, .8], np.float32)
    gate = np.ones((n, t), np.float32)
    gate[0, 1] = 0.
    gate[0, 2] = .2
    gate[~mask] = 0
    gated = raw.copy()
    gated[..., [0, 1, 2, 3, 5, 6]] *= gate[..., None]
    context = np.zeros((n, t, 16), np.float32)
    context[..., :7] = gated
    residual = np.zeros((n, t, 2), np.float32)
    residual[mask] = [2., -1.]
    context[..., 7:9] = residual / f.VSCALE_MPS * gate[..., None]
    context[..., 9] = mask.astype(np.float32) * .4
    context[..., 10] = gate
    context[..., 11:14] = mask[..., None]
    context[..., 14:16][mask] = [.1, -.2]
    plain = context.copy()
    plain[..., :7] = raw
    plain[..., 9:12] = 0
    d = dict(mask=mask, raw=raw, gated=gated, context=context,
             plain_context=plain.copy(), plain_affine_context=plain.copy(),
             residual_mps=residual, background_mean_mps=np.zeros((n,t,2),np.float32),
             background_mean_valid=mask.copy(), context_valid=mask.copy(),
             valid_neighbor_count=mask.astype(np.int32)*5,
             plain_neighbor_count=mask.astype(np.int32)*6,
             boundary_context_valid=mask.copy(), self_gate_weight=gate,
             self_reliability=gate.copy(), kinematic_valid=mask.copy(),
             y=np.array([0,1],np.int8), mmsi=np.array([123,456]),
             window_id=np.array(['a','b']), base_window_id=np.array(['a','b']),
             region=np.array(['houston']*n), behavior_type=np.array(['none','turn_back']),
             noise_type=np.array(['teleport','none']), noise_scope=np.array(['target_only','none']),
             target_noise_effective=np.array([True,False]),
             neighbor_noise_effective=np.array([False,False]),
             observation_noise_effective=np.array([True,False]),
             observation_noise=np.array([True,False]),
             target_sog_kn=mask.astype(float)*8, target_cog_deg=mask.astype(float)*45,
             pre_noise_sog_kn=mask.astype(float)*8, pre_noise_cog_deg=mask.astype(float)*45)
    return d


class ComponentFeaturesTest(unittest.TestCase):
    def test_raw_and_legacy_preserved_exactly(self):
        d=fixture();v,a=f.derive_variants(d)
        for key in ['raw16','raw_meta','mean_meta','affine_meta','reliable_raw',
                    'reliable_gate_residual','reliable_raw_fallback']:
            np.testing.assert_array_equal(v[key][...,:7],d['raw'])
        np.testing.assert_array_equal(v['legacy_full'],d['context'])
        self.assertFalse(a.any())

    def test_zero_gate_does_not_destroy_saved_ungated_residual(self):
        d=fixture();v,_=f.derive_variants(d)
        self.assertEqual(d['self_gate_weight'][0,1],0)
        self.assertTrue(np.all(v['legacy_full'][0,1,7:9]==0))
        np.testing.assert_allclose(v['reliable_raw'][0,1,7:9],np.array([2.,-1.])/f.VSCALE_MPS)
        self.assertTrue(np.isfinite(v['reliable_raw']).all())

    def test_factor_changes_are_exact_and_local(self):
        d=fixture();v,_=f.derive_variants(d)
        np.testing.assert_array_equal(v['raw16'][...,7:],0)
        np.testing.assert_array_equal(v['raw_meta'][...,:9],v['raw16'][...,:9])
        np.testing.assert_array_equal(v['raw_meta'][...,12:],v['raw16'][...,12:])
        np.testing.assert_array_equal(v['gated_meta'][...,7:],v['raw_meta'][...,7:])
        np.testing.assert_array_equal(v['reliable_gate_target'][...,7:],v['reliable_raw'][...,7:])
        np.testing.assert_array_equal(v['reliable_gate_residual'][...,:7],v['reliable_raw'][...,:7])
        np.testing.assert_array_equal(v['reliable_gate_residual'][...,9:],v['reliable_raw'][...,9:])
        np.testing.assert_array_equal(v['legacy_full'][...,:7],v['reliable_gate_target'][...,:7])
        np.testing.assert_array_equal(v['legacy_full'][...,7:],v['reliable_gate_residual'][...,7:])

    def test_fallback_requires_support_mean_and_coverage(self):
        d=fixture()
        for step in range(1,6):
            d['context_valid'][1,step]=False
            d['context'][1,step,12]=0
            d['residual_mps'][1,step]=0
            d['context'][1,step,7:9]=0
            d['context'][1,step,14:16]=0
        d['valid_neighbor_count'][1,2]=3
        d['background_mean_valid'][1,3]=False
        d['boundary_context_valid'][1,4]=False
        d['background_mean_mps'][1,1]=[1.,2.]
        v,a=f.derive_variants(d)
        self.assertEqual(np.flatnonzero(a[1]).tolist(),[1,5])
        expected=d['raw'][1,1,2:4]-np.array([1.,2.])/f.VSCALE_MPS
        np.testing.assert_allclose(v['reliable_raw_fallback'][1,1,7:9],expected,rtol=1e-6,atol=1e-7)
        self.assertEqual(v['reliable_raw_fallback'][1,1,12],1)
        np.testing.assert_array_equal(v['reliable_raw_fallback'][1,1,14:16],0)
        np.testing.assert_array_equal(v['reliable_raw_fallback'][~a],v['reliable_raw'][~a])

    def test_source_padding_and_nonfinite_rejected(self):
        d=fixture();d['raw'][0,19,0]=1
        with self.assertRaisesRegex(ValueError,'padded'):
            f.derive_variants(d)
        d=fixture();d['context'][0,0,9]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            f.derive_variants(d)

    def test_source_gate_mismatch_rejected(self):
        d=fixture();d['gated'][0,0,2]+=1
        with self.assertRaisesRegex(ValueError,'gate'):
            f.derive_variants(d)

    def test_metadata_diagnostic_uses_actual_effect(self):
        d=fixture();_,d['fallback_activated']=f.derive_variants(d)
        x=f.diagnostics(d)
        self.assertEqual(x['all']['low_gate_steps_lt_0_3'],2)
        self.assertEqual(x['all']['low_gate_reported_velocity_unchanged_by_controlled_corruption_steps'],2)
        self.assertIn('target+no_neighbors',x['strata']['actual_effect'])

    def test_allowed_splits_are_explicit_train_val(self):
        self.assertEqual(f.SPLITS,('train','val'))
        self.assertEqual(len(f.VARIANTS),10)


if __name__ == '__main__':
    unittest.main()
