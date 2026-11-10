"""Business gates must defeat model confidence and unrelated neighbour speeds."""
from __future__ import annotations
import copy
import unittest
import numpy as np

from anomaly_workbench.maritime_rules import assess_motion,score_priority,apply_policy,_saturate


def metadata(speeds,headings=None,*,dt=60.,xy=None,weights=None):
    sog=np.asarray(speeds,float);n=len(sog)
    cog=np.full(n,90.) if headings is None else np.asarray(headings,float)
    ts=np.arange(n)*dt
    if xy is None:
        v=np.column_stack((sog*.514444*np.sin(np.deg2rad(cog)),sog*.514444*np.cos(np.deg2rad(cog))))
        xy=np.zeros((n,2))
        for k in range(1,n):xy[k]=xy[k-1]+.5*(v[k]+v[k-1])*dt
    return {'motion_observations':{'sog_kn':sog.tolist(),'cog_deg':cog.tolist(),'xy_m':np.asarray(xy).tolist(),
            'time_seconds':ts.tolist(),'gate_weights':([1.]*n if weights is None else list(weights)),
            'sog_reported_mask':np.isfinite(sog).tolist(),'cog_reported_mask':np.isfinite(cog).tolist()}}


class MaritimeRuleTests(unittest.TestCase):
    def test_original_zero_sog_wins_over_inferred_speed_and_position_noise(self):
        xy=np.column_stack((np.tile([0,900],10),np.zeros(20)))
        meta=metadata(np.zeros(20),np.tile([0,180],10),xy=xy)
        meta['sog_kn']=[18.]*20  # derived/model samples must not replace original SOG
        result=apply_policy(meta,1.,True,local_residual=25.)
        self.assertTrue(result['motion']['is_stationary']);self.assertFalse(result['is_behavior_candidate'])
        self.assertEqual(result['score'],0.);self.assertFalse(result['model_used'])

    def test_isolated_speed_spike_low_median_and_short_real_path_remains_stationary(self):
        sog=np.full(18,.1);sog[8]=4.1
        xy=np.column_stack((np.linspace(0,147.28,18),np.sin(np.arange(18))*.4))
        motion=assess_motion(metadata(sog,xy=xy,dt=67.))
        self.assertTrue(motion['is_stationary']);self.assertEqual(motion['stationary_reason'],'median_below_0_5kn_and_path_below_300m')
        self.assertLess(motion['path_length_m'],300);self.assertIsNone(motion['event_type'])

    def test_stationary_checkbox_does_not_admit_moored_ship_to_behavior_queue(self):
        result=apply_policy(metadata([0.]*20),1.,True,30.,exclude_stationary=False)
        self.assertTrue(result['motion']['is_stationary']);self.assertFalse(result['is_behavior_candidate'])
        self.assertEqual(result['score'],0.)

    def test_mobile_moving_then_stopping_shape_is_not_hardcoded_by_mmsi(self):
        sog=[8.1,7.6,7.6,7.5,7.2,7.,6.3,5.7,4.3,3.5,.7,0,0,0,0,0,0]
        meta=metadata(sog,dt=71.25);meta['mmsi']='367359250'
        motion=assess_motion(meta)
        self.assertFalse(motion['is_stationary']);self.assertTrue(motion['eligible'])
        self.assertIn(motion['event_type'],['stall','speed_drop']);self.assertGreater(motion['initial_sog_kn'],3.)
        self.assertGreater(motion['path_length_m'],500)
        self.assertTrue(any('正常到港' in text for text in motion['reasons']))

    def test_drop_requires_genuine_early_underway_motion(self):
        motion=assess_motion(metadata([.1]*5+[1.]*5+[0.]*10))
        self.assertFalse(motion['eligible']);self.assertFalse(motion['evidence']['dynamic_speed_drop'])
        moving=assess_motion(metadata([8.]*10+[3.]*10))
        self.assertTrue(moving['eligible']);self.assertEqual(moving['event_type'],'speed_drop')

    def test_speed_drop_without_positional_motion_is_not_eligible(self):
        xy=np.zeros((20,2));xy[:,0]=np.arange(20)*.2
        motion=assess_motion(metadata([8.]*10+[0.]*10,xy=xy))
        self.assertFalse(motion['eligible']);self.assertLess(motion['early_path_length_m'],100)

    def test_exactly_fifty_percent_without_low_endpoint_is_not_dynamic_drop(self):
        motion=assess_motion(metadata([8.]*10+[4.]*10))
        self.assertFalse(motion['evidence']['dynamic_speed_drop'])

    def test_true_turnback_requires_moving_path_and_opposite_direction(self):
        headings=[90.]*7+[120.,150.,180.,210.,240.]+[270.]*8
        motion=assess_motion(metadata([8.]*20,headings))
        self.assertTrue(motion['eligible']);self.assertEqual(motion['event_type'],'turn_back')
        self.assertGreater(motion['turnback_angle_deg'],110)
        self.assertAlmostEqual(motion['peak_abs_rot_deg_per_min'],30.)
        self.assertAlmostEqual(motion['heading_span_deg'],180.)
        stationary=assess_motion(metadata([.1]*20,headings))
        self.assertFalse(stationary['eligible'])

    def test_kinematic_jump_cannot_create_500_meter_motion(self):
        xy=np.zeros((20,2));xy[10:]=[3000,0]
        motion=assess_motion(metadata([5.]*20,xy=xy,weights=[1.]*9+[0.,0.,0.]+[1.]*8))
        self.assertLess(motion['continuous_path_length_m'],500);self.assertFalse(motion['eligible'])

    def test_reported_heading_oscillation_without_curved_track_is_not_sinuous(self):
        n=30;headings=np.tile([60.,120.],15)
        xy=np.column_stack((np.arange(n)*200.,np.zeros(n)))
        motion=assess_motion(metadata([8.]*n,headings,xy=xy))
        self.assertFalse(motion['evidence']['sinuous'])

    def test_true_repeated_curved_motion_is_sinuous(self):
        headings=[90.]*4+([120.]*4+[60.]*4)*4+[90.]*4
        motion=assess_motion(metadata([8.]*len(headings),headings))
        self.assertTrue(motion['evidence']['sinuous']);self.assertEqual(motion['event_type'],'sinuous')
        self.assertGreaterEqual(motion['rot_sign_alternations'],3)

    def test_gradual_single_bend_is_not_falsely_called_sinuous(self):
        motion=assess_motion(metadata([8.]*30,np.linspace(50,100,30)))
        self.assertFalse(motion['evidence']['sinuous']);self.assertEqual(motion['rot_sign_alternations'],0)

    def test_sparse_priority_is_invariant_to_model_and_neighbor_values(self):
        motion=assess_motion(metadata([8.]*10+[0.]*10))
        a=score_priority(motion,0.,False,0.,[0.,.1,.2])
        b=score_priority(motion,1.,False,200.,[.99,.999,1.])
        self.assertEqual(a['score'],b['score']);self.assertEqual(a['score_components'],b['score_components'])
        self.assertFalse(a['model_used']);self.assertFalse(b['neighbor_used']);self.assertEqual(a['score_basis'],'self_kinematics')
        self.assertGreaterEqual(a['score'],.5);self.assertLess(a['score'],.99)

    def test_unsupported_low_speed_is_capped_below_half(self):
        motion=assess_motion(metadata([2.1]*20,xy=np.column_stack((np.linspace(0,70,20),np.zeros(20)))))
        result=score_priority(motion,1.,False,100.)
        self.assertFalse(result['is_behavior_candidate']);self.assertLess(result['score'],.5)

    def test_priority_is_smooth_monotone_and_never_rounds_to_one(self):
        base={'eligible':True,'is_stationary':False}
        values=[score_priority({**base,'kinematic_strength':float(s)},1.,False)['score'] for s in np.linspace(0,1,101)]
        self.assertTrue(all(b>a for a,b in zip(values,values[1:])))
        self.assertTrue(all(.5<=v<.99 for v in values));self.assertLess(max(np.diff(values)),.02)
        self.assertTrue(all(f'{v:.3f}'!='1.000' for v in values))

    def test_strength_keeps_real_differences_beyond_old_hard_scales(self):
        # Each measurement exceeds its former hard-clipping scale. Higher real
        # speed/deceleration/distance must still contribute additional evidence.
        for scale in [12.,4.,3000.]:
            values=[_saturate(multiplier*scale,scale) for multiplier in [1.,1.5,2.,4.,8.]]
            self.assertTrue(all(a<b for a,b in zip(values,values[1:])))
            self.assertTrue(all(0<v<1 for v in values))
        self.assertEqual(_saturate(-1.,12.),0.)

    def test_distinct_high_strength_real_drop_tracks_do_not_share_flat_top(self):
        # With the old hard clip these all reached strength=1: initial SOG>=12,
        # deceleration>=4kn/min, and actual continuous track length>=3km.
        motions=[assess_motion(metadata([speed]*10+[0.]*10)) for speed in [12.,16.,20.,24.,28.,32.]]
        self.assertTrue(all(m['eligible'] and m['continuous_path_length_m']>=3000 for m in motions))
        priorities=[score_priority(m,1.,False,100.)['score'] for m in motions]
        self.assertTrue(all(a<b for a,b in zip(priorities,priorities[1:])))
        self.assertEqual(len(set(priorities)),len(priorities))
        self.assertGreater(len({f'{value:.3f}' for value in priorities}),1)
        self.assertTrue(all(.5<=value<.99 for value in priorities))

    def test_supported_confirmation_also_preserves_high_strength_differences(self):
        weaker=assess_motion(metadata([14.]*10+[0.]*10))
        stronger=assess_motion(metadata([28.]*10+[0.]*10))
        low=score_priority(weaker,.999,True,3.)['score']
        high=score_priority(stronger,.999,True,3.)['score']
        self.assertGreater(high,low)
        self.assertNotEqual(f'{high:.3f}',f'{low:.3f}')

    def test_supported_model_cannot_override_absent_motion_event(self):
        motion=assess_motion(metadata([8.]*20))
        result=score_priority(motion,1.,True,50.)
        self.assertFalse(motion['eligible']);self.assertFalse(result['is_behavior_candidate']);self.assertLess(result['score'],.5)

    def test_missing_speeds_or_insufficient_points_are_not_confirmed_stationary(self):
        motion=assess_motion(metadata([np.nan]*20))
        self.assertFalse(motion['is_stationary']);self.assertFalse(motion['eligible'])
        short=assess_motion(metadata([0.]*3))
        self.assertFalse(short['is_stationary']);self.assertEqual(short['movement_state'],'insufficient_observations')

    def test_missing_course_does_not_erase_valid_zero_sog(self):
        meta=metadata([0.]*20,[np.nan]*20)
        meta['motion_observations']['reported_mask']=[False]*20
        motion=assess_motion(meta)
        self.assertTrue(motion['is_stationary']);self.assertEqual(motion['max_sog_kn'],0.)

    def test_identical_physics_has_identical_score_not_batch_rank_jitter(self):
        motion=assess_motion(metadata([7.]*10+[0.]*10))
        first=score_priority(motion,.999,True,2.)
        second=score_priority(copy.deepcopy(motion),.999,True,2.)
        self.assertEqual(first,second)


if __name__=='__main__':unittest.main()
