import unittest
from dataclasses import replace
import numpy as np
from maneuver_generator import (PhysicalLimits,apply_observation_noise,generate_maneuver,
                                innovation_vectors,paired_controls,physical_statistics,
                                sample_parameters,velocity_vectors)


def source(irregular=True):
    t=np.array([0,62,128,191,260,331,402,478,550,626,700,779,861,943,1027,1113,1200],float) if irregular else np.arange(0,1201,60,dtype=float)
    s=10+.12*np.sin(t/140);c=(359+.2*np.sin(t/200))%360
    v=velocity_vectors(s,c);p=np.zeros((len(t),2));p[1:]=np.cumsum((v[1:]+v[:-1])*.5*np.diff(t)[:,None],axis=0)
    # Nonzero realistic observed innovations survive the maneuver injection.
    p+=np.column_stack((3*np.sin(t/55),2*np.cos(t/75)))
    return t,p,s,c


class ManeuverContracts(unittest.TestCase):
    def test_identity_bitwise_and_input_not_mutated(self):
        x=source();before=[v.copy() for v in x]
        r=generate_maneuver(*x,kind='none')
        for a,b,key in zip(x,before,['t','pos','sog_kn','cog_deg']):
            self.assertTrue(np.array_equal(a,b));self.assertTrue(np.array_equal(r[key],b))

    def test_all_five_maneuvers_preserve_vector_innovations(self):
        x=source();original=innovation_vectors(*x)
        for kind in ['turn_back','deviation','stall','sinuous','speed_drop']:
            with self.subTest(kind=kind):
                r=generate_maneuver(*x,kind=kind,onset_s=270,params=sample_parameters(kind,42))
                self.assertTrue(r['accepted'],r['metadata'])
                actual=innovation_vectors(r['t'],r['pos'],r['sog_kn'],r['cog_deg'])
                self.assertTrue(np.allclose(original,actual,atol=1e-10))
                self.assertGreater(np.max(np.linalg.norm(original,axis=1)),.01)
                self.assertTrue(np.array_equal(r['pos'][x[0]<270],x[1][x[0]<270]))
                self.assertGreater(np.linalg.norm(r['pos'][-1]-x[1][-1]),1)

    def test_off_grid_first_affected_interval_is_updated(self):
        x=source();r=generate_maneuver(*x,kind='speed_drop',onset_s=270,params={'factor':.3,'duration_s':120})
        i=np.searchsorted(x[0],270)
        self.assertFalse(np.array_equal(r['pos'][i],x[1][i]))
        self.assertTrue(np.array_equal(r['pos'][i-1],x[1][i-1]))
        self.assertEqual(r['metadata']['onset_bracket_s'],[260.,331.])
        self.assertFalse(r['metadata']['sub_report_physical_onset_resolution_claimed'])

    def test_sample_aligned_smooth_start_has_no_splice_change(self):
        x=source();r=generate_maneuver(*x,kind='turn_back',onset_s=260)
        i=4
        self.assertTrue(np.array_equal(r['pos'][:i+1],x[1][:i+1]))
        self.assertAlmostEqual(r['cog_deg'][i],x[3][i])

    def test_course_wrap_359_to_1_is_small(self):
        s=physical_statistics(np.array([0.,10.]),np.array([10.,10.]),np.array([359.,1.]))
        self.assertAlmostEqual(s['max_turn_rate_deg_s'],.2)
        self.assertLess(s['max_lateral_accel_mps2'],.02)

    def test_strict_budget_rejection_returns_unchanged_input(self):
        x=source();limits=PhysicalLimits(max_turn_rate_deg_s=.001)
        r=generate_maneuver(*x,kind='turn_back',onset_s=260,params={'delta_deg':180,'duration_s':10},limits=limits,adapt=False)
        self.assertFalse(r['accepted']);self.assertEqual(r['reason'],'physical_budget_exceeded')
        self.assertTrue(np.array_equal(r['pos'],x[1]));self.assertFalse(r['metadata']['behavior_injected'])

    def test_duration_adaptation_is_explicit(self):
        x=source();limits=PhysicalLimits(max_turn_rate_deg_s=.7,max_lateral_accel_mps2=.3)
        r=generate_maneuver(*x,kind='turn_back',onset_s=260,params={'delta_deg':150,'duration_s':20},limits=limits,max_adaptations=16)
        self.assertTrue(r['accepted']);self.assertGreater(r['metadata']['applied_params']['duration_s'],20)
        self.assertGreater(len(r['metadata']['adaptation_attempts']),1)
        self.assertLessEqual(r['metadata']['physical_dense']['max_turn_rate_deg_s'],.7+1e-8)

    def test_speed_drop_changes_speed_not_heading(self):
        x=source();r=generate_maneuver(*x,kind='speed_drop',onset_s=260,params={'factor':.4,'duration_s':180})
        self.assertTrue(r['accepted']);self.assertTrue(np.array_equal(r['cog_deg'],x[3]))
        self.assertAlmostEqual(r['sog_kn'][-1]/x[2][-1],.4)

    def test_stall_does_not_add_turn(self):
        x=source();r=generate_maneuver(*x,kind='stall',onset_s=260,params={'target_kn':.7,'duration_s':200})
        self.assertTrue(r['accepted']);self.assertTrue(np.array_equal(r['cog_deg'],x[3]))
        self.assertAlmostEqual(r['sog_kn'][-1],.7)

    def test_matched_controls_have_exact_same_plan_and_semantics(self):
        x=source()
        for kind in ['turn_back','deviation','stall','sinuous','speed_drop']:
            with self.subTest(kind=kind):
                r=paired_controls(*x,kind=kind,onset_s=260,params=sample_parameters(kind,81))
                self.assertTrue(r['accepted'])
                sm,ha=r['smooth'],r['hard_step']
                self.assertEqual(sm['metadata']['applied_params'],ha['metadata']['applied_params'])
                self.assertEqual(sm['metadata']['kind'],ha['metadata']['kind'])
                if kind in ['stall','speed_drop']:
                    self.assertTrue(np.array_equal(ha['cog_deg'],x[3]))
                if kind in ['turn_back','deviation','sinuous']:
                    self.assertTrue(np.array_equal(ha['sog_kn'],x[2]))
                self.assertTrue(np.allclose(innovation_vectors(ha['t'],ha['pos'],ha['sog_kn'],ha['cog_deg']),innovation_vectors(*x),atol=1e-10))
                self.assertFalse(ha['metadata']['constraints_enforced'])

    def test_sinuous_phase_direction_is_controlled(self):
        x=source();plus=generate_maneuver(*x,kind='sinuous',onset_s=260,params={'sign':1})
        minus=generate_maneuver(*x,kind='sinuous',onset_s=260,params={'sign':-1})
        a=(plus['cog_deg']-x[3]+180)%360-180;b=(minus['cog_deg']-x[3]+180)%360-180
        self.assertTrue(np.allclose(a,-b))

    def test_sinuous_ramp_alias_phase_and_sampling_floor(self):
        x=source();r=generate_maneuver(*x,kind='sinuous',onset_s=260,
             params={'ramp_up_s':100,'period_s':100,'phase_rad':.5})
        self.assertTrue(r['accepted'])
        p=r['metadata']['applied_params']
        self.assertEqual(p['ramp_up_s'],p['duration_s'])
        self.assertGreaterEqual(p['period_s'],4*np.median(np.diff(x[0])))
        self.assertGreater(r['metadata']['realized']['realized_sinuous_amplitude_deg'],1)
        rejected=generate_maneuver(*x,kind='sinuous',onset_s=260,params={'period_s':100},adapt=False)
        self.assertFalse(rejected['accepted']);self.assertEqual(rejected['reason'],'temporal_resolution_exceeded')

    def test_sinuous_finite_duration_returns_to_baseline(self):
        x=source();r=generate_maneuver(*x,kind='sinuous',onset_s=260,
                  params={'ramp_up_s':120,'active_duration_s':600,'ramp_down_s':120,'phase_rad':.7})
        self.assertTrue(r['accepted'])
        self.assertTrue(np.array_equal(r['cog_deg'][x[0]>=860],x[3][x[0]>=860]))

    def test_noise_none_is_exact_and_reproducible_noise_is_label_independent(self):
        x=source();none=apply_observation_noise(*x,noise='none')
        for key,value in zip(['t','pos','sog_kn','cog_deg'],x):self.assertTrue(np.array_equal(none[key],value))
        for noise in ['teleport','speed_spike','drift']:
            a=apply_observation_noise(*x,noise=noise,seed=82);b=apply_observation_noise(*x,noise=noise,seed=82)
            self.assertTrue(np.array_equal(a['pos'],b['pos']));self.assertTrue(np.array_equal(a['sog_kn'],b['sog_kn']))
            self.assertEqual(a['metadata']['parameters'],b['metadata']['parameters'])
            self.assertFalse(a['metadata']['behavior_label_changed'])

    def test_no_argument_mutation(self):
        x=source();snap=[v.copy() for v in x];params={'factor':.35,'duration_s':160}
        generate_maneuver(*x,kind='speed_drop',onset_s=300,params=params)
        apply_observation_noise(*x,noise='teleport',seed=7)
        for v,b in zip(x,snap):self.assertTrue(np.array_equal(v,b))
        self.assertEqual(params,{'factor':.35,'duration_s':160})

    def test_speed_spike_is_valid_after_float32_roundtrip(self):
        t,p,s,c=source();s[:]=30.
        r=apply_observation_noise(t,p,s,c,noise='speed_spike',params={'factor':5.,'report_index':3})
        self.assertLess(np.asarray(r['sog_kn'],np.float32).max(),102.2)
        self.assertTrue(r['metadata']['observations_changed'])

    def test_invalid_inputs_fail(self):
        x=list(source());x[0][2]=x[0][1]
        with self.assertRaises(ValueError):generate_maneuver(*x,kind='none')
        x=source()
        with self.assertRaises(ValueError):generate_maneuver(*x,kind='turn_back',onset_s=260,params={'unknown':2})
        with self.assertRaises(ValueError):generate_maneuver(*x,kind='speed_drop',onset_s=x[0][0])


if __name__=='__main__':unittest.main()
