"""Current manoeuvres support limited hypotheses, never confirmed intent."""
from __future__ import annotations

import copy
import unittest
import numpy as np

from anomaly_workbench.maneuver_intent import assess_maneuver_intent


def fixture(speed, headings=None, *, reversal=False, drop=True, dt=60.):
    speed = np.asarray(speed, float)
    n = len(speed)
    headings = np.full(n, 90.) if headings is None else np.asarray(headings, float)
    velocity = np.column_stack((speed * .514444 * np.sin(np.deg2rad(headings)), speed * .514444 * np.cos(np.deg2rad(headings))))
    xy = np.zeros((n, 2))
    for i in range(1, n):
        xy[i] = xy[i-1] + .5 * (velocity[i] + velocity[i-1]) * dt
    return {'id': 'AIS-test', 'end_utc': '2024-07-06T00:20:00Z',
            'motion_observations': {'sog_kn': speed.tolist(), 'source_sog': speed.tolist(),
                                    'cog_deg': headings.tolist(), 'source_cog': headings.tolist(),
                                    'xy_m': xy.tolist(), 'time_seconds': (np.arange(n) * dt).tolist(),
                                    'gate_weights': [1.] * n, 'sog_reported_mask': [True] * n},
            'review_assessment': {'evidence': {'turnback_confirmed': reversal, 'turnback_stable_angle_deg': 170. if reversal else None,
                                               'deceleration': {'confirmed': drop}, 'local_corroboration': False}},
            'maritime_policy': {'neighbor_supported': False},
            'recurrence_evidence': {'repeated': False, 'reference_ids': []}}


class ManeuverIntentTests(unittest.TestCase):
    def test_cold_start_turn_then_hold_remains_a_possible_operational_pattern(self):
        r = fixture([6.] * 12 + [.1] * 6, [0.] * 6 + [180.] * 12, reversal=True)
        result = assess_maneuver_intent(r)
        self.assertEqual(result['evidence']['pattern_code'], 'turn_then_hold')
        self.assertTrue(result['evidence']['normal_maneuver_pattern'])
        self.assertGreaterEqual(result['evidence']['terminal_low_speed_seconds'], 90)
        self.assertLess(result['evidence']['net_path_ratio'], .35)
        self.assertIn('不表示该动作罕见或异常', result['explanation'])
        self.assertIn('原始导航状态', result['missing_context'])
        self.assertTrue(result['normal_operation_not_excluded'])

    def test_ending_at_four_point_six_knots_is_transit_not_completed_berthing(self):
        r = fixture([24.] * 6 + [4.6] * 10)
        result = assess_maneuver_intent(r)
        self.assertEqual(result['evidence']['pattern_code'], 'fast_to_slow_transit')
        self.assertEqual(result['evidence']['terminal_speed_kn'], 4.6)
        self.assertIn('不能据此称已靠泊', result['explanation'])
        self.assertNotIn('已靠泊', result['observed_action'])

    def test_deceleration_then_hold_is_not_a_confirmed_intent(self):
        result = assess_maneuver_intent(fixture([8.] * 8 + [.1] * 8))
        self.assertEqual(result['evidence']['pattern_code'], 'decelerate_then_hold')
        self.assertTrue(result['possible_intents'])
        self.assertIn('具体目的尚未确认', result['explanation'])

    def test_high_speed_turnback_is_not_demoted_to_a_stopping_pattern(self):
        result = assess_maneuver_intent(fixture([8.] * 16, [0.] * 8 + [180.] * 8, reversal=True))
        self.assertFalse(result['evidence']['normal_maneuver_pattern'])
        self.assertEqual(result['evidence']['terminal_low_speed_seconds'], 0)
        self.assertIn('在航回转', result['observed_action'])

    def test_isolated_speed_spike_cannot_prove_a_stable_before_state(self):
        result = assess_maneuver_intent(fixture([4.] * 5 + [24.] + [4.] * 10))
        self.assertFalse(result['evidence']['normal_maneuver_pattern'])
        self.assertFalse(result['evidence']['verified_deceleration'])

    def test_position_jump_or_time_gap_cannot_bridge_the_manoeuvre(self):
        for fault in ['jump', 'time']:
            r = fixture([8.] * 8 + [.1] * 8)
            if fault == 'jump':
                for i in range(8, 16):
                    r['motion_observations']['xy_m'][i][0] += 5000
                r['motion_observations']['gate_weights'][8] = .1
            else:
                for i in range(8, 16):
                    r['motion_observations']['time_seconds'][i] += 1000
            result = assess_maneuver_intent(r)
            self.assertFalse(result['evidence']['normal_maneuver_pattern'])

    def test_original_sog_precedes_display_or_derived_speed(self):
        r = fixture([.1] * 20)
        r['sog_kn'] = [24.] * 6 + [4.] * 14
        r['motion_observations']['sog_kn'] = r['sog_kn']
        result = assess_maneuver_intent(r)
        self.assertEqual(result['evidence']['terminal_speed_kn'], .1)
        self.assertFalse(result['evidence']['normal_maneuver_pattern'])

    def test_missing_or_unreliable_terminal_observations_do_not_infer_intent(self):
        self.assertFalse(assess_maneuver_intent({})['evidence']['normal_maneuver_pattern'])
        r = fixture([8.] * 8 + [.1] * 8)
        r['motion_observations'].pop('time_seconds')
        self.assertFalse(assess_maneuver_intent(r)['evidence']['normal_maneuver_pattern'])
        r = fixture([8.] * 8 + [.1] * 8)
        r['motion_observations']['gate_weights'][-1] = .1
        self.assertFalse(assess_maneuver_intent(r)['evidence']['normal_maneuver_pattern'])

    def test_input_unchanged_and_neighbor_reference_is_not_intent_proof(self):
        r = fixture([24.] * 6 + [4.] * 10)
        r['maritime_policy']['neighbor_supported'] = True
        r['review_assessment']['evidence']['local_corroboration'] = True
        before = copy.deepcopy(r)
        result = assess_maneuver_intent(r)
        self.assertEqual(r, before)
        self.assertTrue(result['evidence']['neighbor_reference_available'])
        self.assertTrue(result['evidence']['local_motion_corroboration'])
        self.assertIn('不证明作业意图', result['evidence']['reference_meaning'])


if __name__ == '__main__':
    unittest.main()
