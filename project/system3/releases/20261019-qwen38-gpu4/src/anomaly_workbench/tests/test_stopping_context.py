"""Historical observed stopping places are context, not verified port labels."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from anomaly_workbench.stopping_context import annotate_stopping_context


def row(vessel, hour, *, candidate=False, offset_m=0, speed=0., duration=1140, n=20):
    start = datetime(2024, 7, 6, tzinfo=timezone.utc) + timedelta(hours=hour)
    xy = np.column_stack((np.linspace(0, 5, n) + offset_m, np.zeros(n)))
    ll = np.column_stack((np.full(n, 30.5), -88 + xy[:, 0] / (111195 * np.cos(np.deg2rad(30.5)))))
    sog = np.full(n, speed)
    times = np.linspace(0, duration, n)
    return {'id': f'{vessel}-{hour}', 'vessel_id': str(vessel), 'start_utc': start.isoformat(),
            'end_utc': (start + timedelta(seconds=duration)).isoformat(), 'motion_candidate': candidate,
            'latlon': ll.tolist(), 'gate_weights': [1.] * n,
            'maritime_policy': {'late_sog_kn': speed},
            'motion_observations': {'sog_kn': sog.tolist(), 'time_seconds': times.tolist(),
                                    'gate_weights': [1.] * n, 'xy_m': xy.tolist(), 'sog_reported_mask': [True] * n}}


def fixture():
    return [row('111', 0, offset_m=10), row('222', 1, offset_m=20), row('333', 2, offset_m=30), row('999', 4, candidate=True)]


class StoppingContextTests(unittest.TestCase):
    def test_three_other_ships_past_stops_are_context_with_provenance(self):
        records = fixture()
        before = copy.deepcopy(records)
        context = annotate_stopping_context(records)['999-4']
        self.assertTrue(context['matches'])
        self.assertEqual(set(context['reference_mmsi']), {'111', '222', '333'})
        self.assertEqual(context['basis'], 'past_observations_only')
        self.assertEqual(context['radius_m'], 200)
        self.assertIn('不是官方港口', context['meaning'])
        self.assertEqual(records, before)

    def test_future_or_not_yet_ended_window_cannot_supply_reference(self):
        for index in [0, 1, 2]:
            records = fixture()
            records[index]['end_utc'] = records[-1]['end_utc']
            self.assertFalse(annotate_stopping_context(records)['999-4']['matches'])

    def test_same_ship_and_repeated_windows_cannot_count_as_other_three(self):
        records = [row('111', 0), row('111', 1), row('111', 2), row('999', 0), row('999', 4, candidate=True)]
        self.assertFalse(annotate_stopping_context(records)['999-4']['matches'])

    def test_reference_must_have_ten_minutes_and_eight_reliable_original_points(self):
        for replacement in [row('333', 2, duration=599), row('333', 2, n=7)]:
            records = fixture()
            records[2] = replacement
            self.assertFalse(annotate_stopping_context(records)['999-4']['matches'])

    def test_target_needs_observed_final_low_speed_for_ninety_seconds(self):
        for wrong in ['tail', 'derived']:
            records = fixture()
            target = records[-1]
            if wrong == 'tail':
                target['motion_observations']['sog_kn'][:-2] = [5.] * 18
            else:
                target['sog_kn'] = [0.] * 20
                target['motion_observations']['sog_kn'] = [5.] * 20
            self.assertFalse(annotate_stopping_context(records)['999-4']['matches'])

    def test_observed_low_speed_tail_wins_over_coarse_window_median(self):
        records = fixture()
        target = records[-1]
        target['maritime_policy']['late_sog_kn'] = 3.1
        target['motion_observations']['sog_kn'][:-3] = [8.] * 17
        self.assertTrue(annotate_stopping_context(records)['999-4']['matches'])

    def test_jump_or_unreliable_positions_do_not_supply_stopping_history(self):
        records = fixture()
        reference = records[2]
        reference['motion_observations']['xy_m'][10:] = [[2000., 0.]] * 10
        self.assertFalse(annotate_stopping_context(records)['999-4']['matches'])
        records = fixture()
        records[2]['motion_observations']['gate_weights'] = [.1] * 20
        self.assertFalse(annotate_stopping_context(records)['999-4']['matches'])

    def test_far_stops_or_real_moving_windows_are_not_stopping_context(self):
        for replacement in [row('333', 2, offset_m=500), row('333', 2, speed=1.)]:
            records = fixture()
            records[2] = replacement
            self.assertFalse(annotate_stopping_context(records)['999-4']['matches'])

    def test_missing_data_and_unreliable_target_endpoint_do_not_match(self):
        for key in ['motion_observations', 'latlon', 'gate_weights']:
            records = fixture()
            records[-1].pop(key)
            self.assertFalse(annotate_stopping_context(records)['999-4']['matches'])
        records = fixture()
        records[-1]['gate_weights'][-1] = .1
        self.assertFalse(annotate_stopping_context(records)['999-4']['matches'])
        self.assertEqual(annotate_stopping_context([None, {}, {'id': 'missing-vessel'}]), {})


if __name__ == '__main__':
    unittest.main()
