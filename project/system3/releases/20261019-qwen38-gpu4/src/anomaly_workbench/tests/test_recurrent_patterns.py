"""Observed repetition can reduce priority without declaring a ship normal."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from anomaly_workbench.recurrent_patterns import annotate_recurrence


def record(hour, *, rid=None, vessel='219000000', x_offset=0., y_offset=0., reverse=False,
           initial=8., late=8., arrival=False, candidate=True, path=1200., n=20):
    utc = datetime(2024, 7, 6, tzinfo=timezone.utc) + timedelta(hours=hour)
    x = np.linspace(0, path, n) + x_offset
    y = np.full(n, y_offset)
    if reverse:
        x = x[::-1]
    ll = np.column_stack((30.5 + y / 111195., -88 + x / (111195. * np.cos(np.deg2rad(30.5)))))
    return {'id': rid or f'window-{hour}', 'vessel_id': vessel, 'start_utc': utc.isoformat(),
            'end_utc': (utc + timedelta(minutes=19)).isoformat(), 'motion_candidate': candidate,
            'is_candidate': candidate, 'latlon': ll.tolist(), 'time_seconds': np.linspace(0, 1140, n).tolist(),
            'sog_kn': np.linspace(initial, late, n).tolist(), 'gate_weights': [1.] * n,
            'maritime_policy': {'stationary': False, 'continuous_path_length_m': path,
                                'initial_sog_kn': initial, 'late_sog_kn': late,
                                'early_path_length_m': 400. if arrival else 200.}}


class RecurrentPatternTests(unittest.TestCase):
    def test_two_distinct_past_passages_are_evidence_and_input_is_unchanged(self):
        records = [record(0, candidate=False), record(2, candidate=False), record(4)]
        before = copy.deepcopy(records)
        result = annotate_recurrence(records)
        self.assertTrue(result['window-4']['repeated'])
        self.assertEqual(result['window-4']['kind'], 'trajectory')
        self.assertEqual(result['window-4']['reference_ids'], ['window-0', 'window-2'])
        self.assertEqual(result['window-4']['independent_passages'], 2)
        self.assertEqual(result['window-4']['basis'], 'past_observations_only')
        self.assertEqual(records, before)

    def test_future_passages_never_establish_target_history(self):
        records = [record(4), record(0), record(2)]
        result = annotate_recurrence(records)
        self.assertFalse(result['window-0']['repeated'])
        self.assertFalse(result['window-2']['repeated'])
        self.assertTrue(result['window-4']['repeated'])

    def test_neighboring_windows_do_not_count_as_independent_passages(self):
        records = [record(0), record(.3), record(.6), record(4)]
        self.assertFalse(annotate_recurrence(records)['window-4']['repeated'])

    def test_one_past_match_is_insufficient(self):
        result = annotate_recurrence([record(0), record(2)])
        self.assertFalse(result['window-2']['repeated'])
        self.assertEqual(result['window-2']['reference_ids'], [])

    def test_opposite_direction_is_a_different_trajectory(self):
        result = annotate_recurrence([record(0, reverse=True), record(2, reverse=True), record(4)])
        self.assertFalse(result['window-4']['repeated'])

    def test_other_vessels_do_not_become_own_operating_history(self):
        result = annotate_recurrence([record(0, vessel='111'), record(2, vessel='111'), record(4)])
        self.assertFalse(result['window-4']['repeated'])

    def test_compact_closed_loop_in_reverse_direction_is_not_a_match(self):
        rows = [record(0, n=40), record(2, n=40), record(4, n=40)]
        theta = np.linspace(0, 2 * np.pi, 40)
        x, y = 120 * np.cos(theta), 120 * np.sin(theta)
        ll = np.column_stack((30.5 + y / 111195., -88 + x / (111195. * np.cos(np.deg2rad(30.5)))))
        for row in rows[:2]:
            row['latlon'] = ll[::-1].tolist()
            row['maritime_policy']['continuous_path_length_m'] = 750.
        rows[2]['latlon'] = ll.tolist()
        rows[2]['maritime_policy']['continuous_path_length_m'] = 750.
        self.assertFalse(annotate_recurrence(rows)['window-4']['repeated'])

    def test_endpoint_match_requires_actual_underway_then_slowdown(self):
        records = [record(0, x_offset=-1200, path=2400, initial=8, late=0, arrival=True),
                   record(2, x_offset=-2400, path=3600, initial=8, late=.2, arrival=True),
                   record(4, initial=8, late=.1, arrival=True)]
        result = annotate_recurrence(records)['window-4']
        self.assertTrue(result['repeated'])
        self.assertEqual(result['kind'], 'slowdown_endpoint')
        self.assertTrue(all(distance < 1 for distance in result['reference_distances_m']))

    def test_stationary_reports_cannot_establish_arrival_recurrence(self):
        records = [record(0, initial=0, late=0), record(2, initial=0, late=0), record(4, initial=8, late=0, arrival=True)]
        for row in records[:2]:
            row['maritime_policy']['stationary'] = True
        self.assertFalse(annotate_recurrence(records)['window-4']['repeated'])

    def test_low_initial_speed_or_missing_early_motion_is_not_arrival_reference(self):
        for missing in ['initial', 'early']:
            records = [record(0, x_offset=-1200, path=2400, initial=8, late=0, arrival=True),
                       record(2, x_offset=-2400, path=3600, initial=8, late=0, arrival=True),
                       record(4, initial=8, late=0, arrival=True)]
            for row in records[:2]:
                row['maritime_policy']['initial_sog_kn' if missing == 'initial' else 'early_path_length_m'] = 0
            self.assertFalse(annotate_recurrence(records)['window-4']['repeated'])

    def test_different_speed_profile_or_remote_endpoint_does_not_match(self):
        cases = [dict(initial=14, late=14), dict(y_offset=600)]
        for kwargs in cases:
            result = annotate_recurrence([record(0, **kwargs), record(2, **kwargs), record(4)])
            self.assertFalse(result['window-4']['repeated'])

    def test_jump_or_low_confidence_cannot_supply_continuous_reference_shape(self):
        records = [record(0), record(2), record(4)]
        for row in records[:2]:
            row['gate_weights'] = [1.] * 6 + [.1] * 8 + [1.] * 6
            for i in range(14, 20):
                row['latlon'][i][1] += .1
        self.assertFalse(annotate_recurrence(records)['window-4']['repeated'])

    def test_missing_observations_are_not_filled_or_trusted(self):
        for key in ['gate_weights', 'sog_kn', 'time_seconds', 'latlon']:
            records = [record(0), record(2), record(4)]
            for row in records[:2]:
                row.pop(key)
            self.assertFalse(annotate_recurrence(records)['window-4']['repeated'])

    def test_duplicate_ids_cannot_count_twice(self):
        result = annotate_recurrence([record(0), record(2, rid='window-0'), record(4)])
        self.assertFalse(result['window-4']['repeated'])

    def test_overlapping_future_coverage_is_not_a_past_reference(self):
        records = [record(0), record(2), record(4)]
        records[0]['end_utc'] = records[2]['end_utc']
        self.assertFalse(annotate_recurrence(records)['window-4']['repeated'])


if __name__ == '__main__':
    unittest.main()
