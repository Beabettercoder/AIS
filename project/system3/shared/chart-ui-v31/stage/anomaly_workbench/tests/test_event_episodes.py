from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import unittest

from anomaly_workbench.event_episodes import merge_motion_episodes


BASE = datetime(2024, 7, 6, tzinfo=timezone.utc)


def utc(seconds):
    return (BASE + timedelta(seconds=seconds)).isoformat().replace('+00:00', 'Z')


def row(index, kind='speed_drop', priority=False, score=.85, vessel='123', **changes):
    # Adjacent windows have a 60 s actual-report gap and contiguous geometry.
    value = {
        'id': f'{vessel}-{index}', 'vessel_id': vessel, 'window_index': index,
        'start_utc': utc(index * 1200), 'end_utc': utc(index * 1200 + 1140),
        'type': kind, 'type_label': 'suspected-' + kind,
        'latlon': [[30., -88. + index * .001], [30., -88. + (index + 1) * .001]],
        'sog_kn': [5., 5.], 'gate_weights': [1., 1.],
        'score': score, 'motion_candidate': True, 'is_candidate': priority,
        'review_assessment': {'qualified': priority},
        'recurrence_evidence': {'matches': 2},
    }
    value.update(changes)
    return value


class MotionEpisodeTests(unittest.TestCase):
    def test_contiguous_different_labels_make_one_episode_with_all_evidence(self):
        first, second = row(0, 'turn_back'), row(1, 'stall')
        result = merge_motion_episodes([second, first])
        self.assertEqual(len(result), 1)
        episode = result[0]
        self.assertEqual(episode['event_window_ids'], ['123-0', '123-1'])
        self.assertEqual(episode['event_types'], ['turn_back', 'stall'])
        self.assertEqual(episode['event_type_labels'], ['suspected-turn_back', 'suspected-stall'])
        self.assertEqual(episode['event_start_utc'], first['start_utc'])
        self.assertEqual(episode['event_end_utc'], second['end_utc'])
        self.assertEqual(episode['review_level'], 'clue')
        self.assertFalse(episode['is_candidate'])
        self.assertEqual(episode['priority_window_ids'], [])

    def test_priority_representative_wins_even_with_lower_original_score(self):
        first, second = row(0, score=.95), row(1, priority=True, score=.82)
        third = row(2, priority=True, score=.88)
        episode = merge_motion_episodes([first, second, third])[0]
        self.assertEqual(episode['id'], third['id'])
        self.assertEqual(episode['score'], .88)
        self.assertEqual(episode['priority_window_ids'], [second['id'], third['id']])
        self.assertEqual(len(episode['event_window_ids']), 3)
        self.assertEqual(episode['review_level'], 'priority')
        self.assertEqual(episode['review_assessment'], third['review_assessment'])
        self.assertEqual(episode['recurrence_evidence'], third['recurrence_evidence'])

    def test_stale_event_fields_are_rebuilt_and_inputs_stay_unchanged(self):
        first = row(0, event_start_utc='wrong', event_window_ids=['wrong'],
                    event_types=['wrong'], event_timeline={'wrong': 1},
                    related_events=[{'wrong': 1}], related_event_count=99)
        second = row(1, score=.9)
        before = deepcopy([first, second])
        result = merge_motion_episodes([first, second])[0]
        self.assertEqual([first, second], before)
        self.assertNotIn('related_events', result)
        self.assertNotIn('event_timeline', result)
        self.assertEqual(result['event_start_utc'], first['start_utc'])
        result['recurrence_evidence']['matches'] = 100
        self.assertEqual(second['recurrence_evidence']['matches'], 2)

    def test_different_vessels_and_missing_vessel_do_not_merge(self):
        self.assertEqual(len(merge_motion_episodes([row(0), row(1, vessel='456')])), 2)
        self.assertEqual(len(merge_motion_episodes([row(0, vessel_id=None), row(1, vessel_id=None)])), 2)

    def test_skipped_or_duplicate_window_is_not_a_continuous_episode(self):
        self.assertEqual(len(merge_motion_episodes([row(0), row(2)])), 2)
        self.assertEqual(len(merge_motion_episodes([row(0), row(0)])), 2)

    def test_different_voyages_or_overlapping_report_times_do_not_merge(self):
        self.assertEqual(len(merge_motion_episodes([row(0), row(1, start_utc=utc(1900))])), 2)
        self.assertEqual(len(merge_motion_episodes([row(0), row(1, start_utc=utc(1100))])), 2)

    def test_report_gap_600_seconds_is_inclusive_but_larger_gap_is_not(self):
        first = row(0, end_utc=utc(600))
        self.assertEqual(len(merge_motion_episodes([first, row(1)])), 1)
        self.assertEqual(len(merge_motion_episodes([row(0, end_utc=utc(599)), row(1)])), 2)

    def test_reachable_distance_uses_both_trustworthy_endpoint_speeds(self):
        first = row(0)
        # Roughly 480 m apart: 5 kn * 60 s + 300 m permits only about 454 m.
        second = row(1, latlon=[[30., -87.994], [30., -87.993]])
        self.assertEqual(len(merge_motion_episodes([first, second])), 2)
        second['sog_kn'] = [10., 10.]
        self.assertEqual(len(merge_motion_episodes([first, second])), 1)

    def test_hard_spatial_limit_blocks_jump_even_with_extreme_reported_speed(self):
        first = row(0, sog_kn=[100., 100.], end_utc=utc(600))
        second = row(1, sog_kn=[100., 100.], latlon=[[30., -87.8], [30., -87.799]])
        self.assertEqual(len(merge_motion_episodes([first, second])), 2)

    def test_missing_bad_or_low_confidence_actual_endpoint_does_not_merge(self):
        cases = [None, [], [[30, -88], [math.nan, -88]], [[30, -88], [91, -88]]]
        for track in cases:
            with self.subTest(track=track):
                first = row(0, latlon=track, repaired_latlon=[[30, -88], [30, -87.999]])
                self.assertEqual(len(merge_motion_episodes([first, row(1)])), 2)
        self.assertEqual(len(merge_motion_episodes([row(0, gate_weights=[1, .29]), row(1)])), 2)

    def test_missing_speed_does_not_create_reachable_motion_allowance(self):
        first = row(0, sog_kn=None)
        second = row(1, latlon=[[30., -87.995], [30., -87.994]])
        self.assertEqual(len(merge_motion_episodes([first, second])), 2)
        self.assertEqual(len(merge_motion_episodes([first, row(1)])), 1)

    def test_missing_or_invalid_time_and_window_indices_never_join(self):
        for changes in [{'start_utc': None}, {'start_utc': '2024-07-06T00:20:00'},
                        {'end_utc': utc(1100)}, {'window_index': None}, {'window_index': True}]:
            with self.subTest(changes=changes):
                self.assertEqual(len(merge_motion_episodes([row(0), row(1, **changes)])), 2)

    def test_empty_input_and_single_record_preserve_original_score(self):
        self.assertEqual(merge_motion_episodes([]), [])
        value = row(0, priority=True, score=.834)
        episode = merge_motion_episodes([value])[0]
        self.assertEqual(episode['score'], .834)
        self.assertEqual(episode['event_window_ids'], [value['id']])
        self.assertEqual(episode['priority_window_ids'], [value['id']])


if __name__ == '__main__':
    unittest.main()
