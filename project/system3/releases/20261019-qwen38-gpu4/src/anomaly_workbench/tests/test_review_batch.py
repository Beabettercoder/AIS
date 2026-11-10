"""Batch qualification must preserve facts while reducing review work."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from anomaly_workbench import review_batch as batch


BASE = datetime(2024, 7, 6, tzinfo=timezone.utc)


def utc(seconds):
    return (BASE + timedelta(seconds=seconds)).isoformat().replace('+00:00', 'Z')


def row(vessel, index, kind='speed_drop', score=.85, motion=True, **updates):
    value = {
        'id': f'{vessel}-{index}', 'vessel_id': vessel, 'window_index': index,
        'start_utc': utc(index * 1200), 'end_utc': utc(index * 1200 + 1140),
        'window_start_utc': utc(index * 1200), 'window_end_utc': utc((index + 1) * 1200),
        'is_candidate': motion, 'type': kind, 'type_label': 'suspected-' + kind,
        'tier': 'Tier-I', 'score': score, 'model_raw_score': None,
        'why_flagged': 'original physical evidence',
        'latlon': [[30., -88. + index * .001], [30., -88. + (index + 1) * .001]],
        'sog_kn': [5., 5.], 'cog_deg': [90., 90.], 'time_seconds': [0., 1140.],
        'gate_weights': [1., 1.], 'mean_gate_weight': 1.,
        'maritime_policy': {'stationary': False, 'continuous_path_length_m': 700., 'source_sog_max_kn': 5.},
        'field_residual_mps': None, 'e_kin_mps': .1, 'has_quality_issue': False,
    }
    value.update(updates)
    return value


PRIORITY_IDS = {'A-0', 'A-2', 'B-1', 'E-0', 'E-2'}
REPEATED_IDS = {'A-1', 'B-0', 'B-1', 'D-0', 'D-1'}


def recurrence(records):
    return {
        item['id']: {'repeated': item['id'] in REPEATED_IDS,
                     'reference_ids': ['past-independent-1', 'past-independent-2'] if item['id'] in REPEATED_IDS else [],
                     'basis': 'past_observations_only'}
        for item in records
    }


def assessment(item):
    return {'eligible': item['id'] in PRIORITY_IDS or item['id'] in REPEATED_IDS,
            'codes': ['confirmed_group_sustained_difference'] if item['id'] == 'B-1' else ['substantial_self_change'],
            'reasons': ['independent sustained group evidence' if item['id'] == 'B-1' else 'self motion evidence']}


class ReviewBatchTests(unittest.TestCase):
    def test_plausible_normal_maneuver_requires_independent_context_for_priority(self):
        normal={'observed_action':'slow maneuver','possible_intents':['unconfirmed operation'],
                'evidence':{'normal_maneuver_pattern':True},'normal_operation_not_excluded':True}
        with patch.object(batch,'assess_maneuver_intent',return_value=normal):
            weak=batch.qualify_records([row('A',0)])[0]
            supported=batch.qualify_records([row('B',1)])[0]
        self.assertFalse(weak['is_candidate'])
        self.assertTrue(weak['motion_candidate'])
        self.assertEqual(weak['intent_assessment'],normal)
        self.assertTrue(supported['is_candidate'])

    def test_intent_receives_already_computed_historical_context(self):
        seen=[]
        def assess(record):
            seen.append(record.get('recurrence_evidence'))
            return {'evidence':{'normal_maneuver_pattern':False}}
        with patch.object(batch,'assess_maneuver_intent',side_effect=assess):
            batch.qualify_records([row('A',1)])
        self.assertTrue(seen[0]['repeated'])

    def test_specific_review_type_changes_tier_but_keeps_original_motion_type(self):
        record = row('A', 0, kind='formation_departure', tier='Tier-II')
        qualified = {'eligible': True, 'review_type': 'stall', 'codes': ['sustained_rapid_deceleration'], 'reasons': ['actual stable speed change']}
        with patch.object(batch, 'assess_review_evidence', return_value=qualified):
            result = batch.qualify_records([record])[0]
        self.assertEqual(result['type'], 'stall')
        self.assertEqual(result['tier'], 'Tier-I')
        self.assertEqual(result['motion_type'], 'formation_departure')
        self.assertEqual(result['motion_tier'], 'Tier-II')

    def test_historical_dwell_context_deprioritizes_only_plain_deceleration(self):
        examples = [row('A', 0), row('A', 2, kind='turn_back')]
        contexts = {r['id']: {'matches': True, 'reference_mmsi': ['past-a', 'past-b', 'past-c']} for r in examples}
        def qualification(r):
            code = 'sustained_rapid_deceleration' if r['id'] == 'A-0' else 'stable_direction_reversal'
            return {'eligible': True, 'codes': [code], 'reasons': ['physical evidence'], 'evidence': {'local_corroboration': True}}
        with patch.object(batch, 'annotate_stopping_context', return_value=contexts), patch.object(batch, 'assess_review_evidence', side_effect=qualification):
            result = batch.qualify_records(examples)
        self.assertFalse(result[0]['is_candidate'])
        self.assertEqual(result[0]['review_level'], 'clue')
        self.assertTrue(result[1]['is_candidate'])
        self.assertEqual(result[0]['stopping_context']['reference_mmsi'], ['past-a', 'past-b', 'past-c'])

    def test_reprocessing_preserves_first_stage_motion_evidence(self):
        rows = [row('A', 1, kind='stall')]
        batch.qualify_records(rows)
        self.assertFalse(rows[0]['is_candidate'])
        batch.qualify_records(rows)
        self.assertTrue(rows[0]['motion_candidate'])
        self.assertEqual(rows[0]['why_flagged'].count('重点核查资格：'), 1)

    def setUp(self):
        self.recurrence_patch = patch.object(batch, 'annotate_recurrence', side_effect=recurrence)
        self.assessment_patch = patch.object(batch, 'assess_review_evidence', side_effect=assessment)
        self.annotate = self.recurrence_patch.start()
        self.assess = self.assessment_patch.start()
        self.addCleanup(self.recurrence_patch.stop)
        self.addCleanup(self.assessment_patch.stop)

    def test_qualification_preserves_scores_and_physical_facts(self):
        records = [row('A', 0, 'turn_back', .84), row('A', 1, 'stall', .97)]
        original = deepcopy(records)
        result = batch.qualify_records(records)
        self.assertIs(result, records)
        for before, after in zip(original, result):
            for key in ['score', 'model_raw_score', 'latlon', 'sog_kn', 'cog_deg', 'gate_weights',
                        'maritime_policy', 'field_residual_mps', 'e_kin_mps', 'start_utc', 'end_utc', 'type']:
                self.assertEqual(after[key], before[key], key)
            self.assertTrue(after['why_flagged'].startswith(before['why_flagged']))
            self.assertEqual(after['original_motion_score'], before['score'])
        self.assertTrue(result[0]['is_candidate'])
        self.assertFalse(result[1]['is_candidate'])
        self.assertEqual(result[1]['review_level'], 'clue')
        self.assertEqual(result[1]['selection_role'], 'motion_clue')
        self.assertEqual(result[1]['recurrence_evidence']['reference_ids'], ['past-independent-1', 'past-independent-2'])
        self.assertIsNone(result[1]['review_priority_score'])

    def test_repeated_group_evidence_is_retained_but_generic_group_word_is_not_enough(self):
        group, repeated_self = row('B', 1, 'formation_departure', .82), row('B', 0, score=.99)
        result = batch.qualify_records([group, repeated_self])
        self.assertTrue(result[0]['is_candidate'])
        self.assertTrue(result[0]['review_assessment']['eligible'])
        self.assertEqual(result[0]['review_priority_score'], .82)
        self.assertFalse(result[1]['is_candidate'])
        with patch.object(batch, 'assess_review_evidence', return_value={
            'eligible': True, 'codes': ['unconfirmed_group_hint'], 'reasons': ['insufficient group evidence']
        }):
            result = batch.qualify_records([row('B', 1)])
        self.assertFalse(result[0]['is_candidate'])

    def test_non_motion_reference_is_never_promoted_or_assessed(self):
        reference = row('C', 0, kind='background', score=.3, motion=False)
        self.assess.side_effect = AssertionError('background reference must not be qualified as an event')
        result = batch.qualify_records([reference])
        self.assertFalse(result[0]['motion_candidate'])
        self.assertFalse(result[0]['is_candidate'])
        self.assertEqual(result[0]['review_level'], 'background')
        self.assertNotIn('review_priority_score', result[0])
        self.annotate.assert_called_once()
        self.assess.assert_not_called()

    def _saved_fixture(self, directory):
        rows = [
            [row('A', 0, 'turn_back', .84), row('B', 0, score=.99), row('D', 0, score=.96),
             row('E', 0, score=.83), row('C', 0, kind='background', score=.3, motion=False),
             row('S', 0, kind='stationary', score=0., motion=False, has_quality_issue=True, e_kin_mps=9.,
                 maritime_policy={'stationary': True, 'continuous_path_length_m': 0.})],
            [row('A', 1, 'stall', .97, has_quality_issue=True, e_kin_mps=2.),
             row('B', 1, 'formation_departure', .81), row('D', 1, score=.95)],
            [row('A', 2, 'deviation', .86), row('E', 2, score=.89)],
        ]
        timeline = [{'index': index, 'start_utc': utc(index * 1200), 'end_utc': utc((index + 1) * 1200),
                     'macro_label': 'no regional congestion', 'candidate_count': 99, 'point_count': len(records) * 2}
                    for index, records in enumerate(rows)]
        folder = Path(directory)
        for index, records in enumerate(rows):
            with gzip.open(folder / f'window-{index}.json.gz', 'wt', encoding='utf-8') as stream:
                json.dump({'records': records, 'summary': deepcopy(timeline[index])}, stream)
        return folder, timeline, {item['id']: deepcopy(item) for records in rows for item in records}

    def test_saved_windows_scatter_episodes_and_counts_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, timeline, before = self._saved_fixture(directory)
            result = batch.refine_saved_windows(folder, timeline, top_k=10)
            saved = []
            for index in range(3):
                with gzip.open(folder / f'window-{index}.json.gz', 'rt', encoding='utf-8') as stream:
                    window = json.load(stream)
                self.assertEqual(window['summary'], timeline[index])
                self.assertEqual(window['summary']['candidate_count'], [2, 1, 2][index])
                self.assertEqual(window['summary']['motion_clue_count'], [4, 3, 2][index])
                saved.extend(window['records'])
            self.assertEqual({item['id'] for item in saved}, set(before))
            by_id = {item['id']: item for item in saved}
            scatter = {item['id']: item for item in result['scatter']}
            self.assertEqual(set(scatter), set(before))
            for rid, original in before.items():
                item = by_id[rid]
                self.assertEqual(item['score'], original['score'])
                self.assertEqual(item['maritime_policy'], original['maritime_policy'])
                self.assertEqual(item['latlon'], original['latlon'])
                self.assertEqual(item['policy_version'], batch.POLICY_VERSION)
                for key in ['score', 'is_candidate', 'motion_candidate', 'review_level', 'maritime_policy']:
                    self.assertEqual(scatter[rid][key], item[key], (rid, key))
            self.assertEqual({item['id'] for item in saved if item['is_candidate']}, PRIORITY_IDS)
            self.assertEqual(result['summary'], {
                'candidate_count': 4, 'candidate_windows': 5, 'candidate_vessel_count': 3,
                'motion_clue_windows': 9, 'motion_episode_count': 5, 'general_clue_event_count': 1,
                'repeated_pattern_windows': 5, 'historical_stopping_area_windows': 0, 'qualification_rejected_windows': 4,
                'ordinary_maneuver_windows': 0,
            })
            self.assertEqual(sum(row['candidate_count'] for row in timeline), result['summary']['candidate_windows'])
            self.assertEqual(result['type_counts'], {'turn_back': 1, 'speed_drop': 2, 'formation_departure': 1,
                                                   'deviation': 1, 'observation_glitch': 2, 'congestion': 0})
            self.assertEqual(sum(result['motion_type_counts'].values()), 9)
            self.assertEqual([r['id'] for r in result['quality_records']], ['S-0', 'A-1'])

    def test_priority_queue_never_uses_clues_to_fill_topk_and_keeps_all_window_links(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, timeline, _ = self._saved_fixture(directory)
            result = batch.refine_saved_windows(folder, timeline, top_k=10)
        self.assertEqual([item['vessel_id'] for item in result['records']], ['E', 'A', 'B'])
        self.assertEqual(len(result['records']), 3)  # Does not pad the queue to ten.
        self.assertTrue(all(item['is_candidate'] and item['review_level'] == 'priority' for item in result['records']))
        self.assertEqual({item['vessel_id'] for item in result['clue_records']}, {'D'})
        motion_ids = {rid for event in result['motion_events'] for rid in event['event_window_ids']}
        self.assertEqual(motion_ids, {'A-0', 'A-1', 'A-2', 'B-0', 'B-1', 'D-0', 'D-1', 'E-0', 'E-2'})
        priority_ids = {rid for event in result['candidate_events'] for rid in event['priority_window_ids']}
        self.assertEqual(priority_ids, PRIORITY_IDS)
        a = next(item for item in result['records'] if item['vessel_id'] == 'A')
        self.assertEqual(a['id'], 'A-2')
        self.assertEqual(a['score'], .86)  # Repeated clue A-1 has higher score .97.
        self.assertEqual(a['event_window_ids'], ['A-0', 'A-1', 'A-2'])
        self.assertEqual(a['event_types'], ['turn_back', 'stall', 'deviation'])
        e = next(item for item in result['records'] if item['vessel_id'] == 'E')
        self.assertEqual(e['related_event_count'], 2)
        self.assertEqual({event['id'] for event in e['related_events']}, {'E-0', 'E-2'})

    def test_topk_limits_vessels_without_losing_full_event_index_or_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, timeline, _ = self._saved_fixture(directory)
            result = batch.refine_saved_windows(folder, timeline, top_k=1)
        self.assertEqual(len(result['records']), 1)
        self.assertEqual(result['records'][0]['vessel_id'], 'E')
        self.assertEqual(result['summary']['candidate_vessel_count'], 3)
        self.assertEqual(len(result['candidate_events']), 4)
        self.assertEqual(len(result['scatter']), 11)


if __name__ == '__main__':
    unittest.main()
