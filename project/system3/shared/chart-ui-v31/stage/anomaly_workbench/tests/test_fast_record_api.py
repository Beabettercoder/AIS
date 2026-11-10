"""Saved AIS evidence must render independently of neural diagnostics."""
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import pytest

from anomaly_workbench import live_service as service
from anomaly_workbench.tests.test_live_api import client  # Shared isolated host DB/runtime fixture.


SCAN_ID = 'f' * 32
FIRST_ID = 'AIS-338416034-1720225200'
SECOND_ID = 'AIS-367006330-1720225200'


def saved_record(record_id, vessel, latitude):
    positions = [[latitude, -94.93], [latitude + .001, -94.929], [latitude + .0001, -94.9299]]
    frames = [{'step': index, 'time_epoch': 1720225200 + index * 60,
               'time_utc': f'2024-07-06T00:0{index}:00Z', 'field_valid': False, 'count': 1,
               'neighbors': [{'mmsi': '900000001', 'position': [latitude + .0002, -94.9301],
                              'sog_kn': 5., 'cog_deg': 90., 'distance_nm': .05, 'weight': 1.}]}
              for index in range(3)]
    return {
        'id': record_id, 'window_id': record_id, 'vessel_id': vessel, 'window_index': 0,
        'dataset_id': 'r1', 'start_utc': '2024-07-06T00:00:00Z', 'end_utc': '2024-07-06T00:02:00Z',
        'window_start_utc': '2024-07-06T00:00:00Z', 'window_end_utc': '2024-07-06T00:20:00Z',
        'latlon': positions, 'repaired_latlon': deepcopy(positions), 'raw_xy': [[0, 0], [100, 100], [10, 10]],
        'sog_kn': [6., .1, .1], 'cog_deg': [90., 0., 0.], 'gate_weights': [1., 1., 1.],
        'time_seconds': [0., 60., 120.], 'field_vectors': [None, None, None],
        'residual_vectors': [None, None, None], 'neighbor_frames': frames,
        'neighbor_tracks': [], 'neighbor_context_version': 2, 'neighbor_revision': 3,
        'neighbor_track_catalog': [{'mmsi': '900000001', 'latlon': [[latitude, -94.9301]], 'timestamps': [1720225200]}],
        'neighbor_mean_sog': [None, None, None], 'quality_steps': [],
        'motion_observations': {'sog_kn': [6., .1, .1], 'cog_deg': [90., 0., 0.],
                                'time_seconds': [0., 60., 120.], 'xy_m': [[0, 0], [100, 100], [10, 10]],
                                'gate_weights': [1., 1., 1.], 'sog_reported_mask': [True, True, True],
                                'cog_reported_mask': [True, True, True]},
        'diagnostics': {'pg_probability': [], 'raw_probability': [], 'gate_weights': [1., 1., 1.],
                        'time_seconds': [0., 60., 120.], 'method': 'saved observations; model not requested'},
        'score': .83, 'is_candidate': False, 'motion_candidate': True, 'review_level': 'clue',
        'selection_role': 'motion_clue', 'type': 'turn_back', 'type_label': '一般回转线索',
        'why_flagged': 'Observed movement only; port-arrival intent is unconfirmed.',
        'maritime_policy': {'stationary': False, 'neighbor_supported': False}, 'policy_version': 5,
    }


@pytest.fixture
def completed_scan(client):
    records = [saved_record(FIRST_ID, '338416034', 29.55), saved_record(SECOND_ID, '367006330', 29.58)]
    scan = {
        'policy_version': 5, 'latest_policy_version': 5,
        'params': {'dataset_id': 'r1', 'policy_version': 5, 'neighbor_revision': 3},
        'dataset': {'id': 'r1', 'name': 'Houston-Beryl DS'}, 'summary': {'window_count': 1},
        'calibration': {}, 'timeline': [{'start_utc': '2024-07-06T00:00:00Z', 'end_utc': '2024-07-06T00:20:00Z'}],
        'records': [], 'clue_records': [], 'motion_events': [],
        'record_index': {item['id']: 0 for item in records},
    }
    folder = service.RUNTIME / 'jobs' / SCAN_ID
    folder.mkdir(parents=True)
    for filename, value in [('result.json.gz', scan), ('window-0.json.gz', {'records': records, 'summary': {}})]:
        with gzip.open(folder / filename, 'wt', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False)
    return {item['id']: deepcopy(item) for item in records}


def record_url(record_id, diagnostic=None):
    url = f'/anomaly-workbench/api/scans/{SCAN_ID}/records/{record_id}'
    return url if diagnostic is None else url + '?diagnostic=' + str(diagnostic).lower()


def forbid_worker(monkeypatch):
    def unavailable(*args, **kwargs):
        pytest.fail('fast saved-evidence read must not start or resolve a model worker')
    monkeypatch.setattr(service, 'python_worker', unavailable)
    monkeypatch.setattr(service.subprocess, 'run', unavailable)


def assert_same_observations(value, saved):
    for key in ['latlon', 'repaired_latlon', 'sog_kn', 'cog_deg', 'gate_weights', 'time_seconds',
                'neighbor_frames', 'neighbor_track_catalog', 'neighbor_tracks', 'field_vectors',
                'residual_vectors', 'motion_observations']:
        assert value[key] == saved[key], key


def test_fast_snapshot_returns_original_track_and_neighbors_without_model(client, completed_scan, monkeypatch):
    forbid_worker(monkeypatch)
    response = client.get(record_url(FIRST_ID, False))
    assert response.status_code == 200
    value = response.json()
    assert value['id'] == FIRST_ID
    assert_same_observations(value, completed_scan[FIRST_ID])
    assert value['diagnostics']['pg_probability'] == []
    assert value['diagnostics']['raw_probability'] == []


def test_two_fast_ship_reads_keep_their_own_saved_geometry(client, completed_scan, monkeypatch):
    forbid_worker(monkeypatch)
    responses = [client.get(record_url(record_id, False)) for record_id in [SECOND_ID, FIRST_ID, SECOND_ID]]
    assert all(response.status_code == 200 for response in responses)
    for response, rid in zip(responses, [SECOND_ID, FIRST_ID, SECOND_ID]):
        value = response.json()
        assert_same_observations(value, completed_scan[rid])
        assert value['vessel_id'] == completed_scan[rid]['vessel_id']
    assert responses[0].json()['latlon'] != responses[1].json()['latlon']


def test_boolean_query_is_forwarded_and_default_preserves_diagnostic_route(client, monkeypatch):
    calls = []
    def loaded(scan_id, record_id, diagnostic=False):
        calls.append((scan_id, record_id, diagnostic))
        return {'policy_version': 5}, {'id': record_id, 'diagnostics': {}}
    monkeypatch.setattr(service, 'record', loaded)
    assert client.get(record_url(FIRST_ID, False)).status_code == 200
    assert client.get(record_url(FIRST_ID, True)).status_code == 200
    assert client.get(record_url(FIRST_ID)).status_code == 200
    assert calls == [(SCAN_ID, FIRST_ID, False), (SCAN_ID, FIRST_ID, True), (SCAN_ID, FIRST_ID, True)]


def test_verified_diagnostics_are_reused_by_true_and_fast_reads_without_new_worker(client, completed_scan, monkeypatch):
    calls = []
    expected = completed_scan[FIRST_ID]
    diagnostic = {'pg_probability': [.1, .3, .7], 'raw_probability': [.2, .4, .8],
                  'gate_weights': [1., 1., 1.], 'time_seconds': [0., 60., 120.], 'method': 'verified test diagnostic'}
    signature = service.evidence_signature(expected['motion_observations'], expected['neighbor_frames'])
    monkeypatch.setattr(service, 'python_worker', lambda: 'fixture-python')
    def run(*args, **kwargs):
        calls.append(json.loads(kwargs['input']))
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            'event': 'detail', 'evidence_signature': signature, 'diagnostics': diagnostic,
            'geometry': {'latlon': [[0, 0]], 'neighbor_frames': []},
        }))
    monkeypatch.setattr(service.subprocess, 'run', run)
    first = client.get(record_url(FIRST_ID, True))
    assert first.status_code == 200
    assert first.json()['diagnostics'] == diagnostic
    assert_same_observations(first.json(), expected)
    assert len(calls) == 1 and calls[0]['record_id'] == FIRST_ID
    forbid_worker(monkeypatch)
    for request_diagnostic in [True, False]:
        response = client.get(record_url(FIRST_ID, request_diagnostic))
        assert response.status_code == 200
        assert response.json()['diagnostics'] == diagnostic
        assert_same_observations(response.json(), expected)
    # A verified diagnostic for the first vessel must not leak to the second.
    other = client.get(record_url(SECOND_ID, False))
    assert other.status_code == 200 and other.json()['diagnostics']['pg_probability'] == []


def test_mismatched_diagnostic_digest_cannot_replace_saved_evidence_or_seed_cache(client, completed_scan, monkeypatch):
    monkeypatch.setattr(service, 'python_worker', lambda: 'fixture-python')
    monkeypatch.setattr(service.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({'event': 'detail', 'evidence_signature': 'wrong-snapshot',
                                        'diagnostics': {'pg_probability': [.999]}})))
    response = client.get(record_url(FIRST_ID, True))
    assert response.status_code == 200
    assert response.json()['diagnostics']['pg_probability'] == []
    assert_same_observations(response.json(), completed_scan[FIRST_ID])
    assert FIRST_ID not in service.job_by_id(SCAN_ID).get('_diagnostics', {})
    forbid_worker(monkeypatch)
    fast = client.get(record_url(FIRST_ID, False))
    assert fast.status_code == 200 and fast.json()['diagnostics']['pg_probability'] == []


def test_worker_unavailable_does_not_make_saved_ship_unreadable(client, completed_scan, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError('test model worker unavailable')
    monkeypatch.setattr(service, 'python_worker', unavailable)
    assert client.get(record_url(FIRST_ID, False)).status_code == 200
    response = client.get(record_url(FIRST_ID, True))
    assert response.status_code == 200
    assert response.json()['diagnostics']['pg_probability'] == []
    assert_same_observations(response.json(), completed_scan[FIRST_ID])


def test_chart_explanations_load_saved_evidence_without_waiting_for_neural_diagnostic(client, completed_scan, monkeypatch):
    forbid_worker(monkeypatch)
    calls = []
    def explain(scan, item):
        calls.append((scan, deepcopy(item)))
        return {'source': 'template_fallback', 'explanations': {
            'motion': {'purpose': '查看当前运动变化。', 'reading': '保存轨迹可读，模型曲线尚未生成。'}
        }}
    monkeypatch.setattr(service, 'generate_chart_explanations', explain)
    url = record_url(FIRST_ID) + '/chart-explanations'
    response = client.post(url)
    assert response.status_code == 200
    assert len(calls) == 1 and calls[0][1]['diagnostics']['pg_probability'] == []
    assert_same_observations(calls[0][1], completed_scan[FIRST_ID])


def test_chart_endpoint_explicitly_requests_nondiagnostic_record(client, monkeypatch):
    calls = []
    def loaded(scan_id, record_id, diagnostic=False):
        calls.append(diagnostic)
        return {'policy_version': 5}, {'id': record_id, 'diagnostics': {}}
    monkeypatch.setattr(service, 'record', loaded)
    monkeypatch.setattr(service, 'generate_chart_explanations', lambda *args: {'source': 'template_fallback', 'explanations': {}})
    response = client.post(record_url(FIRST_ID) + '/chart-explanations')
    assert response.status_code == 200 and calls == [False]


def test_saved_observations_do_not_confirm_arrival_intent_without_context(client, completed_scan, monkeypatch):
    forbid_worker(monkeypatch)
    value = client.get(record_url(SECOND_ID, False)).json()
    assert 'intent_assessment' in value
    # The fixture contains no port, berth, route plan, or human confirmation.
    # Its weak movement sample must remain explicitly unconfirmed.
    description = json.dumps(value['intent_assessment'], ensure_ascii=False)
    assert '未确认' in description
