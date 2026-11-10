from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from anomaly_workbench import attach
from anomaly_workbench import live_service as service
from anomaly_workbench.tests.test_sqlite_pipeline import make_db
from anomaly_workbench.reviews import ReviewStore
import hashlib
import json,gzip
from types import SimpleNamespace

@pytest.fixture
def client(tmp_path,monkeypatch):
    db=tmp_path/'host.sqlite';make_db(db);monkeypatch.setenv('ANOMALY_SQLITE_PATH',str(db))
    monkeypatch.setattr(service,'RUNTIME',tmp_path/'runtime');service.JOBS.clear()
    monkeypatch.setattr(service,'STORE',ReviewStore(tmp_path/'runtime/reviews.sqlite3'))
    app=FastAPI()
    @app.get('/health')
    def health():return {'status':'original-ok'}
    @app.get('/data-management')
    def management():
        from fastapi.responses import HTMLResponse
        return HTMLResponse('<html><body><p id="original">original catalog</p></body></html>')
    attach(app)
    return TestClient(app)

def test_catalog_reads_registered_sqlite_and_metadata(client):
    data=client.get('/anomaly-workbench/api/catalog').json();assert data['datasets'][0]['name']=='danmai'
    entry=client.get('/anomaly-workbench/api/datasets/danmai').json()
    assert entry['row_count']==120 and entry['unique_mmsi']==6
    assert entry['bbox'][0]>=11

def test_removed_gallery_and_offline_routes_are_not_available(client):
    assert client.get('/anomaly-workbench/api/experiments').status_code==404
    assert client.get('/anomaly-workbench/static/workbench.js').status_code==404
    page=client.get('/anomaly-workbench?dataset=danmai')
    assert page.status_code==200 and '跨海域泛化与模型评测舱' not in page.text
    assert 'fig7' not in page.text and 'id="dataset"' in page.text

def test_data_management_hook_preserves_original_page(client):
    response=client.get('/data-management')
    assert '<p id="original">original catalog</p>' in response.text
    assert response.text.count('catalog-entry.js')==1
    assert client.get('/health').json()=={'status':'original-ok'}

def test_cross_origin_and_unknown_dataset_rejected(client):
    assert client.post('/anomaly-workbench/api/jobs',json={'dataset_id':'danmai'},headers={'Origin':'https://other.example'}).status_code==403
    assert client.post('/anomaly-workbench/api/jobs',json={'dataset_id':'houston_observed'}).status_code==422
    assert client.post('/anomaly-workbench/api/jobs',json={'dataset_id':'danmai','far_budget':.99}).status_code==422

def test_failed_worker_does_not_claim_completion_or_break_host(client,monkeypatch):
    monkeypatch.setattr(service,'python_worker',lambda:'configured-python')
    monkeypatch.setattr(service,'run_job',lambda job:job.update(status='error',message='worker unavailable'))
    response=client.post('/anomaly-workbench/api/jobs',json={'dataset_id':'danmai'})
    assert response.status_code==200;identifier=response.json()['id']
    assert response.json()['params']['neighbor_revision']==3
    assert response.json()['params']['exclude_stationary'] is True
    assert client.get('/anomaly-workbench/api/jobs/'+identifier).json()['status']=='error'
    assert client.get('/anomaly-workbench/api/jobs/'+identifier+'/result').status_code==409
    assert client.get('/health').status_code==200

def test_source_directory_traversal_and_future_unknown_job(client):
    assert client.get('/anomaly-workbench/static/engine.py').status_code==404
    assert client.get('/anomaly-workbench/api/jobs/unknown').status_code==404

def mock_evidence(monkeypatch):
    scan={'dataset':{'id':'r1','name':'danmai'},'params':{'dataset_id':'r1'},'calibration':{'threshold':.95},'summary':{}}
    item={'id':'AIS-219000000-1723767600','vessel_id':'219000000','start_utc':'2024-08-16T00:00:00Z','end_utc':'2024-08-16T00:20:00Z','score':.97,'type_label':'疑似减速','selection_role':'candidate','review_history':[]}
    monkeypatch.setattr(service,'record',lambda *a,**k:(scan,item))
    return scan,item

def test_review_archiving_is_isolated_and_source_unchanged(client,monkeypatch):
    mock_evidence(monkeypatch);db=service.sqlite_stream.settings_path();before=hashlib.sha256(db.read_bytes()).hexdigest()
    response=client.post('/anomaly-workbench/api/reviews',json={'scan_id':'unit-scan','record_id':'AIS-219000000-1723767600','status':'normal','elapsed_seconds':8.5,'note':'fixture review'})
    assert response.status_code==200
    assert response.json()['history'][0]['status']=='normal'
    assert service.STORE.path.is_file() and service.STORE.path!=db
    assert hashlib.sha256(db.read_bytes()).hexdigest()==before

def test_evidence_export_contains_provenance_and_digest(client,monkeypatch):
    mock_evidence(monkeypatch)
    response=client.get('/anomaly-workbench/api/scans/unit/records/AIS-219000000-1723767600/export?format=json')
    assert response.status_code==200
    value=response.json();assert len(value['sha256'])==64
    assert value['source']=='system3 SQLite live query' and value['record']['vessel_id']=='219000000'
    assert 'attachment' in response.headers['content-disposition']
    md=client.get('/anomaly-workbench/api/scans/unit/records/AIS-219000000-1723767600/export?format=md')
    assert md.status_code==200 and '人工复核' in md.text

def test_completed_scan_can_be_restored_after_service_restart(client):
    identifier='a'*32;folder=service.RUNTIME/'jobs'/identifier;folder.mkdir(parents=True)
    stored={'params':{'dataset_id':'r1'},'dataset':{'id':'r1','name':'danmai'},'summary':{'window_count':1},
            'timeline':[{'start_utc':'2024-08-16T00:00:00Z'}],'records':[],'record_index':{}}
    with gzip.open(folder/'result.json.gz','wt',encoding='utf-8') as f:json.dump(stored,f)
    service.JOBS.clear()
    response=client.get('/anomaly-workbench/api/jobs/'+identifier)
    assert response.status_code==200 and response.json()['status']=='complete'
    restored=client.get('/anomaly-workbench/api/jobs/'+identifier+'/result').json()
    assert restored['scan_id']==identifier and restored['dataset']['name']=='danmai'
    assert restored['needs_rescan'] is True and restored['records']==[]
    assert client.get('/anomaly-workbench/api/jobs/'+'q'*32).status_code==404

def test_snapshot_verified_diagnostics_are_cached_without_replacing_geometry(client,monkeypatch):
    identifier='b'*32;rid='AIS-219000000-1723767600';folder=service.RUNTIME/'jobs'/identifier;folder.mkdir(parents=True)
    item={'id':rid,'vessel_id':'219000000','window_start_utc':'2024-08-16T00:00:00Z','window_end_utc':'2024-08-16T00:20:00Z',
          'motion_observations':{'sog_kn':[8.]},'neighbor_context_version':2,
          'neighbor_frames':[{'count':1,'neighbors':[{'mmsi':'219000001'}]}]}
    scan={'params':{'dataset_id':'r1'},'dataset':{'id':'r1'},'summary':{},'timeline':[{}],'records':[item],'record_index':{rid:0}}
    with gzip.open(folder/'result.json.gz','wt',encoding='utf-8') as f:json.dump(scan,f)
    with gzip.open(folder/'window-0.json.gz','wt',encoding='utf-8') as f:json.dump({'records':[item]},f)
    calls=[]
    def run(*a,**kwargs):
        calls.append(1)
        return SimpleNamespace(stdout=json.dumps({'event':'detail','diagnostics':{'pg_probability':[.2]},
            'evidence_signature':service.evidence_signature(item['motion_observations'],item['neighbor_frames']),
            'geometry':{'neighbor_frames':[{'neighbors':[{'mmsi':'should-not-replace-snapshot'}]}]}}),returncode=0)
    monkeypatch.setattr(service,'python_worker',lambda:'test-worker');monkeypatch.setattr(service.subprocess,'run',run)
    url=f'/anomaly-workbench/api/scans/{identifier}/records/{rid}'
    assert client.get(url).json()['neighbor_frames'][0]['count']==1
    assert client.get(url).status_code==200 and len(calls)==1


def test_changed_or_unavailable_model_keeps_saved_evidence_visible(client,monkeypatch):
    identifier='d'*32;rid='saved-record';folder=service.RUNTIME/'jobs'/identifier;folder.mkdir(parents=True)
    item={'id':rid,'vessel_id':'ship','window_start_utc':'2024-07-06T00:00:00Z','window_end_utc':'2024-07-06T00:20:00Z',
          'motion_observations':{'sog_kn':[8.]},'neighbor_frames':[], 'latlon':[[30,-88]],'sog_kn':[8.],'gate_weights':[1.]}
    scan={'params':{},'dataset':{'id':'source'},'summary':{},'timeline':[{}],'records':[], 'record_index':{rid:0}}
    with gzip.open(folder/'result.json.gz','wt') as f:json.dump(scan,f)
    with gzip.open(folder/'window-0.json.gz','wt') as f:json.dump({'records':[item]},f)
    monkeypatch.setattr(service,'python_worker',lambda:'worker')
    monkeypatch.setattr(service.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=json.dumps({'event':'detail',
        'evidence_signature':'changed-source','diagnostics':{'pg_probability':[1.]},'geometry':{'latlon':[[0,0]]}})))
    url=f'/anomaly-workbench/api/scans/{identifier}/records/{rid}'
    response=client.get(url);assert response.status_code==200
    actual=response.json();assert actual['latlon']==item['latlon'] and actual['sog_kn']==[8.]
    assert actual['diagnostics']['pg_probability']==[] and actual['diagnostics']['source']=='saved_observations_only'
    def timeout(*a,**k):raise service.subprocess.TimeoutExpired('worker',120)
    monkeypatch.setattr(service.subprocess,'run',timeout)
    assert client.get(url).json()['latlon']==item['latlon']


def test_chart_explanations_use_server_evidence_and_enforce_origin(client,monkeypatch):
    scan,item=mock_evidence(monkeypatch)
    scan['policy_version']=5
    calls=[]
    def explain(actual_scan,actual_item):
        calls.append((actual_scan,actual_item))
        return {'source':'local_llm','explanations':{'motion':'用航速变化定位减速发生的时段。'}}
    monkeypatch.setattr(service,'generate_chart_explanations',explain)
    url='/anomaly-workbench/api/scans/unit/records/'+item['id']+'/chart-explanations'
    assert client.post(url,headers={'Origin':'https://other.example'}).status_code==403
    assert not calls
    response=client.post(url)
    assert response.status_code==200 and response.json()['source']=='local_llm'
    assert calls==[(scan,item)]


def test_chart_explanations_reject_previous_policy_results(client,monkeypatch):
    _,item=mock_evidence(monkeypatch)
    url='/anomaly-workbench/api/scans/unit/records/'+item['id']+'/chart-explanations'
    assert client.post(url).status_code==409


def test_unlisted_episode_member_keeps_complete_evidence_and_own_qualification(client):
    identifier='c'*32;folder=service.RUNTIME/'jobs'/identifier;folder.mkdir(parents=True)
    left={'id':'motion-left','vessel_id':'ship','is_candidate':False,'review_level':'clue'}
    right={'id':'motion-right','vessel_id':'ship','is_candidate':True,'review_level':'priority'}
    episode={'id':'motion-right','review_level':'priority','event_window_ids':['motion-left','motion-right'],
             'priority_window_ids':['motion-right'],'event_start_utc':'2024-07-06T00:00:00Z','event_end_utc':'2024-07-06T00:40:00Z'}
    stored={'params':{},'dataset':{'id':'dataset'},'summary':{},'timeline':[{},{}],
            'records':[],'clue_records':[],'motion_events':[episode],'record_index':{'motion-left':0,'motion-right':1}}
    with gzip.open(folder/'result.json.gz','wt') as f:json.dump(stored,f)
    for index,row in enumerate([left,right]):
        with gzip.open(folder/f'window-{index}.json.gz','wt') as f:json.dump({'records':[row]},f)
    _,actual=service.record(identifier,'motion-left')
    assert actual['event_window_ids']==['motion-left','motion-right']
    assert actual['priority_window_ids']==['motion-right']
    assert actual['episode_review_level']=='priority'
    assert actual['review_level']=='clue' and actual['is_candidate'] is False


def test_queue_endpoint_uses_whole_scan_and_rejects_invalid_window(client,monkeypatch):
    from anomaly_workbench.tests.test_queue_views import fixture
    scan=fixture();scan.update(policy_version=5,timeline=[{}])
    monkeypatch.setattr(service,'result',lambda value:({'_folder':None},scan))
    url='/anomaly-workbench/api/scans/unit/queue'
    result=client.get(url,params={'type':'speed_drop','limit':1})
    assert result.status_code==200 and result.json()['records'][0]['id']=='b0'
    assert result.json()['window_index'] is None
    assert client.get(url,params={'window_index':1}).status_code==404
    assert client.get(url,params={'limit':201}).status_code==422
    assert client.get(url,params={'kind':'unknown'}).status_code==422
