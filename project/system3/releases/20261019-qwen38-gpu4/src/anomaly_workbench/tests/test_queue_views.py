from copy import deepcopy
from anomaly_workbench.queue_views import build_queue


def fixture():
    def row(rid, vessel, kind, score, priority=True, **extra):
        return {'id':rid,'vessel_id':vessel,'type':kind,'score':score,'is_candidate':priority,
                'motion_candidate':True,'review_level':'priority' if priority else 'clue',**extra}
    rows=[row('a0','A','turn_back',.95),row('a1','A','speed_drop',.70),
          row('b0','B','speed_drop',.85),row('c0','C','deviation',.92,False),
          row('d0','D','background',0,False,motion_candidate=False,maritime_policy={'stationary':True})]
    events=[{'id':'a0','vessel_id':'A','review_level':'priority','event_window_ids':['a0','a1'],
             'priority_window_ids':['a0','a1'],'event_types':['turn_back','speed_drop']},
            {'id':'b0','vessel_id':'B','review_level':'priority','event_window_ids':['b0'],'priority_window_ids':['b0']},
            {'id':'c0','vessel_id':'C','review_level':'clue','event_window_ids':['c0'],'priority_window_ids':[]}]
    return {'scatter':rows,'motion_events':events,'records':[rows[0]]}


def test_type_filter_precedes_limit_and_selects_the_matching_member():
    scan=fixture()
    result=build_queue(scan,category='speed_drop',limit=1)
    assert [r['id'] for r in result['records']]==['b0']
    assert result['total_vessels']==2 and result['total_matching']==2
    result=build_queue(scan,category='speed_drop',limit=20)
    assert [r['id'] for r in result['records']]==['b0','a1']
    assert result['records'][1]['event_window_ids']==['a0','a1']


def test_queue_reads_all_evidence_not_saved_top_k_and_preserves_input():
    scan=fixture();before=deepcopy(scan)
    result=build_queue(scan,limit=200)
    assert len(result['records'])==2 and result['matching_unit']=='运动段'
    assert scan==before


def test_focus_cannot_bypass_queue_category_or_stationary_filter():
    scan=fixture()
    assert build_queue(scan,focus_record='c0')['focused_record'] is None
    assert build_queue(scan,category='speed_drop',focus_record='a0')['focused_record'] is None
    result=build_queue(scan,category='speed_drop',focus_record='a1',limit=1)
    assert result['records'][0]['id']=='a1' and result['total_vessels']==2
    assert build_queue(scan,kind='all',hide_stationary=True,focus_record='d0')['focused_record'] is None


def test_static_filter_happens_before_count_and_limit():
    scan=fixture()
    hidden=build_queue(scan,kind='all',hide_stationary=True,limit=200)
    assert hidden['total_vessels']==3 and hidden['total_matching']==4
    assert all(r['vessel_id']!='D' for r in hidden['records'])
    shown=build_queue(scan,kind='all',limit=200)
    assert shown['total_vessels']==4


def test_window_scope_never_falls_back_to_whole_scan_and_quality_is_deduplicated():
    scan=fixture()
    assert build_queue(scan,window={'records':[]})['records']==[]
    rows=[{'id':'q0','vessel_id':'Q','has_quality_issue':True,'e_kin_mps':2},
          {'id':'q1','vessel_id':'Q','has_quality_issue':True,'e_kin_mps':4}]
    result=build_queue(scan,kind='quality',window={'records':rows})
    assert result['scope']=='window' and result['total_matching']==2 and result['total_vessels']==1
    assert result['records'][0]['id']=='q1'
