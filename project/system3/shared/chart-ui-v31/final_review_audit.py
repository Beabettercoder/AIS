from pathlib import Path
from collections import Counter
from datetime import datetime
import gzip,json

root=Path('/mnt/sda/WangZijia/AIS/system3/shared/anomaly-workbench-runtime/jobs')
old=root/'0aff9f4b7b304b5595ca9216be5e0424'
new=root/'79b59e2bc42c4479abecb8462a9a9764'
def read(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)
def epoch(value):return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
scan=read(new/'result.json.gz');baseline=read(old/'result.json.gz')
rows={};raw=0;focus=[]
for index in range(360):
    before={row['id']:row for row in read(old/f'window-{index}.json.gz')['records']}
    current=read(new/f'window-{index}.json.gz')
    for row in current['records']:
        previous=before[row['id']];rows[row['id']]=row
        assert row['motion_candidate']==previous['is_candidate']
        for field in ['sog_kn','latlon','gate_weights']:
            assert row[field]==previous[field]
        assert abs(row['score']-previous['score'])<1e-7
        raw+=row['motion_candidate']
        if row['maritime_policy']['stationary']:assert not row['is_candidate'] and row['score']==0
        if row['is_candidate']:
            assert row['review_assessment']['eligible'];focus.append(row)
            assert not (row.get('stopping_context',{}).get('matches') and row['review_assessment']['codes']==['sustained_rapid_deceleration'])
    assert current['summary']['candidate_count']==sum(row['is_candidate'] for row in current['records'])
assert len(rows)==32446 and raw==2058
for row in rows.values():
    refs=row.get('recurrence_evidence',{})
    if refs.get('repeated'):
        assert len(refs['reference_ids'])>=2
        for rid in refs['reference_ids']:
            reference=rows[rid]
            assert reference['vessel_id']==row['vessel_id']
            assert epoch(reference['end_utc'])<epoch(row['start_utc'])
            assert epoch(row['start_utc'])-epoch(reference['start_utc'])>=3600
    stops=row.get('stopping_context',{})
    if stops.get('matches'):
        assert len(set(stops['reference_mmsi']))>=3
        for rid in stops['reference_ids']:
            reference=rows[rid]
            assert reference['vessel_id']!=row['vessel_id']
            assert epoch(reference['end_utc'])<epoch(row['start_utc'])
seen=[]
for event in scan['motion_events']:
    ids=event['event_window_ids'];seen.extend(ids)
    assert all(rows[rid]['motion_candidate'] for rid in ids)
    assert all(rows[rid]['is_candidate'] for rid in event['priority_window_ids'])
    assert bool(event['priority_window_ids'])==(event['review_level']=='priority')
    assert all(rows[b]['window_index']-rows[a]['window_index']==1 for a,b in zip(ids,ids[1:]))
assert len(seen)==len(set(seen))==2058
assert len(focus)==scan['summary']['candidate_windows']
assert len({r['vessel_id'] for r in focus})==scan['summary']['candidate_vessel_count']
assert len(scan['records'])==min(scan['params']['top_k'],scan['summary']['candidate_vessel_count'])
assert all(r['is_candidate'] for r in scan['records'])
assert len({r['vessel_id'] for r in scan['records']})==len(scan['records'])
types=Counter(r['type'] for r in focus)
assert all(scan['type_counts'].get(kind)==count for kind,count in types.items())
targets={v:{'motion_windows':sum(r['motion_candidate'] and r['vessel_id']==v for r in rows.values()),
            'priority_windows':sum(r['is_candidate'] and r['vessel_id']==v for r in rows.values())} for v in ['367359250','367574880','368212230']}
report={'baseline_summary':baseline['summary'],'focused_summary':scan['summary'],
        'priority_types':dict(types),'priority_episodes_by_day':dict(Counter(r['event_start_utc'][:10] for r in scan['candidate_events'])),
        'audited_window_records':len(rows),'all_original_motion_facts_and_scores_preserved':True,
        'all_motion_windows_preserved_once_in_episodes':True,'all_references_are_past_observations':True,
        'queue_is_not_filled_with_general_clues':True,'targets':targets}
out=Path('/mnt/sda/WangZijia/AIS/system3/shared/chart-ui-v31/review-load-audit.json')
out.write_text(json.dumps(report,indent=2))
print('AUDIT_PASS',json.dumps(report),flush=True)
