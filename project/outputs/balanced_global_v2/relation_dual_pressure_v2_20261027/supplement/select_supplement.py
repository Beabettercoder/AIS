"""Expand capped LA/Haugesund source times without consulting model scores."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import pandas as pd
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');PARENT=ROOT/'relation_dual_pressure_v2_20261027';OUT=PARENT/'supplement'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=json.loads((OUT/'selection_lock.json').read_text());assert all(sha(OUT/k)==v for k,v in lock['local_files_sha256'].items())
prior=json.loads((PARENT/'candidate_pool_manifest.json').read_text());parents=[json.loads(x) for x in (PARENT/'candidate_pool.jsonl').read_text().splitlines()]
byid={(r['region'],r['mmsi']):r for r in parents};previous=pd.read_parquet(ROOT/'release_20261026/selected_windows.parquet');oldids=set(previous.window_id.astype(str));old_reports=set()
for row in previous.itertuples(index=False):
 for t in json.loads(row.timestamps):old_reports.add((str(row.mmsi),int(t)))
result=[];summary={}
for region in ('la','haugesund'):
 reject=[json.loads(x) for x in (PARENT/f'{region}_candidate_rejections.jsonl').read_text().splitlines()];keep={r['window_id'] for r in reject if r['reason']=='per_target_region_hash_cap_24'}
 data=pd.read_parquet(ROOT/'candidates'/f'{region}.parquet');data=data[data.window_id.astype(str).isin(keep)];assert len(data)==len(keep)
 for row in data.itertuples(index=False):
  meta=byid[(region,str(row.mmsi))];ts=json.loads(row.timestamps)
  result.append({**{k:meta[k] for k in ['region','split','mmsi','known_release_target','prior_trainval_background','prior_test_background']},'window_id':str(row.window_id),'base_window_id':str(row.window_id),'source_timestamps':ts,'source_start':int(ts[0]),'source_end':int(ts[-1]),'old_release_window':str(row.window_id) in oldids,'old_release_target_report_overlap_count':sum((str(row.mmsi),int(t)) in old_reports for t in ts),'rank':hashlib.sha256(('rel-v1:'+region+':'+str(row.window_id)).encode()).hexdigest()})
 summary[region]={s:sum(z['region']==region and z['split']==s for z in result) for s in ['train','val','test']}
assert not set(x['window_id'] for x in result)&set(x['window_id'] for x in parents)
with (OUT/'candidate_pool.jsonl').open('x',encoding='utf8') as f:
 for x in sorted(result,key=lambda x:(x['region'],x['split'],x['rank'])):f.write(json.dumps(x,ensure_ascii=False)+'\n')
manifest={'status':'supplemental_candidates_selected_without_model_scores','candidate_pool_sha256':sha(OUT/'candidate_pool.jsonl'),'target_ids':prior['target_ids'],'regions':summary,'selection_lock_sha256':sha(OUT/'selection_lock.json'),'prior_pool_sha256':sha(PARENT/'candidate_pool.jsonl'),'note':'Includes all otherwise-valid LA/Haugesund candidates removed by initial per-target cap; same identities and split, no new target IDs; source windows can overlap prior release/test.'}
with (OUT/'candidate_pool_manifest.json').open('x',encoding='utf8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
print('SUPPLEMENT_POOL',json.dumps(summary),len(result),flush=True)
