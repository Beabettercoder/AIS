"""Independent LA/Haugesund supplementary candidate and combined-source audit."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import copy,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
RUN=ROOT/'relation_dual_pressure_v2_20261027';SUP=RUN/'supplement';OUT=ROOT/'relation_independent_audit_20261027'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def locked(p,key='local_files_sha256'):
 d=json.loads(p.read_text(encoding='utf8'))
 for k,v in d[key].items():assert sha(p.parent/k)==v,k
 for k,v in d.get('prior_frozen_inputs',{}).items():assert sha(Path(k))==v,k
 for k,v in d.get('external_dependencies_sha256',{}).items():assert sha(Path(k))==v,k
 return d
sl=locked(SUP/'selection_lock.json');cl=locked(SUP/'construction_lock.json')
assert datetime.fromisoformat(sl['time_utc'])<datetime.fromtimestamp((SUP/'candidate_pool.jsonl').stat().st_mtime,timezone.utc)
assert datetime.fromisoformat(cl['time_utc'])<datetime.fromtimestamp((SUP/'dataset_manifest.json').stat().st_mtime,timezone.utc)
parent_pool={}
for line in (RUN/'candidate_pool.jsonl').read_text(encoding='utf8').splitlines():
 r=json.loads(line);parent_pool[(r['region'],r['window_id'])]=r
byid={}
for r in parent_pool.values():
 k=(r['region'],r['mmsi']);meta=tuple(r[x] for x in ('split','known_release_target','prior_trainval_background','prior_test_background'))
 if k in byid:assert byid[k]==meta
 byid[k]=meta
previous=pd.read_parquet(ROOT/'release_20261026/selected_windows.parquet',columns=['mmsi','window_id','timestamps'])
oldids=set(previous.window_id.astype(str));oldreports={(str(r.mmsi),int(t)) for r in previous.itertuples(index=False) for t in json.loads(r.timestamps)}
suppool={}
for line in (SUP/'candidate_pool.jsonl').read_text(encoding='utf8').splitlines():
 r=json.loads(line);key=(r['region'],r['window_id']);assert key not in suppool and key not in parent_pool;suppool[key]=r
for region in ('la','haugesund'):
 rejected={}
 with (RUN/f'{region}_candidate_rejections.jsonl').open(encoding='utf8') as f:
  for line in f:
   r=json.loads(line)
   if r['reason']=='per_target_region_hash_cap_24':rejected[r['window_id']]=(r['mmsi'],r['split'])
 df=pd.read_parquet(ROOT/'candidates'/f'{region}.parquet',columns=['mmsi','window_id','timestamps'])
 source={str(r.window_id):r for r in df.itertuples(index=False) if str(r.window_id) in rejected}
 assert len(source)==len(rejected)
 candidates={k[1]:v for k,v in suppool.items() if k[0]==region};assert set(candidates)==set(rejected)
 for wid,row in candidates.items():
  raw=source[wid];mid=str(raw.mmsi);t=json.loads(raw.timestamps);assert rejected[wid]==(mid,row['split'])
  assert (row['split'],row['known_release_target'],row['prior_trainval_background'],row['prior_test_background'])==byid[(region,mid)]
  assert row['source_timestamps']==t and row['source_start']==int(t[0]) and row['source_end']==int(t[-1])
  assert row['old_release_window']==(wid in oldids) and row['old_release_target_report_overlap_count']==sum((mid,int(x)) in oldreports for x in t)
  assert row['rank']==hashlib.sha256(('rel-v1:'+region+':'+wid).encode()).hexdigest()

main=json.loads((RUN/'dataset_manifest.json').read_text(encoding='utf8'));sub=json.loads((SUP/'dataset_manifest.json').read_text(encoding='utf8'));combined=json.loads((RUN/'dataset_manifest_combined.json').read_text(encoding='utf8'))
assert combined['main_manifest_sha256']==sha(RUN/'dataset_manifest.json') and combined['supplement_manifest_sha256']==sha(SUP/'dataset_manifest.json')
assert combined['merge_lock_sha256']==sha(RUN/'merge_lock.json')
expected={}
for is_extra,source in [(False,main),(True,sub)]:
 for row in source['samples']:
  row=copy.deepcopy(row);key=(row['version'],row['region'],row['window_id']);assert key not in expected
  if is_extra:
   row['supplemental_source']=True
   for view in row['files']:row['files'][view]['path']='supplement/'+row['files'][view]['path']
  expected[key]=row
actual={(r['version'],r['region'],r['window_id']):r for r in combined['samples']}
assert len(actual)==len(combined['samples'])==len(expected) and actual==expected
counts=defaultdict(list);background=defaultdict(set);reports=defaultdict(set);targets=defaultdict(set)
for row in combined['samples']:
 key=(row['version'],row['split']);counts[key].append(row);targets[key].add(row['mmsi'])
 with np.load(RUN/row['files']['base']['path'],allow_pickle=False) as z:
  assert set(map(str,z['cohort_mmsi']))==set(row['cohort_mmsi'])
  background[key].update(map(str,z['cohort_mmsi']))
  reports[key].update(zip(z['neighbor_mmsi'].astype(str),z['source_neighbor_t'].astype(float)))
for ver in ('original','relaxed'):
 for split in ('train','val','test'):
  rows=counts[(ver,split)];rep=combined['counts'][ver][split]
  assert rep['sources']==len(rows) and rep['target_mmsi']==len(targets[(ver,split)])
  assert rep['coverage_complete']==sum(r['coverage_complete'] for r in rows)
  assert rep['supplement_sources']==sum(r.get('supplemental_source',False) for r in rows)
  assert rep['prior_trainval_background_sources']==sum(r['prior_trainval_background'] for r in rows)
  assert rep['prior_test_background_sources']==sum(r['prior_test_background'] for r in rows)
  region_counts=Counter(r['region'] for r in rows)
  assert rep['per_region']=={region:region_counts.get(region,0) for region in rep['per_region']}
 for a,b in [('train','val'),('train','test'),('val','test')]:
  assert not targets[(ver,a)]&targets[(ver,b)]
  ob=combined['background_overlap'][ver][a+'_'+b]
  assert ob['shared_background_mmsi']==len(background[(ver,a)]&background[(ver,b)])
  assert ob['shared_background_reports']==len(reports[(ver,a)]&reports[(ver,b)])
out={'status':'passed','supplement_candidate_count':len(suppool),'supplement_accepted':len(sub['samples']),
     'combined_accepted':len(actual),'main_accepted':len(main['samples']),'combined_sha256':sha(RUN/'dataset_manifest_combined.json'),
     'candidate_selection_utc':sl['time_utc'],'construction_lock_utc':cl['time_utc'],
     'scope':'full supplemental candidate reconstruction, lock hashes/timing, exact merged records, target identity, counts and raw background report overlaps',
     'limit':'supplement chosen after main source-feasibility counts; this is not a once-preregistered pristine dataset',
     'script_sha256':sha(Path(__file__))}
with (OUT/'supplement_and_merge_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False,indent=2))
