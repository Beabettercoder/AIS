"""Independently replay candidate identity, quality, time and hash-cap rules."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,gzip,hashlib,json
import pandas as pd

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=ROOT/'relation_independent_audit_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();RUN=args.run
protocol=json.loads((RUN/'protocol.json').read_text(encoding='utf8'))
assert '24 windows' in protocol['candidate_policy'] and 'duration_s>=660' in protocol['candidate_policy']
manifest=json.loads((RUN/'candidate_pool_manifest.json').read_text(encoding='utf8'))
assert manifest['candidate_pool_sha256']==sha(RUN/'candidate_pool.jsonl') and manifest['protocol_sha256']==sha(RUN/'protocol.json')
previous=pd.read_parquet(ROOT/'release_20261026/selected_windows.parquet',columns=['mmsi','window_id','timestamps'])
selected=set(previous.mmsi.astype(str));old_windows=set(previous.window_id.astype(str))
old_reports={(str(r.mmsi),int(t)) for r in previous.itertuples(index=False) for t in json.loads(r.timestamps)}
group=pd.read_parquet(ROOT/'release_20261026/global_all_mmsi_split.parquet',columns=['mmsi','split'])
mapping={str(m):str(s) for m,s in zip(group.mmsi,group.split)}
exposed={s:set() for s in ('train','val','test')}
for region in REGIONS:
 for split in exposed:
  p=ROOT/'labeling/labeled_v1/features_scene'/region/f'{split}_context_audit.jsonl.gz'
  with gzip.open(p,'rt',encoding='utf8') as f:
   for line in f:
    d=json.loads(line)
    exposed[split].update(map(str,d['plain_context_exposure_mmsi']));exposed[split].update(map(str,d['context_exposure_mmsi']))
trainval=exposed['train']|exposed['val']
pool={}
for line in (RUN/'candidate_pool.jsonl').read_text(encoding='utf8').splitlines():
 row=json.loads(line);k=(row['region'],row['window_id']);assert k not in pool;pool[k]=row
assert len(pool)==len(set(pool))
counts={};split_ids={s:set() for s in exposed};new_test_ids=set();release_test_exposed=set();sampled_hashes=[]
for region in REGIONS:
 p=ROOT/'candidates'/f'{region}.parquet'
 df=pd.read_parquet(p,columns=['mmsi','window_id','all_points_core','duration_s','max_gap_s','timestamps'])
 grouped=defaultdict(list);reasons=Counter();expected={};rejected={}
 for r in df.itertuples(index=False):
  mid=str(r.mmsi);wid=str(r.window_id);split=mapping.get(mid)
  assert wid not in rejected
  valid=bool(r.all_points_core) and float(r.duration_s)>=660 and float(r.max_gap_s)<=300 and split is not None
  allow=(mid in selected) or (split=='test' and mid not in trainval)
  if not valid:
   reasons['invalid_source_window_or_unmapped_identity']+=1;rejected[wid]=(mid,str(split) if split is not None else 'nan','invalid_source_window_or_unmapped_identity');continue
  if not allow:
   reasons['identity_not_in_fixed_target_pool']+=1;rejected[wid]=(mid,str(split),'identity_not_in_fixed_target_pool');continue
  rank=hashlib.sha256(('rel-v1:'+region+':'+wid).encode()).hexdigest()
  grouped[mid].append((rank,wid,split,r.timestamps))
 for mid,rows in grouped.items():
  rows.sort(key=lambda x:(x[0],x[1]));reasons['per_target_region_hash_cap_24']+=max(0,len(rows)-24)
  for rank,wid,split,t in rows[24:]:rejected[wid]=(mid,str(split),'per_target_region_hash_cap_24')
  for rank,wid,split,t in rows[:24]:
   k=(region,wid);expected[k]=(mid,split,rank,json.loads(t))
 actual={k:v for k,v in pool.items() if k[0]==region};assert set(actual)==set(expected),(region,len(actual),len(expected),len(set(actual)^set(expected)))
 for k,row in actual.items():
  mid,split,rank,t=expected[k]
  assert (row['mmsi'],row['split'],row['rank'])==(mid,split,rank) and row['source_timestamps']==t
  assert row['source_start']==int(t[0]) and row['source_end']==int(t[-1])
  assert row['old_release_window']==(row['window_id'] in old_windows)
  assert row['known_release_target']==(mid in selected)
  assert row['prior_trainval_background']==(mid in trainval)
  assert row['prior_test_background']==(mid in exposed['test'])
  assert row['old_release_target_report_overlap_count']==sum((mid,int(ti)) in old_reports for ti in t)
  split_ids[split].add(mid)
  if split=='test' and mid not in selected:new_test_ids.add(mid);assert mid not in trainval
  if split=='test' and mid in selected and mid in trainval:release_test_exposed.add(mid)
 expected_counts=manifest['counts'][region]['reject_reasons']
 for k,v in reasons.items():assert expected_counts.get(k,0)==v,(region,k,v,expected_counts)
 assert expected_counts.get('',0)==len(actual)
 saved={}
 with (RUN/f'{region}_candidate_rejections.jsonl').open(encoding='utf8') as f:
  for line in f:
   r=json.loads(line);assert r['region']==region and r['window_id'] not in saved
   saved[r['window_id']]=(r['mmsi'],r['split'],r['reason'])
 assert saved==rejected,(region,len(saved),len(rejected))
 counts[region]={'all_candidates':len(df),'selected':len(actual),'rejections':dict(reasons),'candidate_sha256':sha(p)}
 print('AUDIT_CANDIDATES',region,len(actual),flush=True)
for a,b in [('train','val'),('train','test'),('val','test')]:assert not split_ids[a]&split_ids[b]
out={'status':'passed','scope':'candidate identity/quality/time/hash-cap rules; no neighborhood or quartet replay',
 'protocol_sha256':sha(RUN/'protocol.json'),'candidate_pool_sha256':sha(RUN/'candidate_pool.jsonl'),
 'candidate_count':len(pool),'regions':counts,'target_mmsi':{s:len(v) for s,v in split_ids.items()},
 'new_test_target_ids_not_seen_in_prior_main_trainval_background':len(new_test_ids),
 'release_test_target_ids_seen_in_prior_main_trainval_background':sorted(release_test_exposed),
 'prior_trainval_background_scope':'only prior labeled_v1 features_scene train/val exposure; not all historical models or prior pilot/test',
 'script_sha256':sha(Path(__file__))}
OUT.mkdir(parents=True,exist_ok=True)
dest=OUT/(RUN.name+'_candidate_rejection_audit.json')
with dest.open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k not in ('regions','release_test_target_ids_seen_in_prior_main_trainval_background')},indent=2),flush=True)
