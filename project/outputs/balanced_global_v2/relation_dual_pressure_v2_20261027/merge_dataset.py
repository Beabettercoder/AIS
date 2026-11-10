"""Merge frozen source-feasibility supplement without changing base records."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import json,hashlib
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');OUT=ROOT/'relation_dual_pressure_v2_20261027'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=json.loads((OUT/'merge_lock.json').read_text());assert all(sha(OUT/k)==v for k,v in lock['inputs'].items())
main=json.loads((OUT/'dataset_manifest.json').read_text());supp=json.loads((OUT/'supplement/dataset_manifest.json').read_text());samples=list(main['samples'])
for r in supp['samples']:
 q={**r,'supplemental_source':True,'files':{v:{**data,'path':'supplement/'+data['path']} for v,data in r['files'].items()}};samples.append(q)
assert len(samples)==len({(r['version'],r['region'],r['window_id']) for r in samples})
versions=['original','relaxed'];splits=['train','val','test'];targets={v:{s:set() for s in splits} for v in versions};backgrounds={v:{s:set() for s in splits} for v in versions};reports={v:{s:set() for s in splits} for v in versions};counts={v:{s:{} for s in splits} for v in versions}
for r in samples:
 v,s=r['version'],r['split'];targets[v][s].add(r['mmsi']);backgrounds[v][s].update(r['cohort_mmsi']);p=OUT/r['files']['base']['path'];assert sha(p)==r['files']['base']['sha256']
 with np.load(p,allow_pickle=False) as z:reports[v][s].update((str(a),float(b)) for a,b in zip(z['neighbor_mmsi'],z['source_neighbor_t']))
for v in versions:
 for a,b in [('train','val'),('train','test'),('val','test')]:
  assert not targets[v][a]&targets[v][b]
  assert not targets[v][a]&backgrounds[v][b] and not targets[v][b]&backgrounds[v][a]
 for s in splits:
  rows=[r for r in samples if r['version']==v and r['split']==s];counts[v][s]={'sources':len(rows),'target_mmsi':len(targets[v][s]),'per_region':{r:sum(x['region']==r for x in rows) for r in ['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai']},'coverage_complete':sum(x['coverage_complete'] for x in rows),'supplement_sources':sum(x.get('supplemental_source',False) for x in rows),'prior_trainval_background_sources':sum(x['prior_trainval_background'] for x in rows),'prior_test_background_sources':sum(x['prior_test_background'] for x in rows)}
overlap={v:{a+'_'+b:{'shared_background_mmsi':len(backgrounds[v][a]&backgrounds[v][b]),'shared_background_reports':len(reports[v][a]&reports[v][b])} for a,b in [('train','val'),('train','test'),('val','test')]} for v in versions}
result={'status':'frozen_combined_pool_before_any_new_model_training_or_predictions','time_utc':datetime.now(timezone.utc).isoformat(),'main_manifest_sha256':sha(OUT/'dataset_manifest.json'),'supplement_manifest_sha256':sha(OUT/'supplement/dataset_manifest.json'),'merge_lock_sha256':sha(OUT/'merge_lock.json'),'counts':counts,'background_overlap':overlap,'samples':samples,'note':'Prior v2 main and supplemental pools retained separately; additional LA/Haugesund sources used identical pre-score rule, after initial source-feasibility counts were viewed.'}
with (OUT/'dataset_manifest_combined.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('COMBINED',json.dumps(counts,ensure_ascii=False),flush=True)
