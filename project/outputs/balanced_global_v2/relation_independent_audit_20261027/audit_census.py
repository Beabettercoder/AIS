"""Independent Arrow raw-source and selected target-vessel census replay."""
from pathlib import Path
import hashlib,json
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
REL=ROOT/'release_20261026'; CENSUS=ROOT/'relation_feasibility_census_20261027'
OUT=ROOT/'relation_independent_audit_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
registry=json.loads((REL/'source_registry.json').read_text(encoding='utf8'))['regions']
records={r['region']:r for r in map(json.loads,(CENSUS/'报文船舶清点结果.jsonl').read_text(encoding='utf8').splitlines())}
w=pq.read_table(REL/'selected_windows.parquet',columns=['region','mmsi','split'])
regions=w['region'].to_numpy();ids=w['mmsi'].to_numpy();splits=w['split'].to_numpy()
report=[]
for region in REGIONS:
 path=Path(registry[region]['canonical_file'])
 table=pq.read_table(path,columns=['mmsi','sog','cog','timestamp'])
 sog=table['sog'].to_numpy();cog=table['cog'].to_numpy();valid=(sog>=0)&(sog<102.2)&(cog>=0)&(cog<360)
 rawm=int(pc.count_distinct(table['mmsi'],mode='only_valid').as_py())
 tt=table['timestamp']; lo=int(pc.min(tt).as_py());hi=int(pc.max(tt).as_py())
 expected=records[region];rr=regions==region
 assert len(table)==expected['records'] and int(valid.sum())==expected['valid_sog_cog_records'] and rawm==expected['raw_mmsi']
 assert lo==expected['timestamp_min_s'] and hi==expected['timestamp_max_s']
 assert int(rr.sum())==expected['selected_windows'] and len(np.unique(ids[rr]))==expected['selected_mmsi']
 split_data={}
 for split in ['train','val','test']:
  m=rr&(splits==split);actual={'windows':int(m.sum()),'target_mmsi':int(len(np.unique(ids[m])))}
  assert actual==expected['splits'][split],(region,split,actual,expected['splits'][split]);split_data[split]=actual
 report.append({'region':region,'source_file':str(path),'recorded_source_sha256':registry[region]['sha256'],
  'raw_messages':len(table),'valid_sog_cog_messages':int(valid.sum()),'unique_raw_mmsi':rawm,
  'timestamp_min_s':lo,'timestamp_max_s':hi,'selected_windows':int(rr.sum()),'selected_mmsi':len(np.unique(ids[rr])), 'splits':split_data})
 print('CENSUS_OK',region,flush=True)
out={'status':'passed','scope':'independent Arrow read of 8 raw sources and selected windows; no cohort feasibility or field-quality assessment',
     'prior_census_sha256':sha(CENSUS/'报文船舶清点结果.jsonl'),'selected_windows_sha256':sha(REL/'selected_windows.parquet'),
     'regions':report,'script_sha256':sha(Path(__file__))}
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'audit_census.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
