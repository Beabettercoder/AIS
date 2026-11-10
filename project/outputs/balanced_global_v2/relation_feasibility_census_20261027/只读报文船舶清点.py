"""Read-only eight-region standardized message/identity counts; prints JSONL."""
from pathlib import Path
import json
import pandas as pd

root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/release_20261026')
registry=json.loads((root/'source_registry.json').read_text(encoding='utf8'))['regions']
windows=pd.read_parquet(root/'selected_windows.parquet')
regions=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
for region in regions:
    d=pd.read_parquet(registry[region]['canonical_file'],columns=['mmsi','sog','cog','timestamp'])
    w=windows[windows.region==region]
    valid=(d.sog>=0)&(d.sog<102.2)&(d.cog>=0)&(d.cog<360)
    print(json.dumps({'region':region,'canonical_file':registry[region]['canonical_file'],
     'canonical_recorded_sha256':registry[region]['sha256'],'records':len(d),'valid_sog_cog_records':int(valid.sum()),
     'raw_mmsi':int(d.mmsi.nunique()),'timestamp_min_s':int(d.timestamp.min()),'timestamp_max_s':int(d.timestamp.max()),
     'selected_windows':len(w),'selected_mmsi':int(w.mmsi.nunique()),'splits':{
       str(k):{'windows':len(v),'target_mmsi':int(v.mmsi.nunique())} for k,v in w.groupby('split')}},ensure_ascii=False),flush=True)
