"""Conservative identity exposure census; prints JSONL, reads no predictions."""
from pathlib import Path
import gzip
import json
import pandas as pd

root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
release=root/'release_20261026'
reg=json.loads((release/'source_registry.json').read_text(encoding='utf8'))['regions']
regions=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
windows=pd.read_parquet(release/'selected_windows.parquet')
ids=pd.read_parquet(release/'global_all_mmsi_split.parquet')
test_ids=set(ids.loc[ids.split=='test','mmsi'].astype(str))
selected_targets=set(windows.mmsi.astype(str))
prior_background=set();lines=0
for region in regions:
    for split in ('train','val'):
        path=root/'labeling/labeled_v1/features_scene'/region/(split+'_context_audit.jsonl.gz')
        with gzip.open(path,'rt',encoding='utf8') as rows:
            for row in rows:
                item=json.loads(row)
                prior_background.update(map(str,item['plain_context_exposure_mmsi']))
                prior_background.update(map(str,item['context_exposure_mmsi']))
                lines+=1
print(json.dumps({'scope':'prior train/val context exposure; presence by identity only; no model score or new windows',
    'audit_windows':lines,'prior_background_mmsi':len(prior_background),'release_selected_target_mmsi':len(selected_targets)},ensure_ascii=False),flush=True)
for region in regions:
    data=pd.read_parquet(reg[region]['canonical_file'],columns=['mmsi','sog','cog'])
    moving=set(data.loc[(data.sog>=3)&(data.sog<102.2)&(data.cog>=0)&(data.cog<360),'mmsi'].astype(str))
    eligible_test_ids=moving&test_ids
    other=eligible_test_ids-selected_targets
    print(json.dumps({'region':region,'global_test_ids_with_at_least_one_valid_moving_report':len(eligible_test_ids),
        'not_release_selected_target':len(other),'not_release_target_and_not_trainval_background':len(other-prior_background),
        'not_release_target_but_trainval_background':len(other&prior_background),
        'caveat':'single report/identity upper bound, not valid window or never-exposed proof'},ensure_ascii=False),flush=True)
