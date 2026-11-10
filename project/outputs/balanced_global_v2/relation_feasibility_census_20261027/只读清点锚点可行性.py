"""Read-only census of existing test-source anchor feasibility; prints JSONL.

Run with /mnt/sda/WangZijia/AIS/.venv/bin/python on the 6000D host.
The upper bound does not run quartet construction or examine model scores.
"""
from pathlib import Path
from collections import Counter
import gc
import json
import sys
import numpy as np
import pandas as pd

root = Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
sys.path.insert(0, str(root/'original_method_and_relation_v1_20261026/relation/vendor'))
from audit_neighborhood import Source, REGIONS, wrap, NM

registry = json.loads((root/'release_20261026/source_registry.json').read_text(encoding='utf8'))['regions']
windows = pd.read_parquet(root/'release_20261026/selected_windows.parquet')
ids = set(pd.read_parquet(root/'release_20261026/global_all_mmsi_split.parquet').mmsi.astype(str))
train_targets = set(windows.loc[windows.split=='train','mmsi'].astype(str))
val_targets = set(windows.loc[windows.split=='val','mmsi'].astype(str))
allowed = ids-train_targets-val_targets

for region in REGIONS:
    source = Source(registry[region])
    selected = windows[(windows.region==region)&(windows.split=='test')]
    counts = Counter()
    targets = {k:set() for k in ['orig4','relaxed3','relaxed4']}
    for row in selected.itertuples(index=False):
        t = np.array(json.loads(row.timestamps),float)
        p, s, c = source.source_target(row.mmsi,t)
        anchor = max(2,len(t)//2-1)
        nearest = source.latest(float(t[anchor]),row.mmsi,allowed)
        d = np.linalg.norm(source.p[nearest]-p[anchor],axis=1)
        direction = np.abs(wrap(source.c[nearest]-c[anchor]))
        speed = np.abs(source.s[nearest]-s[anchor])
        eligible = source.good[nearest]&source.valid[nearest]&(source.w[nearest]>=.3)&(source.s[nearest]>=3)
        prefix = bool(s[anchor]>=3 and np.median(s[:anchor+1])>=3 and t[-1]-t[anchor]>=180)
        old = int(np.sum(eligible&(d<=4*NM)&(direction<=30)&(speed<=3)))
        relaxed = int(np.sum(eligible&(d<=5*NM)&(direction<=30)&(speed<=4.5)))
        counts['candidates'] += 1
        counts['past_coexisting_member_occurrences'] += len(nearest)
        counts['orig_eligible_member_occurrences'] += old
        counts['relaxed_eligible_member_occurrences'] += relaxed
        for name,yes in [('orig4',prefix and old>=4),('relaxed3',prefix and relaxed>=3),('relaxed4',prefix and relaxed>=4)]:
            if yes:
                counts[name] += 1
                targets[name].add(str(row.mmsi))
    print(json.dumps({'region':region,'counts':dict(counts),
          'distinct_target_mmsi_by_rule':{k:len(v) for k,v in targets.items()},
          'scope':'anchor-only upper bounds; no future quartet construction or model scores'},ensure_ascii=False),flush=True)
    del source
    gc.collect()
