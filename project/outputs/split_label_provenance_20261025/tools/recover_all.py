from pathlib import Path
import json, time, traceback
from recover_window_provenance import recover

BASE = Path('/mnt/sda/WangZijia/AIS')
OUT = BASE/'outputs/split_label_provenance_20261025'
REGIONS = {
    'houston': ('houston_202407', 'eval_v2_graph2'),
    **{r: (r, r+'_dataset') for r in ['ny','la','mississippi','honolulu','puget','miami','goldengate','chesapeake','gom_open','atl_open','pac_open','cpacific']}
}
excluded = json.loads((BASE/'outputs/phase1/events_ground_truth.json').read_text())['clean_pool_excluded_dates']
status = {}
for region, (source, dataset) in REGIONS.items():
    started = time.time()
    print(json.dumps({'event':'region_start','region':region}), flush=True)
    try:
        value = recover(BASE/f'data/interim/ais_{source}_enriched.parquet', BASE/'outputs/phase1'/dataset,
                        OUT/'original'/region, excluded)
        status[region] = {'complete':value['complete'], 'seconds':round(time.time()-started,2),
                          'samples':sum(value[s]['total_samples'] for s in ['train','val','test']),
                          'matched':sum(value[s]['unique_matched'] for s in ['train','val','test'])}
    except Exception:
        status[region] = {'complete':False,'error':traceback.format_exc()}
    (OUT/'recovery_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps({'event':'region_done','region':region,**status[region]}),flush=True)
raise SystemExit(0 if all(v['complete'] for v in status.values()) else 2)
