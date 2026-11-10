from pathlib import Path
import json, itertools, hashlib

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/split_label_provenance_20261025')
regions=['houston','ny','la','mississippi','honolulu','puget','miami','goldengate','chesapeake','gom_open','atl_open','pac_open','cpacific']
SPLITS=['train','val','test']
rows={}
for r in regions:
    s=json.loads((ROOT/'original'/r/'provenance_summary.json').read_text())
    assert s['complete'], r
    rows[r]={k:[json.loads(x) for x in (ROOT/'original'/r/(k+'_provenance.jsonl')).read_text().splitlines()] for k in SPLITS}

def ids(items): return {x['mmsi'] for x in items}
def compare(a,b):
    overlap=ids(a)&ids(b)
    return {'shared_mmsi':sorted(overlap),'shared_mmsi_count':len(overlap),
            'target_windows':len(b),'overlap_target_windows':sum(x['mmsi'] in overlap for x in b),
            'retained_target_windows':sum(x['mmsi'] not in overlap for x in b)}

within={r:{a+'__'+b:compare(rows[r][a],rows[r][b]) for a,b in itertools.combinations(SPLITS,2)} for r in regions}
pool=['la','mississippi','honolulu','miami','goldengate','chesapeake','gom_open','cpacific']
scenarios={'houston_to_12':{'sources':['houston'],'targets':regions[1:]},
           'joint4_to_8':{'sources':['houston','ny','puget','atl_open'],'targets':pool}}
for src in ['ny','la','puget','miami','gom_open','atl_open']:
    scenarios['rotation_'+src]={'sources':[src],'targets':[r for r in pool if r!=src]}
for name,scenario in scenarios.items():
    source_train=[x for r in scenario['sources'] for x in rows[r]['train']]
    source_train_val=source_train+[x for r in scenario['sources'] for x in rows[r]['val']]
    scenario['source_train_windows']=len(source_train)
    scenario['source_train_mmsi']=len(ids(source_train))
    scenario['source_train_val_mmsi']=len(ids(source_train_val))
    scenario['audit']={r:{'vs_source_train':compare(source_train,rows[r]['test']),
                           'vs_source_train_and_val':compare(source_train_val,rows[r]['test'])} for r in scenario['targets']}
    # These exact index files identify which existing samples can be evaluated
    # without overlapping the historical source train+validation targets.
    keepdir=ROOT/'historical_holdout_indices'/name
    keepdir.mkdir(parents=True,exist_ok=True)
    seen=ids(source_train_val)
    for r in scenario['targets']:
        original=rows[r]['test'];keep=[x for x in original if x['mmsi'] not in seen]
        payload={'region':r,'original_npz':original[0]['npz_file'],'original_npz_sha256':original[0]['npz_sha256'],
                 'scope':'target vessel MMSI, source train and validation excluded',
                 'keep_indices':[x['sample_index'] for x in keep],
                 'removed_indices':[x['sample_index'] for x in original if x['mmsi'] in seen],
                 'kept_mmsi':sorted(ids(keep)),'source_mmsi_intersection':sorted(ids(keep)&seen)}
        assert not payload['source_mmsi_intersection']
        (keepdir/(r+'.json')).write_text(json.dumps(payload,indent=2))

result={'scope':'Recovered target-vessel identity of actual stored samples, not neighbor-context identity.',
        'all_regions':regions,'recovered_samples':sum(len(x) for v in rows.values() for x in v.values()),
        'unique_mmsi':len(ids([x for v in rows.values() for group in v.values() for x in group])),
        'within_region':within,'scenarios':scenarios}
(ROOT/'historical_actual_sample_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'samples':result['recovered_samples'],'unique_mmsi':result['unique_mmsi'],
                  'within_region_train_test':{r:within[r]['train__test']['shared_mmsi_count'] for r in regions},
                  'houston_to_12':{r:v['vs_source_train'] for r,v in scenarios['houston_to_12']['audit'].items()},
                  'joint4_to_8':{r:v['vs_source_train_and_val'] for r,v in scenarios['joint4_to_8']['audit'].items()}},indent=2))
