"""Create additive, reproducible target-vessel-disjoint copies of recovered data.

No old dataset is overwritten. All old feature/label values are preserved;
only membership changes and identity metadata are added. This is explicitly
a target-vessel split, not a claim of graph-neighbor input isolation.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, itertools
import numpy as np

BASE=Path('/mnt/sda/WangZijia/AIS')
ROOT=BASE/'outputs/split_label_provenance_20261025'
OUT=ROOT/'global_mmsi_v1'
REGIONS=['houston','ny','la','mississippi','honolulu','puget','miami','goldengate','chesapeake','gom_open','atl_open','pac_open','cpacific']
SPLITS=['train','val','test']
SEED=20261025

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

if OUT.exists():
    raise SystemExit('Refusing to replace an existing split version: '+str(OUT))
all_rows={}
for region in REGIONS:
    summary=json.loads((ROOT/'original'/region/'provenance_summary.json').read_text())
    assert summary['complete']
    all_rows[region]={s:[json.loads(line) for line in (ROOT/'original'/region/(s+'_provenance.jsonl')).read_text().splitlines()] for s in SPLITS}

vessels=np.array(sorted({r['mmsi'] for d in all_rows.values() for group in d.values() for r in group}),dtype=np.int64)
order=np.random.default_rng(SEED).permutation(vessels)
a,b=int(.70*len(order)),int(.85*len(order))
groups={'train':order[:a],'val':order[a:b],'test':order[b:]}
assignment={int(v):s for s,vs in groups.items() for v in vs}
for region,parts in all_rows.items():
    combined=[r for s in SPLITS for r in parts[s]]
    for split in SPLITS:
        chosen=[r for r in combined if assignment[r['mmsi']]==split]
        assert len(chosen)>0, (region,split,'empty split')
        assert {bool(r['label_class']) for r in chosen}=={False,True}, (region,split,'single binary class')

OUT.mkdir(parents=True)
(OUT/'mmsi_assignment.json').write_text(json.dumps({'seed':SEED,'unit':'global target-vessel MMSI',
    'ratios':[.7,.15,.15],'vessels':{s:sorted(map(int,vs)) for s,vs in groups.items()}},indent=2))
report={'version':'global_mmsi_v1','seed':SEED,'unit':'global target-vessel MMSI',
        'mmsi_counts':{s:len(vs) for s,vs in groups.items()},'total_unique_mmsi':len(vessels),
        'input_provenance_root':str(ROOT/'original'),'sample_values':'All original features and labels preserved exactly.',
        'scope':'Target vessel identities and their source observations are disjoint globally. Neighbor context arrays are inherited without reconstruction.',
        'evaluation_status':'New split, no model has been retrained or evaluated for this split.',
        'regions':{},'original_files':{},'new_files':{}}
global_ids={s:set() for s in SPLITS}
global_windows={s:set() for s in SPLITS}
global_observations={s:set() for s in SPLITS}

for region,parts in all_rows.items():
    combined=[r for s in SPLITS for r in parts[s]]
    original={}
    for split in SPLITS:
        path=Path(parts[split][0]['npz_file'])
        assert digest(path)==parts[split][0]['npz_sha256']
        with np.load(path,allow_pickle=False) as z:original[split]={k:z[k] for k in z.files}
        report['original_files'][str(path)]=parts[split][0]['npz_sha256']
    keys=set(original['train'])
    assert all(set(original[s])==keys for s in SPLITS)
    n0=[len(parts[s]) for s in SPLITS]
    for split in SPLITS:
        for key,arr in original[split].items():assert arr.ndim and len(arr)==len(parts[split]),(region,split,key)
    merged={k:np.concatenate([original[s][k] for s in SPLITS],axis=0) for k in keys}
    folder=OUT/region;folder.mkdir()
    region_report={}
    for split in SPLITS:
        take=np.array([i for i,r in enumerate(combined) if assignment[r['mmsi']]==split],dtype=int)
        selected=[combined[i] for i in take]
        arrays={k:v[take] for k,v in merged.items()}
        absolute=np.zeros(arrays['mask'].shape,dtype=np.int64)
        for i,r in enumerate(selected):absolute[i,:r['valid_steps']]=r['timestamps']
        arrays.update(mmsi=np.array([r['mmsi'] for r in selected],dtype=np.int64),
                      absolute_t=absolute,
                      behavior_onset_timestamp=np.array([r['behavior_onset_timestamp'] if r['behavior_onset_timestamp'] is not None else -1 for r in selected],dtype=np.int64),
                      trip_id=np.array([r['trip_id'] for r in selected]),
                      source_window_sha256=np.array([r['source_window_sha256'] for r in selected]),
                      origin_split=np.array([r['split'] for r in selected]),
                      origin_index=np.array([r['sample_index'] for r in selected],dtype=np.int64),
                      region_id=np.full(len(selected),region))
        target=folder/(split+'.npz')
        np.savez_compressed(target,**arrays)
        # Reopen the actual written data, not just sets computed before saving.
        with np.load(target,allow_pickle=False) as check:
            assert np.array_equal(check['mmsi'],arrays['mmsi'])
            assert np.array_equal(check['absolute_t'],absolute)
            for k in keys:assert np.array_equal(check[k],merged[k][take],equal_nan=True),(region,split,k)
            mm=set(map(int,check['mmsi']))
            global_ids[split]|=mm
            global_windows[split]|=set(map(str,check['source_window_sha256']))
            for i,ship in enumerate(check['mmsi']):
                global_observations[split].update((int(ship),int(t)) for t in check['absolute_t'][i][check['mask'][i]])
            labels=check['label_class'];cells=check['label_cell']
            region_report[split]={'windows':len(labels),'vessels':len(mm),
                'class_counts':{str(k):int((labels==k).sum()) for k in range(6)},
                'cell_counts':{str(k):int((cells==k).sum()) for k in range(4)},
                'gradient_training_subset_cells_0_2':int(np.isin(cells,[0,2]).sum()) if split=='train' else None,
                'sha256':digest(target)}
        report['new_files'][str(target)]=region_report[split]['sha256']
        with open(folder/(split+'_provenance.jsonl'),'w') as f:
            for i,r in enumerate(selected):f.write(json.dumps({**r,'new_split':split,'new_sample_index':i,'new_npz_file':str(target)})+'\n')
    report['regions'][region]=region_report
    (folder/'manifest.json').write_text(json.dumps(region_report,indent=2))
    print(json.dumps({'region':region,'split_windows':{s:v['windows'] for s,v in region_report.items()},
                      'split_vessels':{s:v['vessels'] for s,v in region_report.items()}}),flush=True)

report['global_overlap_checks']={}
for a,b in itertools.combinations(SPLITS,2):
    values={'target_mmsi':len(global_ids[a]&global_ids[b]),'source_windows':len(global_windows[a]&global_windows[b]),
            'source_observations':len(global_observations[a]&global_observations[b])}
    assert not any(values.values()),(a,b,values)
    report['global_overlap_checks'][a+'__'+b]=values

target_pool=['la','mississippi','honolulu','miami','goldengate','chesapeake','gom_open','cpacific']
protocols={'houston_to_12':(['houston'],REGIONS[1:]),'joint4_to_8':(['houston','ny','puget','atl_open'],target_pool)}
for source in ['ny','la','puget','miami','gom_open','atl_open']:
    protocols['rotation_'+source]=([source],[r for r in target_pool if r!=source])
report['protocols']={}
for name,(sources,targets) in protocols.items():
    seen=set()
    for region in sources:
        for split in ['train','val']:
            with np.load(OUT/region/(split+'.npz')) as z:seen.update(map(int,z['mmsi']))
    values={}
    for region in targets:
        with np.load(OUT/region/'test.npz') as z:
            overlap=seen&set(map(int,z['mmsi']))
            assert not overlap,(name,region,overlap)
            values[region]={'test_windows':len(z['mmsi']),'test_vessels':len(set(map(int,z['mmsi']))),'overlap_mmsi_count':len(overlap)}
    report['protocols'][name]={'sources':sources,'targets':targets,'source_train_val_mmsi':len(seen),'target_checks':values}

report['original_hashes_unchanged']=all(digest(Path(p))==h for p,h in report['original_files'].items())
assert report['original_hashes_unchanged']
report['total_windows']=sum(v[s]['windows'] for v in report['regions'].values() for s in SPLITS)
report['split_windows']={s:sum(v[s]['windows'] for v in report['regions'].values()) for s in SPLITS}
assert report['total_windows']==sum(len(v) for d in all_rows.values() for v in d.values())
(OUT/'validation_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'complete':True,'total_windows':report['total_windows'],'mmsi_counts':report['mmsi_counts'],
                  'split_windows':report['split_windows'],'global_overlap_checks':report['global_overlap_checks'],
                  'original_hashes_unchanged':report['original_hashes_unchanged']}),flush=True)
