"""Paired MMSI-cluster intervals conditional on four fixed seed-0 models."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/framework_realignment_pilot_20261029')
NAMES=('tcn','gru','transformer','itransformer'); ARMS=('raw','causal_bundle','f0_bundle','f1_bundle');VIEWS=('base','natural','suffix','combined_l100_m600')
result={'scope':'Exposed development validation, conditional on one trained seed; no multiplicity correction','bootstrap':'paired target-MMSI cluster resampling 2000 draws, seed 20261029','models':{}}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
arm_path=lambda n,a: OUT/('relation_rate_runs' if a in ('f0_bundle','f1_bundle') else 'relation_temporal_runs')/n/a
result['summary_sha256']=sha(Path(__file__));result['data_sha256']={s:sha(OUT/(s+'_relation_field.npz')) for s in ('train','val')}
for name in NAMES:
    arrays={};reports={}
    for arm in ARMS:
        path=arm_path(name,arm)
        with np.load(path/'val_scores.npz',allow_pickle=False) as z:arrays[arm]={k:z[k] for k in z.files}
        reports[arm]=json.loads((path/'result.json').read_text())
    base=arrays['raw']; ids,inv=np.unique(base['mmsi'].astype(str),return_inverse=True);rng=np.random.default_rng(20261029); draw=rng.integers(0,len(ids),size=(2000,len(ids)))
    for arm in ARMS:
        for key in ('y','source','case','mmsi','region'):np.testing.assert_array_equal(arrays[arm][key],base[key])
    output={'values':{v:{arm:reports[arm]['views'][v] for arm in ARMS} for v in VIEWS},'intervals':{},'init_hashes':{a:reports[a]['init_sha256'] for a in ARMS},'predict_sha256':{a:sha(arm_path(name,a)/'val_scores.npz') for a in ARMS}}
    assert len(set(output['init_hashes'].values()))==1
    for view in VIEWS:
        keys=[view+'_score'];y=base['y'];vals={arm:arrays[arm][keys[0]] for arm in ARMS};diffs={k:[] for k in ('f1_vs_f0','f1_vs_raw','f1_vs_causal')}
        for sample in draw:
            weight=np.bincount(sample,minlength=len(ids))[inv];ap={arm:average_precision_score(y,vals[arm],sample_weight=weight) for arm in ARMS}
            diffs['f1_vs_f0'].append(ap['f1_bundle']-ap['f0_bundle']);diffs['f1_vs_raw'].append(ap['f1_bundle']-ap['raw']);diffs['f1_vs_causal'].append(ap['f1_bundle']-ap['causal_bundle'])
        output['intervals'][view]={k:{'lo':float(np.quantile(d,.025)),'hi':float(np.quantile(d,.975))} for k,d in diffs.items()}
    result['models'][name]=output
    print(name,'done',flush=True)
path=OUT/'rate_development_summary.json'
with path.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(path,flush=True)
