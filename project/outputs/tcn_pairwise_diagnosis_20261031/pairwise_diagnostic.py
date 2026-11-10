"""Exploratory within-source TCN ranking diagnostic. No test data."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score

BASE=Path('/mnt/sda/WangZijia/AIS/outputs')
OUT=BASE/'tcn_pairwise_diagnosis_20261031'
DATA=BASE/'relation_repair_20261030/val_relation_field.npz'
old=BASE/'heading_alignment_20261030/runs/aligned_embed/tcn'
extra=BASE/'tcn_suffix_stability_20261030/runs/aligned_embed'
with np.load(DATA,allow_pickle=False) as z:
    d={k:z[k] for k in z.files}
sources, inv = np.unique(d['source'],return_inverse=True)
assert len(sources)==272
groups=[np.flatnonzero(inv==i) for i in range(len(sources))]
assert all(len(g)==4 and d['y'][g].sum()==2 for g in groups)
assert all(len(np.unique(d['mmsi'][g]))==1 for g in groups)
boats=np.array([d['mmsi'][g[0]] for g in groups])
score={}
for seed in range(3):
    for arm in ('F0','F1'):
        path=(old if seed==0 else extra/f'seed{seed}'/'tcn')/arm/'val_scores.npz'
        with np.load(path,allow_pickle=False) as z:
            for k in ('y','mmsi','region','source','case'): assert np.array_equal(z[k],d[k]),(path,k)
            score[seed,arm]={v:z[v+'_score'].astype(float) for v in ('base','natural','suffix','combined_l100_m600')}

def within(y,p):
    pos=p[y==1]; neg=p[y==0]
    return float((np.greater(pos[:,None],neg[None,:])+.5*np.equal(pos[:,None],neg[None,:])).mean())
def boot(a,b):
    unique=np.unique(boats); ids=[np.flatnonzero(boats==x) for x in unique]
    rng=np.random.default_rng(20261031); dif=[]
    for _ in range(2000):
        ii=np.concatenate([ids[k] for k in rng.integers(len(unique),size=len(unique))])
        dif.append((a[ii]-b[ii]).mean())
    return np.quantile(dif,[.025,.975]).tolist()

out={'scope':'post-hoc exploratory validation; within-source pairwise ranking, not AP; 272 sources, 86 targets, labels are constructed; MMSI bootstrap fixed three models', 'views':{}}
for view in ('base','natural','suffix','combined_l100_m600'):
    rank={a:np.stack([[within(d['y'][g],score[s,a][view][g]) for g in groups] for s in range(3)]).mean(0) for a in ('F0','F1')}
    shift={}
    source_shift={}
    for arm in ('F0','F1'):
        shifts=np.stack([score[s,arm][view]-score[s,arm]['base'] for s in range(3)])
        shift[arm]={'mean_abs_score_change':float(np.abs(shifts).mean()),'positive_mean_shift':float(shifts[:,d['y']==1].mean()),'negative_mean_shift':float(shifts[:,d['y']==0].mean())}
        source_shift[arm]=np.array([np.abs(shifts[:,g]).mean() for g in groups])
    out['views'][view]={'within_pairwise_F0':float(rank['F0'].mean()),'within_pairwise_F1':float(rank['F1'].mean()),'F1_minus_F0_ci':boot(rank['F1'],rank['F0']),'source_gains':int((rank['F1']>rank['F0']).sum()),'source_losses':int((rank['F1']<rank['F0']).sum()),'source_ties':int((rank['F1']==rank['F0']).sum()),'score_shift_vs_base':shift,'F1_minus_F0_abs_score_shift':float((source_shift['F1']-source_shift['F0']).mean()),'F1_minus_F0_abs_score_shift_ci':boot(source_shift['F1'],source_shift['F0'])}

records={r['source']:r for r in json.load(open(BASE/'tcn_suffix_stability_20261030/seed0_diagnostic.json',encoding='utf8'))['source_records']}
out['subgroups']={}
for name,ix in [('3_neighbors',np.array([i for i,s in enumerate(sources) if records[s]['candidate_members']==3])),('4plus_neighbors',np.array([i for i,s in enumerate(sources) if records[s]['candidate_members']>=4]))]:
    rank={a:np.stack([[within(d['y'][groups[i]],score[s,a]['suffix'][groups[i]]) for i in ix] for s in range(3)]).mean(0) for a in ('F0','F1')}
    rows=np.concatenate([groups[i] for i in ix])
    out['subgroups'][name]={'sources':int(len(ix)),'boats':int(len(np.unique(boats[ix]))),'within_pairwise_F0':float(rank['F0'].mean()),'within_pairwise_F1':float(rank['F1'].mean()),'pooled_AP_mean':{a:float(np.mean([average_precision_score(d['y'][rows],score[s,a]['suffix'][rows]) for s in range(3)])) for a in ('F0','F1')}}

OUT.mkdir(exist_ok=True)
(OUT/'pairwise_diagnostic.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(out,indent=2,ensure_ascii=False))
