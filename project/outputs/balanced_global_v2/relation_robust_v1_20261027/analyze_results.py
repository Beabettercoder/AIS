"""Group-aware validation resampling and observable-report suppression diagnostics."""
from pathlib import Path
import csv,json
import numpy as np
from sklearn.metrics import average_precision_score
P=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_robust_v1_20261027')
def read(name):
    with (P/name).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
rows=read('statistics_scores.csv');neural=read('neural_predictions.csv');quality=read('report_weight_diagnostics.csv');prov=json.loads((P/'provenance.json').read_text());rng=np.random.default_rng(20261027)
val=[r for r in rows if r['split']=='val'];ids=sorted(set(r['mmsi'] for r in val));assert len(ids)==58
draws=np.array([rng.integers(0,len(ids),len(ids)) for _ in range(2000)]);out={'bootstrap_unit':'validation target MMSI, all source windows and quartet variants retained; conditional on current 58 targets, not shared scene independent','seed':20261027,'resamples':2000,'comparisons':{},'quality':{},'views':{}}
for view in ('base','existing_speed','high_speed'):
    r=[x for x in val if x['view']==view];groups={i:np.array([j for j,x in enumerate(r) if x['mmsi']==i],dtype=int) for i in ids};y=np.array([int(x['y']) for x in r]);
    for a,b in [('mixed_weighted_median','median'),('mixed_report_weighted_median','median'),('mixed_weighted_mean','absolute_weighted_mean')]:
        x=np.array([float(t[a]) for t in r]);z=np.array([float(t[b]) for t in r]);base=average_precision_score(y,x)-average_precision_score(y,z);diff=[]
        for d in draws:
            ix=np.concatenate([groups[ids[j]] for j in d]);diff.append(average_precision_score(y[ix],x[ix])-average_precision_score(y[ix],z[ix]))
        out['comparisons'][f'{view}:{a}-{b}']={'difference_AP':base,'CI_95_percentile':np.quantile(diff,[.025,.975]).tolist()}
    pv=[p for p in prov if p['split']=='val'];out['views'][view]={'val_sources':len(pv)}
    if view=='high_speed':out['views'][view].update(requested_members=sum(len(p['selected_members']) for p in pv),suffix_overlap_sources=sum(p['suffix_overlap'] for p in pv))
for view in ('base','existing_speed','high_speed'):
    rr=[x for x in quality if x['view']==view and x['split']=='val'];out['quality'][view]={}
    for band,fun in [('under_5kn',lambda x:x<5),('5_to_10kn',lambda x:5<=x<10),('at_least_10kn',lambda x:x>=10)]:
        for sel in ('all','changed'):
            zz=[x for x in rr if fun(float(x['base_sog_kn'])) and (sel=='all' or x['changed']=='True')];n=len(zz)
            out['quality'][view][band+'_'+sel]={'assessable_report_states':n,'unique_source_member_reports':len({(x['source'],x['neighbor'],x['time']) for x in zz}),'abs_w_below_0_3':sum(float(x['w_abs'])<.3 for x in zz)/n if n else None,'mixed_w_below_0_3':sum(float(x['w_mixed'])<.3 for x in zz)/n if n else None}
(P/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False))
