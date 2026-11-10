"""Post-protocol diagnostic: same robust estimators with original absolute-only weights."""
from pathlib import Path
import csv,json
import numpy as np
from sklearn.metrics import average_precision_score
from run_relation import OLD,OUT,OBSERVED,observed,weights,median_weighted,high_select,sha
import build_speed_stress as stress

manifest=json.loads((OLD/'all_sources_v1/relation_manifest.json').read_text());old=json.loads((OLD/'neighbor_speed_stress_v1/stress_manifest.json').read_text());lookup={x['file']:x for x in old['samples']};rows=[]
for item in manifest['samples']:
    p=OLD/'all_sources_v1'/item['file'];assert sha(p)==item['sha256']
    with np.load(p,allow_pickle=False) as z:
        o=observed(z);members=z['cohort_mmsi'].copy();wid=str(z['source_window_id']);metadata={k:str(z[k]) for k in ('target_mmsi','split','region')};y=z['y'].astype(int)
    q=OLD/'neighbor_speed_stress_v1'/item['file'];assert sha(q)==lookup[item['file']]['sha256']
    with np.load(q,allow_pickle=False) as z:existing=observed(z)
    stress.select_members=high_select;high,_=stress.stress_one({**o,'cohort_mmsi':members},wid)
    for view,obs in [('base',o),('existing_speed',existing),('high_speed',high)]:
        for case in range(4):
            t=obs['target_t'][case];s=obs['target_sog_kn'][case];anchor=max(2,len(t)//2-1);pre=np.arange(anchor+1);post=np.arange(anchor+1,len(t))[-3:];target=float(s[post].mean()/max(s[pre].mean(),.5)-1);change=[];mw=[];report=[]
            for lo,hi in zip(obs['member_offsets'][:-1],obs['member_offsets'][1:]):
                lo=int(lo);hi=int(hi);nt=obs['neighbor_t'][case,lo:hi];ns=obs['neighbor_sog_kn'][case,lo:hi];_,w,_,_,_=weights(nt,obs['neighbor_pos'][case,lo:hi],ns,obs['neighbor_cog_deg'][case,lo:hi]);ix=np.searchsorted(nt,t,side='right')-1;safe=np.maximum(ix,0);valid=(ix>=0)&(t>=nt[safe])&(t-nt[safe]<=300);ai=pre[valid[pre]];bi=post[valid[post]]
                if len(ai)<2 or len(bi)<2:continue
                j=safe[ai];k=safe[bi];change.append(float(ns[k].mean()/max(ns[j].mean(),.5)-1));mw.append(float(np.sqrt(w[j].mean()*w[k].mean())))
                p=float(np.average(ns[j],weights=w[j])) if w[j].sum()>1e-12 else float(ns[j].mean());q=float(np.average(ns[k],weights=w[k])) if w[k].sum()>1e-12 else float(ns[k].mean());report.append(q/max(p,.5)-1)
            rows.append(dict(view=view,source=wid,mmsi=metadata['target_mmsi'],region=metadata['region'],split=metadata['split'],case=case,y=int(y[case]),absolute_weighted_median=abs(target-median_weighted(change,mw)),absolute_report_weighted_median=abs(target-float(np.median(report)))))
with (OUT/'absolute_gate_robust_scores.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUT/'statistics_scores.csv').open(newline='',encoding='utf8') as f:main=list(csv.DictReader(f))
assert [(r['view'],r['source'],str(r['case'])) for r in rows]==[(r['view'],r['source'],r['case']) for r in main]
for r,m in zip(rows,main):r.update(mixed_weighted_median=float(m['mixed_weighted_median']),mixed_report_weighted_median=float(m['mixed_report_weighted_median']))
ids=sorted({r['mmsi'] for r in rows if r['split']=='val'});rng=np.random.default_rng(20261027);draws=rng.integers(0,len(ids),(2000,len(ids)));out={'status':'completed','scope':'post-protocol factorial control; absolute-only vs combined relative+absolute within same robust aggregator','bootstrap_unit':'target MMSI, 2000 draws','views':{}}
for view in ['base','existing_speed','high_speed']:
    r=[x for x in rows if x['split']=='val' and x['view']==view];groups={i:np.array([j for j,x in enumerate(r) if x['mmsi']==i]) for i in ids};yy=np.array([x['y'] for x in r]);out['views'][view]={}
    for mixed,absolute in [('mixed_weighted_median','absolute_weighted_median'),('mixed_report_weighted_median','absolute_report_weighted_median')]:
        a=np.array([x[mixed] for x in r]);b=np.array([x[absolute] for x in r]);dif=[]
        for d in draws:
            ix=np.concatenate([groups[ids[j]] for j in d]);dif.append(average_precision_score(yy[ix],a[ix])-average_precision_score(yy[ix],b[ix]))
        out['views'][view][mixed]={'absolute_AP':average_precision_score(yy,b),'mixed_AP':average_precision_score(yy,a),'difference':average_precision_score(yy,a)-average_precision_score(yy,b),'CI_95_percentile':np.quantile(dif,[.025,.975]).tolist()}
(OUT/'absolute_gate_robust_control.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
