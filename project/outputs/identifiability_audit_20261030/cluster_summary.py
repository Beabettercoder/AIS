"""Cluster uncertainty for training-only proxy statistics, not a deployment threshold."""
from pathlib import Path
import json,numpy as np
from sklearn.metrics import roc_auc_score
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/identifiability_audit_20261030')
r=np.load(OUT/'records.npz')['r'];y=r['y'];c=r['ecur'];f=r['futurepos']
scores={'e_cur':c,'two_side_min':np.minimum(c,f),'cur_minus_next':c-f,'2cur_minus_next':2*c-f}
neighbors, group=np.unique(r['neighbor'],return_inverse=True); rng=np.random.default_rng(20261030)
draw=rng.integers(0,len(neighbors),size=(1000,len(neighbors))); cnt=np.zeros((1000,len(neighbors)),np.int32)
for i in range(1000):cnt[i]=np.bincount(draw[i],minlength=len(neighbors))
def weighted_q(x,w,q):
    ix=np.argsort(x,kind='mergesort');x,w=x[ix],w[ix];cw=np.cumsum(w);return float(x[np.searchsorted(cw,cw[-1]*q,side='left')])
result={}
for name,s in scores.items():
    q=np.quantile(s[y==0],[.90,.95,.99]); neg=s[y==0];pos=s[y==1]
    by_next=[]
    for z in (0,1):
        mask=(y==1)&(r['nextchanged']==z);by_next.append(dict(next_report_contaminated=bool(z),n=int(mask.sum()),targets=int(len(np.unique(r['target'][mask]))),neighbors=int(len(np.unique(r['neighbor'][mask]))),above_q95=float((s[mask]>q[1]).mean())))
    metrics=[]
    for b in range(1000):
        w=cnt[b,group]; nw=w[y==0];pw=w[y==1]
        if nw.sum()==0 or pw.sum()==0:continue
        qb=weighted_q(neg,nw,.95)
        tpr=float(np.sum(pw*(pos>qb))/pw.sum())
        auc=float(roc_auc_score(y,s,sample_weight=w))
        metrics.append((qb,tpr,auc))
    metrics=np.array(metrics)
    result[name]={'neg_q90_q95_q99':q.tolist(),'positive_above_each_q':[(float((pos>v).mean())) for v in q], 'negative_fraction_above_q95':float((neg>q[1]).mean()),'positive_by_next':by_next,'roc_auc':float(roc_auc_score(y,s)),'bootstrap_95_neighbor_mmsi':{'q95':np.quantile(metrics[:,0],[.025,.975]).tolist(),'tpr':np.quantile(metrics[:,1],[.025,.975]).tolist(),'auc':np.quantile(metrics[:,2],[.025,.975]).tolist()}}
out={'scope':'training-only post-hoc thresholds fitted and evaluated on same proxy groups; repeated target-source reports and shared backgrounds; cluster by neighbor MMSI only. Unmodified group is not verified clean or true maneuver; positive is controlled reported-speed intervention. Future variables require waiting until next observation and are never online at event time. q-based TPR is descriptive not independent estimate. subgroup true baseline low speed is for diagnostic only.', 'n':int(len(r)), 'positive':int(y.sum()), 'negative':int((1-y).sum()),'target_mmsi':int(len(np.unique(r['target']))),'neighbor_mmsi':int(len(neighbors)),'unique_source':int(len(np.unique(r['source']))),'resamples':1000,'methods':result}
(OUT/'cluster_summary.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
