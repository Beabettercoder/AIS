"""Scoring functions consume observed quartet arrays only, never source truth."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,precision_recall_fscore_support,confusion_matrix

NAMES=('target_only','neighbor_only','relative_mean','relative_median','relative_reliable')
OBSERVED=('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def velocity(s,c):
    a=np.deg2rad(c);return (s*1852/3600)[:,None]*np.column_stack((np.sin(a),np.cos(a)))
def kinematic_weights(t,p,s,c):
    t,p,s,c=map(np.asarray,(t,p,s,c));w=np.zeros(len(t));v=velocity(s,c)
    dt=np.diff(t);ok=(dt>0)&(dt<=300)&np.isfinite(p[:-1]).all(1)&np.isfinite(p[1:]).all(1)&np.isfinite(v[:-1]).all(1)&np.isfinite(v[1:]).all(1)
    ix=np.flatnonzero(ok)+1
    e=np.linalg.norm((p[ix]-p[ix-1])/dt[ix-1,None]-(v[ix]+v[ix-1])/2,axis=1)
    w[ix]=np.exp(-np.maximum(0,e-2.5)**2/8)
    return w

def score_observed(o):
    if set(o)!=set(OBSERVED):raise ValueError('Only whitelisted observed arrays accepted')
    rows=[]
    for case in range(4):
        t=o['target_t'][case];s=o['target_sog_kn'][case];n=len(t);anchor=max(2,n//2-1)
        pre=np.arange(anchor+1);post=np.arange(anchor+1,n)[-3:]
        if len(post)<2:raise ValueError('At least two post-anchor reports required')
        target=float(np.mean(s[post])/max(float(np.mean(s[pre])),.5)-1)
        changes=[];reliability=[];usable=[]
        for member,(lo,hi) in enumerate(zip(o['member_offsets'][:-1],o['member_offsets'][1:])):
            nt=o['neighbor_t'][case,lo:hi];ns=o['neighbor_sog_kn'][case,lo:hi]
            nw=kinematic_weights(nt,o['neighbor_pos'][case,lo:hi],ns,o['neighbor_cog_deg'][case,lo:hi])
            ii=np.searchsorted(nt,t,side='right')-1
            valid=ii>=0;safe=np.maximum(ii,0);valid &= (t-nt[safe]<=300)&(t>=nt[safe])&np.isfinite(ns[safe])
            a=pre[valid[pre]];b=post[valid[post]]
            if len(a)<2 or len(b)<2:continue
            jj=safe[a];kk=safe[b]
            changes.append(float(ns[kk].mean()/max(float(ns[jj].mean()),.5)-1))
            reliability.append(float(np.sqrt(nw[jj].mean()*nw[kk].mean())))
            usable.append(member)
        if not changes:
            rows.append(dict(available=False,anchor_step=anchor,target_change=target,usable_members=[],member_changes=[],member_weights=[]));continue
        changes=np.array(changes);reliability=np.array(reliability)
        avg=float(changes.mean());med=float(np.median(changes));fallback=not(reliability.sum()>0)
        trusted=avg if fallback else float(np.average(changes,weights=reliability))
        rows.append(dict(available=True,anchor_step=anchor,target_change=target,neighbor_mean_change=avg,neighbor_median_change=med,neighbor_reliable_change=trusted,
                         usable_members=usable,member_changes=changes.tolist(),member_weights=reliability.tolist(),reliable_fallback=fallback,
                         target_only=abs(target),neighbor_only=abs(avg),relative_mean=abs(target-avg),relative_median=abs(target-med),relative_reliable=abs(target-trusted)))
    return rows

def metric(y,score,threshold):
    y=np.asarray(y);score=np.asarray(score);pred=score>=threshold
    pr,re,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    both=len(np.unique(y))==2
    return dict(samples=len(y),positive=int(y.sum()),AP=float(average_precision_score(y,score)) if both else None,AUROC=float(roc_auc_score(y,score)) if both else None,
                F1=float(f1),precision=float(pr),recall=float(re),confusion=confusion_matrix(y,pred,labels=[0,1]).tolist())
def threshold(y,score):
    pr,re,ts=precision_recall_curve(y,score);v=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-15)
    return float(ts[np.flatnonzero(np.isclose(v,v.max(),rtol=0,atol=1e-12))[-1]])
def evaluate(rows):
    report={}
    for name in NAMES:
        tr=[r for r in rows if r['split']=='train'];th=threshold([r['y'] for r in tr],[r[name] for r in tr])
        report[name]={'threshold_train_max_F1':th,'splits':{}}
        for split in ['train','val']:
            rr=[r for r in rows if r['split']==split]
            domains={}
            for region in sorted(set(r['region'] for r in rr)):
                z=[r for r in rr if r['region']==region];domains[region]=metric([r['y'] for r in z],[r[name] for r in z],th)
                domains[region].update(source_windows=len(set(r['base_window_id'] for r in z)),target_mmsi=len(set(r['mmsi'] for r in z)))
            pooled=metric([r['y'] for r in rr],[r[name] for r in rr],th)
            report[name]['splits'][split]={'pooled':pooled,'macro':{m:float(np.mean([v[m] for v in domains.values() if v[m] is not None])) for m in ['AP','AUROC','F1','precision','recall']},'per_region':domains}
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);manifest=json.loads(a.manifest.read_text());rows=[];provenance=[];unavailable=[];details=[]
    for sample in manifest['samples']:
        p=a.samples/sample['file'];assert sha(p)==sample['sha256']
        with np.load(p,allow_pickle=False) as z:
            observed={key:z[key] for key in OBSERVED};sc=score_observed(observed)
            # Only after prediction has been formed do labels enter evaluation.
            y=z['y'].copy();region=str(z['region']);split=str(z['split']);wid=str(z['base_window_id']);mmsi=str(z['target_mmsi'])
        assert split in ('train','val')
        provenance.append(dict(file=str(p),sha256=sha(p),base_window_id=wid,split=split))
        if not all(x['available'] for x in sc):unavailable.append(dict(base_window_id=wid,region=region,split=split,reason='one_or_more_siblings_lack_usable_neighbors'));continue
        assert sc[0]['target_only']==sc[1]['target_only'] and sc[2]['target_only']==sc[3]['target_only']
        assert sc[0]['neighbor_only']==sc[2]['neighbor_only'] and sc[1]['neighbor_only']==sc[3]['neighbor_only']
        for i,x in enumerate(sc):
            meta=dict(base_window_id=wid,mmsi=mmsi,region=region,split=split,case_index=i,y=int(y[i]))
            rows.append({**meta,**{n:x[n] for n in NAMES}});details.append({**meta,**x})
    assert rows and not set(r['mmsi'] for r in rows if r['split']=='train')&set(r['mmsi'] for r in rows if r['split']=='val')
    summary={'status':'completed','scope':'Observed-input fixed statistics; no trained model and no test access. Source four-cell dependence retained.','protocol_sha256':sha(a.protocol),'code_sha256':sha(Path(__file__)),
             'source_manifest_sha256':sha(a.manifest),'source_windows':len(set(r['base_window_id'] for r in rows)),'scenarios':len(rows),'unavailable_sources':unavailable,
             'counts':{s:dict(source_windows=len(set(r['base_window_id'] for r in rows if r['split']==s)),mmsi=len(set(r['mmsi'] for r in rows if r['split']==s)),scenarios=sum(r['split']==s for r in rows)) for s in ['train','val']},
             'metrics':evaluate(rows),'sources':provenance}
    for name in ['target_only','neighbor_only']:
        for split in ['train','val']:
            v=summary['metrics'][name]['splits'][split]['pooled'];assert abs(v['AP']-.5)<1e-12 and abs(v['AUROC']-.5)<1e-12
    for name,data in [('statistics_summary.json',summary),('score_components.json',details)]:
        (a.out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (a.out/'scores.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({n:summary['metrics'][n]['splits']['val']['macro'] for n in NAMES}),flush=True)
if __name__=='__main__':main()
