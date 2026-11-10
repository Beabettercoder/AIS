#!/usr/bin/env python3
"""Review actual target observations; candidate labels are rules, never ground truth."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv, hashlib, json, math
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS')
SRC=ROOT/'outputs/balanced_global_v2/relation_dual_pressure_v2_20261027'
OUT=ROOT/'outputs/real_low_speed_review_20261031'
HEADING=ROOT/'outputs/heading_coverage_20261030/heading_train.npz'
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def circ(x):
    z=np.exp(1j*np.deg2rad(x)).mean()
    return float(np.degrees(np.angle(z))%360),float(abs(z))
def signed_delta(a,b):return float((b-a+180)%360-180)
def safe(x):return None if not math.isfinite(float(x)) else float(x)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    mf=SRC/'dataset_manifest_combined.json'
    rows=[r for r in json.loads(mf.read_text())['samples'] if r.get('split')=='train' and r.get('version')=='relaxed']
    bym=defaultdict(list)
    for r in rows:bym[str(r['mmsi'])].append(r)
    chosen=[]
    for m, items in sorted(bym.items()):
        chosen.extend(sorted(items,key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())[:2])
    # This selection precedes all event labels and Heading availability.
    H=np.load(HEADING,allow_pickle=False); hm={str(x):i for i,x in enumerate(H['window_id'])}
    assert len(hm)==len(H['window_id'])
    stats=Counter(); rec=[]
    for row in chosen:
        key=row['window_id']; fi=SRC/row['files']['base']['path']; assert sha(fi)==row['files']['base']['sha256']
        with np.load(fi,allow_pickle=False) as z:
            t=z['source_target_t'].astype(float);p=z['source_target_pos'].astype(float)
            s=z['source_target_sog_kn'].astype(float); c=z['source_target_cog_deg'].astype(float)
        n=len(t);stats['selected_before_filters']+=1
        if n<9 or not (np.isfinite(t).all() and np.isfinite(p).all() and np.isfinite(s).all() and np.isfinite(c).all()):stats['short_or_nonfinite']+=1;continue
        dt=np.diff(t)
        if not np.all((dt>0)&(dt<=300)):stats['interval_outside_0_300']+=1;continue
        med=float(np.median(s)); displacement=float(np.linalg.norm(p[-1]-p[0]))
        if not (0.5<=med<=5 and (s<2).sum()>=2 and displacement>=100):stats['outside_low_speed_scope']+=1;continue
        v=(s*0.5144444444)[:,None]*np.column_stack((np.sin(np.deg2rad(c)),np.cos(np.deg2rad(c))))
        residual=np.linalg.norm(np.diff(p,axis=0)/dt[:,None]-(v[1:]+v[:-1])/2,axis=1)
        if np.quantile(residual,.9)>4:stats['failed_kinematic_quality']+=1;continue
        k=n//3; pre=np.arange(k);post=np.arange(n-k,n); pv=pre[s[pre]>=1];qv=post[s[post]>=1]
        if len(pv)<2 or len(qv)<2:stats['insufficient_moving_cog']+=1;continue
        ca,cr=circ(c[pv]);cb,cs=circ(c[qv]);
        if min(cr,cs)<.5:stats['ambiguous_cog_direction']+=1;continue
        dc=signed_delta(ca,cb); magnitude=abs(dc)
        if magnitude>=30:group='turn_rule'
        elif magnitude<=10:group='straight_control'
        else:stats['intermediate_cog_change']+=1;continue
        idx=hm[key]; assert str(H['target_mmsi'][idx])==str(row['mmsi'])
        np.testing.assert_array_equal(H['target_t'][idx,:n],t)
        heading=H['target_heading'][idx,:n].astype(float); hv=H['target_valid'][idx,:n].astype(bool)
        assert np.all(np.isfinite(heading[hv])) and np.all((heading[hv]>=0)&(heading[hv]<360))
        hp=pre[hv[pre]]; hq=post[hv[post]]; known=len(hp)>=2 and len(hq)>=2
        dh=float('nan'); agree=float('nan');resultants=[]
        if known:
            ha,hr=circ(heading[hp]);hb,hs=circ(heading[hq]);known=min(hr,hs)>=.5
            if known:
                dh=signed_delta(ha,hb); agree=float(np.sign(dh)==np.sign(dc)) if magnitude>0 else float('nan');resultants=[hr,hs]
        item=dict(window_id=key,mmsi=str(row['mmsi']),region=row['region'],group=group,
          start_utc=datetime.fromtimestamp(t[0],tz=timezone.utc).isoformat(),end_utc=datetime.fromtimestamp(t[-1],tz=timezone.utc).isoformat(),
          n_reports=n,median_sog_kn=med,n_below2=int((s<2).sum()),displacement_m=displacement,
          e90_mps=float(np.quantile(residual,.9)),cog_delta_deg=dc,cog_start_resultant=cr,cog_end_resultant=cs,
          n_heading_valid=int(hv.sum()),heading_first_valid=int(len(hp)),heading_last_valid=int(len(hq)),heading_usable=bool(known),
          heading_delta_deg=safe(dh),direction_agrees=safe(agree),source_npz=str(fi),source_sha256=row['files']['base']['sha256'])
        rec.append(item);stats[group]+=1;stats['heading_usable_'+group]+=int(known)
    cols=list(rec[0]) if rec else []
    with (OUT/'review_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rec)
    groups={}
    for label in ('turn_rule','straight_control'):
        x=[r for r in rec if r['group']==label];use=[r for r in x if r['heading_usable']]
        groups[label]=dict(windows=len(x),mmsi=len(set(r['mmsi'] for r in x)),heading_usable_windows=len(use),heading_usable_mmsi=len(set(r['mmsi'] for r in use)),
          by_region=dict(Counter(r['region'] for r in x)),heading_abs_delta_median_deg=float(np.median([abs(r['heading_delta_deg']) for r in use])) if use else None,
          heading_change_at_least_20deg=int(sum(abs(r['heading_delta_deg'])>=20 for r in use)),
          same_turn_direction=int(sum(r['direction_agrees']==1 for r in use)))
    # Predict the pre-defined COG-based group using heading change; this is not truth about manoeuvres or AIS contamination.
    use=[r for r in rec if r['heading_usable']]; y=np.array([r['group']=='turn_rule' for r in use]); score=np.array([abs(r['heading_delta_deg']) for r in use]);ids=np.array([r['mmsi'] for r in use]);boot=None; metrics=None
    if y.sum()>=2 and (~y).sum()>=2:
        from sklearn.metrics import roc_auc_score,average_precision_score
        metrics=dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()),auc=float(roc_auc_score(y,score)),ap=float(average_precision_score(y,score)))
        unique=np.unique(ids); rng=np.random.default_rng(20261031); accum=[]
        for b in range(2000):
            draws=rng.choice(unique,len(unique),replace=True);cnt=Counter(draws);w=np.array([cnt[z] for z in ids],float)
            if w[y].sum()>0 and w[~y].sum()>0:accum.append(float(roc_auc_score(y,score,sample_weight=w)))
        boot=dict(resamples_requested=2000,resamples_valid=len(accum),unit='target MMSI',auc_95=[float(x) for x in np.quantile(accum,[.025,.975])]) if accum else None
    result=dict(protocol=protocol,protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(Path(__file__)),manifest_sha256=sha(mf),heading_cache_sha256=sha(HEADING),
                selection=dict(input_windows=len(rows),train_mmsi=len(bym),maximum_windows_per_mmsi=2,selected=len(chosen),counts=dict(stats)),groups=groups,heading_score_vs_cog_rule=metrics,bootstrap=boot,
                limitations=['COG-rule candidates are not externally verified manoeuvre labels','source AIS may contain natural errors','low-speed heading and COG differ physically','train/val background vessels can overlap','at most two train windows per target MMSI; windows may overlap','heading not verified to be sensor-independent','no human or real sensor fault ground truth used'])
    (OUT/'review_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
