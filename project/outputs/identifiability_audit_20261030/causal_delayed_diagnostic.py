"""Train-source observational proxies. Delayed evidence is diagnostic only."""
from pathlib import Path
import json, hashlib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/identifiability_audit_20261030')
manifest=json.loads((ROOT/'dataset_manifest_combined.json').read_text())
rows=sorted([r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='train'],key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())[:400]
records=[]
for k,row in enumerate(rows):
    a={}
    for view in ['base','suffix']:
        with np.load(ROOT/row['files'][view]['path'],allow_pickle=False) as z:
            a[view]={key:z[key].copy() for key in ('neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets','cohort_mmsi')}
    b,s=a['base'],a['suffix']
    for case in range(4):
        for j,(lo,hi) in enumerate(zip(b['member_offsets'][:-1],b['member_offsets'][1:])):
            lo,hi=int(lo),int(hi)
            t=b['neighbor_t'][case,lo:hi];p=b['neighbor_pos'][case,lo:hi]
            sb=b['neighbor_sog_kn'][case,lo:hi];sp=s['neighbor_sog_kn'][case,lo:hi]
            c=b['neighbor_cog_deg'][case,lo:hi]; ang=np.deg2rad(c)
            v=sp[:,None]*0.514444444444*np.column_stack([np.sin(ang),np.cos(ang)])
            for i in range(3,len(t)-1):
                dt=t[i]-t[i-1];dn=t[i+1]-t[i]
                if not (0<dt<=300 and 0<dn<=300):continue
                if not np.isfinite(p[i-1:i+2]).all() or not np.isfinite(v[i-3:i+2]).all() or not np.isfinite(sb[i-3:i+2]).all():continue
                h=np.median(sp[i-3:i]);
                if h>=2 or sb[i]>=2:continue
                changed=abs(sp[i]-sb[i])>1e-9;prev=abs(sp[i-1]-sb[i-1])>1e-9;nextchanged=abs(sp[i+1]-sb[i+1])>1e-9
                if not changed and (prev or nextchanged):continue
                # Positives are injected reports; negatives are untouched reports
                # selected by base-observed abrupt change as a manoeuvre proxy.
                if not changed and abs(sb[i]-sb[i-1])<1:continue
                vp=(p[i]-p[i-1])/dt;vn=(p[i+1]-p[i])/dn
                ecur=np.linalg.norm(vp-v[i]);emid=np.linalg.norm(vp-(v[i]+v[i-1])*.5)
                enext=np.linalg.norm(vn-v[i]);enext_report=np.linalg.norm(v[i+1]-v[i])
                ratio=abs(sp[i]-h)/(h+.5)
                records.append((str(row['window_id']),str(row['mmsi']),str(b['cohort_mmsi'][j]),float(t[i]),int(case),int(changed),int(prev),int(nextchanged),float(ecur),float(emid),float(ratio),float(enext),float(enext_report),float(h),float(sb[i]),float(sp[i])))
    if (k+1)%100==0:print('progress',k+1,flush=True)
dtype=[('source','U40'),('target','U30'),('neighbor','U30'),('time','f8'),('case','i4'),('y','i4'),('prevchanged','i4'),('nextchanged','i4'),('ecur','f8'),('emid','f8'),('ratio','f8'),('futurepos','f8'),('future_report_delta','f8'),('hist_sog','f8'),('base_sog','f8'),('sog','f8')]
r=np.array(records,dtype=dtype); np.savez_compressed(OUT/'records.npz',r=r)
def metrics(mask,key):
    y=r['y'][mask];x=r[key][mask];return dict(n=int(len(y)),positive=int(y.sum()),negative=int((1-y).sum()),target_mmsi=int(len(np.unique(r['target'][mask]))),neighbor_mmsi=int(len(np.unique(r['neighbor'][mask]))),roc_auc=float(roc_auc_score(y,x)),ap=float(average_precision_score(y,x)),positive_q=np.quantile(x[y==1],[.1,.5,.9]).tolist(),negative_q=np.quantile(x[y==0],[.1,.5,.9]).tolist())
allmask=np.ones(len(r),bool)
out={'scope':'400 SHA-ordered training source windows; controlled suffix velocity-only corruption vs baseline unmodified acceleration proxy. base reference may contain unknown noise. ground truth on corruption only, no real maneuver labels. Duplicate reports across sources/four cases, AP prevalence depends on case selection. Next-observation statistics are OFFLINE diagnostics, unavailable at current time.', 'data':{key:metrics(allmask,key) for key in ('emid','ecur','ratio','futurepos','future_report_delta')},'sources':len(rows),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'causal_delayed_summary.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
