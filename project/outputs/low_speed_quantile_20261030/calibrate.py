"""Unlabelled, train/Base-only low-speed consistency residual calibration."""
import numpy as np, json, hashlib
from pathlib import Path
R=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
O=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_quantile_20261030')
m=json.loads((R/'dataset_manifest_combined.json').read_text())
rows=sorted([r for r in m['samples'] if r['version']=='relaxed' and r['split']=='train'],key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())
seen=set(); es=[]; nseen=0
for row in rows:
 with np.load(R/row['files']['base']['path'],allow_pickle=False) as z:
  t=z['neighbor_t']; p=z['neighbor_pos']; s=z['neighbor_sog_kn']; c=z['neighbor_cog_deg']; off=z['member_offsets']; ids=z['cohort_mmsi']
  for ci in range(4):
   for j,(lo,hi) in enumerate(zip(off[:-1],off[1:])):
    lo,hi=int(lo),int(hi); tt=t[ci,lo:hi]; pp=p[ci,lo:hi]; ss=s[ci,lo:hi]; cc=c[ci,lo:hi]
    theta=np.deg2rad(cc); v=ss[:,None]*.514444444444*np.column_stack([np.sin(theta),np.cos(theta)])
    for i in range(1,len(tt)):
     dt=tt[i]-tt[i-1]
     if not (0<dt<=300 and np.isfinite(pp[i-1:i+1]).all() and np.isfinite(v[i-1:i+1]).all()): continue
     vp=(pp[i]-pp[i-1])/dt; speed=min(np.linalg.norm(vp),np.linalg.norm(v[i]))/.514444444444
     if not speed<2: continue
     nseen+=1
     key=(str(ids[j]),float(tt[i-1]),float(tt[i]),pp[i-1:i+1].tobytes(),v[i-1:i+1].tobytes())
     if key in seen: continue
     seen.add(key); es.append(float(np.linalg.norm(vp-(v[i]+v[i-1])/2)))
if len(es)<100: raise SystemExit('insufficient low-speed observations; stop rather than infer threshold')
q95,q99=np.quantile(es,[.95,.99]); assert q99>q95 and np.isfinite([q95,q99]).all()
d={'training_only':'all train sources sorted deterministically by SHA256 window ID; Base view only, four constructed cases, no labels','unique_records':len(es),'before_dedup':nseen,'q95':float(q95),'q99':float(q99),'criterion':'min(position-difference speed, current reported speed)<2 knots','valid_dt':'0<dt<=300 seconds','dedup':'MMSI plus previous/current time, positions, velocity vectors','risk':'Base observations include possible unknown natural errors; four constructed cases and background identities can recur; quantiles not calibrated FAR','fallback':'if q99<=q95 or fewer than 100 observations, stop'}
O.mkdir(parents=True,exist_ok=True); f=O/'calibration_full.json'
if f.exists(): raise FileExistsError(f)
f.write_text(json.dumps(d,indent=2),encoding='utf8'); print(json.dumps(d,indent=2))

