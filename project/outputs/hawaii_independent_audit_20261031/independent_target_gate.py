from pathlib import Path
import sys,json,numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs');D=R/'hawaii_event_feasibility_20261031';O=R/'hawaii_independent_audit_20261031';sys.path.insert(0,str(O));import independent_eligibility as a
with np.load(D/'ood_inputs_v2.npz',allow_pickle=False) as z: stored=z['causal7'];mask=z['mask'];times=z['time_s']
rows=json.loads((D/'ood_windows_v2.json').read_text())['all_windows'];maxerr=0.;mismatch=[];steps=0; cap=13.73566722869873
for row in rows:
 i=row['event_observed_window_index'];t=np.asarray(row['timestamps'],float); ix=np.array([a.lookup[(row['mmsi'],int(q))] for q in t]);p=np.column_stack([a.px[ix],a.py[ix]]);s=a.s[ix];c=a.c[ix];v=np.column_stack([s*np.sin(np.deg2rad(c)),s*np.cos(np.deg2rad(c))])*.514444444444; raw=np.column_stack([p-p[0],v,s,np.sin(np.deg2rad(c)),np.cos(np.deg2rad(c))]).astype(np.float32)
 dt=np.diff(t);valid=np.r_[False,(dt>0)&(dt<=300)&np.isfinite(p[:-1]).all(1)&np.isfinite(p[1:]).all(1)&np.isfinite(s[:-1])&np.isfinite(s[1:])&np.isfinite(c[:-1])&np.isfinite(c[1:])];ii=np.flatnonzero(valid);e=np.linalg.norm((p[ii]-p[ii-1])/dt[ii-1,None]-(v[ii]+v[ii-1])/2,axis=1);w=np.zeros(len(t));w[ii]=np.exp(-np.maximum(0,e-2.5)**2/8)
 pred=raw.copy();current=np.zeros(2);prev=[]
 def bounded(vec):return vec*min(1.,cap/max(np.linalg.norm(vec),1e-12))
 for k in range(1,len(t)):
  dp=raw[k,:2].astype(float)-raw[k-1,:2].astype(float);wt=float(w[k]);delta=float(t[k]-t[k-1]);ok=bool(valid[k] and delta>0 and np.isfinite(delta) and np.isfinite(raw[k-1:k+1,2:4]).all());step=wt*dp
  if ok:
   report=.5*(raw[k-1,2:4].astype(float)+raw[k,2:4].astype(float))
   if wt>=.3:step=wt*dp+(1-wt)*report*delta
   elif len(prev)>=3:
    ref=np.median(np.asarray(prev[-5:]),axis=0);pos=dp/delta;dpv=np.linalg.norm(pos-ref);drv=np.linalg.norm(report-ref)
    if dpv<2.5 and dpv<.5*drv:step=dp
    elif drv<2.5 and drv<.5*dpv:step=wt*dp+(1-wt)*bounded(report)*delta
    else:step=wt*dp+(1-wt)*bounded(ref)*delta
  current+=step;pred[k,:2]=current
  if valid[k] and float(w[k])>=.7 and np.isfinite(raw[k,2:4]).all():prev.append(raw[k,2:4].astype(float).copy())
 for j in range(1,len(t)):
  if w[j]<.2:pred[j,2:7]=pred[j-1,2:7]
 error=float(np.max(np.abs(pred-stored[i,:len(t)])));maxerr=max(maxerr,error);steps+=len(t)
 if error>1e-5:mismatch.append({'index':i,'max_error':error})
 assert mask[i,:len(t)].all() and np.array_equal(times[i,:len(t)],t)
assert not mismatch
out={'windows':len(rows),'target_steps':steps,'max_absolute_error':maxerr,'mismatches':mismatch,'method':'independent original raw source target position, velocity, kinematic weight, original causal branch, velocity hold; no label data','constraint':'full-window OOD; time-unit seconds and position meters; no event-level labels used as model inputs'};(O/'independent_target_gate.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
