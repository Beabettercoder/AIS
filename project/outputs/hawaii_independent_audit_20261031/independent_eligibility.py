import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/mnt/sda/WangZijia/AIS/outputs/hawaii_event_feasibility_20261031'); O=Path('/mnt/sda/WangZijia/AIS/outputs/hawaii_independent_audit_20261031');O.mkdir(exist_ok=True)
a=pd.read_parquet(R/'canonical_context_v2.parquet');a['_idx']=np.arange(len(a));a=a.sort_values(['mmsi','timestamp','_idx'],kind='stable').drop_duplicates(['mmsi','timestamp'],keep='last').reset_index(drop=True); mid=a.mmsi.astype(str).to_numpy(); t=a.timestamp.to_numpy(float); s=a.sog.to_numpy(float);c=a.cog.to_numpy(float); lat=a.lat.to_numpy(float);lon=a.lon.to_numpy(float);px=(lon+156.5)*111195*np.cos(np.deg2rad(21));py=(lat-21)*111195; good=np.isfinite(px)&np.isfinite(py)&np.isfinite(s)&np.isfinite(c)&(s>=0)&(s<102.2)&(c>=0)&(c<360);valid=np.zeros(len(a),bool);w=np.zeros(len(a));groups={}
for m,g in a.groupby('mmsi',sort=False):
 ix=g.index.to_numpy();groups[str(m)]=ix;dt=np.diff(t[ix]);ok=good[ix[:-1]]&good[ix[1:]]&(dt>0)&(dt<=300);j=np.flatnonzero(ok)+1;valid[ix[j]]=True;vx=s[ix]*1852/3600*np.sin(np.deg2rad(c[ix]));vy=s[ix]*1852/3600*np.cos(np.deg2rad(c[ix]));err=np.hypot((px[ix[j]]-px[ix[j-1]])/dt[j-1]-(vx[j]+vx[j-1])/2,(py[ix[j]]-py[ix[j-1]])/dt[j-1]-(vy[j]+vy[j-1])/2);w[ix[j]]=np.exp(-np.maximum(0,err-2.5)**2/8)
lookup={(m,int(tt)):i for i,(m,tt) in enumerate(zip(mid,t))};cand=json.loads((R/'candidate_targets.json').read_text())['windows']; prior=json.loads((R/'cohort_count_diagnostics.json').read_text())['rows'];expected={(x['mmsi'],x['anchor_time_s']):x for x in prior};res=[];dis=[]
for x in cand:
 ts=np.asarray(x['timestamps'],dtype=int);aidx=max(2,len(ts)//2-1);ix=np.array([lookup[(x['mmsi'],int(q))] for q in ts]);ai=ix[aidx];at=t[ai]
 if not(s[ai]>=3 and np.median(s[ix[:aidx+1]])>=3 and t[ix[-1]]-at>=180):continue
 vals=[]
 for mm,jj in groups.items():
  if mm==x['mmsi']:continue
  k=np.searchsorted(t[jj],at,side='right')-1
  if k<0:continue
  q=jj[k]
  if at-t[q]>300:continue
  if not(good[q] and valid[q] and w[q]>=.3 and s[q]>=3):continue
  if np.hypot(px[q]-px[ai],py[q]-py[ai])>5*1852:continue
  if abs((c[q]-c[ai]+180)%360-180)>30:continue
  if abs(s[q]-s[ai])>4.5:continue
  vals.append(mm)
 row={'name':x['name'],'mmsi':x['mmsi'],'anchor_time_s':int(at),'count':len(vals),'members':sorted(vals)};res.append(row);key=(row['mmsi'],row['anchor_time_s']); old=expected.get(key)
 if old is None or old['n_plus_speed_match']!=len(vals):dis.append({'key':key,'ours':len(vals),'old':None if old is None else old['n_plus_speed_match']})
summary={'method':'Independent pandas per-ship latest via searchsorted, independent kinematic and equirectangular computations; no model scores and no target labels','total_windows':len(res),'old_windows':len(prior),'diff':dis,'count_ge3':sum(x['count']>=3 for x in res),'groups':{n:{'windows':len(q),'max':max((x['count'] for x in q),default=0),'ge3':sum(x['count']>=3 for x in q)} for n in sorted(set(x['name'] for x in res)) for q in [[r for r in res if r['name']==n]]},'duplicate_target_note':'SEEKER duplicate time 1484006362 has identical position/speed/cog/heading; first vs last immaterial for current data','source_context_bounds_warning':'context received within incident associated intervals +/-1800 seconds, no certified receiver coverage; observation gaps are not proof of no ships'}
assert len(res)==len(prior) and not dis
(O/'independent_eligibility.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
