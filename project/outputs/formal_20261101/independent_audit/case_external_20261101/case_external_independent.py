from pathlib import Path
import datetime, hashlib, json, math, sys
import numpy as np
import pandas as pd

R=Path('/mnt/sda/WangZijia/AIS/outputs')
D=R/'formal_20261101/cases/v6'
OUT=R/'formal_20261101/independent_audit/case_external_20261101'
OUT.mkdir(parents=True, exist_ok=True)
KNOT=1852/3600; RADIUS=8*1852; SIGMA=4*1852
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def fromiso(s): return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def frame_load(case):
 cols=['mmsi','timestamp','lat','lon','sog','cog']
 if case=='MATSON':
  d=pd.read_parquet(R/'hawaii_event_feasibility_20261031/canonical_context_v2.parquet',columns=cols)
  box=(-159.,19.,-154.,23.); ident='366792000'
 else:
  p=Path('/mnt/sda/WangZijia/AIS/data/interim/ais_miami_enriched.parquet')
  d=pd.read_parquet(p,columns=cols); d=d[d.mmsi.astype(str)!='334979000']
  raw=pd.read_csv(R/'external_event_evidence_20261031/sara_regina_20240716_1200_2200_utc_noaa_raw.csv',dtype={'MMSI':str})
  raw['timestamp']=[fromiso(s) for s in raw.utc_iso]
  raw=raw.rename(columns={'MMSI':'mmsi','LAT':'lat','LON':'lon','SOG':'sog','COG':'cog'})[cols]
  d=pd.concat([d,raw],ignore_index=True);box=(-80.30,25.40,-79.80,26.20);ident='334979000'
 d=d.copy();d['mmsi']=d.mmsi.astype(str)
 for k in ('timestamp','lat','lon','sog','cog'): d[k]=pd.to_numeric(d[k],errors='coerce')
 d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=['timestamp']);d['_row']=np.arange(len(d))
 d=d.sort_values(['mmsi','timestamp','_row'],kind='stable').drop_duplicates(['mmsi','timestamp'],keep='last').reset_index(drop=True)
 lat0=(box[1]+box[3])/2;lon0=(box[0]+box[2])/2
 d['x']=(d.lon.to_numpy()-lon0)*111195*math.cos(math.radians(lat0));d['y']=(d.lat.to_numpy()-lat0)*111195
 theta=np.deg2rad(d.cog.to_numpy()); speed=d.sog.to_numpy();d['vx']=speed*KNOT*np.sin(theta);d['vy']=speed*KNOT*np.cos(theta)
 pp=d[['x','y']].to_numpy();vv=d[['vx','vy']].to_numpy();time=d.timestamp.to_numpy(float);ids=d.mmsi.to_numpy(str)
 valid=np.isfinite(pp).all(1)&np.isfinite(speed)&(speed>=0)&(speed<102.2)&np.isfinite(d.cog.to_numpy())&(d.cog.to_numpy()>=0)&(d.cog.to_numpy()<360)
 pair=np.zeros(len(d),bool);dt=np.diff(time);pair[1:]=(ids[1:]==ids[:-1])&(dt>0)&(dt<=300)&valid[1:]&valid[:-1]
 w=np.zeros(len(d));ii=np.flatnonzero(pair)
 e=np.linalg.norm((pp[ii]-pp[ii-1])/dt[ii-1,None]-(vv[ii]+vv[ii-1])/2,axis=1);w[ii]=np.exp(-np.maximum(0,e-2.5)**2/8)
 d['reliability']=w;d['pairvalid']=pair;d['obsvalid']=valid
 d=d.sort_values('timestamp',kind='stable').reset_index(drop=True)
 return d,ident

def independent_gated(d,tt,cap):
 n=len(tt); xyz=d[['x','y']].to_numpy(); vel=d[['vx','vy']].to_numpy();spd=d.sog.to_numpy();cog=d.cog.to_numpy()
 raw=np.zeros((20,7),np.float32);raw[:n]=np.column_stack([xyz-xyz[0],vel,spd,np.sin(np.deg2rad(cog)),np.cos(np.deg2rad(cog))])
 delta=np.diff(tt);valid=np.r_[False,(delta>0)&(delta<=300)]
 err=np.linalg.norm(np.diff(xyz,axis=0)/delta[:,None]-(vel[1:]+vel[:-1])/2,axis=1)
 w=np.ones(n);w[1:]=np.where(valid[1:],np.exp(-np.maximum(0,err-2.5)**2/8),1.);w=w.astype(np.float32)
 result=raw.copy(); acc=np.zeros(2);hist=[]
 def bounded(v): return v*min(1.,cap/max(float(np.linalg.norm(v)),1e-12))
 for k in range(1,n):
  dp=raw[k,:2].astype(float)-raw[k-1,:2].astype(float);wt=float(w[k]);dt=float(tt[k]-tt[k-1]);step=wt*dp
  good=bool(valid[k] and dt>0 and np.isfinite(dt) and np.isfinite(raw[k-1:k+1,2:4]).all())
  if good:
   rep=.5*(raw[k-1,2:4].astype(float)+raw[k,2:4].astype(float))
   if wt>=.3:step=wt*dp+(1-wt)*rep*dt
   elif len(hist)>=3:
    ref=np.median(np.asarray(hist[-5:]),axis=0);pos=dp/dt;dpos=np.linalg.norm(pos-ref);drep=np.linalg.norm(rep-ref)
    if dpos<2.5 and dpos<.5*drep:step=dp
    elif drep<2.5 and drep<.5*dpos:step=wt*dp+(1-wt)*bounded(rep)*dt
    else:step=wt*dp+(1-wt)*bounded(ref)*dt
  acc+=step;result[k,:2]=acc
  if valid[k] and float(w[k])>=.7 and np.isfinite(raw[k,2:4]).all():hist.append(raw[k,2:4].astype(float).copy())
 for k in range(1,n):
  if w[k]<.2:result[k,2:7]=result[k-1,2:7]
 result[n:]=0
 return result

def last_per_ship(d,t,ident):
 times=d.timestamp.to_numpy(float);left=np.searchsorted(times,t-300,'left');right=np.searchsorted(times,t,'right'); q=d.iloc[left:right]
 # Timestamp sort is stable; last() takes the actual latest row even when invalid.
 q=q.drop_duplicates('mmsi',keep='last');return q[q.mmsi!=ident]

def affine(p,v,w):
 if len(p)<4 or not np.isfinite(p).all() or not np.isfinite(v).all() or not np.isfinite(w).all() or w.sum()<=1e-12:return None
 w=w/w.sum();X=np.column_stack([np.ones(len(p)),p/RADIUS]);G=X.T@(w[:,None]*X)
 if not np.isfinite(np.linalg.cond(G)) or np.linalg.cond(G)>1e8:return None
 A=np.vstack([np.sqrt(w)[:,None]*X, np.sqrt(.001)*np.array([[0,1,0],[0,0,1]])]);Y=np.vstack([np.sqrt(w)[:,None]*v,np.zeros((2,2))])
 return np.linalg.lstsq(A,Y,rcond=None)[0]

def context(d,t,ident,pos,speed,cog,gated):
 q=last_per_ship(d,t,ident);assert not (q.mmsi==ident).any();assert (q.timestamp<=t).all()
 p=q[['x','y']].to_numpy(float)-pos;dist=np.linalg.norm(p,axis=1);inside=np.isfinite(dist)&(dist<=RADIUS);ngeo=int(inside.sum())
 if speed<.5: return ngeo,0,0,False,np.zeros(4),0.,[]
 dif=np.abs((q.cog.to_numpy(float)-cog+180)%360-180)
 plain=inside&q.obsvalid.to_numpy(bool)&(q.sog.to_numpy(float)>=.5)&(dif<=45)
 pq=q.loc[plain]; pp=p[plain];dd=dist[plain]; vv=pq[['vx','vy']].to_numpy(float); ww=pq.reliability.to_numpy(float); pv=pq.pairvalid.to_numpy(bool)
 trusted=pv&(ww>=.3); ntr=int(trusted.sum()); valid=False; co=None
 if ntr>=4:
  tp=pp[trusted];tw=ww[trusted]*np.exp(-.5*(dd[trusted]/SIGMA)**2);tw=tw/tw.sum();center=tp-(tw[:,None]*tp).sum(0);eig=np.linalg.eigvalsh(center.T@(tw[:,None]*center));X=np.column_stack([np.ones(len(tp)),tp/RADIUS]);cond=np.linalg.cond(X.T@(tw[:,None]*X))
  geom=eig[0]>=50**2 and np.isfinite(cond) and cond<=1e5
  if geom:
   dw=np.exp(-.5*(dd/SIGMA)**2); b0=affine(pp,vv,dw); b1=affine(pp,vv,dw*np.where(pv,ww,0.))
   if b0 is not None and b1 is not None:valid=True;co=b1
 if valid:
  grad=co[1:,:].T/RADIUS;field=np.r_[gated[2:4]-co[0],np.trace(grad),grad[1,0]-grad[0,1]];rho=min(1.,len(ww)/4)*float(np.where(pv,ww,0.).mean())
 else:field=np.zeros(4);rho=0.
 return ngeo,len(pq),ntr,valid,field,rho,pq.mmsi.astype(str).tolist()

def select_sara_nonoverlap(d,ident):
 s=d[(d.mmsi==ident)&d.obsvalid&d.lat.between(-90,90)&d.lon.between(-180,180)].sort_values('timestamp');times=s.timestamp.to_numpy(float);segments=[];seg=[]
 for t in times:
  if seg and t-seg[-1]>1800:segments.append(seg);seg=[]
  seg.append(t)
 if seg:segments.append(seg)
 windows=[]
 for seg in segments:
  start=seg[0];end=seg[-1]
  while start<=end:
   vals=[t for t in seg if start<=t<start+1380];start+=1380;sel=[]
   for t in vals:
    if not sel or t-sel[-1]>=60:sel.append(t)
   sel=sel[-20:]
   if len(sel)>=8:windows.append(sel)
 return windows

def select_rolling(d,ident,lo,hi):
 s=d[(d.mmsi==ident)&d.obsvalid&d.lat.between(-90,90)&d.lon.between(-180,180)].sort_values('timestamp');times=s.timestamp.to_numpy(float);out=[];last=-1e20
 for t in times[(times>=lo)&(times<=hi)]:
  if t-last<300:continue
  candidates=times[(times>=t-1380)&(times<=t)];sel=[]
  for x in candidates[::-1]:
   if not sel or sel[-1]-x>=60:sel.append(float(x))
  sel=sel[:20][::-1]
  if len(sel)>=8:out.append(sel);last=t
 return out

def main():
 cap=json.loads((R/'balanced_global_v2/causal_gate_main_v1_20261027/gate_cap.json').read_text())['cap_mps'];audit={};source_locks={}
 for name in ('case_input_code_lock.json','rolling_input_code_lock.json','case_models_prescore_lock.json','rolling_models_prescore_lock.json'):
  obj=json.loads((D/name).read_text()); source_locks[name]=digest(D/name)
  if 'sha256' in obj:
   for p,h in obj['sha256'].items(): assert digest(p)==h,(name,p)
 fr=json.loads((R/'hawaii_event_feasibility_20261031/observed_event_windows.json').read_text())['windows']; mat=[w['timestamps'] for w in fr if str(w['mmsi'])=='366792000'];assert len(mat)==5
 for case in ('MATSON','SARA'):
  df,ident=frame_load(case);target=df[df.mmsi==ident].copy();target_index=target.set_index('timestamp',drop=False,verify_integrity=True)
  non=json.loads((D/'realcase_input_audit.json').read_text())['cases'][case];roll=json.loads((D/'rolling_input_audit.json').read_text())['cases'][case]
  non_times=[[fromiso(x) for x in row['times_utc']] for row in non];roll_times=[[fromiso(x) for x in row['times_utc']] for row in roll]
  expected_non=mat if case=='MATSON' else select_sara_nonoverlap(df,ident);assert all(np.array_equal(a,b) for a,b in zip(non_times,expected_non)) and len(non_times)==len(expected_non)
  interval=('2017-01-13T16:01:56+00:00','2017-01-13T19:58:54+00:00') if case=='MATSON' else ('2024-07-16T12:00:00+00:00','2024-07-16T22:00:00+00:00')
  expected_roll=select_rolling(df,ident,*[fromiso(x) for x in interval]);assert len(roll_times)==len(expected_roll) and all(np.array_equal(a,b) for a,b in zip(roll_times,expected_roll))
  audit[case]={}
  for subset,rows in (('nonoverlap',non),('rolling',roll)):
   npz=D/('realcase_inputs.npz' if subset=='nonoverlap' else 'rolling_inputs.npz');z=np.load(npz,allow_pickle=False);X={k:z[case+'__'+k] for k in ('self','mask','time_s','field','rho','field_valid')}
   count=0;max_self=0.;max_field=0.;max_rho=0.;valid_count=0;max_age=0.;ids_checked=0
   for i,row in enumerate(rows):
    tt=np.asarray([fromiso(x) for x in row['times_utc']],float);n=len(tt);assert n==row['count'] and 8<=n<=20 and np.all(np.diff(tt)>=60) and X['mask'][i].sum()==n
    d=target_index.loc[tt];assert len(d)==n and np.all(d.mmsi==ident) and np.all(d.timestamp.to_numpy()==tt)
    g=independent_gated(d,tt,cap);max_self=max(max_self,float(np.max(np.abs(g-X['self'][i]))));assert np.max(np.abs(X['time_s'][i,:n]-(tt-tt[0])))<1e-8 and np.all(X['time_s'][i,n:]==0)
    for j,t in enumerate(tt):
     p=d[['x','y']].to_numpy()[j];speed=float(d.sog.iloc[j]);cog=float(d.cog.iloc[j]);ng,npn,nt,ok,ff,rr,ids=context(df,t,ident,p,speed,cog,g[j]);assert ng==row['neighbors_geo_per_step'][j];assert npn==row['neighbors_plain_per_step'][j];assert nt==row['neighbors_trusted_per_step'][j];assert ok==bool(X['field_valid'][i,j]);assert set(ids)==set(row['neighbor_ids_per_step'][j]);ids_checked+=len(ids)
     max_field=max(max_field,float(np.max(np.abs(ff-X['field'][i,j]))));max_rho=max(max_rho,abs(rr-float(X['rho'][i,j])));valid_count+=ok;count+=1
   assert max_self<2e-4,(case,subset,'self',max_self);assert max_field<2e-5,(case,subset,'field',max_field);assert max_rho<1e-6
   audit[case][subset]={'windows':len(rows),'steps':count,'valid_field_steps':valid_count,'self_max_abs_diff':max_self,'field_max_abs_diff':max_field,'rho_max_abs_diff':max_rho,'plain_neighbor_selections_checked':ids_checked,'sampling_independently_reconstructed':True,'no_future_neighbor':True,'target_excluded':True}
  print(case,json.dumps(audit[case]),flush=True)
 out={'scope':'Independent pandas/numpy reimplementation; all windows and all steps; raw source not model labels. No assumption that event time is pointwise truth.','audit':audit,'source_locks_sha256':source_locks}
 p=OUT/'case_input_independent.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8')

if __name__=='__main__':main()
