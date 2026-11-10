from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');D=ROOT/'hawaii_event_feasibility_20261031';O=ROOT/'hawaii_independent_audit_20261031';sys.path.insert(0,str(O));import independent_eligibility as data
with np.load(D/'ood_inputs_v2.npz',allow_pickle=False) as z:actual={k:z[k] for k in z.files}
rows=json.loads((D/'ood_windows_v2.json').read_text())['all_windows']; a=data.a; records_checked=0; zeros=0; ferr=0.;herr=0.; mismatches=[];excluded_target=0

def features(t,p,s,c,member_ids):
 v=np.stack([s*np.sin(np.deg2rad(c)),s*np.cos(np.deg2rad(c))],axis=-1)*.514444444444;n=len(t);field=np.zeros((2,n,8),np.float64);heading=np.zeros((2,n,5),np.float64);count=np.zeros(n,dtype=int);obs=[]
 for m in member_ids:
  assert m!=target
  idx=data.groups[m];idx=idx[(data.t[idx]>=t[0]-300)&(data.t[idx]<=t[-1])&data.good[idx]];nt=data.t[idx];pos=np.stack([data.px[idx],data.py[idx]],axis=-1);speed=data.s[idx];course=data.c[idx];nv=np.stack([speed*np.sin(np.deg2rad(course)),speed*np.cos(np.deg2rad(course))],axis=-1)*.514444444444
  dt=np.diff(nt);valid=np.r_[False,(dt>0)&(dt<=300)]; ww=np.zeros(len(nt));ii=np.flatnonzero(valid);e=np.linalg.norm((pos[ii]-pos[ii-1])/dt[ii-1,None]-(nv[ii]+nv[ii-1])/2,axis=1);ww[ii]=np.exp(-(np.maximum(e-2.5,0)**2)/8)
  hv=pd.to_numeric(a.iloc[idx].heading,errors='coerce').to_numpy(float);hok=np.isfinite(hv)&(hv>=0)&(hv<360);yaw=np.zeros(len(nt));yok=np.zeros(len(nt),bool);q=(dt>0)&(dt<=300)&hok[1:]&hok[:-1];yaw[1:][q]=np.deg2rad((hv[1:][q]-hv[:-1][q]+180)%360-180)/dt[q];yok[1:]=q
  obs.append((m,nt,pos,nv,course,ww,hv,hok,yaw,yok))
 for j in range(n):
  phase=t[0] if j<n//2 else t[n//2];rec=[]
  for m,nt,pos,nv,course,w,h,ok,yaw,yok in obs:
   qi=np.searchsorted(nt,t[j],side='right')-1
   if qi<0 or not(0<=t[j]-nt[qi]<=300):continue
   ids=np.flatnonzero((nt>=phase)&(nt<=t[j])); ids=ids if len(ids) else np.array([qi]);weights=w[ids]; sm=weights.sum();v0=nv[ids].mean(axis=0);v1=np.average(nv[ids],axis=0,weights=weights) if sm>1e-12 else nv[qi]; d=pos[qi]-p[j];dist=np.exp(-.5*(d@d)/(4*1852)**2);delta=np.deg2rad((h[qi]-course[qi]+180)%360-180) if ok[qi] and np.isfinite(course[qi]) else 0.;rec.append((dist,dist*weights.mean(),v0,v1,bool(ok[qi]) and np.isfinite(course[qi]),np.sin(delta),np.cos(delta),bool(yok[qi]),yaw[qi],weights.mean()))
  count[j]=len(rec)
  if not rec:continue
  assert len(rec)<4
  d0=sum(r[0] for r in rec);d1=sum(r[1] for r in rec)
  if min(d0,d1)<=1e-12:continue
  rho=min(1,len(rec)/4)*np.mean([r[9] for r in rec])
  for k in (0,1):
   ws=np.asarray([r[k] for r in rec]);vel=np.asarray([r[2+k] for r in rec]);bg=np.average(vel,axis=0,weights=ws);field[k,j]=[*bg,*(v[j]-bg),0,0,rho,1]
   hvalid=np.asarray([r[4] for r in rec]);yw=np.asarray([r[7] for r in rec]);den=(ws*hvalid).sum();yd=(ws*yw).sum();
   if den>1e-12:heading[k,j,0]=sum(ws[z]*rec[z][5]*hvalid[z] for z in range(len(rec)))/den;heading[k,j,1]=sum(ws[z]*rec[z][6]*hvalid[z] for z in range(len(rec)))/den
   heading[k,j,2]=den/ws.sum()
   if yd>1e-12:heading[k,j,3]=sum(ws[z]*rec[z][8]*yw[z] for z in range(len(rec)))/yd
   heading[k,j,4]=yd/ws.sum()
 return field,heading,count
for r in rows:
 target=r['mmsi'];t=np.asarray(r['timestamps'],dtype=float);idx=np.asarray([data.lookup[(target,int(q))] for q in t]);p=np.stack([data.px[idx],data.py[idx]],axis=-1);s=data.s[idx];c=data.c[idx];expected_f,expected_h,count=features(t,p,s,c,r['cohort']);idxr=r['event_observed_window_index'];n=len(t);fe=float(np.max(np.abs(actual['field'][idxr,:,:n]-expected_f)));he=float(np.max(np.abs(actual['heading'][idxr,:,:n]-expected_h)));ferr=max(ferr,fe);herr=max(herr,he);records_checked+=n;zeros+=int((count==0).sum());
 if fe>2e-5 or he>2e-5 or not np.array_equal(actual['count'][idxr,:n],count):mismatches.append({'window':idxr,'field_error':fe,'heading_error':he})
assert not mismatches
out={'windows':len(rows),'target_steps_recomputed':records_checked,'no_members_steps':zeros,'max_abs_field_error':ferr,'max_abs_heading_error':herr,'issues':mismatches,'method':'Independent full pandas/reconstructed kinematic weights, latest alignment, stage means, weighted zero-order field and heading features; no calls to original field builder. All anchor cohorts <4, so no affine coefficients are estimated.','caveat':'Retrospective out-of-domain weak event-linked windows only, not a measurement of detection performance.'};(O/'independent_source_features.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
