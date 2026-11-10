"""Reconstruct observed-only sequence features and relative statistics independently."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');RUN=ROOT/'relation_dual_pressure_v2_20261027';STUDY=RUN/'model_study';OUT=ROOT/'relation_independent_audit_20261027'
VIEWS=('base','natural','suffix');MODES=('target','mean','median','absolute_median','absolute_report_median','mixed_report_median')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def weighted_median(values,weights):
 order=np.argsort(values,kind='stable');x=np.asarray(values)[order];w=np.asarray(weights)[order]
 if w.sum()<=1e-12:return float(np.median(x))
 i=int(np.searchsorted(np.cumsum(w),w.sum()/2,side='left'));return float(x[min(i,len(x)-1)])
def compute_weights(t,p,s,c):
 a=np.deg2rad(c);v=s[:,None]*(1852/3600)*np.stack((np.sin(a),np.cos(a)),axis=1)
 dt=np.diff(t);good=(dt>0)&(dt<=300)&np.isfinite(p[:-1]).all(1)&np.isfinite(p[1:]).all(1)&np.isfinite(s[:-1])&np.isfinite(s[1:])&np.isfinite(c[:-1])&np.isfinite(c[1:])
 ix=np.flatnonzero(good)+1;pv=(p[ix]-p[ix-1])/dt[ix-1,None];rv=(v[ix]+v[ix-1])/2;err=np.linalg.norm(pv-rv,axis=1)
 rel=err/(np.minimum(np.linalg.norm(pv,axis=1),np.linalg.norm(rv,axis=1))+.3)
 wa=np.zeros(len(t));wm=np.zeros(len(t));wa[ix]=np.exp(-np.maximum(0,err-2.5)**2/8);wm[ix]=np.minimum(wa[ix],np.exp(-np.maximum(0,rel-1)**2/2))
 return wa,wm
def features(z,case):
 t=z['target_t'][case];s=z['target_sog_kn'][case];c=z['target_cog_deg'][case];n=len(t);anchor=max(2,n//2-1);pre=np.arange(anchor+1);post=np.arange(anchor+1,n)[-3:]
 target_change=float(s[post].mean()/max(float(s[pre].mean()),.5)-1)
 changes=[];wa_members=[];wm_members=[];rca=[];rcm=[];grid=[];grid_a=[];grid_m=[]
 for lo,hi in zip(z['member_offsets'][:-1],z['member_offsets'][1:]):
  lo=int(lo);hi=int(hi);nt=z['neighbor_t'][case,lo:hi];ns=z['neighbor_sog_kn'][case,lo:hi]
  wa,wm=compute_weights(nt,z['neighbor_pos'][case,lo:hi],ns,z['neighbor_cog_deg'][case,lo:hi]);ix=np.searchsorted(nt,t,side='right')-1;safe=np.maximum(ix,0)
  present=(ix>=0)&(t>=nt[safe])&(t-nt[safe]<=300);ai=pre[present[pre]];bi=post[present[post]]
  if len(ai)<2 or len(bi)<2:continue
  j=safe[ai];k=safe[bi];changes.append(float(ns[k].mean()/max(float(ns[j].mean()),.5)-1))
  wa_members.append(float(np.sqrt(wa[j].mean()*wa[k].mean())));wm_members.append(float(np.sqrt(wm[j].mean()*wm[k].mean())))
  for weights,dest in [(wa,rca),(wm,rcm)]:
   x=float(np.average(ns[j],weights=weights[j])) if weights[j].sum()>1e-12 else float(ns[j].mean())
   y=float(np.average(ns[k],weights=weights[k])) if weights[k].sum()>1e-12 else float(ns[k].mean())
   dest.append(y/max(x,.5)-1)
  gs=np.full(n,np.nan);ga=np.zeros(n);gm=np.zeros(n);gs[present]=ns[safe[present]];ga[present]=wa[safe[present]];gm[present]=wm[safe[present]]
  grid.append(gs);grid_a.append(ga);grid_m.append(gm)
 assert changes
 changes=np.asarray(changes);wa_members=np.asarray(wa_members);wm_members=np.asarray(wm_members)
 mean=float(changes.mean());med=float(np.median(changes))
 absolute=float(np.average(changes,weights=wa_members)) if wa_members.sum()>1e-12 else mean
 mixed=float(np.average(changes,weights=wm_members)) if wm_members.sum()>1e-12 else mean
 stats={'target_only':abs(target_change),'neighbor_only':abs(mean),'mean':abs(target_change-mean),'median':abs(target_change-med),
  'absolute_weighted_mean':abs(target_change-absolute),'mixed_weighted_mean':abs(target_change-mixed),
  'mixed_weighted_median':abs(target_change-weighted_median(changes,wm_members)),
  'mixed_report_weighted_median':abs(target_change-float(np.median(rcm))),
  'absolute_weighted_median':abs(target_change-weighted_median(changes,wa_members)),
  'absolute_report_weighted_median':abs(target_change-float(np.median(rca)))}
 grid=np.asarray(grid);grid_a=np.asarray(grid_a);grid_m=np.asarray(grid_m)
 contexts={name:np.zeros((n,3),np.float32) for name in MODES[1:]}
 for k in range(n):
  mask=np.isfinite(grid[:,k]);values=grid[mask,k];w=grid_a[mask,k]
  if len(values):
   for name,val in [('mean',values.mean()),('median',np.median(values)),('absolute_median',weighted_median(values,w))]:
    contexts[name][k,0]=val/10;contexts[name][k,2]=len(values)/8
  for name,weights in [('absolute_report_median',grid_a),('mixed_report_median',grid_m)]:
   vals=[]
   for j in range(len(grid)):
    if not mask[j]:continue
    start=0 if k<=anchor else anchor+1;ii=np.arange(start,k+1);ii=ii[np.isfinite(grid[j,ii])];ww=weights[j,ii];z0=grid[j,ii]
    vals.append(float(np.average(z0,weights=ww)) if ww.sum()>1e-12 else float(z0.mean()))
   if vals:contexts[name][k,0]=np.median(vals)/10;contexts[name][k,2]=len(vals)/8
 for v in contexts.values():v[1:,1]=np.diff(v[:,0])
 a=np.deg2rad(c);target=np.column_stack((s/10,np.sin(a),np.cos(a),np.r_[0,np.diff(s)]/10)).astype(np.float32)
 output={'target':np.concatenate((target,np.zeros((n,3),np.float32)),axis=1)}
 output.update({name:np.concatenate((target,v),axis=1) for name,v in contexts.items()})
 return output,stats

