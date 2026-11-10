"""Independent loop reconstruction of train/validation causal gate candidates."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');SRC=ROOT/'gating_training_v2_20261027/data';BASE=ROOT/'causal_gate_main_v1_20261027';RUN=ROOT/'causal_gate_followup_v2_20261027';SCENE=ROOT/'labeling/labeled_v1/features_scene';OUT=ROOT/'gate_followup_independent_audit_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai');METHODS=('short3','smooth_guard','smooth_guard_cold')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def cap_vector(x,maxnorm):
 size=np.sqrt(np.dot(x,x));return x*min(1,maxnorm/max(size,1e-12))
def independent(raw,w,mask,valid,time,cap,ac,method):
 n=int(mask.sum());coords=np.zeros((len(mask),2),dtype=np.float64);codes=np.zeros(len(mask),np.int8);past=[];current=np.zeros(2)
 for k in range(1,n):
  d=raw[k,:2].astype(float)-raw[k-1,:2].astype(float);v0=raw[k-1,2:4].astype(float);v1=raw[k,2:4].astype(float);dt=float(time[k]-time[k-1]);wt=float(w[k]);part=wt*d
  good=bool(valid[k] and dt>0 and np.isfinite(dt) and np.isfinite(v0).all() and np.isfinite(v1).all())
  if good:
   vmid=(v0+v1)/2;speed_ok=bool(k>=2 and valid[k-1] and dt<=300 and np.linalg.norm(v0)<=cap and np.linalg.norm(v1)<=cap and np.linalg.norm(raw[k,2:4]-raw[k-1,2:4])/dt<=ac)
   if wt>=.3:
    part=part+(1-wt)*vmid*dt;codes[k]=1
   elif len(past)<3:
    if method=='smooth_guard_cold' and speed_ok:part=part+(1-wt)*cap_vector(vmid,cap)*dt;codes[k]=7
    else:codes[k]=5
   else:
    h=np.median(np.asarray(past[-3 if method=='short3' else -5:]),axis=0)
    delta_position=np.linalg.norm(d/dt-h);delta_report=np.linalg.norm(vmid-h)
    if delta_position<2.5 and 2*delta_position<delta_report:part=d;codes[k]=2
    elif delta_report<2.5 and 2*delta_report<delta_position:part=part+(1-wt)*cap_vector(vmid,cap)*dt;codes[k]=3
    elif method!='short3' and speed_ok and 2*delta_report<delta_position:part=part+(1-wt)*cap_vector(vmid,cap)*dt;codes[k]=6
    else:part=part+(1-wt)*cap_vector(h,cap)*dt;codes[k]=4
  current=current+part;coords[k]=current
  if valid[k] and float(w[k])>=.7 and np.isfinite(v1).all():past.append(v1.copy())
 coords[n:]=current
 return coords,codes

