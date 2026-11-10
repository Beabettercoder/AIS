from pathlib import Path
import numpy as np,sys,importlib.util,json
R=Path('/mnt/sda/WangZijia/AIS/outputs');path=R/'balanced_global_v2/labeling/labeled_v1/features_scene/houston/train.npz';saved=R/'balanced_global_v2/causal_gate_main_v1_20261027/gate_data/houston/train.npz'
s=importlib.util.spec_from_file_location('gate',R/'balanced_global_v2/causal_gate_main_v1_20261027/gate_builder.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in ('mask','time_s','target_pos','target_sog_kn','target_cog_deg','self_gate_weight','kinematic_valid','window_id')}
with np.load(saved,allow_pickle=False) as z: expected=z['causal7']
errw=0.;errx=0.;errrest=0.;comp=0
for i in range(len(d['mask'])):
 m=d['mask'][i];n=int(m.sum());t=d['time_s'][i,:n];p=d['target_pos'][i,:n];sog=d['target_sog_kn'][i,:n];c=d['target_cog_deg'][i,:n];theta=np.deg2rad(c);v=np.column_stack([sog*np.sin(theta),sog*np.cos(theta)])*1852/3600
 raw=np.zeros((20,7),np.float32);raw[:n]=np.column_stack([p-p[0],v,sog,np.sin(theta),np.cos(theta)]);tm=np.zeros(20);tm[:n]=t;mask=np.arange(20)<n;valid=np.zeros(20,bool);dt=np.diff(t);valid[1:n]=(dt>0)&(dt<=300);w=np.zeros(20,dtype=np.float32);w[:n]=1;err=np.linalg.norm(np.diff(p,axis=0)/dt[:,None]-(v[1:]+v[:-1])/2,axis=1);w[1:n]=np.where(valid[1:n],np.exp(-np.maximum(0,err-2.5)**2/8),1.);assert np.array_equal(valid,d['kinematic_valid'][i]);errw=max(errw,float(np.max(np.abs(w[:n]-d['self_gate_weight'][i,:n]))));out,_=g.causal(raw,w,mask,valid,tm,13.73566722869873)
 for j in range(1,n):
  if w[j]<.2:out[j,2:7]=out[j-1,2:7]
 errx=max(errx,float(np.max(np.abs(out[:n,:2]-expected[i,:n,:2]))));errrest=max(errrest,float(np.max(np.abs(out[:n,2:]-expected[i,:n,2:]))));comp+=1
print(json.dumps({'train_houston_windows':comp,'max_abs_self_weight':errw,'max_abs_gated_position_m':errx,'max_abs_other_channels':errrest}))
