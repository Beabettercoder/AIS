from pathlib import Path
import importlib.util,json
import numpy as np

root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
work=root/'causal_gate_main_v1_20261027';src=root/'gating_training_v2_20261027/data'
out=root/'independent_audit_causal_v1_20261027';regions=['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai']
capdata=json.loads((work/'gate_cap.json').read_text())
speeds=[]
for r in regions:
 with np.load(src/r/'train.npz',allow_pickle=False) as z:
  mask=z['mask']&z['kinematic_valid']&(z['w_self']>=.7)
  speeds.append(np.linalg.norm(z['raw_legacy7'][...,2:4][mask],axis=1))
cap=float(np.quantile(np.concatenate(speeds),.995))
assert abs(cap-capdata['cap_mps'])<1e-12

def independently_gate(raw,weight,mask,valid,t):
 n=int(mask.sum()); generated=np.array(raw,copy=True);past=[];out=np.zeros((n,2)); branch=np.zeros(len(mask),np.int8)
 for k in range(1,n):
  dp=raw[k,:2].astype(float)-raw[k-1,:2].astype(float)
  dt=float(t[k]-t[k-1]);wk=float(weight[k]);step=wk*dp
  ok=bool(valid[k] and dt>0 and np.isfinite(dt) and np.isfinite(raw[k-1:k+1,2:4]).all())
  if ok:
   v=(raw[k-1,2:4].astype(float)+raw[k,2:4].astype(float))/2
   if wk>=.3:
    step+=((1-wk)*dt)*v;branch[k]=1
   elif len(past)>=3:
    reference=np.median(np.asarray(past[-5:]),axis=0)
    p_dist=float(np.linalg.norm(dp/dt-reference));v_dist=float(np.linalg.norm(v-reference))
    if p_dist<2.5 and p_dist<.5*v_dist:step=dp;branch[k]=2
    elif v_dist<2.5 and v_dist<.5*p_dist:
     step+=((1-wk)*dt)*v*min(1.,cap/max(float(np.linalg.norm(v)),1e-12));branch[k]=3
    else:
     step+=((1-wk)*dt)*reference*min(1.,cap/max(float(np.linalg.norm(reference)),1e-12));branch[k]=4
   else:branch[k]=5
  out[k]=out[k-1]+step
  if valid[k] and wk>=.7 and np.isfinite(raw[k,2:4]).all():past.append(raw[k,2:4].astype(float))
 generated[:n,:2]=out;generated[n:,:2]=out[-1]
 return generated,branch

module_spec=importlib.util.spec_from_file_location('candidate_gate_module',work/'gate_builder.py');module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
rng=np.random.default_rng(278120);maxerror=0.;max_prefix=0.;nwindows=0;trials=0;branch_counts=np.zeros(6,dtype=int)
for r in regions:
 for split in ('train','val'):
  with np.load(src/r/(split+'.npz'),allow_pickle=False) as z, np.load(work/'gate_data'/r/(split+'.npz'),allow_pickle=False) as g:
   assert np.array_equal(z['window_id'],g['window_id'])
   for i in range(len(z['window_id'])):
    raw=z['raw_legacy7'][i];mask=z['mask'][i];w=z['w_self'][i];valid=z['kinematic_valid'][i];t=z['time_s'][i]
    expect,b=independently_gate(raw,w,mask,valid,t)
    maxerror=max(maxerror,float(np.max(np.abs(expect[:,:2]-g['causal7'][i,:,:2]))))
    assert np.array_equal(g['causal7'][i,:,2:],z['old_gate9'][i,:,2:7])
    assert np.array_equal(b,g['branches'][i]);branch_counts+=np.bincount(b,minlength=6);nwindows+=1
    if i<25 and mask.sum()>3:
     n=int(mask.sum());cut=int(rng.integers(2,n));base,_=module.causal(raw,w,mask,valid,t,cap)
     rr=raw.copy();ww=w.copy();vv=valid.copy();rr[cut:,:2]+=rng.normal(0,10000,rr[cut:,:2].shape);rr[cut:,2:4]+=rng.normal(0,100,rr[cut:,2:4].shape);ww[cut:]=rng.random(len(ww)-cut);vv[cut:]=~vv[cut:]
     alter,_=module.causal(rr,ww,mask,vv,t,cap);max_prefix=max(max_prefix,float(np.max(np.abs(base[:cut]-alter[:cut]))));trials+=1
assert maxerror<.01 and max_prefix==0
result={'status':'passed','cap_mps':cap,'cap_steps':sum(map(len,speeds)),'source_windows':nwindows,
 'max_absolute_coordinate_difference_m':maxerror,'branch_counts':branch_counts.tolist(),
 'suffix_perturbation_cases':trials,'max_past_difference':max_prefix,
 'scope':'independent formula replay of all train/val; suffix interventions sample only; does not establish validity of physical fault identification'}
(out/'audit_gate.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
