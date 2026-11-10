"""Train-only cap, causal gate and validation counterfactual diagnostic."""
from pathlib import Path
import json, hashlib
from collections import Counter
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
SRC=ROOT/'gating_training_v2_20261027/data'
REFERENCE=ROOT/'labeling/labeled_v1/features_scene'
OUT=ROOT/'causal_gate_main_v1_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def bounded(v,cap):
 n=np.linalg.norm(v)
 return v*min(1.,cap/max(n,1e-12))

def causal(raw,w,mask,valid,times,cap):
 n=int(mask.sum()); out=np.array(raw,copy=True); current=np.zeros(2); previous=[]; branches=np.zeros(len(mask),np.int8)
 for k in range(1,n):
  dp=raw[k,:2].astype(float)-raw[k-1,:2].astype(float); wt=float(w[k]); dt=float(times[k]-times[k-1]); good=bool(valid[k] and dt>0 and np.isfinite(dt) and np.isfinite(raw[k-1:k+1,2:4]).all())
  step=wt*dp
  if good:
   report=.5*(raw[k-1,2:4].astype(float)+raw[k,2:4].astype(float))
   if wt>=.3:
    step=wt*dp+(1-wt)*report*dt; branches[k]=1
   elif len(previous)>=3:
    ref=np.median(np.array(previous[-5:]),axis=0); pos=dp/dt
    dpos=np.linalg.norm(pos-ref); drep=np.linalg.norm(report-ref)
    if dpos<2.5 and dpos<.5*drep:
     step=dp; branches[k]=2
    elif drep<2.5 and drep<.5*dpos:
     step=wt*dp+(1-wt)*bounded(report,cap)*dt; branches[k]=3
    else:
     step=wt*dp+(1-wt)*bounded(ref,cap)*dt; branches[k]=4
   else: branches[k]=5
  current+=step;out[k,:2]=current
  if valid[k] and float(w[k])>=.7 and np.isfinite(raw[k,2:4]).all():previous.append(raw[k,2:4].astype(float).copy())
 out[n:,:2]=current
 return out,branches

def main():
 vals=[]
 for r in REGIONS:
  with np.load(SRC/r/'train.npz',allow_pickle=False) as z:
   m=z['mask']&z['kinematic_valid']&(z['w_self']>=.7)
   vals.append(np.linalg.norm(z['raw_legacy7'][...,2:4][m],axis=1))
 cap=float(np.quantile(np.concatenate(vals),.995)); (OUT/'gate_cap.json').write_text(json.dumps({'train_only':True,'quantile':.995,'cap_mps':cap,'eligible_steps':sum(len(x) for x in vals),'protocol_sha256':sha(OUT/'protocol.json')},indent=2),encoding='utf8')
 records=[];manifest=[]
 for r in REGIONS:
  for split in ('train','val'):
   p=SRC/r/(split+'.npz');refp=REFERENCE/r/(split+'.npz')
   with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
   with np.load(refp,allow_pickle=False) as z:ref={k:z[k] for k in ['window_id','pre_noise_pos','target_pos','target_noise_effective','behavior_type','noise_type']}
   assert np.array_equal(d['window_id'],ref['window_id'])
   x=d['old_gate9'][...,:7].copy(); b=np.zeros(d['mask'].shape,np.int8)
   for i in range(len(x)):
    # Only coordinate slots change; all non-coordinate slots match old gate.
    candidate,branch=causal(d['raw_legacy7'][i],d['w_self'][i],d['mask'][i],d['kinematic_valid'][i],d['time_s'][i],cap)
    x[i,:,:2]=candidate[:,:2];b[i]=branch
    if split=='val':
     n=int(d['mask'][i].sum()); truth=ref['pre_noise_pos'][i,:n]-ref['pre_noise_pos'][i,0]
     row={'region':r,'window_id':str(d['window_id'][i]),'mmsi':str(d['mmsi'][i]),'behavior':str(ref['behavior_type'][i]),'noise_type':str(ref['noise_type'][i]),'target_noise':bool(ref['target_noise_effective'][i]),'branches':np.bincount(branch,minlength=6).tolist()}
     for name,v in [('raw',d['raw_legacy7'][i,:n,:2]),('old',d['old_gate9'][i,:n,:2]),('comp',d['comp_gate9'][i,:n,:2]),('causal',x[i,:n,:2])]:
      error=np.linalg.norm(v.astype(float)-truth,axis=1);row[name+'_endpoint_m']=float(error[-1]);row[name+'_mean_path_m']=float(error.mean())
     records.append(row)
   dest=OUT/'gate_data'/r/(split+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(dest,window_id=d['window_id'],causal7=x,branches=b)
   manifest.append({'region':r,'split':split,'source_sha256':sha(p),'reference_sha256':sha(refp),'output_sha256':sha(dest),'branch_counts':np.bincount(b.ravel(),minlength=6).tolist()})
 (OUT/'gate_input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 (OUT/'trajectory_validation_rows.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 groups={'all':records,'target_noise':[v for v in records if v['target_noise']], 'target_not_noise':[v for v in records if not v['target_noise']]}
 for typ in ('speed_spike','teleport','drift'):
  groups[typ]=[v for v in records if v['target_noise'] and v['noise_type']==typ]
 for typ in ('turn_back','deviation','stall','speed_drop','sinuous'):
  groups['no_target_noise_'+typ]=[v for v in records if not v['target_noise'] and v['behavior']==typ]
 summary={}
 for name,rows in groups.items():
  if not rows:continue
  item={'windows':len(rows),'unique_target_mmsi':len({v['mmsi'] for v in rows})}
  for method in ('raw','old','comp','causal'):
   for metric in ('endpoint','mean_path'):
    a=np.array([v[method+'_'+metric+'_m'] for v in rows]);item[method+'_'+metric]={'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.quantile(a,.9))}
  summary[name]=item
 (OUT/'trajectory_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'cap_mps':cap,'summary':summary},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
