"""Frozen candidate construction; reads train/val only, oracle used only to score validation."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=ROOT/'causal_gate_followup_v2_20261027'
SRC=ROOT/'gating_training_v2_20261027/data'
BASE=ROOT/'causal_gate_main_v1_20261027'
SCENE=ROOT/'labeling/labeled_v1/features_scene'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
METHODS=('short3','smooth_guard','smooth_guard_cold')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(path,value):
 with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def bounded(x,cap):return x*min(1.,cap/max(float(np.linalg.norm(x)),1e-12))

def generate(raw,w,mask,valid,times,cap,accel_cap,method):
 n=int(mask.sum());out=np.array(raw,copy=True);current=np.zeros(2);history=[];branches=np.zeros(len(mask),np.int8)
 for k in range(1,n):
  dp=raw[k,:2].astype(float)-raw[k-1,:2].astype(float);wt=float(w[k]);dt=float(times[k]-times[k-1]);ok=bool(valid[k] and dt>0 and np.isfinite(dt) and np.isfinite(raw[k-1:k+1,2:4]).all());step=wt*dp
  if ok:
   report=.5*(raw[k-1,2:4].astype(float)+raw[k,2:4].astype(float))
   speed_ok=bool(k>=2 and valid[k-1] and 0<dt<=300 and np.linalg.norm(raw[k-1,2:4])<=cap and np.linalg.norm(raw[k,2:4])<=cap and np.linalg.norm(raw[k,2:4]-raw[k-1,2:4])/dt<=accel_cap)
   if wt>=.3:step=wt*dp+(1-wt)*report*dt;branches[k]=1
   elif len(history)>=3:
    ref=np.median(np.array(history[-(3 if method=='short3' else 5):]),axis=0);dpos=float(np.linalg.norm(dp/dt-ref));drep=float(np.linalg.norm(report-ref))
    if dpos<2.5 and dpos<.5*drep:step=dp;branches[k]=2
    elif drep<2.5 and drep<.5*dpos:step=wt*dp+(1-wt)*bounded(report,cap)*dt;branches[k]=3
    elif method!='short3' and speed_ok and drep<.5*dpos:step=wt*dp+(1-wt)*bounded(report,cap)*dt;branches[k]=6
    else:step=wt*dp+(1-wt)*bounded(ref,cap)*dt;branches[k]=4
   elif method=='smooth_guard_cold' and speed_ok:step=wt*dp+(1-wt)*bounded(report,cap)*dt;branches[k]=7
   else:branches[k]=5
  current+=step;out[k,:2]=current
  if valid[k] and float(w[k])>=.7 and np.isfinite(raw[k,2:4]).all():history.append(raw[k,2:4].astype(float).copy())
 out[n:,:2]=current
 return out,branches

def main():
 cap=json.loads((BASE/'gate_cap.json').read_text(encoding='utf8'))['cap_mps'];values=[];inputs=[]
 for r in REGIONS:
  p=SRC/r/'train.npz';inputs.append({'region':r,'split':'train','sha256':sha(p)})
  with np.load(p,allow_pickle=False) as z:
   v=z['raw_legacy7'][...,2:4].astype(np.float64);mask=z['mask'];valid=z['kinematic_valid'];w=z['w_self'];dt=np.diff(z['time_s'],axis=1);q=mask[:,1:]&mask[:,:-1]&valid[:,1:]&valid[:,:-1]&(w[:,1:]>=.7)&(w[:,:-1]>=.7)&(dt>0)&(dt<=300);acc=np.linalg.norm(np.diff(v,axis=1),axis=-1)/np.where(dt>0,dt,1.);values.extend(acc[q].tolist())
 assert len(values)>100
 ac=float(np.quantile(values,.99));save(OUT/'fitted_parameters.json',{'speed_cap_mps':cap,'acceleration_q99_mps2':ac,'train_valid_pairs':len(values),'train_files':inputs,'protocol_sha256':sha(OUT/'protocol.json')})
 print('FITTED',len(values),cap,ac,flush=True)
 records=[];manifest=[]
 for r in REGIONS:
  for split in ('train','val'):
   p=SRC/r/(split+'.npz');prior=BASE/'gate_data'/r/(split+'.npz')
   with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
   with np.load(prior,allow_pickle=False) as z:old={k:z[k] for k in z.files}
   assert np.array_equal(d['window_id'],old['window_id'])
   if split=='val':
    o=SCENE/r/'val.npz'
    with np.load(o,allow_pickle=False) as z:ref={k:z[k] for k in ['window_id','pre_noise_pos','target_pos','target_sog_kn','target_cog_deg','target_noise_effective','observation_noise','y','noise_type','behavior_type']}
    assert np.array_equal(ref['window_id'],d['window_id'])
   n=len(d['window_id']);out={'window_id':d['window_id']}
   for name in METHODS:out[name]=d['old_gate9'][...,:7].copy();out[name+'_branch']=np.zeros_like(d['mask'],np.int8)
   for i in range(n):
    raw=d['raw_legacy7'][i];w=d['w_self'][i];mask=d['mask'][i];valid=d['kinematic_valid'][i];times=d['time_s'][i];N=int(mask.sum())
    for name in METHODS:
     z,b=generate(raw,w,mask,valid,times,cap,ac,name);out[name][i,:,:2]=z[:,:2];out[name+'_branch'][i]=b
    if split=='val':
     truth=ref['pre_noise_pos'][i,:N]-ref['pre_noise_pos'][i,0];sog=ref['target_sog_kn'][i,:N];cog=ref['target_cog_deg'][i,:N];change=np.abs((cog-cog[0]+180)%360-180)
     row={'window_id':str(d['window_id'][i]),'mmsi':str(d['mmsi'][i]),'region':r,'noise_type':str(ref['noise_type'][i]),'behavior_type':str(ref['behavior_type'][i]),'target_noise':bool(ref['target_noise_effective'][i]),'observation_noise':bool(ref['observation_noise'][i]),'y':int(ref['y'][i]),'observed_turn45':bool(np.max(change)>=45),'observed_speed3kn':bool(np.max(np.abs(sog-sog[0]))>=3)}
     arr={'raw':raw[:N,:2],'old':d['old_gate9'][i,:N,:2],'comp':d['comp_gate9'][i,:N,:2],'causal':old['causal7'][i,:N,:2],**{k:out[k][i,:N,:2] for k in METHODS}}
     for k,v in arr.items():
      e=np.linalg.norm(v.astype(float)-truth,axis=1);row[k+'_endpoint_m']=float(e[-1]);row[k+'_path_m']=float(e.mean())
     records.append(row)
   dest=OUT/'inputs'/r/(split+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**out)
   manifest.append({'region':r,'split':split,'source_sha256':sha(p),'prior_sha256':sha(prior),'output_sha256':sha(dest),'windows':n})
 save(OUT/'input_manifest.json',manifest);save(OUT/'val_trajectory_rows.json',records)
 groups={'target_noise':[r for r in records if r['target_noise']], 'target_not_noise':[r for r in records if not r['target_noise']],**{k:[r for r in records if r['target_noise'] and r['noise_type']==k] for k in ('speed_spike','teleport','drift')},'natural_turn45':[r for r in records if r['y']==0 and not r['target_noise'] and r['observed_turn45']],'natural_speed3kn':[r for r in records if r['y']==0 and not r['target_noise'] and r['observed_speed3kn']], **{'synthetic_no_target_noise_'+k:[r for r in records if not r['target_noise'] and r['behavior_type']==k] for k in ('turn_back','deviation','stall','speed_drop','sinuous')}}
 summary={}
 for group,rr in groups.items():
  if not rr:continue
  x={'windows':len(rr),'mmsi':len({v['mmsi'] for v in rr})}
  for method in ('raw','old','comp','causal',*METHODS):
   x[method]={}
   for metric in ('endpoint','path'):
    a=np.array([v[method+'_'+metric+'_m'] for v in rr]);x[method][metric]={'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.quantile(a,.9))}
   diff=np.array([v[method+'_endpoint_m']-v['causal_endpoint_m'] for v in rr]);x[method]['vs_causal_endpoint']={'better':int((diff< -1e-6).sum()),'worse':int((diff>1e-6).sum()),'equal':int((np.abs(diff)<=1e-6).sum())}
  summary[group]=x
 save(OUT/'trajectory_summary.json',summary)
 groupscore=['speed_spike','teleport','drift'];scores={}
 for method in METHODS:
  vals=[]
  for group in groupscore:
   for metric in ('mean','p90'):vals.append(summary[group][method]['endpoint'][metric]/max(1.,summary[group]['causal']['endpoint'][metric]))
  vals.append(summary['target_not_noise'][method]['endpoint']['mean']/max(1.,summary['target_not_noise']['causal']['endpoint']['mean']))
  scores[method]=float(np.mean(vals))
 selected=min(METHODS,key=lambda n:(scores[n],METHODS.index(n)))
 save(OUT/'development_selection.json',{'status':'locked_before_test','candidate':selected,'scores':scores,'metric':'7 equal-weight endpoint ratios; synthetic noise type mean and p90 plus target_not_noise mean; denominator causal floored at 1m','protocol_sha256':sha(OUT/'protocol.json'),'summary_sha256':sha(OUT/'trajectory_summary.json'),'warning':'this validation data informed ideas; exploratory, not independent selection'})
 print('DEVELOPMENT_CANDIDATE',selected,scores,flush=True)
if __name__=='__main__':main()
