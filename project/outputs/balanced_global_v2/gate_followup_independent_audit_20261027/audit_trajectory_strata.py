"""Independently verify natural-observation strata and prior 18+1 gate traces."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
from independent_gate import independent
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');SRC=ROOT/'gating_training_v2_20261027/data';BASE=ROOT/'causal_gate_main_v1_20261027';RUN=ROOT/'causal_gate_followup_v2_20261027';SCENE=ROOT/'labeling/labeled_v1/features_scene';OUT=ROOT/'gate_followup_independent_audit_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai');METHODS=('raw','old','comp','causal','short3','smooth_guard','smooth_guard_cold')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
valrows=json.loads((RUN/'val_trajectory_rows.json').read_text(encoding='utf8'));rowmap={(r['region'],r['window_id']):r for r in valrows}
assert len(rowmap)==1680
data={}
for r in REGIONS:
 with np.load(SRC/r/'val.npz',allow_pickle=False) as z:d={k:z[k] for k in ('window_id','mmsi','raw_legacy7','old_gate9','comp_gate9','w_self','mask','kinematic_valid','time_s','target_pos','target_sog_kn','target_cog_deg')}
 with np.load(BASE/'gate_data'/r/'val.npz',allow_pickle=False) as z:g={k:z[k] for k in ('window_id','causal7','branches')}
 with np.load(SCENE/r/'val.npz',allow_pickle=False) as z:o={k:z[k] for k in ('window_id','pre_noise_pos','pre_noise_sog_kn','pre_noise_cog_deg','target_noise_effective','observation_noise','y','noise_type','behavior_type')}
 assert np.array_equal(d['window_id'],g['window_id']) and np.array_equal(d['window_id'],o['window_id'])
 data[r]=(d,g,o,{str(x):i for i,x in enumerate(d['window_id'])})
 for i,wid in enumerate(d['window_id'].astype(str)):
  row=rowmap[(r,wid)];N=int(d['mask'][i].sum());s=d['target_sog_kn'][i,:N];c=d['target_cog_deg'][i,:N];ang=np.abs((c-c[0]+180)%360-180)
  assert row['observed_turn45']==bool(np.max(ang)>=45) and row['observed_speed3kn']==bool(np.max(np.abs(s-s[0]))>=3)
  assert row['target_noise']==bool(o['target_noise_effective'][i]) and row['observation_noise']==bool(o['observation_noise'][i]) and row['y']==int(o['y'][i])
  assert row['noise_type']==str(o['noise_type'][i]) and row['behavior_type']==str(o['behavior_type'][i])
  truth=o['pre_noise_pos'][i,:N]-o['pre_noise_pos'][i,0]
  for method,X in [('raw',d['raw_legacy7'][i,:N,:2]),('old',d['old_gate9'][i,:N,:2]),('comp',d['comp_gate9'][i,:N,:2]),('causal',g['causal7'][i,:N,:2])]:
   E=np.linalg.norm(X.astype(float)-truth,axis=1);assert abs(float(E[-1])-row[method+'_endpoint_m'])<1e-8 and abs(float(E.mean())-row[method+'_path_m'])<1e-8
summary=json.loads((RUN/'trajectory_summary.json').read_text(encoding='utf8'))
groups={'natural_turn45':[r for r in valrows if r['y']==0 and not r['target_noise'] and r['observed_turn45']],
        'natural_speed3kn':[r for r in valrows if r['y']==0 and not r['target_noise'] and r['observed_speed3kn']]}
for k in ('turn_back','deviation','stall','speed_drop','sinuous'):
 groups['synthetic_no_target_noise_'+k]=[r for r in valrows if not r['target_noise'] and r['behavior_type']==k]
for name,rr in groups.items():
 assert summary[name]['windows']==len(rr) and summary[name]['mmsi']==len({r['mmsi'] for r in rr})
 for method in METHODS:
  for metric in ('endpoint','path'):
   a=np.array([r[method+'_'+metric+'_m'] for r in rr]);x=summary[name][method][metric]
   assert abs(a.mean()-x['mean'])<1e-9 and abs(np.median(a)-x['median'])<1e-9 and abs(np.quantile(a,.9)-x['p90'])<1e-9
  delta=np.array([r[method+'_endpoint_m']-r['causal_endpoint_m'] for r in rr]);x=summary[name][method]['vs_causal_endpoint']
  assert x=={'better':int((delta< -1e-6).sum()),'worse':int((delta>1e-6).sum()),'equal':int((np.abs(delta)<=1e-6).sum())}

trace=json.loads((OUT/'branch_trace_input.json').read_text(encoding='utf8'));assert trace['windows']==19 and len(trace['traces'])==19
for x in trace['inputs']:assert sha(Path(x['path']))==x['sha256']
prior=json.loads((BASE/'trajectory_validation_rows.json').read_text(encoding='utf8'))
expected={r['window_id'] for r in prior if r['noise_type']=='teleport' and r['target_noise'] and r['causal_endpoint_m']>r['comp_endpoint_m']+1e-6}
assert len(expected)==19 and {r['window_id'] for r in trace['traces']}==expected
maxerror=0.;branch4=[];cold=set();stepcount=0
for item in trace['traces']:
 d,g,o,index=data[item['region']];i=index[item['window_id']];raw=d['raw_legacy7'][i];w=d['w_self'][i];mask=d['mask'][i];valid=d['kinematic_valid'][i];time=d['time_s'][i];n=int(mask.sum());truth=o['pre_noise_pos'][i,:n]-o['pre_noise_pos'][i,0];past=[]
 assert len(item['steps'])==n-1
 for step in item['steps']:
  k=step['step'];dt=float(time[k]-time[k-1]);delta=raw[k,:2].astype(float)-raw[k-1,:2].astype(float);pvel=delta/dt if dt>0 else np.zeros(2);rep=(raw[k,2:4].astype(float)+raw[k-1,2:4].astype(float))/2
  h=np.median(np.array(past[-5:]),axis=0) if len(past)>=3 else None;dp=None if h is None else float(np.linalg.norm(pvel-h));dr=None if h is None else float(np.linalg.norm(rep-h))
  assert step['branch']==int(g['branches'][i,k]) and step['trusted_history_count']==len(past)
  assert step['time_s']==float(time[k]) and step['dt_s']==dt and step['weight']==float(w[k]) and step['valid']==bool(valid[k])
  for field,value in [('observed_position_m',d['target_pos'][i,k]),('observed_displacement_m',delta),('position_velocity_mps',pvel),('reported_velocity_midpoint_mps',rep),('pre_injection_step_m',truth[k]-truth[k-1])]:
   err=float(np.max(np.abs(np.asarray(step[field])-value)));maxerror=max(maxerror,err);assert err<1e-8
  assert (h is None)==(step['past_reference_mps'] is None)
  if h is not None:
   assert np.max(np.abs(np.asarray(step['past_reference_mps'])-h))<1e-8 and abs(step['d_pos_mps']-dp)<1e-8 and abs(step['d_rep_mps']-dr)<1e-8
  for field,X in [('original_gate_step_m',d['old_gate9'][i,:,:2]),('old_comp_step_m',d['comp_gate9'][i,:,:2]),('causal_step_m',g['causal7'][i,:,:2])]:
   err=float(np.max(np.abs(np.asarray(step[field])-(X[k]-X[k-1]))));maxerror=max(maxerror,err);assert err<1e-8
  if step['branch']==5:cold.add(item['window_id'])
  if step['branch']==4:branch4.append((item['window_id'],step))
  if valid[k] and float(w[k])>=.7 and np.isfinite(raw[k,2:4]).all():past.append(raw[k,2:4].astype(float).copy())
  stepcount+=1
assert len(cold)==1
focus=[r for mid,r in branch4 if mid not in cold]
assert len(focus)==trace['steps_branch4_in_18']==24 and sum(r['d_rep_mps']>=2.5 for r in focus)==trace['branch4_drep_ge_2_5']==23
for label,op in [('history_better_step',lambda r:r['causal_step_reference_error_m']<r['old_comp_step_reference_error_m']-1e-6),('history_worse_step',lambda r:r['causal_step_reference_error_m']>r['old_comp_step_reference_error_m']+1e-6),('history_equal_step',lambda r:abs(r['causal_step_reference_error_m']-r['old_comp_step_reference_error_m'])<=1e-6)]:assert sum(op(r) for r in focus)==trace['branch4_reference_errors'][label]
out={'status':'passed','validation_rows':len(valrows),'natural_and_synthetic_group_sizes':{k:len(v) for k,v in groups.items()},
 'trace_windows':len(trace['traces']),'trace_steps':stepcount,'cold_start_windows':len(cold),'branch4_steps_18':len(focus),
 'branch4_report_distance_ge_2_5_steps':sum(r['d_rep_mps']>=2.5 for r in focus),'max_step_field_error':maxerror,
 'scope':'all validation natural/synthetic observation strata and 19 previously selected trace source values; target pre-corruption coordinates are synthetic counterfactual only',
 'script_sha256':sha(Path(__file__)),'trace_sha256':sha(OUT/'branch_trace_input.json')}
with (OUT/'audit_trajectory_strata.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,indent=2,ensure_ascii=False))
