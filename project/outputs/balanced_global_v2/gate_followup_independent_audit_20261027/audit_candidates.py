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

fit=json.loads((RUN/'fitted_parameters.json').read_text(encoding='utf8'));origcap=json.loads((BASE/'gate_cap.json').read_text(encoding='utf8'))['cap_mps'];assert fit['speed_cap_mps']==origcap
values=[]
for r in REGIONS:
 with np.load(SRC/r/'train.npz',allow_pickle=False) as z:
  v=z['raw_legacy7'][...,2:4].astype(float);mask=z['mask'];kin=z['kinematic_valid'];w=z['w_self'];dt=np.diff(z['time_s'],axis=1)
  eligible=mask[:,1:]&mask[:,:-1]&kin[:,1:]&kin[:,:-1]&(w[:,1:]>=.7)&(w[:,:-1]>=.7)&(dt>0)&(dt<=300)
  values.extend((np.linalg.norm(np.diff(v,axis=1),axis=-1)/np.where(dt>0,dt,1))[eligible].tolist())
q=float(np.quantile(values,.99));assert len(values)==fit['train_valid_pairs'] and abs(q-fit['acceleration_q99_mps2'])<1e-12
input_manifest=json.loads((RUN/'input_manifest.json').read_text(encoding='utf8'));index={(x['region'],x['split']):x for x in input_manifest}
valrows=json.loads((RUN/'val_trajectory_rows.json').read_text(encoding='utf8'));valmap={(r['region'],r['window_id']):r for r in valrows};assert len(valmap)==1680
maxdiff=0.;max_err=0.;cases=0;prefix_tests=0;prefix_error=0.;rng=np.random.default_rng(20261028);totals={}
for region in REGIONS:
 for split in ('train','val'):
  source=SRC/region/f'{split}.npz';path=RUN/'inputs'/region/f'{split}.npz';prior=BASE/'gate_data'/region/f'{split}.npz';entry=index[(region,split)]
  assert sha(source)==entry['source_sha256'] and sha(path)==entry['output_sha256'] and sha(prior)==entry['prior_sha256']
  with np.load(source,allow_pickle=False) as z:d={k:z[k] for k in ('window_id','raw_legacy7','old_gate9','w_self','mask','kinematic_valid','time_s')}
  with np.load(path,allow_pickle=False) as z:o={k:z[k] for k in z.files}
  assert np.array_equal(o['window_id'],d['window_id'])
  if split=='val':
   with np.load(SCENE/region/'val.npz',allow_pickle=False) as z:truth=z['pre_noise_pos'].copy();vid=z['window_id'].copy()
   assert np.array_equal(vid,d['window_id'])
  chosen=set(rng.choice(len(d['window_id']),size=min(25,len(d['window_id'])),replace=False).tolist())
  for i in range(len(d['window_id'])):
   raw=d['raw_legacy7'][i];w=d['w_self'][i];m=d['mask'][i];v=d['kinematic_valid'][i];t=d['time_s'][i];N=int(m.sum())
   for name in METHODS:
    pos,branch=independent(raw,w,m,v,t,origcap,q,name)
    err=float(np.max(np.abs(o[name][i,:,:2]-pos)));maxdiff=max(maxdiff,err);assert err<.005,(region,split,name,i,err)
    assert np.array_equal(branch,o[name+'_branch'][i])
    assert np.array_equal(o[name][i,:,2:],d['old_gate9'][i,:,2:7])
    if split=='val':
     r=valmap[(region,str(d['window_id'][i]))];reference=truth[i,:N]-truth[i,0];es=np.linalg.norm(o[name][i,:N,:2].astype(float)-reference,axis=1)
     err=max(abs(float(es[-1])-r[name+'_endpoint_m']),abs(float(es.mean())-r[name+'_path_m']));max_err=max(max_err,err);assert err<1e-8
    if i in chosen and N>=4:
     cut=int(rng.integers(2,N));r2=raw.copy();w2=w.copy();v2=v.copy();t2=t.copy();r2[cut:,:2]+=rng.normal(0,1e5,r2[cut:,:2].shape);r2[cut:,2:4]+=rng.normal(0,30,r2[cut:,2:4].shape);w2[cut:]=rng.random(len(w2)-cut);v2[cut:]=~v2[cut:]
     p2,_=independent(r2,w2,m,v2,t2,origcap,q,name);err=float(np.max(np.abs(pos[:cut]-p2[:cut])));prefix_error=max(prefix_error,err);assert err==0;prefix_tests+=1
    cases+=1
  totals[region+'|'+split]=len(d['window_id'])
 print('CANDIDATES_OK',region,flush=True)
summary=json.loads((RUN/'trajectory_summary.json').read_text(encoding='utf8'));sel=json.loads((RUN/'development_selection.json').read_text(encoding='utf8'))
groups={'target_noise':[r for r in valrows if r['target_noise']], 'target_not_noise':[r for r in valrows if not r['target_noise']]}
for k in ('speed_spike','teleport','drift'):groups[k]=[r for r in valrows if r['target_noise'] and r['noise_type']==k]
for k,group in groups.items():
 assert summary[k]['windows']==len(group) and summary[k]['mmsi']==len({r['mmsi'] for r in group})
 for method in ('raw','old','comp','causal',*METHODS):
  for metric in ('endpoint','path'):
   a=np.array([r[method+'_'+metric+'_m'] for r in group]);x=summary[k][method][metric]
   assert abs(a.mean()-x['mean'])<1e-9 and abs(np.median(a)-x['median'])<1e-9 and abs(np.quantile(a,.9)-x['p90'])<1e-9
scores={}
for method in METHODS:
 ratios=[summary[g][method]['endpoint'][m]/max(1.,summary[g]['causal']['endpoint'][m]) for g in ('speed_spike','teleport','drift') for m in ('mean','p90')]
 ratios.append(summary['target_not_noise'][method]['endpoint']['mean']/max(1.,summary['target_not_noise']['causal']['endpoint']['mean']))
 scores[method]=float(np.mean(ratios));assert abs(scores[method]-sel['scores'][method])<1e-12
assert sel['candidate']==min(METHODS,key=lambda x:(scores[x],METHODS.index(x)))
out={'status':'passed','train_valid_windows':sum(totals.values()),'candidate_window_replays':cases,'causality_suffix_perturbation_cases':prefix_tests,
 'train_pairs_for_acceleration':len(values),'speed_cap_mps':origcap,'acceleration_q99_mps2':q,'max_position_float32_difference_m':maxdiff,
 'max_val_error_calculation_difference_m':max_err,'max_future_to_past_difference_m':prefix_error,'chosen_candidate':sel['candidate'],
 'selection_scores':scores,'scope':'all train/val candidates and branches, candidate trajectory errors and summary, train-only fitted q99, randomized future perturbations',
 'limits':'training eligibility flags from previously audited input; pre-noise data are synthetic counterfactuals, not human sensor truth; candidate chosen after prior inspection of validation',
 'protocol_sha256':sha(RUN/'protocol.json'),'script_sha256':sha(Path(__file__))}
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'audit_candidates.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps(out,indent=2,ensure_ascii=False),flush=True)
