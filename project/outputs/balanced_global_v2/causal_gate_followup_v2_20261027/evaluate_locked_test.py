"""Locked candidate evaluation; historic test previously exposed in earlier pilots."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');BASE=ROOT/'causal_gate_main_v1_20261027';OUT=ROOT/'causal_gate_followup_v2_20261027';SCENE=ROOT/'labeling/labeled_v1/features_scene'
sys.path.insert(0,str(BASE));sys.path.insert(0,str(OUT));import evaluate_test as old
from build_candidates import generate
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=json.loads((OUT/'frozen_test_plan.json').read_text(encoding='utf8'));assert plan['status']=='frozen_before_test';assert plan['evaluator_sha256']==sha(Path(__file__));assert plan['builder_sha256']==sha(OUT/'build_candidates.py')
 for f,h in plan['locks'].items():assert sha(f)==h,(f,sha(f),h)
 data,proof=old.prep();y=data['y'].astype(int);ids=data['window_id'].astype(str);mmsi=data['mmsi'].astype(str);regions=data['region'].astype(str);pars=json.loads((OUT/'fitted_parameters.json').read_text());candidate=plan['candidate'];xs=[];wids=[]
 for r in old.regions:
  p=SCENE/r/'test.npz'
  with np.load(p,allow_pickle=False) as z:
   local=np.flatnonzero(regions==r);assert np.array_equal(ids[local],z['window_id'].astype(str))
   for j,i in enumerate(local):
    raw=data['features']['raw'][i,:,:7];mask=z['mask'][j];w=z['self_gate_weight'][j].copy();w[~mask]=0
    g,_=generate(raw,w,mask,z['kinematic_valid'][j],z['time_s'][j],pars['speed_cap_mps'],pars['acceleration_q99_mps2'],candidate)
    x=data['features']['causal'][i].copy();x[:,:2]=g[:,:2];xs.append(x);wids.append(ids[i])
 assert np.array_equal(wids,ids);X=np.stack(xs);torch.set_num_threads(4)
 old.train.hm.ARMS['candidate']=dict(key='candidate',nf=12,graph=False,recipe=False,masked=False)
 with np.load(BASE/'formal_test_results/all_test_predictions.npz',allow_pickle=False) as z:baseline={k:z[k] for k in z.files}
 assert np.array_equal(baseline['window_id'].astype(str),ids) and np.array_equal(baseline['y'],y) and np.array_equal(baseline['mmsi'].astype(str),mmsi)
 weights,unique=old.boot_weights(y,mmsi);regions_set=sorted(set(regions));run_results=[];score_rows=[];diff_all={}
 # Reject bootstrap draws without a positive and negative example in any region;
 # do not treat a missing class as a zero-valued AP/FAR observation.
 for r in regions_set:
  sel=regions==r
  if np.any(np.sum(weights[:,sel]*(y[sel]==1)[None,:],axis=1)==0) or np.any(np.sum(weights[:,sel]*(y[sel]==0)[None,:],axis=1)==0):raise ValueError('Bootstrap region lacks class: '+r)
 def boots(s):
  pooled=old.bootstrap_one(y,s,weights,{})['AP'];per=[]
  for r in regions_set:
   sel=regions==r;per.append(old.bootstrap_one(y[sel],s[sel],weights[:,sel],{})['AP'])
  return pooled,np.mean(per,axis=0)
 for run in plan['runs']:
  cp=Path(run['checkpoint']);c=torch.load(cp,map_location='cpu',weights_only=False);assert c['epoch']==25 and c['arm']=='candidate'
  model=old.train.hm.HistoricalModel(run['architecture'],'candidate');model.load_state_dict(c['state_dict']);model.eval();x=(X-c['normalizer_mean'].numpy())/c['normalizer_std'].numpy();p=[]
  with torch.no_grad():
   for j in range(0,len(y),128):
    s,*_=model(torch.from_numpy(x[j:j+128]),torch.from_numpy(data['mask'][j:j+128]));p.extend(torch.sigmoid(s).numpy().tolist())
  score=np.asarray(p);score_rows.append(score);met=old.metric(y,score,regions,run['thresholds']);bb=old.bootstrap_one(y,score,weights,run['thresholds']);met['bootstrap_95']={k:np.quantile(v,[.025,.975]).tolist() for k,v in bb.items()}
  result={'architecture':run['architecture'],'seed':run['seed'],**met,'paired':{}};cpool,cmacro=boots(score)
  for arm in ('raw','old','comp','causal'):
   ix=np.flatnonzero((baseline['architecture']==run['architecture'])&(baseline['seed']==run['seed'])&(baseline['arm']==arm));assert len(ix)==1;ref=baseline['scores'][ix[0]];bpool,bmacro=boots(ref);dpool=cpool-bpool;dmacro=cmacro-bmacro
   macro=lambda a:float(np.mean([average_precision_score(y[regions==r],a[regions==r]) for r in regions_set]))
   result['paired'][arm]={'pooled_AP_difference':float(average_precision_score(y,score)-average_precision_score(y,ref)),'pooled_AP_difference_CI95':np.quantile(dpool,[.025,.975]).tolist(), 'macro_AP_difference':macro(score)-macro(ref),'macro_AP_difference_CI95':np.quantile(dmacro,[.025,.975]).tolist()}
   diff_all[(run['architecture'],run['seed'],arm)]=(dpool,dmacro)
  run_results.append(result);print('TEST',run['architecture'],run['seed'],met['macro_AP'],flush=True)
 aggregate=[]
 for arch in ('tcn','gru'):
  for arm in ('raw','old','comp','causal'):
   p=np.mean([diff_all[(arch,s,arm)][0] for s in (0,1,2)],axis=0);m=np.mean([diff_all[(arch,s,arm)][1] for s in (0,1,2)],axis=0);rs=[v['paired'][arm] for v in run_results if v['architecture']==arch]
   aggregate.append({'architecture':arch,'reference':arm,'pooled_AP_difference':float(np.mean([v['pooled_AP_difference'] for v in rs])),'pooled_AP_difference_CI95':np.quantile(p,[.025,.975]).tolist(), 'macro_AP_difference':float(np.mean([v['macro_AP_difference'] for v in rs])),'macro_AP_difference_CI95':np.quantile(m,[.025,.975]).tolist()})
 dest=OUT/'test_results';dest.mkdir(exist_ok=False)
 np.savez_compressed(dest/'predictions.npz',scores=np.asarray(score_rows),window_id=ids,mmsi=mmsi,region=regions,y=y,architecture=np.array([r['architecture'] for r in plan['runs']]),seed=np.array([r['seed'] for r in plan['runs']]))
 result={'status':'complete','scope':'previously-exposed 8-region test; model selected on explored validation; not a new blind test','plan_sha256':sha(OUT/'frozen_test_plan.json'),'test_windows':len(y),'target_mmsi':len(unique),'bootstrap_draws':len(weights),'bootstrap_unit':'target MMSI; shared background and seed sampling not covered','proof':proof,'runs':run_results,'seed_mean_paired':aggregate}
 with (dest/'test_results.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
 print('COMPLETE',json.dumps(aggregate,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
