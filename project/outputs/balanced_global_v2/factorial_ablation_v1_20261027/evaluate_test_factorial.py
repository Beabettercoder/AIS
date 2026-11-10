"""Evaluate all frozen new arms on the previously exposed main test split."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');NEW=ROOT/'factorial_ablation_v1_20261027';OLD=ROOT/'causal_gate_main_v1_20261027'
sys.path.insert(0,str(OLD));import evaluate_test as previous
sys.path.insert(0,str(NEW));import train_factorial as train
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((NEW/'frozen_test_plan.json').read_text());assert plan['status']=='frozen_before_test'
assert sha(Path(__file__))==plan['evaluator_sha256'] and sha(NEW/'train_factorial.py')==plan['training_sha256']
assert sha(OLD/'evaluate_test.py')==plan['prior_test_preparation_sha256']
dest=NEW/'historical_test_results';dest.mkdir(exist_ok=False)
data,proof=previous.prep();y=data['y'].astype(int);regions=data['region'].astype(str);mmsi=data['mmsi'].astype(str);f=data['features']
f['raw_common']=f['raw'].copy();f['raw_common'][...,-1]=f['causal_common'][...,-1]
for arm,base in train.BASE.items():f[arm]=f[base].copy()
prior=['raw','old','comp','causal','causal_common','causal_common_bundle']
oldplan=json.loads((OLD/'frozen_test_plan.json').read_text());oldruns=[r for r in oldplan['runs'] if r['arm'] in prior]
with np.load(OLD/'formal_test_results/all_test_predictions.npz',allow_pickle=False) as z:
 assert np.array_equal(data['window_id'],z['window_id']) and np.array_equal(y,z['y']) and np.array_equal(mmsi,z['mmsi'].astype(str))
 oldpred={(str(a),int(s),str(k)):p.copy() for a,s,k,p in zip(z['architecture'],z['seed'],z['arm'],z['scores']) if str(k) in prior}
assert len(oldruns)==36 and len(oldpred)==36
runs=oldruns+plan['runs'];weights,ids=previous.boot_weights(y,mmsi);torch.set_num_threads(4);summaries=[];preds=[];boot={};rgn=np.unique(regions)
def clustered(s,thresholds):
 pooled=previous.bootstrap_one(y,s,weights,thresholds)
 macro=[];valid=np.ones(len(weights),dtype=bool)
 for rg in rgn:
  q=regions==rg;yw=y[q];ww=weights[:,q]
  pos=np.sum(ww*(yw==1),axis=1);neg=np.sum(ww*(yw==0),axis=1);valid &= (pos>0)&(neg>0)
  macro.append(previous.bootstrap_one(yw,s[q],ww,{})['AP'])
 allmacro=np.mean(macro,axis=0);allmacro[~valid]=np.nan
 return {**pooled,'macro_AP':allmacro}
for run in runs:
 key=(run['architecture'],int(run['seed']),run['arm'])
 if run in oldruns:
  s=oldpred[key]
 else:
  ckpt=Path(run['checkpoint']);assert sha(ckpt)==run['checkpoint_sha256'];saved=torch.load(ckpt,map_location='cpu',weights_only=False)
  model=train.hm.HistoricalModel(run['architecture'],run['arm']);model.load_state_dict(saved['state_dict']);model.eval();x=(f[run['arm']]-saved['normalizer_mean'].numpy())/saved['normalizer_std'].numpy();pred=[]
  with torch.no_grad():
   for lo in range(0,len(y),128):
    xb=torch.from_numpy(x[lo:lo+128]);mask=torch.from_numpy(data['mask'][lo:lo+128]);logit,*_=model(xb,mask);pred.extend(torch.sigmoid(logit).numpy().tolist())
  s=np.array(pred)
 met=previous.metric(y,s,regions,run['thresholds']);dist=clustered(s,run['thresholds'])
 for j in (0,1,2):assert abs(dist['AP'][j]-average_precision_score(y,s,sample_weight=weights[j]))<1e-10
 met['bootstrap_95']={name:np.nanquantile(a,[.025,.975]).tolist() for name,a in dist.items()}
 met['bootstrap_valid_macro_draws']=int(np.isfinite(dist['macro_AP']).sum())
 boot[key]=dist;summaries.append({'architecture':key[0],'seed':key[1],'arm':key[2],**met});preds.append(s)
 print('TEST',*key,round(met['macro_AP'],5),flush=True)
pairs=[('causal','raw'),('raw_common','raw'),('causal_common','causal'),('raw_bundle','raw'),('causal_bundle','causal'),
 ('raw_common_bundle','raw_common'),('causal_common_bundle','causal_common'),('raw_common_bundle','raw_bundle'),('causal_common_bundle','causal_bundle'),
 ('old_bundle','old'),('comp_bundle','comp'),('raw_bundle_self','raw_bundle_bgz'),('causal_bundle_self','causal_bundle_bgz'),
 ('causal_common_bundle_self','causal_common_bundle_bgz'),('raw_bundle_bgz','raw_bundle'),('causal_bundle_bgz','causal_bundle'),('causal_common_bundle_bgz','causal_common_bundle')]
by={(r['architecture'],r['seed'],r['arm']):r for r in summaries};comparisons=[]
for arch in ('tcn','gru'):
 for a,b in pairs:
  for metric in ('AP','macro_AP'):
   diff=np.stack([boot[(arch,s,a)][metric]-boot[(arch,s,b)][metric] for s in (0,1,2)]);avg=np.mean(diff,axis=0)
   values=[by[(arch,s,a)]['pooled_AP' if metric=='AP' else 'macro_AP']-by[(arch,s,b)]['pooled_AP' if metric=='AP' else 'macro_AP'] for s in (0,1,2)]
   comparisons.append({'architecture':arch,'a':a,'b':b,'metric':metric,'mean_difference':float(np.mean(values)),'seed_differences':values,'CI95':np.nanquantile(avg,[.025,.975]).tolist(),'valid_draws':int(np.isfinite(avg).sum()),'meaning':'MMSI clustered, same 3 fixed seeds; not uncertainty over seeds or shared background'})
interactions=[]
definitions=[('gate_by_bundle_support_off','causal_bundle','causal','raw_bundle','raw'),('gate_by_bundle_support_on','causal_common_bundle','causal_common','raw_common_bundle','raw_common'),
             ('support_by_bundle_raw','raw_common_bundle','raw_common','raw_bundle','raw'),('support_by_bundle_causal','causal_common_bundle','causal_common','causal_bundle','causal')]
for arch in ('tcn','gru'):
 for name,a,b,c,d in definitions:
  for metric in ('AP','macro_AP'):
   dist=np.stack([boot[(arch,s,a)][metric]-boot[(arch,s,b)][metric]-boot[(arch,s,c)][metric]+boot[(arch,s,d)][metric] for s in (0,1,2)])
   avg=np.mean(dist,axis=0);key='pooled_AP' if metric=='AP' else 'macro_AP'
   values=[by[(arch,s,a)][key]-by[(arch,s,b)][key]-by[(arch,s,c)][key]+by[(arch,s,d)][key] for s in (0,1,2)]
   interactions.append({'architecture':arch,'name':name,'formula':f'({a} − {b}) − ({c} − {d})','metric':metric,'mean_interaction':float(np.mean(values)),
     'seed_interactions':values,'CI95':np.nanquantile(avg,[.025,.975]).tolist(),'valid_draws':int(np.isfinite(avg).sum()),'meaning':'MMSI clustered, fixed 3 seeds; exploratory historically exposed test'})
np.savez_compressed(dest/'all_test_predictions.npz',scores=np.asarray(preds),window_id=data['window_id'],mmsi=data['mmsi'],region=data['region'],y=y,architecture=np.array([r['architecture'] for r in runs]),seed=np.array([r['seed'] for r in runs]),arm=np.array([r['arm'] for r in runs]))
out={'status':'complete','scope':'historically exposed main test, not blind; no method selected on test','frozen_plan_sha256':sha(NEW/'frozen_test_plan.json'),'source_proof':proof,'test_windows':len(y),'target_mmsi':len(ids),'bootstrap_draws':len(weights),'results':summaries,'paired_AP':comparisons,'interactions':interactions}
(dest/'test_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');print('TEST_ALL_COMPLETE',flush=True)
