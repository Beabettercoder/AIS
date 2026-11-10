from pathlib import Path
from collections import defaultdict
import json,csv,hashlib
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score

root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');work=root/'causal_gate_main_v1_20261027';src=root/'gating_training_v2_20261027/data';field=root/'robust_background_v1_20261027';out=root/'independent_audit_causal_v1_20261027'
regions=['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai']
proto=json.loads((work/'training_protocol.json').read_text());plan=json.loads((work/'frozen_test_plan.json').read_text());arms=proto['arms']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert plan['protocol_sha256']==sha(work/'training_protocol.json')
means={};stds={};parts={a:[] for a in arms};ms=[]
for r in regions:
 with np.load(src/r/'train.npz',allow_pickle=False) as z,np.load(work/'gate_data'/r/'train.npz',allow_pickle=False) as g,np.load(field/r/'train.npz',allow_pickle=False) as f:
  assert np.array_equal(z['window_id'],g['window_id']) and np.array_equal(z['window_id'],f['window_id'])
  w=z['w_self'].astype(np.float32);v=z['kinematic_valid'].astype(np.float32);common=f['common_valid'].astype(np.float32);q=np.zeros_like(w);q2=np.zeros((*w.shape,2),np.float32)
  def pack(x,rf,ww,vv,c):return np.concatenate((x,rf,ww[...,None],vv[...,None],c[...,None]),axis=-1).astype(np.float32)
  feat={'raw':pack(z['raw_legacy7'],q2,q,q,q),'old':pack(z['old_gate9'][...,:7],q2,w,v,q),'comp':pack(z['comp_gate9'][...,:7],q2,w,v,q),'causal':pack(g['causal7'],q2,w,v,q),'causal_common':pack(g['causal7'],q2,w,v,common)}
  for a in ('affine','mean','median','weighted_median'):feat[a]=pack(g['causal7'],f['residual_'+a+'_mps'],w,v,common)
  for a in ('causal_common','affine','weighted_median'):feat[a+'_bundle']=feat[a]
  for a in arms:parts[a].append(feat[a]);ms.append(z['mask']) if a==arms[0] else None
mask=np.concatenate(ms)
for a in arms:
 x=np.concatenate(parts[a])[mask];means[a]=x.mean(0,keepdims=True);stds[a]=x.std(0,keepdims=True)+np.float32(1e-6)
maxnorm=0;maxmetric=0;total=0;orders={};param={};max_bundle_error=0;reported=[]
for run in plan['runs']:
 a=run['arm'];arch=run['architecture'];seed=run['seed'];d=work/'runs'/arch/('seed_'+str(seed))/a
 res=json.loads((d/'result.json').read_text());cfg=json.loads((d/'run_configuration.json').read_text());normal=np.load(d/'normalizer.npz',allow_pickle=False)
 assert sha(d/'checkpoint_final.pt')==run['checkpoint_sha256'] and sha(d/'val_predictions_final.csv')==run['validation_predictions_sha256']
 maxnorm=max(maxnorm,float(np.max(np.abs(normal['mean']-means[a]))),float(np.max(np.abs(normal['std']-stds[a]))))
 assert cfg['training']['policy']=='all' and cfg['training']['epochs']==25 and cfg['arm_spec']['nf']==12
 if not a.endswith('_bundle'):
  for hist in res['history']:
   max_bundle_error=max(max_bundle_error,abs(hist['train_total']-hist['train_binary']))
   assert hist['train_type_ce']==0 and hist['train_js']==0 and hist['train_aug_binary']==0 and hist['train_hidden_smooth']==0
 assert res['gradient_training_windows']==7840 and res['optimizer_steps_per_epoch']==62
 orders.setdefault((arch,seed),set()).add(res['batch_order_sha256'])
 ck=torch.load(d/'checkpoint_final.pt',map_location='cpu',weights_only=False);nparams=sum(int(v.numel()) for v in ck['state_dict'].values());assert nparams==res['parameters'];param[a+':'+arch]=nparams
 with (d/'val_predictions_final.csv').open(newline='',encoding='utf8') as h:r=list(csv.DictReader(h))
 y=np.array([int(z['y']) for z in r]);s=np.array([float(z['score']) for z in r]);g=np.array([z['region'] for z in r]);assert len(y)==1680;total+=len(y)
 ap=average_precision_score(y,s);auc=roc_auc_score(y,s);mac=np.mean([average_precision_score(y[g==x],s[g==x]) for x in regions])
 maxmetric=max(maxmetric,abs(ap-res['final_validation']['pooled']['AP']),abs(auc-res['final_validation']['pooled']['AUROC']),abs(mac-res['final_validation']['macro']['AP']))
 neg=np.sort(s[y==0])[::-1]
 for alpha,tx in run['thresholds'].items():
  k=int(np.floor(float(alpha)*len(neg)));expected=np.nextafter(neg[min(k,len(neg)-1)],np.inf)
  assert tx['threshold']==expected and int(np.sum(neg>=expected))==tx['validation_false_alarms']
 reported.append({'architecture':arch,'seed':seed,'arm':a,'macro_AP':mac})
assert len(plan['runs'])==72 and total==120960 and all(len(x)==1 for x in orders.values())
assert maxnorm<1e-6 and maxmetric<1e-10 and max_bundle_error<1e-7
result={'status':'passed','runs':len(plan['runs']),'validation_predictions':total,'max_normalizer_error':maxnorm,'max_metric_error':maxmetric,
 'batch_order_equal_within_arch_seed':True,'bce_train_total_max_deviation':max_bundle_error,'parameters':param,
 'thresholds_recomputed_from_validation_negatives':True,'checkpoint_hashes_match_frozen_plan':True,'per_run':reported}
(out/'audit_main_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ('per_run','parameters')}))
