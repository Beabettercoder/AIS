"""Lock all pre-existing models and validation negative thresholds prior to Hawaii scoring."""
from pathlib import Path
import json,hashlib,math,numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OUT=ROOT/'hawaii_event_feasibility_20261031'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
models=[]
for name in ('tcn','gru','transformer','itransformer'):
 for seed in ((0,1,2) if name in ('tcn','itransformer') else (0,)):
  for training in ('base','mixed'):
   source={'mixed':'mixed_stress_probe_20261031','base':('itrans_sourcebalanced_20261031' if name=='itransformer' else 'heading_evidence_pilot_20261030')}[training]
   for arm in ('F0_heading','F1_heading'):
    p=ROOT/source/'runs'/name/f'seed{seed}'/arm
    meta=json.loads((p/'result.json').read_text());assert meta['last_epoch']==(75 if name=='itransformer' else 25)
    weights=p/'final.pt';vp=p/'val_scores.npz';assert meta['weights_sha256']==sha(weights) and meta['prediction_sha256']==sha(vp)
    with np.load(vp,allow_pickle=False) as z:y=z['y'];pred=z['base_score'];neg=np.sort(pred[y==0]); assert len(neg)>0
    threshold={}
    for alpha in (.01,.05):
     allowed=math.floor(alpha*len(neg));index=max(0,len(neg)-allowed-1);threshold[str(alpha)]={'threshold':float(neg[index]),'n_neg':len(neg),'empirical_val_far':float(np.mean(neg>neg[index]))}
    models.append({'name':name,'seed':seed,'arm':arm,'training':training,'model':str(weights),'model_sha256':sha(weights),'val_predictions':str(vp),'val_prediction_sha256':sha(vp),'model_protocol':str(p.parent/'protocol.json'),'model_protocol_sha256':sha(p.parent/'protocol.json'),'thresholds':threshold})
f={'scope':'Frozen OOD diagnostic, all 4 predefined event-vessels and all observed windows with >=8 raw observations; fixed source cohort may have fewer than3; target may be stationary. No report-window truth, no recall/FAR, no online or conforming-domain claim.','models':models,'selection':'all TCN/iTransformer seeds0,1,2 and GRU/Transformer seed0, each F0+heading and F1+heading, both old base and mixed training; no model score was accessed during event or window selection','threshold_policy':'old validation/base negative synthetic XOR labels only, empirical order statistic floor(alpha*n), strictly greater, alpha=.01/.05. Not calibrated probabilities and no real-event FAR guarantee.','windows_sha256':sha(OUT/'observed_event_windows.json'),'preselection_protocol_sha256':sha(OUT/'frozen_protocol.json'),'preselection_addendum_sha256':sha(OUT/'frozen_protocol_addendum.json'),'source_original_sha256':'9d6b47b92aef7c377000a1043e4fe02dc5f1e9cf8dd41241b3dfa30a1548e30f','context_parquet_sha256':sha(OUT/'canonical_context_v2.parquet'),'result_instruction':'Do not select/omit windows or models by event score. Do not train or recalibrate. Entire trajectory time and cohort are offline and may use whole-window eligibility. Treat missing/invalid heading and kinematics per original code. Event2 no positive-event labels or externally established precise maneuver interval.'}
(OUT/'frozen_ood_models_thresholds.json').write_text(json.dumps(f,indent=2),encoding='utf8')
print('FROZEN_MODELS',len(models),'sha',sha(OUT/'frozen_ood_models_thresholds.json'))
