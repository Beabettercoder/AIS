from pathlib import Path
import json,hashlib,math,numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'hawaii_event_feasibility_20261031'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
f=json.loads((O/'frozen_ood_models_thresholds.json').read_text()); f['amendment']='Before any external model scoring, add original base-trained C0 target-only models for every matching seed as a descriptive no-context control; no synthetic mixed C0 existed.'
for name in ('tcn','gru','transformer','itransformer'):
 for seed in ((0,1,2) if name in ('tcn','itransformer') else (0,)):
  src='itrans_sourcebalanced_20261031' if name=='itransformer' else 'heading_evidence_pilot_20261030';p=R/src/'runs'/name/f'seed{seed}'/'C0';m=json.loads((p/'result.json').read_text()); w=p/'final.pt';vp=p/'val_scores.npz';assert m['weights_sha256']==sha(w) and m['prediction_sha256']==sha(vp)
  with np.load(vp,allow_pickle=False) as z:
   y=z['y'];pred=z['base_score'];neg=np.sort(pred[y==0]); thresh={}
   for alpha in (.01,.05):
    index=max(0,len(neg)-math.floor(alpha*len(neg))-1);thresh[str(alpha)]={'threshold':float(neg[index]),'n_neg':len(neg),'empirical_val_far':float(np.mean(neg>neg[index]))}
  f['models'].append({'name':name,'seed':seed,'arm':'C0','training':'base','model':str(w),'model_sha256':sha(w),'val_predictions':str(vp),'val_prediction_sha256':sha(vp),'model_protocol':str(p.parent/'protocol.json'),'model_protocol_sha256':sha(p.parent/'protocol.json'),'thresholds':thresh})
f['prior_frozen_model_sha256']=sha(O/'frozen_ood_models_thresholds.json');(O/'frozen_ood_models_thresholds_v2.json').write_text(json.dumps(f,indent=2),encoding='utf8');print('MODELS',len(f['models']),'sha',sha(O/'frozen_ood_models_thresholds_v2.json'))
