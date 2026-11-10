import json,hashlib,math
from pathlib import Path
import numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs');D=R/'hawaii_event_feasibility_20261031';O=R/'hawaii_independent_audit_20261031'; f=json.loads((D/'frozen_ood_models_thresholds_v2.json').read_text()); rows=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for m in f['models']:
 assert sha(m['model'])==m['model_sha256'] and sha(m['val_predictions'])==m['val_prediction_sha256'] and sha(m['model_protocol'])==m['model_protocol_sha256']
 with np.load(m['val_predictions'],allow_pickle=False) as z:
  y=z['y'];v=z['base_score']; neg=np.sort(v[y==0]); assert len(neg)==544
  for level in (0.01,0.05):
   idx=len(neg)-math.floor(level*len(neg))-1; thr=float(neg[idx]); t=m['thresholds'][str(level)]; assert thr==t['threshold'] and abs(float(np.mean(neg>thr))-t['empirical_val_far'])<1e-12
 rows.append((m['name'],m['seed'],m['arm'],m['training']))
assert len(rows)==len(set(rows))==40
out={'models_checked':len(rows),'weight_prediction_protocol_hashes':'all matched','thresholds':'all reproduce exactly from stored development validation/Base y=0 predictions, strict >; no Hawaii input','validation_negative_examples_per_model':544,'warning':'Thresholds from previously exposed synthetic development negatives; not actual OOD false alarm guarantee.'};(O/'independent_thresholds.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
