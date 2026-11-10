"""Freeze all new checkpoint/validation thresholds before historical test reads."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=ROOT/'factorial_ablation_v1_20261027'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=json.loads((OUT/'protocol.json').read_text());runs=[]
for arch in spec['common_budget']['backbones']:
 for seed in spec['common_budget']['seeds']:
  for arm in spec['arms']:
   d=OUT/'runs'/arch/f'seed_{seed}'/arm
   result=json.loads((d/'result.json').read_text());assert result['status']=='completed_validation_only'
   with (d/'val_predictions_final.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
   neg=np.sort(np.array([float(r['score']) for r in rows if int(r['y'])==0]))[::-1];thresholds={}
   for alpha in (.01,.05):
    k=int(np.floor(alpha*len(neg)));t=float(np.nextafter(neg[min(k,len(neg)-1)],np.inf));actual=float(np.mean(neg>=t));assert actual<=alpha
    thresholds[str(alpha)]={'threshold':t,'validation_false_alarms':int(np.sum(neg>=t)),'validation_negatives':len(neg),'validation_FAR':actual}
   runs.append({'architecture':arch,'seed':seed,'arm':arm,'checkpoint':str(d/'checkpoint_final.pt'),'checkpoint_sha256':sha(d/'checkpoint_final.pt'),
    'validation_predictions_sha256':sha(d/'val_predictions_final.csv'),'validation_macro_AP':result['final_validation']['macro']['AP'],'thresholds':thresholds})
assert len(runs)==72
plan={'status':'frozen_before_test','frozen_utc':datetime.now(timezone.utc).isoformat(),'scope':'historically exposed main test, not blind; all predetermined 12 new arms evaluated; no selection from test',
 'protocol_sha256':sha(OUT/'protocol.json'),'training_sha256':sha(OUT/'train_factorial.py'),'evaluator_sha256':sha(OUT/'evaluate_test_factorial.py'),
 'prior_test_preparation_sha256':sha(ROOT/'causal_gate_main_v1_20261027/evaluate_test.py'),
 'threshold_rule':'per-model pooled validation negatives; nextafter first disallowed score, >= threshold, ties conservative',
 'bootstrap':'2000 target MMSI cluster draws shared across models, paired AP differences computed without treating seeds as independent subjects','runs':runs}
dest=OUT/'frozen_test_plan.json'
with dest.open('x',encoding='utf8') as f:json.dump(plan,f,ensure_ascii=False,indent=2,allow_nan=False)
print('FROZEN',len(runs),sha(dest),flush=True)
