"""Freeze validation-only selection and operating thresholds before test access."""
from pathlib import Path
from datetime import datetime, timezone
import csv,hashlib,json
import numpy as np
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/causal_gate_main_v1_20261027')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((OUT/'training_protocol.json').read_text());arms=protocol['arms'];runs=[]
for arch in protocol['backbones']:
 for seed in protocol['seeds']:
  for arm in arms:
   d=OUT/'runs'/arch/f'seed_{seed}'/arm
   result=json.loads((d/'result.json').read_text());assert result['status']=='completed_validation_only'
   with (d/'val_predictions_final.csv').open(encoding='utf8',newline='') as f:r=list(csv.DictReader(f))
   neg=np.array([float(x['score']) for x in r if int(x['y'])==0],np.float64);thresholds={}
   for alpha in (.01,.05):
    k=int(np.floor(alpha*len(neg)));ordered=np.sort(neg)[::-1];threshold=float(np.nextafter(ordered[min(k,len(ordered)-1)],np.inf))
    fpr=float(np.mean(neg>=threshold));assert fpr<=alpha
    thresholds[str(alpha)]={'threshold':threshold,'validation_false_alarms':int(np.sum(neg>=threshold)),'validation_negatives':len(neg),'validation_empirical_far':fpr}
   runs.append({'architecture':arch,'seed':seed,'arm':arm,'checkpoint':str(d/'checkpoint_final.pt'),'checkpoint_sha256':sha(d/'checkpoint_final.pt'),
    'validation_predictions_sha256':sha(d/'val_predictions_final.csv'),'val_macro_AP':result['final_validation']['macro']['AP'],'thresholds':thresholds})
means={a:float(np.mean([r['val_macro_AP'] for r in runs if r['arm']==a])) for a in arms}
best=max(arms,key=lambda a:(means[a],-arms.index(a)))
plan={'status':'frozen_before_test','frozen_utc':datetime.now(timezone.utc).isoformat(),'validation_scope':'development validation has been explored repeatedly; this is not a preregistered final study', 'protocol_sha256':sha(OUT/'training_protocol.json'),'gate_protocol_sha256':sha(OUT/'protocol.json'),
 'test_evaluator_sha256':sha(OUT/'evaluate_test.py'),'test_background_builder_sha256':sha(OUT/'build_backgrounds_test.py'),
 'population':'eight region earlier test splits; previously viewed in pilot and paired-view evaluations; not a never-exposed independent blind test',
 'primary_arm':best,'selection':'highest mean fixed-epoch validation macro AP averaged equally over the 2 backbones and 3 seeds; one shared arm',
 'validation_arm_means':means,'all_predeclared_arms_evaluated_as_controls':arms,'per_run_thresholds':'pooled validation negatives only; nextafter score of first disallowed negative; test applies score>=threshold; ties conservative',
 'bootstrap':'2000 seeded target-MMSI cluster draws, same draws across methods; paired AP comparison for causal-raw, weighted_median-causal_common, weighted_median-affine, weighted_median_bundle-weighted_median',
 'runs':runs}
path=OUT/'frozen_test_plan.json'
with path.open('x',encoding='utf8') as f:json.dump(plan,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'primary_arm':best,'validation_arm_means':means,'runs':len(runs),'plan_sha256':sha(path)},indent=2),flush=True)
