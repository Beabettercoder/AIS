"""Freeze selected candidate, checkpoints, thresholds and evaluator before test evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');OUT=ROOT/'causal_gate_followup_v2_20261027';BASE=ROOT/'causal_gate_main_v1_20261027';HIST=ROOT/'gating_training_v2_20261027/historical'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
selection=json.loads((OUT/'development_selection.json').read_text(encoding='utf8'));assert selection['status']=='locked_before_test'
files=[OUT/n for n in ['protocol.json','development_selection.json','fitted_parameters.json','build_candidates.py','train_candidate.py','evaluate_locked_test.py']]+[BASE/'evaluate_test.py',BASE/'train_controls.py',BASE/'formal_test_results/all_test_predictions.npz',HIST/'historical_model.py',HIST/'train_historical.py',HIST/'vendor/35_graph_ours_final.py']
runs=[]
for arch in ('tcn','gru'):
 for seed in (0,1,2):
  d=OUT/'runs'/arch/f'seed_{seed}'/'candidate';result=json.loads((d/'result.json').read_text(encoding='utf8'));assert result['status']=='completed_validation_only' and result['primary_epoch']==25 and result['epochs']==25
  p=d/'val_predictions_final.csv';rows=list(csv.DictReader(p.open(encoding='utf8',newline='')));neg=np.array([float(x['score']) for x in rows if int(x['y'])==0]);assert len(neg)>0;th={}
  for alpha in (.01,.05):
   k=int(np.floor(alpha*len(neg)));s=np.sort(neg)[::-1];threshold=float(np.nextafter(s[min(k,len(s)-1)],np.inf));actual=float(np.mean(neg>=threshold));assert actual<=alpha
   th[str(alpha)]={'threshold':threshold,'validation_FAR':actual,'validation_negatives':len(neg),'validation_false_alarms':int(np.sum(neg>=threshold))}
  cp=d/'checkpoint_final.pt';files.extend([cp,p,d/'result.json']);runs.append({'architecture':arch,'seed':seed,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'thresholds':th,'validation_macro_AP':result['final_validation']['macro']['AP']})
locks={str(p):sha(p) for p in files};plan={'status':'frozen_before_test','frozen_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'test source historically exposed; selected gate designed with prior exploration of validation; all 3 candidates retained; no independent blind-test claim',
 'candidate':selection['candidate'],'selection_sha256':sha(OUT/'development_selection.json'),'protocol_sha256':sha(OUT/'protocol.json'),
 'builder_sha256':sha(OUT/'build_candidates.py'),'evaluator_sha256':sha(OUT/'evaluate_locked_test.py'),
 'thresholds':'pooled validation negative scores, nextafter first disallowed score; compare test score >= threshold',
 'fixed_epoch':25,'comparison_arms':['raw','old','comp','causal'],'bootstrap':'paired 2000 draws target MMSI; seed averaging conditional on existing three seeds; shared backgrounds not accounted',
 'locks':locks,'runs':runs}
with (OUT/'frozen_test_plan.json').open('x',encoding='utf8') as f:json.dump(plan,f,ensure_ascii=False,indent=2,allow_nan=False)
print('FROZEN',plan['candidate'],len(runs),len(locks),flush=True)
