from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
P=Path('/mnt/sda/WangZijia/AIS/outputs/gate_step_diagnostic_20261030/train_sample_records.npz')
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_conditioned_20261030')
with np.load(P) as z: a=z['suffix']
out={'scope':'Exploratory train-only 400 sources; report-level rows repeated across case/source. speed before modification used ONLY for evaluation; observed inputs only for scoring. AUC not causal performance on actual naturally labelled AIS.', 'bins':{}}
for name,sel in [('lt2',(a['speed']<2)),('2to5',(a['speed']>=2)&(a['speed']<5)),('5to10',(a['speed']>=5)&(a['speed']<10)),('ge10',(a['speed']>=10))]:
 s=sel & ((a['kind']==1)|(a['kind']==0))
 z=a[s]; y=(z['kind']==1).astype(int)
 if not len(z) or y.sum()==0:continue
 result={'n':len(z),'positives':int(y.sum()),'targets':len(np.unique(z['mmsi'])),'metrics':{}}
 for f in ['e0','ec','es','rel']:
  score=z[f]; auc=roc_auc_score(y,score); ap=average_precision_score(y,score)
  vals={}
  for lab in [0,1]:
   q=score[y==lab]; vals[str(lab)]={'q10':float(np.quantile(q,.1)), 'median':float(np.median(q)), 'q90':float(np.quantile(q,.9))}
  q99=np.quantile(score[y==0],.99)
  result['metrics'][f]={'auc':float(auc),'AP':float(ap),'q99_unmodified':float(q99),'modified_above_unmodified_q99':float(np.mean(score[y==1]>q99)),'quantiles':vals}
 result['elapsed_seconds']={'unchanged_median':float(np.median(z['dt'][y==0])),'changed_median':float(np.median(z['dt'][y==1]))}
 out['bins'][name]=result
OUT.mkdir(exist_ok=True)
(OUT/'distribution.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
