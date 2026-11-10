"""Recompute all backend AP/AUROC differences from saved per-example predictions."""
from pathlib import Path
import csv,json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
out=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_independent_audit_20261027/tf32_sensitivity_all')
with np.load(out/'replayed_predictions.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
with (out/'per_model_view.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
assert d['tf32_on'].shape==d['tf32_off'].shape==(108,3,880) and len(rows)==324
err=0.;actual_score_max=0.;actual_ap_max=0.
for i in range(108):
 for j,view in enumerate(d['views']):
  r=rows[i*3+j];assert (r['architecture'],r['training_exposure'],r['context'],int(r['seed']),r['view'])==(str(d['architecture'][i]),str(d['exposure'][i]),str(d['context'][i]),int(d['seed'][i]),str(view))
  on=d['tf32_on'][i,j];off=d['tf32_off'][i,j];ap_on=float(average_precision_score(d['y'],on));ap_off=float(average_precision_score(d['y'],off));auc_on=float(roc_auc_score(d['y'],on));auc_off=float(roc_auc_score(d['y'],off));actual_score_max=max(actual_score_max,float(np.max(np.abs(off-on))));actual_ap_max=max(actual_ap_max,abs(ap_off-ap_on))
  for a,b in [(ap_on,float(r['pooled_AP_on'])),(ap_off,float(r['pooled_AP_off'])),(auc_on,float(r['AUROC_on'])),(auc_off,float(r['AUROC_off'])),(ap_off-ap_on,float(r['delta_AP_off_minus_on']))]:err=max(err,abs(a-b))
result={'status':'passed','test_model_views':len(rows),'max_recomputed_AP_AUROC_delta_error':err,'max_recomputed_AP_delta':actual_ap_max,'max_recomputed_prediction_delta':actual_score_max,'scope':'independent metric calculation from saved replayed predictions; no independent model forward'}
with (out/'independent_recompute.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(result)
