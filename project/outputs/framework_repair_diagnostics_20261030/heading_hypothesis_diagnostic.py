import numpy as np, importlib.util, json
from pathlib import Path
from sklearn.metrics import average_precision_score
base=Path('/mnt/sda/WangZijia/AIS/outputs')
spec=importlib.util.spec_from_file_location('prev_relation_train',base/'relation_repair_20261030/train.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def audit(split):
 d=m.read(split); x,cm=m.features(d,'base','F1'); n=np.asarray(d['length']); i=np.arange(len(n)); j=n-1; y=d['y'].astype(int); last=x[i,j]; theta=np.stack([last[:,3],last[:,4]],-1); ds=last[:,7]; bgd=last[:,10:12]; valid=cm[i,j];bg_parallel=(bgd*theta).sum(-1);score=np.abs(ds-bg_parallel); vars={'background_delta_x':last[:,10],'background_delta_y':last[:,11],'background_delta_parallel':bg_parallel,'target_minus_background_delta_abs':score}; out={'split':split,'n_cases':len(y),'last_step_supported':int(valid.sum())}
 for name,v in vars.items():
  r=np.corrcoef(v[valid],y[valid])[0,1];ap=average_precision_score(y[valid],v[valid]);ap_neg=average_precision_score(y[valid],-v[valid]);out[name]={'pearson_label':float(r),'positive_ap':float(ap),'negative_ap':float(ap_neg),'std':float(np.std(v[valid]))}
 return out
p=base/'framework_repair_diagnostics_20261030/heading_hypothesis_diagnostic.json';p.write_text(json.dumps({'method':'last-valid-step correlation, observational and post-hoc, not a source-isolated cause','results':[audit('train'),audit('val')]},indent=2));print(p.read_text())
