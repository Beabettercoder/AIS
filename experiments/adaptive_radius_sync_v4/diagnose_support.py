from pathlib import Path
import json,numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
import train_relation_v2 as rel
root=Path('/home/ps/AIS_adaptive_sync_20261110');d,_=rel.data(root,'test');y=d['y'];mask=d['mask'];result=[]
for view in ('base','natural','suffix','combined_l100_m600'):
 for arm in ('tight_plain','wide_plain','adaptive_plain_fallback','adaptive_credible_fallback'):
  n=d[view+'__'+arm+'__count'];mode=d[view+'__'+arm+'__field'][:,:,7];length=mask.sum(1);first=np.take_along_axis(n,np.zeros((len(n),1),int),1)[:,0];last=np.take_along_axis(n,(length-1)[:,None],1)[:,0]
  for name,score in [('mean_count',(n*mask).sum(1)/length),('last_minus_first_count',last-first),('coverage_fraction',((mode>0)&mask).sum(1)/length)]:
   result.append({'view':view,'arm':arm,'diagnostic':name,'AP_high':float(average_precision_score(y,score)),'AP_low':float(average_precision_score(y,-score)),'AUROC_high':float(roc_auc_score(y,score))})
path=root/'reports/support_only_diagnostic.json';path.write_text(json.dumps({'status':'complete','role':'exploratory deterministic statistics, sign choices shown for audit, not trained model or independent test','samples':len(y),'results':result},indent=2));print('SUPPORT_DIAGNOSTIC_COMPLETE',len(result))
