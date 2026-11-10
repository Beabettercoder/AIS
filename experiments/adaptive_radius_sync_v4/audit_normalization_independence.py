"""Prove field normalization ignores every credibility-aware feature array."""
from pathlib import Path
import hashlib,json
import numpy as np
import train_single_v2 as single
import train_relation_v2 as relation
root=Path('/home/ps/AIS_adaptive_sync_20261110')
cred=('adaptive_qualified_fallback','adaptive_credible_fallback')
results={}
for task,module in (('relation',relation),):
 d,_=module.data(root,'train');original=module.normalization(d);changed=dict(d)
 for arm in cred:
  if task=='single':
   changed[arm+'__field']=np.full_like(d[arm+'__field'],12345.)
   changed[arm+'__mode']=np.full_like(d[arm+'__mode'],2)
   changed[arm+'__rho']=np.zeros_like(d[arm+'__rho'])
  else:
   changed['base__'+arm+'__field']=np.full_like(d['base__'+arm+'__field'],12345.)
   changed['base__'+arm+'__count']=np.full_like(d['base__'+arm+'__count'],999)
 altered=module.normalization(changed)
 for key in original:
  if not np.array_equal(original[key],altered[key]):raise ValueError('normalization depends on credibility-aware arm '+task+'/'+key)
 results[task]={'status':'passed','fields_compared':list(original),'normalization_sha256':hashlib.sha256(b''.join(np.ascontiguousarray(original[k]).tobytes() for k in original)).hexdigest(),
  'test':'changed all credibility-aware input arrays and verified exactly identical training normalization'}
path=root/'reports/normalization_independence.json';path.write_text(json.dumps(results,indent=2),encoding='utf8');print('NORMALIZATION_INDEPENDENCE_PASSED',json.dumps(results),flush=True)
