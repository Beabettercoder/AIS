import json,hashlib
from pathlib import Path
import numpy as np
root=Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')
rows=[]
for name in ['tcn','gru','transformer','itransformer']:
 cases={}
 for arm in ['raw','causal_bundle','f0_bundle','f1_bundle']:
  p=root/'runs'/name/arm
  r=json.loads((p/'result.json').read_text());z=np.load(p/'final_val.npz');cases[arm]=(r,z)
 ref=cases['raw'][1]
 for arm,(r,z) in cases.items():
  for k in ['y','mmsi','region','window_id']:assert np.array_equal(z[k],ref[k]),(name,arm,k)
  assert r['init']==cases['raw'][0]['init']
  rows.append({'name':name,'arm':arm,'final_macro_ap':r['final']['macro_AP'],'final_pooled_ap':r['final']['pooled_AP'],'final_auroc':r['final']['AUROC'],'best_epoch':r['best_epoch'],'best_macro_ap':r['best_macro_AP'],'alpha':r['alpha']})
result={'comparison':'single seed0, fixed25 primary; validation only, exposed sources','rows':rows}
(root/'pilot_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
for name in ['tcn','gru','transformer','itransformer']:
 r={x['arm']:x for x in rows if x['name']==name}
 print(name,*(f"{arm}={r[arm]['final_macro_ap']:.4f}" for arm in r),f"field_minus_causal={(r['f1_bundle']['final_macro_ap']-r['causal_bundle']['final_macro_ap'])*100:+.2f}pp",f"f1_minus_f0={(r['f1_bundle']['final_macro_ap']-r['f0_bundle']['final_macro_ap'])*100:+.2f}pp")
