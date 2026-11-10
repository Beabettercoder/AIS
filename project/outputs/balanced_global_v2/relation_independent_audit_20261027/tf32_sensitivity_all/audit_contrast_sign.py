"""Compare fixed three-seed mean AP contrast signs before and after TF32 toggle."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib
out=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_independent_audit_20261027/tf32_sensitivity_all');p=out/'per_model_view.csv'
with p.open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
groups=defaultdict(list)
for row in rows:groups[(row['architecture'],row['training_exposure'],row['view'],row['context'])].append(row)
res=[]
for (arch,exposure,view,context),entries in groups.items():
 if context=='median':continue
 base={int(r['seed']):r for r in groups[(arch,exposure,view,'median')]}
 assert len(entries)==3 and set(base)=={0,1,2}
 delta={key:sum(float(r['pooled_AP_'+key])-float(base[int(r['seed'])]['pooled_AP_'+key]) for r in entries)/3 for key in ('on','off')}
 res.append({'architecture':arch,'exposure':exposure,'view':view,'context':context,'delta_on':delta['on'],'delta_off':delta['off'],'strict_sign_flip':delta['on']*delta['off']<0})
result={'status':'complete','comparisons':len(res),'strict_sign_flips':sum(r['strict_sign_flip'] for r in res),'per_model_view_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'entries':res}
with (out/'contrast_signs.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print('CONTRASTS',result['comparisons'],'FLIPS',result['strict_sign_flips'])
