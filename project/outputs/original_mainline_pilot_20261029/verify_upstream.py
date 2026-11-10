import json,hashlib
from pathlib import Path
p=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028');o=Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029');manifest=json.loads((o/'audit_manifest.json').read_text());core=json.loads((p/'code_lock.json').read_text());fld=json.loads((p/'field_extension/code_lock.json').read_text())
def sha(f):
 h=hashlib.sha256()
 with open(f,'rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
items=[p/'models.py',p/'extra_data/field_member_provenance.jsonl.gz',p/'code_lock.json',p/'field_extension/code_lock.json',o/'pilot_audit_stats.py',o/'pilot_lock.sha256',o/'pilot_results.json',o/'pilot_bootstrap.json']
for x in items:manifest[str(x)]=sha(x)
checks={}
for path,want in [(p/'models.py',core['files']['models.py'])]+[(p/'data/individual'/f'{s}.npz',core['data_files'][f'data/individual/{s}.npz']) for s in ('train','val')]+[(p/'extra_data'/f'field_{s}.npz',fld['field_files'][f'extra_data/field_{s}.npz']) for s in ('train','val')]:
 got=sha(path);checks[str(path)]={'actual':got,'expected_prior_lock':want,'match':got==want};assert got==want
(o/'audit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');(o/'upstream_lock_check.json').write_text(json.dumps(checks,indent=2),encoding='utf8');print('upstream_lock_checks',len(checks),'all_true')
