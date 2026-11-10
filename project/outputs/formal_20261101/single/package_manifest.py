import json,hashlib
from pathlib import Path
p=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');result={}
for x in sorted(p.rglob('*')):
 if not x.is_file() or '__pycache__' in x.parts or x.name=='delivery_manifest.json':continue
 h=hashlib.sha256()
 with x.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 result[x.relative_to(p).as_posix()]={'size':x.stat().st_size,'sha256':h.hexdigest()}
(p/'delivery_manifest.json').write_text(json.dumps({'files':result},indent=2));print('MANIFEST_FILES',len(result),'BYTES',sum(x['size'] for x in result.values()))
