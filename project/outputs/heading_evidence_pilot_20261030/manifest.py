"""Hash all deliverables after completion. This does not include itself's manifest."""
import hashlib,json
from pathlib import Path
root=Path('/mnt/sda/WangZijia/AIS/outputs/heading_evidence_pilot_20261030')
records={}
for p in sorted(root.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.name=='complete_manifest.json':continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 records[p.relative_to(root).as_posix()]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
out={'scope':'Heading side-evidence exploratory pilot, train/val only','files':records}
(root/'complete_manifest.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(len(records))
