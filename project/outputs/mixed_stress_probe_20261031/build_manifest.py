from pathlib import Path
import hashlib,json
D=Path('/mnt/sda/WangZijia/AIS/outputs/mixed_stress_probe_20261031')
assets={}
for p in sorted(D.rglob('*')):
 if p.is_file() and p.name!='complete_manifest.json' and '__pycache__' not in p.parts:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  assets[str(p.relative_to(D))]={'sha256':h.hexdigest(),'size':p.stat().st_size}
(D/'complete_manifest.json').write_text(json.dumps({'files':assets,'count':len(assets)},indent=2),encoding='utf8')
print('files',len(assets))
