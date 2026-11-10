from pathlib import Path
import os,json,hashlib,datetime
O=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/relation');files={}
for p in sorted(O.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.name=='release_manifest.json' or p.suffix=='.pyc':continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 files[str(p.relative_to(O))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'file_count':len(files),'description':'train/val/test outputs, released after pipeline completed; excludes itself and Python caches','files':files}
(O/'release_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8');print('file_count',len(files))
