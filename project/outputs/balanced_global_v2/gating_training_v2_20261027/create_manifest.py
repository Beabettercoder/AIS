from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
files=[]
for f in sorted(p.rglob('*')):
 if not f.is_file() or f.name in ('integrity_manifest.json','prepare_code.py') or '__pycache__' in f.parts:continue
 h=hashlib.sha256()
 with f.open('rb') as x:
  for block in iter(lambda:x.read(2**20),b''):h.update(block)
 files.append({'path':f.relative_to(p).as_posix(),'size':f.stat().st_size,'sha256':h.hexdigest()})
(p/'integrity_manifest.json').write_text(json.dumps({'count':len(files),'files':files},indent=2),encoding='utf-8')
print('files',len(files))
