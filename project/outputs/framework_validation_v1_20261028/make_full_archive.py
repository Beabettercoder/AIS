from pathlib import Path
import hashlib,json,zipfile,time
R=Path(__file__).resolve().parent;dest=R.parent/'framework_execution_20261029.zip'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
files=[]
for p in R.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts or p.name.endswith('.tmp'):continue
    if not p.resolve().is_relative_to(R.resolve()):raise RuntimeError('External path found')
    files.append(p)
manifest={str(p.relative_to(R)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files if p.name!='execution_manifest.json'}
(R/'execution_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
files=[p for p in files if p.name!='execution_manifest.json']+[R/'execution_manifest.json']
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for i,p in enumerate(files):
        z.write(p,p.relative_to(R))
        if i%1000==0:print('PACK',i,'/',len(files),flush=True)
with zipfile.ZipFile(dest) as z:assert z.testzip() is None
result={'path':str(dest),'files':len(files),'bytes':dest.stat().st_size,'sha256':sha(dest),'external_raw_AIS_included':False,'scope':'stage files including final/best/resume checkpoints, per-window predictions, caches, frozen source/config, audits and reports; original AIS/predecessor modules remain prerequisites'}
(R/'execution_archive.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False),flush=True)
