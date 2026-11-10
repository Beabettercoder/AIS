from pathlib import Path
import urllib.request, concurrent.futures, hashlib, json, time
BASE='https://s3.eu-central-1.amazonaws.com/aisdata.ais.dk/'
OUT=Path('/mnt/sda/WangZijia/AIS/data/raw/external/dma_2024')
ITEMS=[('2024/aisdk-2024-08-17.zip',685121317)]
def get(item):
 key,size=item; target=OUT/Path(key).name; part=target.with_suffix('.part'); url=BASE+key
 if not target.exists():
  n=part.stat().st_size if part.exists() else 0
  req=urllib.request.Request(url,headers={'Range':f'bytes={n}-'} if n else {})
  with urllib.request.urlopen(req,timeout=60) as r:
   if n and r.status!=206: raise RuntimeError('server did not honor Range')
   with part.open('ab' if n else 'wb') as f:
    while True:
     b=r.read(64*1024)
     if not b: break
     f.write(b)
  if part.stat().st_size!=size: raise RuntimeError(f'size mismatch {part}')
  part.rename(target)
 h=hashlib.sha256()
 with target.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 result={'path':str(target),'url':url,'expected_bytes':size,'bytes':target.stat().st_size,'sha256':h.hexdigest(),'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'license':'Official freely downloadable historical AIS; Danish public sector information terms; attribution Danish Maritime Authority / Danish Emergency Management Agency.'}
 target.with_suffix('.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
 return result
OUT.mkdir(exist_ok=True,parents=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(get,ITEMS))
(OUT/'manifest.json').write_text(json.dumps(results,indent=2))
