"""Resumable verified HTTP-range download of the official public DMA archive."""
from pathlib import Path
import urllib.request, concurrent.futures, hashlib, json, time, os, zipfile, argparse, math
BASE='https://s3.eu-central-1.amazonaws.com/aisdata.ais.dk/2024/aisdk-2024-08-17.zip'
OUT=Path('/mnt/sda/WangZijia/AIS/data/raw/external/dma_2024')
SIZE=685121317
PART=OUT/'aisdk-2024-08-17.part'
FINAL=OUT/'aisdk-2024-08-17.zip'
CHUNKS=OUT/'aisdk-2024-08-17.ranges'

def read_range(a,b,path):
 expected=b-a+1
 if path.exists() and path.stat().st_size==expected:return {'cached':True,'bytes':expected}
 for attempt in range(8):
  n=path.stat().st_size if path.exists() else 0
  if n>expected:raise ValueError('oversized chunk')
  if n==expected:return {'bytes':expected}
  req=urllib.request.Request(BASE,headers={'Range':f'bytes={a+n}-{b}','Accept-Encoding':'identity'})
  try:
   started=time.monotonic()
   with urllib.request.urlopen(req,timeout=40) as r:
    actual=r.headers.get('Content-Range')
    wanted=f'bytes {a+n}-{b}/{SIZE}'
    if r.status!=206 or actual!=wanted:raise ValueError(f'wrong range {r.status} {actual}, expected {wanted}')
    with path.open('ab') as f:
     while True:
      data=r.read(64*1024)
      if not data:break
      if f.tell()+len(data)>expected:raise ValueError('response larger than requested range')
      f.write(data)
   if path.stat().st_size!=expected:raise IOError('truncated response')
   return {'bytes':expected,'downloaded_this_attempt':expected-n,'seconds':time.monotonic()-started}
  except Exception as e:
   print(json.dumps({'retry':attempt,'range':[a,b],'error':repr(e)}),flush=True)
   if attempt==7:raise
   time.sleep(min(20,2*(attempt+1)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');args=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);CHUNKS.mkdir(exist_ok=True)
 if args.probe:
  start=200*1024*1024;length=1024*1024; t=time.monotonic()
  jobs=[(start+i*length,start+(i+1)*length-1,CHUNKS/f'probe_{i}.bin') for i in range(4)]
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:res=list(pool.map(lambda x:read_range(*x),jobs))
  elapsed=time.monotonic()-t
  report={'parallel_connections':4,'bytes':4*length,'elapsed_seconds':elapsed,'aggregate_bytes_per_second':4*length/elapsed,'responses':res}
  (CHUNKS/'probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
 (CHUNKS/'download.pid').write_text(str(os.getpid()))
 if not FINAL.exists():
  prefix=PART.stat().st_size if PART.exists() else 0
  if prefix>SIZE:raise ValueError('oversized prefix')
  layout_path=CHUNKS/'layout.json'
  if layout_path.exists():
   layout=json.loads(layout_path.read_text()); assert layout['prefix']==prefix
  else:
   length=math.ceil((SIZE-prefix)/16)
   layout={'url':BASE,'total':SIZE,'prefix':prefix,'ranges':[[a,min(SIZE-1,a+length-1)] for a in range(prefix,SIZE,length)]}
   layout_path.write_text(json.dumps(layout,indent=2))
  jobs=[(a,b,CHUNKS/f'{a}_{b}.part') for a,b in layout['ranges']]
  started=time.monotonic()
  with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
   futures={pool.submit(read_range,*j):j for j in jobs}
   for f in concurrent.futures.as_completed(futures):
    print(json.dumps({'completed_range':list(futures[f][:2]),'result':f.result(),'elapsed':time.monotonic()-started}),flush=True)
  combined=OUT/'aisdk-2024-08-17.assembling'
  h=hashlib.sha256()
  with combined.open('wb') as dst:
   for path in ([PART] if prefix else [])+[j[2] for j in jobs]:
    with path.open('rb') as src:
     for data in iter(lambda:src.read(8*1024*1024),b''):dst.write(data);h.update(data)
  assert combined.stat().st_size==SIZE
  with zipfile.ZipFile(combined) as z:
   assert z.testzip() is None
   entries=[{'filename':i.filename,'uncompressed_bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':i.CRC} for i in z.infolist()]
  combined.rename(FINAL)
 else:
  h=hashlib.sha256()
  with FINAL.open('rb') as src:
   for data in iter(lambda:src.read(8*1024*1024),b''):h.update(data)
  with zipfile.ZipFile(FINAL) as z:
   assert z.testzip() is None
   entries=[{'filename':i.filename,'uncompressed_bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':i.CRC} for i in z.infolist()]
 report={'url':BASE,'path':str(FINAL),'bytes':SIZE,'sha256':h.hexdigest(),'zip_crc_passed':True,'entries':entries,'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source':'Danish Maritime Authority official historical public AIS archive'}
 FINAL.with_suffix('.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)

if __name__=='__main__':main()
