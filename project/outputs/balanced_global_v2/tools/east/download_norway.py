"""Anonymous read-only historical AIS request and UTC-filtered normalization."""
from pathlib import Path
import urllib.request, json, gzip, hashlib, time
import pandas as pd
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS')
RAW=ROOT/'data/raw/external/kystverket_2024'; RAW.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'outputs/balanced_global_v2/east'
URL='https://kystdatahuset.kystverket.no/ws/api/ais/positions/within-bbox-time'
query={'bbox':'4.65,58.95,5.65,59.65','start':'202401040000','end':'202401042359','minSpeed':0}
(RAW/'20240104_request.json').write_text(json.dumps(query,indent=2))
target=RAW/'20240104_response.json.gz'; part=target.with_suffix('.part')
started=time.time()
if not target.exists():
 req=urllib.request.Request(URL,data=json.dumps(query).encode(),headers={'Content-Type':'application/json','Accept-Encoding':'gzip'})
 with urllib.request.urlopen(req,timeout=90) as r:
  headers=dict(r.headers)
  if headers.get('Content-Encoding')!='gzip' and headers.get('content-encoding')!='gzip': raise RuntimeError('Expected gzip response')
  (RAW/'20240104_headers.json').write_text(json.dumps(headers,indent=2))
  size=0
  with part.open('wb') as f:
   while True:
    b=r.read(65536)
    if not b: break
    f.write(b); size+=len(b)
    if size>300*1024**2 or time.time()-started>2400: raise RuntimeError('bounded acquisition limit reached; incomplete .part preserved')
 part.rename(target)
obj=json.loads(gzip.decompress(target.read_bytes()))
assert obj.get('success'),obj.get('msg')
cols=['mmsi','time_raw','lon','lat','cog','sog','ais_message','calc_speed','seconds_previous','distance_previous','heading','rot']
d=pd.DataFrame(obj['data'],columns=cols); raw_n=len(d)
t=pd.to_datetime(d.pop('time_raw'),utc=True,errors='coerce')
good=(t>=pd.Timestamp('2024-01-04',tz='UTC'))&(t<pd.Timestamp('2024-01-05',tz='UTC'))
d=d.loc[good].copy(); d['timestamp']=t.loc[good].dt.as_unit('ns').astype('int64')//10**9
for c in ['mmsi','lon','lat','sog','cog','heading']: d[c]=pd.to_numeric(d[c],errors='coerce')
d=d.loc[d.mmsi.between(100000000,999999999)&d.lon.between(4.65,5.65)&d.lat.between(58.95,59.65)].copy()
d['mmsi']=d.mmsi.astype('int64'); d['timestamp']=d.timestamp.astype('int64')
for c,top in [('sog',102.3),('cog',360),('heading',360)]: d.loc[~d[c].between(0,top,inclusive='left'),c]=np.nan
for c in ['lon','lat','sog','cog','heading']: d[c]=d[c].astype('float64')
d['source_file']=str(target); d['nav_status']=pd.Series(pd.NA,index=d.index,dtype='string'); d['mobile_type']=pd.Series(pd.NA,index=d.index,dtype='string'); d['moving']=d.sog.ge(2)&d.cog.notna()
keep=['mmsi','timestamp','lon','lat','sog','cog','heading','source_file','nav_status','mobile_type','moving']
d=d[keep].sort_values(['mmsi','timestamp']).drop_duplicates(['mmsi','timestamp'])
d.to_parquet(OUT/'norway_20240104.parquet',index=False)
combined=pd.concat([pd.read_parquet(OUT/'norway_existing.parquet'),d],ignore_index=True).sort_values(['mmsi','timestamp','source_file'],kind='stable').drop_duplicates(['mmsi','timestamp'])
combined.to_parquet(OUT/'norway_expanded.parquet',index=False)
result={'url':URL,'query':query,'raw_path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'raw_rows':raw_n,'filtered_rows':len(d),'filtered_unique_mmsi':int(d.mmsi.nunique()),'combined_rows':len(combined),'combined_unique_mmsi':int(combined.mmsi.nunique()),'license':'NLOD 1.0 as stated by API; attribution Kystverket','runtime_seconds':time.time()-started}
(OUT/'norway_api_acquisition.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
