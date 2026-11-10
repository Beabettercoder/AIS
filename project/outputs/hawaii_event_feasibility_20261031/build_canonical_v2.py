from pathlib import Path
import pandas as pd,numpy as np,hashlib,json
out=Path('/mnt/sda/WangZijia/AIS/outputs/hawaii_event_feasibility_20261031')
p=out/'external_event_context.csv.gz';a=pd.read_csv(p,dtype={'MMSI':'string'})
a=a.rename(columns={'MMSI':'mmsi','speed_over_ground_knots':'sog','course_over_ground_deg':'cog','heading_deg':'heading'})
a['timestamp']=pd.to_datetime(a.pop('datetime_utc'),utc=True).dt.as_unit('ns').astype('int64')//10**9
a['mmsi']=pd.to_numeric(a.mmsi,errors='coerce');a=a.dropna(subset=['mmsi','timestamp','lat','lon','sog','cog']);a['mmsi']=a.mmsi.astype('int64');a=a.reset_index(drop=True)
f=out/'canonical_context_v2.parquet';assert not f.exists();a.to_parquet(f,index=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as x:
  for b in iter(lambda:x.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
json.dump({'input_gzip_sha256':sha(p),'canonical_sha256':sha(f),'rows':len(a),'mmsi_unique':int(a.mmsi.nunique()),'duplicates':int(a.duplicated(['mmsi','timestamp']).sum()),'core_bbox':[-159.0,19.0,-154.0,23.0],'core_bbox_note':'Coordinate projection origin only; no proven receiver coverage. This is not a user regional training definition.','time_range':[int(a.timestamp.min()),int(a.timestamp.max())]},(out/'canonical_context_v2.json').open('x'),indent=2)
print('DONE',len(a),a.mmsi.nunique(),int(a.duplicated(['mmsi','timestamp']).sum()),sha(f),flush=True)

