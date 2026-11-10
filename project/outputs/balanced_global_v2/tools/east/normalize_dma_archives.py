"""Stream complete official daily ZIPs into buffered regional Parquet files.

Incomplete .part files are deliberately not accepted as complete day sources.
"""
from pathlib import Path
import json, zipfile, hashlib, math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path('/mnt/sda/WangZijia/AIS'); OUT=ROOT/'outputs/balanced_global_v2/east'
RAW=ROOT/'data/raw/external/dma_2024'
CONFIG=json.loads((OUT/'source_protocol.json').read_text())['expanded_danish_candidates']
MAP={'# Timestamp':'time_raw','Type of mobile':'mobile_type','MMSI':'mmsi','Latitude':'lat','Longitude':'lon','Navigational status':'nav_status','SOG':'sog','COG':'cog','Heading':'heading'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 report=[]
 for archive in sorted(RAW.glob('aisdk-*.zip')):
  day=archive.stem[-10:]; writers={}; counts={k:0 for k in CONFIG}; rows=0; valid=0
  try:
   with zipfile.ZipFile(archive) as z:
    members=[x for x in z.namelist() if x.lower().endswith('.csv')]
    assert len(members)==1,members
    member=members[0]
    with z.open(member) as f:
     for raw in pd.read_csv(f,usecols=list(MAP),chunksize=250000,low_memory=False):
      row_base=rows; rows+=len(raw); d=raw.rename(columns=MAP)
      d['source_row']=np.arange(row_base,rows,dtype=np.int64)
      for c in ['mmsi','lon','lat','sog','cog','heading']: d[c]=pd.to_numeric(d[c],errors='coerce')
      tm=pd.to_datetime(d.pop('time_raw'),format='%d/%m/%Y %H:%M:%S',errors='coerce',utc=True)
      good=tm.notna()&d.mmsi.between(100000000,999999999)&d.lon.between(-180,180)&d.lat.between(-90,90)
      d=d.loc[good].copy(); valid+=len(d)
      d['timestamp']=tm.loc[good].dt.as_unit('ns').astype('int64')//10**9; d['mmsi']=d.mmsi.astype('int64')
      for c,top in [('sog',102.3),('cog',360),('heading',360)]: d.loc[~d[c].between(0,top,inclusive='left'),c]=np.nan
      for c in ['lon','lat','sog','cog','heading']: d[c]=d[c].astype('float64')
      d['nav_status']=d.nav_status.astype('string'); d['mobile_type']=d.mobile_type.astype('string'); d['source_file']=str(archive)+'!'+member; d['moving']=d.sog.ge(2)&d.cog.notna()
      for region,cfg in CONFIG.items():
       x0,y0,x1,y1=cfg['core_bbox_wgs84']
       by=8/60+0.02; bx=8/60/math.cos(math.radians(max(abs(y0),abs(y1))+by))+0.02
       s=d.loc[d.lon.between(x0-bx,x1+bx)&d.lat.between(y0-by,y1+by)].copy()
       if not len(s): continue
       table=pa.Table.from_pandas(s,preserve_index=False); dest=OUT/f'{region}_{day}.parquet'
       if region not in writers: writers[region]=pq.ParquetWriter(dest,table.schema,compression='snappy')
       writers[region].write_table(table); counts[region]+=len(s)
      if rows%5000000==0: print(f'{archive.name}: {rows} input rows',flush=True)
  finally:
   for w in writers.values(): w.close()
  report.append({'raw_path':str(archive),'raw_sha256':sha(archive),'input_rows':rows,'valid_position_rows':valid,'regional_rows':counts,'completed_csv_crc_check':True})
  print(json.dumps(report[-1]),flush=True)
 for region in CONFIG:
  paths=sorted(OUT.glob(region+'_2024-*.parquet'))
  if region=='great_belt': paths.insert(0,OUT/'denmark_existing.parquet')
  if not paths: continue
  d=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True).sort_values(['mmsi','timestamp','source_file'],kind='stable')
  before=len(d); d=d.drop_duplicates(['mmsi','timestamp']); dest=OUT/(region+'_expanded.parquet'); d.to_parquet(dest,index=False)
  x0,y0,x1,y1=CONFIG[region]['core_bbox_wgs84']; by=8/60+0.02; bx=8/60/math.cos(math.radians(max(abs(y0),abs(y1))+by))+0.02
  days=sorted(pd.to_datetime(d.timestamp,unit='s',utc=True).dt.strftime('%Y-%m-%d').unique().tolist())
  coverage={day:([x0,y0,x1,y1] if region=='great_belt' and day in ['2024-08-15','2024-08-16'] else [x0-bx,y0-by,x1+bx,y1+by]) for day in days}
  report.append({'region':region,'path':str(dest),'sha256':sha(dest),'rows':len(d),'duplicates_removed':before-len(d),'unique_mmsi':int(d.mmsi.nunique()),'days':days,'core_bbox':CONFIG[region]['core_bbox_wgs84'],'latitude_buffer_degrees':by,'longitude_buffer_degrees':bx,'coverage_bbox_by_utc_day':coverage,'existing_days_without_buffer':region=='great_belt'})
  print(json.dumps(report[-1]),flush=True)
 (OUT/'dma_normalization.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
