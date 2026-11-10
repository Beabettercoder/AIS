import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,zipfile,hashlib,math
import pandas as pd
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS')
OUT=ROOT/'outputs/balanced_global_v2/west'
RAW=Path('/mnt/sda/WangZijia/houston_galveston_2024_07/data/raw/noaa_ais')
REGIONS={
 'houston': {'lat':[29.2,29.8],'lon':[-95.1,-94.6]},
 'ny':{'lat':[40.35,40.95],'lon':[-74.35,-73.65]},
 'la':{'lat':[33.6,34.05],'lon':[-118.55,-117.85]},
 'puget':{'lat':[47.5,48.5],'lon':[-124,-122.3]}}
for r in REGIONS.values():
 r['lat_buffer_deg']=8/60+0.02
 r['lon_buffer_deg']=8/60/math.cos(math.radians(max(abs(x) for x in r['lat'])))+0.02
 r['coverage_bbox']={'lat':[r['lat'][0]-r['lat_buffer_deg'],r['lat'][1]+r['lat_buffer_deg']],'lon':[r['lon'][0]-r['lon_buffer_deg'],r['lon'][1]+r['lon_buffer_deg']]}
items={k:[] for k in REGIONS}
manifest={'regions':REGIONS,'buffer_rule':{'context_radius_nm':8,'latitude_degrees':'8/60+0.02','longitude_degrees':'8/60/cos(max(abs(core_latitude)))+0.02','extra_margin_degrees':0.02,'target_core_unchanged':True},'source_days':[],'minimum_mmsi':100000000,'maximum_mmsi':999999999,'source_quality_rules':'valid ID/time/geography; finite SOG in [0,102.2]; COG in [0,360). All motion states retained. Invalid Heading set null.'}
for day in [15,16,17]:
 p=RAW/f'AIS_2024_07_{day:02d}.zip'
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 with zipfile.ZipFile(p) as z:
  members=[x for x in z.namelist() if x.lower().endswith('.csv')]
  assert len(members)==1,members
  entry=members[0];source=f'{p}!{entry}'
  counters={k:0 for k in REGIONS};n=0
  with z.open(entry) as f:
   for d in pd.read_csv(f,chunksize=400000,usecols=['MMSI','BaseDateTime','LAT','LON','SOG','COG','Heading','VesselType','Status'],low_memory=False):
    d.columns=['mmsi','datetime','lat','lon','sog','cog','heading','vessel_type','nav_status']
    # Input row number (CSV data row, zero based) gives exact record provenance.
    d['source_row']=np.arange(n,n+len(d),dtype=np.int64);n+=len(d)
    for region,r in REGIONS.items():
     v=d[d.lat.between(*r['coverage_bbox']['lat'])&d.lon.between(*r['coverage_bbox']['lon'])].copy()
     if not len(v):continue
     for c in ['mmsi','lat','lon','sog','cog','heading']:v[c]=pd.to_numeric(v[c],errors='coerce')
     v=v[v.mmsi.between(100000000,999999999)&v.sog.between(0,102.2)&v.cog.between(0,359.999999)]
     v['datetime']=pd.to_datetime(v.datetime,utc=True,errors='coerce')
     v=v.dropna(subset=['mmsi','datetime','lat','lon','sog','cog'])
     v['timestamp']=(v.datetime.astype('datetime64[ns, UTC]').astype('int64')//10**9).astype('int64')
     assert v.timestamp.between(1721001600,1721260799).all(), 'UTC epoch unit/range check'
     v['mmsi']=v.mmsi.astype('int64')
     v.loc[~v.heading.between(0,359.999999),'heading']=np.nan
     v['source_file']=source
     v['in_core_roi']=v.lat.between(*r['lat'])&v.lon.between(*r['lon'])
     v=v.drop(columns='datetime')
     items[region].append(v);counters[region]+=len(v)
  info=z.getinfo(entry)
  manifest['source_days'].append({'path':str(p),'sha256':h.hexdigest(),'zip_entry':entry,'zip_crc32':info.CRC,'csv_total_rows':n,'retained_rows':counters})
  print(day,json.dumps(counters),flush=True)
for region,ls in items.items():
 d=pd.concat(ls,ignore_index=True).sort_values(['mmsi','timestamp','source_file','source_row'],kind='stable')
 n=len(d);d=d.drop_duplicates(['mmsi','timestamp']).reset_index(drop=True)
 path=OUT/f'{region}_raw_standard.parquet';d.to_parquet(path,index=False)
 manifest['regions'][region].update({'rows':len(d),'dropped_same_ship_time_rows':n-len(d),'ships':int(d.mmsi.nunique()),'core_rows':int(d.in_core_roi.sum()),'timestamp_min':int(d.timestamp.min()),'timestamp_max':int(d.timestamp.max()),'standard_path':str(path)})
 print(region,len(d),d.mmsi.nunique(),flush=True)
(OUT/'raw_extraction_manifest.json').write_text(json.dumps(manifest,indent=2))
