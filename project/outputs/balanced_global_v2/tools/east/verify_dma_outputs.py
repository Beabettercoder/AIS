from pathlib import Path
import json,zipfile,hashlib
import pandas as pd
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS');OUT=ROOT/'outputs/balanced_global_v2/east';RAW=ROOT/'data/raw/external/dma_2024'
metadata=json.loads((OUT/'dma_capacity_verified.json').read_text());report={};samples=[]
for region,meta in metadata.items():
 d=pd.read_parquet(meta['file']);c=pd.read_parquet(OUT/f'{region}_capacity_candidates.parquet')
 assert not d.duplicated(['mmsi','timestamp']).any()
 assert d[['mmsi','timestamp']].equals(d.sort_values(['mmsi','timestamp'])[['mmsi','timestamp']])
 assert d.lat.between(-90,90).all() and d.lon.between(-180,180).all()
 prior={};n=0
 arrays={key:d[key].to_numpy() for key in ['mmsi','timestamp','sog','lat','lon']}
 for r in c.itertuples():
  ix=np.array(json.loads(r.row_indices),dtype=np.int64);tt=json.loads(r.timestamps)
  assert np.all(arrays['mmsi'][ix]==r.mmsi) and arrays['timestamp'][ix].tolist()==tt
  assert 8<=len(ix)<=20 and np.diff(tt).min()>=60 and tt[-1]-tt[0]>=660 and tt[-1]-tt[0]<1380
  assert (arrays['sog'][ix]>=2).sum()>=4 and arrays['sog'][ix].max()>=3
  x0,y0,x1,y1=meta['core_bbox'];j=ix[-1]
  assert x0<=arrays['lon'][j]<=x1 and y0<=arrays['lat'][j]<=y1
  assert tt[0]>prior.get(r.mmsi,-np.inf);prior[r.mmsi]=tt[-1];n+=len(ix)
 src=d[d.source_file.str.contains('aisdk-2024-08-17.zip!',regex=False)]
 assert src.source_row.notna().all() and src.source_row.between(0,19250382).all()
 take=src.sample(min(300,len(src)),random_state=20261026).copy();take['region']=region;samples.append(take)
 report[region]={'candidate_windows_verified':len(c),'selected_observations_verified':n,'exact_source_row_pointers_present_new_date':True,'unique_mmsi_time':True,'sorted_mmsi_time':True,'nonoverlap_windows':True,'static_rows_sog_below_05_retained':int((d.sog<.5).sum()),'canonical_sha256':hashlib.sha256(Path(meta['file']).read_bytes()).hexdigest()}
 print(region,json.dumps(report[region]),flush=True)
wanted=pd.concat(samples,ignore_index=True).set_index('source_row');found=[]
archive=RAW/'aisdk-2024-08-17.zip'
mapping={'# Timestamp':'time_raw','MMSI':'mmsi','Latitude':'lat','Longitude':'lon','SOG':'sog','COG':'cog','Heading':'heading'}
with zipfile.ZipFile(archive) as z:
 with z.open('aisdk-2024-08-17.csv') as f:
  for chunk in pd.read_csv(f,usecols=list(mapping),chunksize=500000,low_memory=False):
   hit=chunk.index.intersection(wanted.index)
   if len(hit):found.append(chunk.loc[hit].rename(columns=mapping))
raw=pd.concat(found).sort_index()
raw['timestamp']=pd.to_datetime(raw.time_raw,format='%d/%m/%Y %H:%M:%S',utc=True).dt.as_unit('ns').astype('int64')//10**9
for key,top in [('sog',102.3),('cog',360),('heading',360)]:raw.loc[~raw[key].between(0,top,inclusive='left'),key]=np.nan
for i,row in wanted.iterrows():
 r=raw.loc[i]
 for key in ['mmsi','timestamp','lat','lon','sog','cog','heading']:
  assert (pd.isna(row[key]) and pd.isna(r[key])) or row[key]==r[key],(i,key,row[key],r[key])
out={'passed':True,'regions':report,'source_samples_checked':len(wanted),'source_sample_fields':['mmsi','timestamp','lat','lon','sog','cog','heading'],'source_samples_exact_match':True,'sample_seed':20261026,'note':'All candidate source indices and time sequences audited. Independent CSV field re-extraction checked 300 deterministic random new-date observations per region.'}
(OUT/'dma_output_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
