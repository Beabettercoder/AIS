"""Normalize existing read-only European AIS intakes; no production writes."""
from pathlib import Path
import pandas as pd
import numpy as np
import json, hashlib

ROOT=Path('/mnt/sda/WangZijia/AIS')
OUT=ROOT/'outputs/balanced_global_v2/east'
INTAKE=ROOT/'system3/current/storage/intake'
FILES={
 'denmark_existing':['276d8046-ce10-414e-9f5b-e8f4162133e3.bin','93ac07ae-8968-4ff2-9d2f-5760a00bfed4.bin'],
 'norway_existing':['f03b0d9c-b9fd-4866-9f9b-e90f3496ef03.bin','304b3a1f-74ac-405a-8ce9-a6bea83d61dd.bin','4fe7e5ff-a7fb-402a-9eb7-b1429babd903.bin','cb3d089c-c6c8-489b-82a3-1e9a97e819e1.bin']}
MAPS={
 'denmark_existing':{'timestamp_source':'time_raw','mmsi':'mmsi','longitude':'lon','latitude':'lat','sog_kn':'sog','cog_deg':'cog','heading_deg':'heading','navigational_status':'nav_status','mobile_type':'mobile_type'},
 'norway_existing':{'date_time_utc':'time_raw','mmsi':'mmsi','longitude':'lon','latitude':'lat','speed_over_ground':'sog','course_over_ground':'cog','true_heading':'heading','status':'nav_status','ais_class':'mobile_type'}}

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def capacity(df):
 """Provisional 60s greedy thinning; nonoverlapping source windows <=23min."""
 records=[]
 move=df[df.sog.between(2,60)&df.cog.between(0,360,inclusive='left')]
 for m,g in move.groupby('mmsi',sort=False):
  ts=g.timestamp.to_numpy(); selected=[]; last=-10**15
  for i,t in enumerate(ts):
   if t-last>=60: selected.append(i); last=t
  t=ts[selected]; n=0; i=0
  while i<len(t):
   end=np.searchsorted(t,t[i]+1380,side='right')
   if end-i>=8:
    n+=1; i=end
   else: i+=1
  if n: records.append({'mmsi':int(m),'nonoverlap_windows':n,'thinned_points':len(t)})
 return records

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=[]; summary={}
 for region,files in FILES.items():
  pieces=[]
  for name in files:
   p=INTAKE/name; cnt=0; kept=0
   for raw in pd.read_csv(p,usecols=list(MAPS[region]),chunksize=200000,low_memory=False):
    cnt+=len(raw); d=raw.rename(columns=MAPS[region])
    for c in ['mmsi','lon','lat','sog','cog','heading']: d[c]=pd.to_numeric(d[c],errors='coerce')
    tm=pd.to_datetime(d.pop('time_raw'),format='%d/%m/%Y %H:%M:%S' if region.startswith('denmark') else 'mixed',utc=True,errors='coerce')
    good=tm.notna()&d.mmsi.between(100000000,999999999)&d.lon.between(-180,180)&d.lat.between(-90,90)
    d=d.loc[good].copy(); d['timestamp']=(tm.loc[good].dt.as_unit('ns').astype('int64')//10**9).astype('int64'); d['mmsi']=d.mmsi.astype('int64')
    for c,top in [('sog',102.3),('cog',360),('heading',360)]: d.loc[~d[c].between(0,top,inclusive='left'),c]=np.nan
    for c in ['lon','lat','sog','cog','heading']: d[c]=d[c].astype('float64')
    d['source_file']=str(p); d['nav_status']=d.nav_status.astype('string'); d['mobile_type']=d.mobile_type.astype('string')
    d['moving']=d.sog.ge(2)&d.cog.notna(); kept+=len(d); pieces.append(d)
   entry={'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size,'input_rows':cnt,'valid_position_rows':kept}
   manifest.append(entry); print(entry,flush=True)
  d=pd.concat(pieces,ignore_index=True).sort_values(['mmsi','timestamp','source_file'],kind='stable')
  pre=len(d); d=d.drop_duplicates(['mmsi','timestamp'],keep='first').reset_index(drop=True)
  p=OUT/(region+'.parquet'); d.to_parquet(p,index=False)
  cap=capacity(d); pd.DataFrame(cap).to_csv(OUT/(region+'_capacity.csv'),index=False)
  summary[region]={'path':str(p),'sha256':digest(p),'rows':len(d),'duplicate_positions_removed':pre-len(d),'unique_mmsi':int(d.mmsi.nunique()),'start_utc':str(pd.to_datetime(d.timestamp.min(),unit='s',utc=True)),'end_utc':str(pd.to_datetime(d.timestamp.max(),unit='s',utc=True)),'bbox':[float(d.lon.min()),float(d.lat.min()),float(d.lon.max()),float(d.lat.max())],'qualified_vessels':len(cap),'nonoverlap_windows':sum(r['nonoverlap_windows'] for r in cap),'capped20_capacity':sum(min(20,r['nonoverlap_windows']) for r in cap)}
  print(json.dumps(summary[region]),flush=True)
  (OUT/'existing_summary.json').write_text(json.dumps(summary,indent=2))
  (OUT/'existing_raw_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
