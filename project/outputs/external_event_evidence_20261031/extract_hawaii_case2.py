from pathlib import Path
import io,csv,zipfile,urllib.request,time,datetime,json,hashlib,math,collections
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
out=Path('/mnt/sda/WangZijia/AIS/outputs/external_event_evidence_20261031/hawaii_case2');out.mkdir(parents=True,exist_ok=True)
url='https://zenodo.org/api/records/8253611/files/HawaiiCoast_GT.zip/content';size=2962047276; block=4*1024*1024
class Remote(io.RawIOBase):
 def __init__(self):self.pos=0;self.cache={};self.downloaded=0;self.calls=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,x,whence=0):self.pos=x if whence==0 else (self.pos+x if whence==1 else size+x);return self.pos
 def read(self,n=-1):
  if n<0:n=size-self.pos
  remain=min(n,size-self.pos);parts=[]
  while remain>0:
   start=(self.pos//block)*block
   if start not in self.cache:
    last=min(start+block,size)-1
    for attempt in range(4):
     try:
      with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={start}-{last}'}),timeout=90) as r:
       if r.status!=206:raise ValueError('range not honored')
       data=r.read()
      if len(data)!=last-start+1:raise ValueError('incorrect range length')
      self.cache={start:data};self.downloaded+=len(data);self.calls+=1;break
     except Exception:
      if attempt==3:raise
      time.sleep(2*(attempt+1))
   data=self.cache[start];at=self.pos-start;take=min(remain,len(data)-at);parts.append(data[at:at+take]);remain-=take;self.pos+=take
  return b''.join(parts)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
r=Remote();target=[];context=[];total=0
start='2017-01-13 14:00:00+00:00';end='2017-01-13 22:00:00+00:00'
with zipfile.ZipFile(r) as z:
 name='HawaiiCoast_GT/AIS_data/Hawaii_2017_01.csv';info=z.getinfo(name)
 assert info.file_size==380885602 and info.compress_size==64592497
 with z.open(info) as b:
  rd=csv.DictReader(io.TextIOWrapper(b,encoding='utf-8-sig',newline=''))
  for row in rd:
   total+=1;t=row['datetime_utc']; mid=row['MMSI']
   if mid=='366792000' and t.startswith('2017-01-13'):target.append(row)
   if start<=t<=end:
    try:lat=float(row['lat']);lon=float(row['lon'])
    except ValueError:continue
    if 20.5<=lat<=21.8 and -158.8<=lon<=-157.2:context.append(row)
 print('CRC verified by full entry read',flush=True)
target=pd.DataFrame(target).sort_values('datetime_utc'); context=pd.DataFrame(context).sort_values(['MMSI','datetime_utc'])
target.to_csv(out/'matson_navigator_20170113_utc.csv',index=False)
context.to_csv(out/'oahu_20170113_1400_2200_utc.csv',index=False)
assert len(target)>0 and len(context)>0
# Evidence bounds are labels of correlated interval, not point-by-point actions.
lo=pd.Timestamp('2017-01-13 16:01:56+00:00');hi=pd.Timestamp('2017-01-13 19:58:54+00:00');tt=pd.to_datetime(target.datetime_utc,utc=True)
event=target[(tt>=lo)&(tt<=hi)].copy(); event['t']=pd.to_datetime(event.datetime_utc,utc=True)
for c in ['lat','lon','speed_over_ground_knots','course_over_ground_deg','heading_deg']:
 event[c]=pd.to_numeric(event[c],errors='coerce');context[c]=pd.to_numeric(context[c],errors='coerce')
context['t']=pd.to_datetime(context.datetime_utc,utc=True);groups=[]
for mid,d in context.groupby('MMSI'):
 if mid=='366792000':continue
 d=d.sort_values('t');groups.append((mid,d,d.t.astype('int64').to_numpy()))
counts=[]
for row in event.itertuples():
 t=row.t.value; n=0; codir=0
 for mid,d,ns in groups:
  k=np.searchsorted(ns,t,side='right')-1
  if k<0 or t-ns[k]>300*10**9:continue
  v=d.iloc[k];lat2=v['lat'];lon2=v['lon']
  if not np.isfinite(lat2) or not np.isfinite(lon2):continue
  dist=math.hypot((lat2-row.lat)*111195,(lon2-row.lon)*111195*math.cos(math.radians(row.lat)))
  if dist>3.5*1852:continue
  n+=1;c=v['course_over_ground_deg'];a=row.course_over_ground_deg
  if np.isfinite(c) and np.isfinite(a) and 0<=c<360 and 0<=a<360 and abs((c-a+180)%360-180)<45:codir+=1
 counts.append({'datetime_utc':str(row.t),'lat':row.lat,'lon':row.lon,'sog_kn':row.speed_over_ground_knots,'cog_deg':row.course_over_ground_deg,'heading_deg':row.heading_deg,'valid_heading':bool(np.isfinite(row.heading_deg) and 0<=row.heading_deg<360),'neighbors_3_5nm_300s':n,'same_direction_45deg':codir})
pairs=pd.DataFrame(counts);pairs.to_csv(out/'event_target_neighbor_counts.csv',index=False)
splits=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/release_20261026/global_all_mmsi_split.parquet');sp=pq.read_table(splits).to_pandas(); cols={x.lower():x for x in sp.columns}; match=sp[sp[cols.get('mmsi','mmsi')].astype(str)=='366792000']; split_summary=match.to_dict('records'); split_summary=[{str(k):str(v) for k,v in row.items()} for row in split_summary]
summary={'dataset':'HawaiiCoast_GT v1.0 CC-BY-4.0','dataset_source':url,'dataset_doi':'https://doi.org/10.5281/zenodo.8253611','zenodo_complete_zip_md5':'e7dd0951d1512357af7ca50cad710ebe','member':name,'member_uncompressed_bytes':info.file_size,'member_compressed_bytes':info.compress_size,'member_crc32_hex':f'{info.CRC:08x}','member_crc_validation':'Full ZipExtFile streamed to EOF; Python zipfile CRC check passed','http_bytes_downloaded':r.downloaded,'http_requests':r.calls,'total_csv_rows':total,'official_incident_csv':'https://github.com/sandialabs/HawaiiCoast_GT_Code_Generation/blob/main/hawaii_primary_trajectories_of_interest_2017_2020.csv','incident_id':2,'mmsi':'366792000','vessel_name':'MATSON NAVIGATOR','event_type':'Loss/reduction of propulsion','evidence_grade':'High evidence','reported_event_time_hst':'2017-01-13T08:00:00-10:00','reported_event_time_utc':'2017-01-13T18:00:00+00:00','ais_associated_interval_hst':['2017-01-13T06:01:56-10:00','2017-01-13T09:58:54-10:00'],'ais_associated_interval_utc':[lo.isoformat(),hi.isoformat()],'event_target_raw_points':len(event),'target_first_event_utc':str(event.t.min()),'target_last_event_utc':str(event.t.max()),'event_target_median_sog_kn':float(event.speed_over_ground_knots.median()),'event_target_sog_lt2_n':int((event.speed_over_ground_knots<2).sum()),'event_target_heading_valid_n':int(pairs.valid_heading.sum()),'event_target_heading_valid_fraction':float(pairs.valid_heading.mean()),'gap_seconds_median':float(event.t.diff().dt.total_seconds().median()),'gap_seconds_p90':float(event.t.diff().dt.total_seconds().quantile(.9)),'gap_seconds_max':float(event.t.diff().dt.total_seconds().max()),'oahu_context_points':len(context),'oahu_context_unique_mmsi':int(context.MMSI.nunique()),'target_event_points_with_at_least_4_neighbors_3_5nm_300s':int((pairs.neighbors_3_5nm_300s>=4).sum()),'target_event_points_with_at_least_4_same_dir_neighbors':int((pairs.same_direction_45deg>=4).sum()),'target_event_points_any_neighbor':int((pairs.neighbors_3_5nm_300s>=1).sum()),'existing_project_mmsi_split_match':split_summary,'files_sha256':{p.name:sha(p) for p in [out/'matson_navigator_20170113_utc.csv',out/'oahu_20170113_1400_2200_utc.csv',out/'event_target_neighbor_counts.csv']},'important':['MMSI+official incident time is an event-level association, not a claim that every point is anomalous or that cause is sensor pollution.','Neighborhood counts are coarse geometry/temporal feasibility only; no AIS reliability/affine geometry/model training.','Background samples include vessel MMSIs with no complete event labels; do not treat unlabelled as known negative.','This is an external 2017 dataset distinct from existing 2023/24 windows.','Heading validity is counted numerically; independence and true quality are not guaranteed.']}
(out/'case_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
