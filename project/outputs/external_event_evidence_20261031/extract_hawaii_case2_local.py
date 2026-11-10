import io,zipfile,urllib.request,json,struct,zlib,hashlib,time,concurrent.futures,csv,datetime,math,bisect,collections,os,runpy,contextlib
from pathlib import Path
base=Path(r'E:\AIS\outputs\external_event_evidence_20261031');out=base/'hawaii_case2';out.mkdir(exist_ok=True)
u='https://zenodo.org/api/records/8253611/files/HawaiiCoast_GT.zip/content'; size=2962047276
class R(io.RawIOBase):
 def __init__(self):self.pos=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,x,w=0):self.pos=x if w==0 else (self.pos+x if w==1 else size+x);return self.pos
 def read(self,n=-1):
  if n<0:n=size-self.pos
  if n<=0:return b''
  end=min(self.pos+n,size)-1
  with urllib.request.urlopen(urllib.request.Request(u,headers={'Range':f'bytes={self.pos}-{end}'}),timeout=45) as f:
   if f.status!=206:raise RuntimeError('range unsupported')
   d=f.read()
  self.pos+=len(d);return d
with zipfile.ZipFile(R()) as z:
 info=z.getinfo('HawaiiCoast_GT/AIS_data/Hawaii_2017_01.csv');print('MEMBER',info.file_size,info.compress_size,hex(info.CRC),flush=True)
 rr=R();rr.seek(info.header_offset);header=rr.read(30);v=struct.unpack('<IHHHHHIIIHH',header);assert v[0]==0x04034b50
 begin=info.header_offset+30+v[-2]+v[-1];cs=info.compress_size;crc=info.CRC;us=info.file_size
 assert cs==64592497 and us==380885602
chunk=4*1024*1024;parts=[]
for i,at in enumerate(range(0,cs,chunk)):parts.append((i,begin+at,begin+min(at+chunk,cs)-1))
def pull(t):
 i,lo,hi=t;p=out/f'part_{i:03d}.bin';expected=hi-lo+1
 if p.exists() and p.stat().st_size==expected:return i,p
 for k in range(4):
  try:
   req=urllib.request.Request(u,headers={'Range':f'bytes={lo}-{hi}','User-Agent':'research-audit/1.0'})
   with urllib.request.urlopen(req,timeout=90) as r:
    if r.status!=206:raise RuntimeError('unexpected status')
    data=r.read()
   if len(data)!=expected:raise RuntimeError('wrong range length')
   p.write_bytes(data);return i,p
  except Exception:
   if k==3:raise
   time.sleep(3*(k+1))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
 for i,p in ex.map(pull,parts):print('PART',i,p.stat().st_size,flush=True)
pfile=out/'Hawaii_2017_01.csv';cc=0;length=0;dc=zlib.decompressobj(-15);sh=hashlib.sha256()
with pfile.open('wb') as f:
 for i,_,_ in parts:
  with (out/f'part_{i:03d}.bin').open('rb') as q:
   for b in iter(lambda:q.read(1024*1024),b''):
    x=dc.decompress(b);f.write(x);cc=zlib.crc32(x,cc);length+=len(x);sh.update(x)
 x=dc.flush();f.write(x);cc=zlib.crc32(x,cc);length+=len(x);sh.update(x)
assert dc.eof and length==us and cc==crc,(dc.eof,length,hex(cc),hex(crc))
print('CRC_VALID',length,hex(cc),flush=True)
start=datetime.datetime.fromisoformat('2017-01-13T14:00:00+00:00');end=datetime.datetime.fromisoformat('2017-01-13T22:00:00+00:00'); lo=datetime.datetime.fromisoformat('2017-01-13T16:01:56+00:00');hi=datetime.datetime.fromisoformat('2017-01-13T19:58:54+00:00')
target=[];context=[];nrows=0
with pfile.open(encoding='utf-8-sig',newline='') as f:
 rd=csv.DictReader(f);fields=rd.fieldnames
 for row in rd:
  nrows+=1;ds=row['datetime_utc'];
  if not ds.startswith('2017-01-13'):continue
  try:t=datetime.datetime.fromisoformat(ds)
  except ValueError:continue
  if t.tzinfo is None:t=t.replace(tzinfo=datetime.timezone.utc)
  if row['MMSI']=='366792000':target.append(row)
  if start<=t<=end:
   try:la=float(row['lat']);lo_=float(row['lon'])
   except ValueError:continue
   if 20.5<=la<=21.8 and -158.8<=lo_<=-157.2:context.append(row)
print('FILTERED',nrows,len(target),len(context),flush=True)
def write_csv(name,rows,fields):
 p=out/name
 with p.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 return p
assert target and context
target.sort(key=lambda x:x['datetime_utc']);context.sort(key=lambda x:(x['MMSI'],x['datetime_utc']))
tp=write_csv('matson_navigator_20170113_utc.csv',target,fields);cp=write_csv('oahu_20170113_1400_2200_utc.csv',context,fields)
event=[x for x in target if lo<=datetime.datetime.fromisoformat(x['datetime_utc'])<=hi]
def num(x):
 try:return float(x)
 except (ValueError,TypeError):return float('nan')
groups={}
for row in context:
 mid=row['MMSI']
 if mid=='366792000':continue
 if mid not in groups:groups[mid]=([],[])
 groups[mid][0].append(datetime.datetime.fromisoformat(row['datetime_utc']).timestamp());groups[mid][1].append(row)
counts=[]
for row in event:
 t=datetime.datetime.fromisoformat(row['datetime_utc']).timestamp();la=num(row['lat']);lon=num(row['lon']);cog=num(row['course_over_ground_deg']);n=co=0
 for mid,(ts,rows) in groups.items():
  j=bisect.bisect_right(ts,t)-1
  if j<0 or t-ts[j]>300:continue
  r=rows[j];la2=num(r['lat']);lo2=num(r['lon']);d=math.hypot((la2-la)*111195,(lo2-lon)*111195*math.cos(math.radians(la)))
  if not math.isfinite(d) or d>3.5*1852:continue
  n+=1;c=num(r['course_over_ground_deg'])
  if math.isfinite(c) and math.isfinite(cog) and 0<=c<360 and 0<=cog<360 and abs((c-cog+180)%360-180)<45:co+=1
 h=num(row['heading_deg']); counts.append({'datetime_utc':row['datetime_utc'],'lat':row['lat'],'lon':row['lon'],'sog_kn':row['speed_over_ground_knots'],'cog_deg':row['course_over_ground_deg'],'heading_deg':row['heading_deg'],'heading_valid':int(math.isfinite(h) and 0<=h<360),'neighbors_3_5nm_300s':n,'same_direction_45deg':co})
np=write_csv('event_target_neighbor_counts.csv',counts,list(counts[0].keys()))
def filehash(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()
def percentile(s,q):
 s=sorted(s);p=(len(s)-1)*q;i=int(p);a=p-i;return s[i]*(1-a)+s[min(i+1,len(s)-1)]*a
etime=[datetime.datetime.fromisoformat(x['datetime_utc']).timestamp() for x in event]; gaps=[b-a for a,b in zip(etime,etime[1:])]; speeds=[num(x['speed_over_ground_knots']) for x in event];speeds=[x for x in speeds if math.isfinite(x)];assert speeds and event
j={'dataset':'HawaiiCoast_GT v1.0, CC BY 4.0','dataset_doi':'https://doi.org/10.5281/zenodo.8253611','dataset_url':u,'whole_archive_md5_from_zenodo':'e7dd0951d1512357af7ca50cad710ebe','member':'HawaiiCoast_GT/AIS_data/Hawaii_2017_01.csv','compressed_size':cs,'uncompressed_size':us,'member_crc32_expected':f'{crc:08x}','member_crc32_actual':f'{cc:08x}','uncompressed_sha256':sh.hexdigest(),'crc_verified':True,'member_rows':nrows,'incident_number':2,'mmsi':'366792000','event_name':'MATSON NAVIGATOR','source':'USCG IIR via Sandia event list, IIR.2017.01.13 / 6075786','incident_csv_source':'https://github.com/sandialabs/HawaiiCoast_GT_Code_Generation/blob/main/hawaii_primary_trajectories_of_interest_2017_2020.csv','report_time_hst':'2017-01-13T08:00:00-10:00','report_time_utc':'2017-01-13T18:00:00+00:00','ais_bound_utc':[lo.isoformat(),hi.isoformat()],'event_points':len(event),'target_first_utc':event[0]['datetime_utc'],'target_last_utc':event[-1]['datetime_utc'],'median_sog_kn':percentile(speeds,.5),'sog_lt2_n':sum(x<2 for x in speeds),'heading_valid_n':sum(x['heading_valid'] for x in counts),'heading_valid_fraction':sum(x['heading_valid'] for x in counts)/len(counts),'gap_seconds_median':percentile(gaps,.5),'gap_seconds_p90':percentile(gaps,.9),'gap_seconds_max':max(gaps),'coarse_context_window_utc':[start.isoformat(),end.isoformat()],'context_points':len(context),'context_vessels':len(groups),'target_steps_any_neighbor':sum(x['neighbors_3_5nm_300s']>=1 for x in counts),'target_steps_4neighbors':sum(x['neighbors_3_5nm_300s']>=4 for x in counts),'target_steps_4same_direction':sum(x['same_direction_45deg']>=4 for x in counts),'artifacts_sha256':{p.name:filehash(p) for p in [tp,cp,np]},'limitations':['Incidents are report-linked intervals, not labels for each point or measurements of true sensor contamination.','Context counts use 3.5nm and most recent ≤300s, same-direction <45°, but not quality gate, duplicates/condition number or trained model.','A 0.918-style score cannot be inferred from this one event; unlabeled other vessels not verified normal.','Heading numeric validity does not establish independent sensor origin or physical correctness.','Only one month selected for evidence exploration; no model tuning or testing.']}
(out/'case_summary.json').write_text(json.dumps(j,indent=2),encoding='utf8');print(json.dumps(j,indent=2),flush=True)
