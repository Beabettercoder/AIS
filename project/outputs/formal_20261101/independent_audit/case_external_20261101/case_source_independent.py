from pathlib import Path
import csv,datetime,io,json,zipfile,hashlib
import pandas as pd,numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs');D=R/'formal_20261101/cases/v6';O=R/'formal_20261101/independent_audit/case_external_20261101'
haw=R/'external_event_evidence_20261031/hawaii_case2/matson_navigator_20170113_utc.csv';can=R/'hawaii_event_feasibility_20261031/canonical_context_v2.parquet';raw=R/'external_event_evidence_20261031/sara_regina_20240716_1200_2200_utc_noaa_raw.csv';zipp=Path('/mnt/sda/WangZijia/AIS/data/raw/noaa/AIS_2024_07_16.zip')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ts(s):return int(datetime.datetime.fromisoformat(str(s).replace('Z','+00:00')).replace(tzinfo=datetime.timezone.utc).timestamp())
z=zipfile.ZipFile(zipp);names=z.namelist();assert names==['AIS_2024_07_16.csv'];ex=[]
with z.open(names[0]) as f:
 for q in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline='')):
  if q['MMSI']!='334979000':continue
  t=ts(q['BaseDateTime']);
  if ts('2024-07-16T12:00:00+00:00')<=t<=ts('2024-07-16T22:00:00+00:00'):ex.append(q)
print('NOAA_ROWS',len(ex),flush=True)
saved=list(csv.DictReader(raw.open(newline='',encoding='utf8')));assert len(ex)==len(saved)==406
rawdict={ts(q['BaseDateTime']):q for q in ex};assert len(rawdict)==len(ex)
for q in saved:
 r=rawdict[ts(q['utc_iso'])];assert r['MMSI']==q['MMSI']=='334979000' and r['VesselName'].strip().upper()=='SARA REGINA' and r['IMO'].strip().upper()=='IMO9142655'
 for k in ('LAT','LON','SOG','COG','Heading'):
  a,b=r[k],q[k]
  if a=='' or b=='':assert a==b
  else:assert float(a)==float(b),(k,a,b)
assert all(float(q['Heading'])==511 for q in ex)
h=pd.read_csv(haw,dtype={'MMSI':str}); assert h.MMSI.eq('366792000').all() and h.vessel_name.str.strip().str.upper().eq('MATSON NAVIGATOR').all()
h['timestamp']=[ts(x) for x in h.datetime_utc];h=h.drop_duplicates('timestamp',keep='last').set_index('timestamp'); c=pd.read_parquet(can);c=c[c.mmsi.astype(str)=='366792000'].copy();c['timestamp']=c.timestamp.astype(int);c=c.sort_values('timestamp').drop_duplicates('timestamp',keep='last');checked=0
for row in c.itertuples():
 if row.timestamp not in h.index:continue
 q=h.loc[row.timestamp]
 for key,other in [('lat','lat'),('lon','lon'),('sog','speed_over_ground_knots'),('cog','course_over_ground_deg')]:
  x,y=float(getattr(row,key)),float(q[other]);assert np.isclose(x,y,rtol=0,atol=1e-8,equal_nan=True),(row.timestamp,key,x,y)
 checked+=1
assert checked>0
for meta in ('realcase_input_audit.json','rolling_input_audit.json'):
 obj=json.loads((D/meta).read_text())['cases']
 for r in obj['MATSON']:
  assert all(ts(t) in h.index for t in r['times_utc'])
 for r in obj['SARA']:
  assert all(ts(t) in rawdict for t in r['times_utc'])
src=R/'external_event_evidence_20261031/hawaii_case2/case_summary.json';s=json.loads(src.read_text());assert s['mmsi']=='366792000' and s['report_time_utc']=='2017-01-13T18:00:00+00:00';assert s['event_points']==77 and s['sog_lt2_n']==74
data={'NOAA_sara_independent_source_records':len(ex),'NOAA_all_heading_missing':True,'NOAA_name':'SARA REGINA','NOAA_MMSI':'334979000','NOAA_IMO':'IMO9142655','NOAA_zip_sha256':sha(zipp),'SARA_export_sha256':sha(raw),'MATSON_target_records_checked_against_canonical':checked,'MATSON_event_utc':'2017-01-13T18:00:00+00:00','MATSON_source_target_export_sha256':sha(haw),'MATSON_event_table_original_report_not_independently_obtained':True,'MATSON_all_window_times_present_in_source':True,'SARA_all_window_times_present_in_source':True,'warnings':['SARA event time is not specified by the retrieved public report; cannot label an AIS point as the collision.','MATSON event reference is event-table report time, not per-message ground truth.','Heading is an available field, not proven sensor-independent; models here use COG, not AIS Heading.']}
p=O/'case_source_independent.json';assert not p.exists();p.write_text(json.dumps(data,indent=2),encoding='utf8');print(json.dumps(data,indent=2),flush=True)
