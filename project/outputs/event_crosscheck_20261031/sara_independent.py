import csv,io,zipfile,datetime,math,json,hashlib
from pathlib import Path
Z=Path('/mnt/sda/WangZijia/AIS/data/raw/noaa/AIS_2024_07_16.zip')
D=Path('/mnt/sda/WangZijia/AIS/outputs/external_event_evidence_20261031')
O=Path('/mnt/sda/WangZijia/AIS/outputs/event_crosscheck_20261031')
bridge=(25.76992,-80.18997)
def dist(lat,lon):
 a,b=map(math.radians,bridge); c,d=map(math.radians,(lat,lon)); x=math.sin((c-a)/2)**2+math.cos(a)*math.cos(c)*math.sin((d-b)/2)**2; return 12742000*math.asin(min(1,math.sqrt(x)))
records=[]
with zipfile.ZipFile(Z) as z:
 assert z.namelist()==['AIS_2024_07_16.csv'],z.namelist()
 with io.TextIOWrapper(z.open(z.namelist()[0]),encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r['MMSI']!='334979000':continue
   timestamp=datetime.datetime.fromisoformat(r['BaseDateTime']).replace(tzinfo=datetime.timezone.utc)
   if not datetime.datetime(2024,7,16,12,tzinfo=datetime.timezone.utc)<=timestamp<datetime.datetime(2024,7,16,22,tzinfo=datetime.timezone.utc):continue
   try: lat,lon=float(r['LAT']),float(r['LON']); sog=float(r['SOG'])
   except (ValueError,TypeError):continue
   records.append({'time':timestamp.isoformat(),'lat':lat,'lon':lon,'sog':sog,'heading':r['Heading'],'name':r['VesselName'],'imo':r['IMO'],'distance_approx_m':dist(lat,lon)})
records.sort(key=lambda r:r['time'])
a=[r for r in records if '2024-07-16T16:00:00'<=r['time']<'2024-07-16T19:00:00']
nearest=min(a,key=lambda r:r['distance_approx_m']) if a else None
with (D/'sara_regina_20240716_1200_2200_utc_noaa_raw.csv').open(encoding='utf8',newline='') as f: orig=list(csv.DictReader(f))
keys1=[(r['time'].replace('T',' ').replace('+00:00','+00:00'),r['lat'],r['lon'],r['sog']) for r in records]
keys2=[(r['utc_iso'],float(r['LAT']),float(r['LON']),float(r['SOG'])) for r in orig]
validhead=sum(1 for r in records if r['heading'] not in ('','511','511.0') and 0<=float(r['heading'])<360)
h=hashlib.sha256()
with Z.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
result={'source':str(Z),'zip_sha256':h.hexdigest(),'raw_count':len(records),'unique_timestamps':len(set(r['time'] for r in records)),'external_extract_count':len(orig),'raw_records_equal_external_extract':keys1==keys2,'vessel_names':sorted(set(r['name'] for r in records)),'imo_values':sorted(set(r['imo'] for r in records)),'heading_valid':validhead,'broad_early_afternoon_utc_16_19_count':len(a),'nearest_raw_point':nearest,'bridge_reference':[25.76992,-80.18997],'bridge_reference_source':'https://bridgereportcard.com/bridge/fl-870759/','event_source':'https://twitter.com/USCGSoutheast/status/1813348095775666612','event_reprint':'https://www.local10.com/news/local/2024/07/17/fdot-inspects-bridge-struck-by-cargo-ship-in-miamis-brickell-area/','warnings':['The source reports early Tuesday afternoon, not an exact collision time.','The approximate bridge point is not the location of impact.','The extracted period includes pre-event and post-event records; no individual observation is an event label.','The Miami case is not one of the prior eight-area selected candidate samples.']}
(O/'sara_independent.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))

