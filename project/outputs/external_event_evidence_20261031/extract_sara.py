from pathlib import Path
import csv,zipfile,hashlib,json
from datetime import datetime,timezone
import pandas as pd
import pyarrow.parquet as pq
root=Path('/mnt/sda/WangZijia/AIS')
out=root/'outputs/external_event_evidence_20261031'
out.mkdir(parents=True,exist_ok=True)
zfile=root/'data/raw/noaa/AIS_2024_07_16.zip'
pfile=root/'data/interim/ais_miami_enriched.parquet'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
# Deliberately broad analyst review interval; event minute not established.
start=pd.Timestamp('2024-07-16 12:00:00',tz='UTC');end=pd.Timestamp('2024-07-16 22:00:00',tz='UTC')
records=[]
with zipfile.ZipFile(zfile) as z:
 names=[n for n in z.namelist() if n.lower().endswith('.csv')]
 assert len(names)==1,names
 import io
 with z.open(names[0]) as b:
  for row in csv.DictReader(io.TextIOWrapper(b,encoding='utf-8-sig',newline='')):
   if row['MMSI']!='334979000':continue
   t=pd.to_datetime(row['BaseDateTime'],utc=True)
   if start<=t<=end: records.append(row)
raw=pd.DataFrame(records)
assert not raw.empty
assert set(raw.VesselName.str.strip().str.upper())=={'SARA REGINA'}
assert set(raw.IMO.str.strip().str.upper())=={'IMO9142655'}
assert not raw.BaseDateTime.duplicated().any()
raw=raw.sort_values('BaseDateTime').reset_index(drop=True)
raw.insert(2,'utc_iso',pd.to_datetime(raw.BaseDateTime,utc=True).astype(str))
raw['Heading_is_valid']=pd.to_numeric(raw.Heading,errors='coerce').between(0,359,inclusive='both')
raw['gap_prev_s']=pd.to_datetime(raw.BaseDateTime,utc=True).diff().dt.total_seconds()
raw.to_csv(out/'sara_regina_20240716_1200_2200_utc_noaa_raw.csv',index=False)
d=pq.read_table(pfile,columns=['mmsi','base_datetime','lat','lon','sog','cog','heading']).to_pandas()
d=d[(d.mmsi==334979000)&(d.base_datetime>=start)&(d.base_datetime<=end)].copy().sort_values('base_datetime')
joined=raw.merge(d,left_on='utc_iso',right_on=d.base_datetime.astype(str),how='left',validate='one_to_one',indicator=True)
# Approximate OSM point, not official collision coordinates or exact bridge geometry.
import math
latref=25.7699;lonref=-80.19006
lat=pd.to_numeric(raw.LAT);lon=pd.to_numeric(raw.LON)
raw['distance_to_osm_reference_m']=((lat-latref)**2+(lon-lonref)**2*(math.cos(math.radians(latref))**2))**.5*111195
k=int(raw['distance_to_osm_reference_m'].idxmin());near=raw.loc[k]
mask=raw.utc_iso.between('2024-07-16 16:00:00+00:00','2024-07-16 19:00:00+00:00')
summary={'source_noaa_zip':str(zfile),'source_noaa_zip_sha256':sha(zfile),'source_parquet':str(pfile),'source_parquet_sha256':sha(pfile),'zip_csv_member':names[0],'raw_extraction_interval_utc':[start.isoformat(),end.isoformat()],'records':len(raw),'heading_valid_records':int(raw.Heading_is_valid.sum()),'name_values':sorted(raw.VesselName.unique().tolist()),'imo_values':sorted(raw.IMO.unique().tolist()),'mmsi':'334979000','unique_raw_utc':int(raw.BaseDateTime.nunique()),'parquet_records':len(d),'raw_to_parquet_matched':int((joined._merge=='both').sum()),'first_utc':str(raw.utc_iso.iloc[0]),'last_utc':str(raw.utc_iso.iloc[-1]),'report_possible_notification_range_utc':'Approx 2024-07-16 16:00–19:00 UTC (12:00–15:00 EDT); descriptive reporting, not exact event interval', 'broad_afternoon_records':int(mask.sum()),'gap_seconds':{'median':float(raw.gap_prev_s.median()),'p90':float(raw.gap_prev_s.quantile(.9)),'max':float(raw.gap_prev_s.max())},'osm_reference_point':[latref,lonref],'osm_reference_url':'https://mapcarta.com/20643382','nearest_raw_report':{'time_utc':str(near.utc_iso),'latitude':float(near.LAT),'longitude':float(near.LON),'sog_kn':float(near.SOG),'cog_deg':float(near.COG),'distance_m_approx':float(near.distance_to_osm_reference_m)},'event_primary_social_url':'https://twitter.com/USCGSoutheast/status/1813348095775666612','event_accessible_source':'https://www.local10.com/news/local/2024/07/17/fdot-inspects-bridge-struck-by-cargo-ship-in-miamis-brickell-area/','official_vessel_registry':'https://www.uscg.mil/Portals/0/NPFC/COFR/COFR%20Vsl%20Status%20Report.pdf?ver=a7P-d1NFN4Ii_zidczDduw%3D%3D','limitations':['Event exact time and geographic collision point are not specified by the quoted official notice.','OSM reference is an approximate bridge coordinate; distance is not distance to the impact point.','Heading values are source-reported sentinel 511 where invalid; do not treat as heading=0.','Miami source is not one of the eight train/validation regions or the existing 297 selected candidate windows.','A verified event does not determine which AIS reports are anomalous, nor does it identify low-speed intent.']}
(out/'sara_regina_case_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
